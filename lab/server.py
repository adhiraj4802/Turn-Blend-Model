"""
turn-blend: end-of-turn detection from Silero VAD + Smart Turn v3.2 + turn-1-mini, with a browser demo.

The browser streams 16 kHz mono mic audio (512-sample float32 chunks) over a WebSocket. The server runs
Silero VAD on every chunk and, while a turn is open, scores it with Smart Turn v3.2 and turn-1-mini. A
logistic blend of the two scores and the elapsed silence decides when the turn has ended (see BLEND).
Everything runs through onnxruntime, on the CPU by default; CUDA and TensorRT are optional.

Run:  ./run.sh   ->  http://localhost:8000   (PORT=8001 ./run.sh for another port)
"""

import asyncio
import json
import math
import os
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download

import mini as turn_mini

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

RATE = 16000
CHUNK = 512  # Silero VAD needs exactly 512 samples @ 16 kHz (32 ms)
CHUNK_MS = CHUNK / RATE * 1000.0
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad_16k_op15.onnx"
SMART_TURN_REPO, SMART_TURN_FILE = "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx"  # fp32 weights


# ------------------------------------------------------------ backends ---
try:
    ort.preload_dlls()  # load the pip-installed CUDA/cuDNN libraries (onnxruntime-gpu)
except Exception:
    pass
GPU_OK = "CUDAExecutionProvider" in ort.get_available_providers()


def _load_tensorrt():
    """TensorRT 10 libs from the tensorrt-cu13 pip package."""
    if not GPU_OK or "TensorrtExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        import ctypes
        import tensorrt_libs
        d = Path(tensorrt_libs.__file__).parent
        for name in ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]:
            ctypes.CDLL(str(d / name), mode=ctypes.RTLD_GLOBAL)
        return True
    except Exception:
        return False


TRT_OK = _load_tensorrt()
TRT_CACHE = MODELS / "trt_cache"  # built engines are cached here (first build takes ~1 min)
BACKENDS = (["trt"] if TRT_OK else []) + (["gpu"] if GPU_OK else []) + ["cpu"]
# Smart Turn also has a half-precision TensorRT engine. Not offered for the VAD or the feature
# graph: fp16 buys nothing on the tiny VAD and overflows to inf in the feature graph.
ST_BACKENDS = (["trt", "trt-fp16"] if TRT_OK else []) + (["gpu"] if GPU_OK else []) + ["cpu"]
BEST = BACKENDS[0]


def providers(backend):
    """trt = TensorRT engine, gpu = CUDA provider, cpu = CPU provider."""
    if backend in ("trt", "trt-fp16") and TRT_OK:
        TRT_CACHE.mkdir(exist_ok=True)
        return [("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(TRT_CACHE),
            "trt_fp16_enable": backend == "trt-fp16",
        }), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if backend == "gpu" and GPU_OK:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


DEFAULTS = {
    # --- VAD (same meaning as Pipecat VADParams) ---
    "vad_confidence": 0.7,   # Silero speech-probability threshold
    "vad_start_secs": 0.2,   # speech must persist this long -> "speech started"
    "vad_stop_secs": 0.2,    # silence must persist this long -> "speech stopped" (triggers Smart Turn)
    "vad_min_db": -80.0,     # chunks quieter than this (RMS dBFS) never count as speech
    "vad_backend": "cpu",    # fastest for a model this small (0.09 ms vs 0.19 ms on TensorRT)
    # --- Smart Turn (same meaning as Pipecat SmartTurnParams) ---
    "st_threshold": 0.5,     # probability above which the turn is "complete"
    "st_stop_secs": 3.0,     # silence after an "incomplete" verdict that force-ends the turn
    "st_pre_speech_ms": 500, # audio kept from before speech start
    "st_max_duration_secs": 8.0,  # audio window given to the model (model max = 8)
    "st_backend": "cpu",     # all models on CPU by default; trt / gpu stay selectable on the page
    # --- when Smart Turn runs ---
    # "vad_stop": once, when the VAD reports speech stopped (Pipecat's way).
    # "streaming": on every chunk from speech start; a turn ends when the verdict is positive during silence.
    # "blend": streaming, but the decision uses BLEND below (Smart Turn + turn-1-mini + elapsed silence).
    "st_mode": "blend",
    "blend_threshold": 0.85,        # commit: blended probability above this ...
    "blend_min_silence_ms": 320,    # ... after at least this much silence ...
    "blend_confirm_chunks": 2,      # ... for this many chunks in a row
    "tent_threshold": 0.5,          # tentative end of turn (where an agent could start its LLM, output held)
    "tent_min_silence_ms": 160,
    "stream_every_chunks": 2,       # run Smart Turn every N chunks (1 = every 32 ms; CPU cannot keep up with 1)
    "stream_min_silence_ms": 64,    # a positive only counts after this much silence (positives mid-speech are unreliable)
    "stream_confirm_chunks": 2,     # consecutive counted positives needed to end the turn
    "st_threads": 6,         # onnxruntime intra-op threads, CPU backend only
    "fe_backend": "cpu",     # log-mel feature extraction graph (features.py)
}


# Blend of the two detectors, fitted by logistic regression on the 400 English turns of LiveKit's eot-bench
# (every non-speech chunk inside a turn; label = "the turn is really over"):
#   p = sigmoid(bias + a * logit(P_smart_turn) + b * logit(P_turn_1_mini) + c * ln(silence in seconds))
# Cross-validated on that set: 15% of turns cut off early at ~600 ms mean delay with the default commit rule,
# against 30% / ~1100 ms for Smart Turn once at VAD stop. English only (turn-1-mini is English only).
BLEND = {"bias": 2.8007, "smart_turn": 0.2939, "mini": 0.4873, "log_silence": 1.2880}


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def blend_prob(p_st, p_mini, silence_s):
    z = BLEND["bias"] + BLEND["smart_turn"] * _logit(p_st) + BLEND["mini"] * _logit(p_mini) \
        + BLEND["log_silence"] * math.log(max(silence_s, CHUNK / RATE))
    return 1 / (1 + math.exp(-z))


# --------------------------------------------------------------- models ---
def silero_path():
    """Silero VAD v6.2, 16 kHz graph, with shapes fixed and its control flow folded away.

    The published graph has dynamic shapes and If nodes that TensorRT cannot parse. Fixing the input to
    one 576-sample window (64 context + 512 new), making the sample rate a constant and letting
    onnxruntime constant-fold leaves a plain static graph (Conv / LSTM / Sigmoid) with identical output.
    """
    dst = MODELS / "silero_vad_16k_static.onnx"
    if dst.exists():
        return dst
    import onnx
    from onnx import numpy_helper
    src = MODELS / "silero_vad_16k_op15.onnx"
    if not src.exists():
        print("Downloading Silero VAD...")
        for attempt in range(4):   # GitHub sometimes stalls; never wait on it forever
            try:
                with urllib.request.urlopen(SILERO_URL, timeout=20) as r:
                    data = r.read()
                src.write_bytes(data)
                break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"Could not download Silero VAD from {SILERO_URL}: {e}") from e
                print(f"  retrying ({e})")
    m = onnx.load(str(src))
    g = m.graph

    def set_shape(vi, shape):
        dims = vi.type.tensor_type.shape
        del dims.dim[:]
        for s in shape:
            dims.dim.add().dim_value = s

    shapes = {"input": [1, 64 + CHUNK], "state": [2, 1, 128], "output": [1, 1], "stateN": [2, 1, 128]}
    for vi in list(g.input) + list(g.output):
        if vi.name in shapes:
            set_shape(vi, shapes[vi.name])
    g.input.remove([i for i in g.input if i.name == "sr"][0])
    g.initializer.append(numpy_helper.from_array(np.array(RATE, dtype=np.int64), "sr"))
    tmp = MODELS / "_silero_tmp.onnx"
    onnx.save(m, str(tmp))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    so.log_severity_level = 3
    ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    tmp.unlink()
    return dst


def features_path():
    """Whisper log-mel extraction as an ONNX graph (see features.py)."""
    dst = MODELS / "whisper_features_8s.onnx"
    if not dst.exists():
        from transformers import WhisperFeatureExtractor
        from features import build_feature_model
        build_feature_model(dst, WhisperFeatureExtractor(chunk_length=8).mel_filters)
    return dst


_sessions = {}


def session(kind, backend, threads=1):
    """One cached onnxruntime session per (model, backend, threads). kind: vad | features | smart_turn."""
    key = (kind, backend, threads)
    if key in _sessions:
        return _sessions[key]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = threads
    if kind == "vad":
        path = silero_path()
        feed = {"input": np.zeros((1, 64 + CHUNK), np.float32), "state": np.zeros((2, 1, 128), np.float32)}
    elif kind == "mini":
        path = hf_hub_download(turn_mini.REPO, turn_mini.FILE, local_dir=MODELS / "p99")
        feed = None
    elif kind == "features":
        path = features_path()
        feed = {"audio": np.zeros((1, 8 * RATE), np.float32)}
    else:
        path = hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_dir=MODELS)
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        feed = {"input_features": np.zeros((1, 80, 800), np.float32)}
    s = ort.InferenceSession(str(path), sess_options=so, providers=providers(backend))
    if feed is None:
        turn_mini.Turn1Mini(s).push(np.zeros(turn_mini.STEP * 3, np.float32))
    for _ in range(3 if feed else 0):  # warm-up (first GPU runs are slow; TensorRT builds its engine here)
        s.run(None, feed)
    _sessions[key] = s
    return s


def model_threads(backend, threads):
    return threads if backend == "cpu" else 1


def load_models(cfg):
    """Load (or build) every session the config needs; returns the VAD session."""
    session("smart_turn", cfg["st_backend"], model_threads(cfg["st_backend"], cfg["st_threads"]))
    session("features", cfg["fe_backend"])
    if cfg["st_mode"] == "blend":
        session("mini", "cpu")
    return session("vad", cfg["vad_backend"])


class SileroVAD:
    def __init__(self, backend):
        self.session = session("vad", backend)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.last_reset = time.monotonic()

    def prob(self, chunk):
        x = np.concatenate((self.context, chunk.reshape(1, -1)), axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state})
        self.context = x[:, -64:]
        if time.monotonic() - self.last_reset >= 5.0:  # same periodic reset as upstream/Pipecat
            self.reset()
        return float(np.ravel(out)[0])


def predict_endpoint(audio, backend, threads, max_secs, fe_backend):
    """Same pre-processing as upstream inference.py; returns probability + timings."""
    model = session("smart_turn", backend, model_threads(backend, threads))
    fe = session("features", fe_backend)
    window = int(min(max(max_secs, 0.5), 8.0) * RATE)
    audio = audio[-window:]
    full = 8 * RATE
    if len(audio) < full:  # zero-pad at the beginning, audio sits at the end
        audio = np.pad(audio, (full - len(audio), 0))

    t0 = time.perf_counter()
    feats = fe.run(None, {"audio": audio[None].astype(np.float32)})[0]
    t1 = time.perf_counter()
    prob = float(np.ravel(model.run(None, {"input_features": feats})[0])[0])
    t2 = time.perf_counter()
    return prob, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


# -------------------------------------------------------------------- app ---
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/info")
def info():
    return {"defaults": DEFAULTS, "backends": BACKENDS, "st_backends": ST_BACKENDS, "chunk_ms": CHUNK_MS,
            "tensorrt": TRT_OK, "cuda": GPU_OK}


@app.get("/api/benchmark")
def benchmark(st_backend: str = BEST, fe_backend: str = BEST, vad_backend: str = BEST, threads: int = 4, n: int = 30):
    """Raw speed of the three graphs on this machine."""
    if st_backend not in ST_BACKENDS or fe_backend not in BACKENDS or vad_backend not in BACKENDS:
        return {"error": "unknown backend"}
    threads = max(1, min(threads, 16))
    n = max(1, min(n, 200))
    audio = (np.random.randn(8 * RATE) * 0.05).astype(np.float32)
    predict_endpoint(audio, st_backend, threads, 8, fe_backend)
    fe, inf, vad = [], [], []
    for _ in range(n):
        _, a, b = predict_endpoint(audio, st_backend, threads, 8, fe_backend)
        fe.append(a)
        inf.append(b)
    v = SileroVAD(vad_backend)
    for i in range(n * 4):
        t0 = time.perf_counter()
        v.prob(audio[i * CHUNK:(i + 1) * CHUNK])
        vad.append((time.perf_counter() - t0) * 1000.0)
    stat = lambda x: {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95))}
    return {"st_backend": st_backend, "fe_backend": fe_backend, "vad_backend": vad_backend, "n": n,
            "smart_turn_ms": stat(inf), "feature_ms": stat(fe), "vad_ms": stat(vad)}


def bench_turn(audio, timestamps, cfg):
    """Benchmark hook: run the live pipeline over one whole turn and report what it knows at each timestamp.

    Same steps as the WebSocket loop, in audio time: Silero on every 32 ms chunk (state reset every 5 s),
    turn start after vad_start_secs of speech with the pre-speech buffer, turn-1-mini fed every chunk,
    Smart Turn on the turn so far (last 8 s), blend from both scores and the silence since the last speech
    chunk. A timestamp is answered from the last complete chunk at or before it, which is all the live
    system has at that moment. The blend is only defined while a turn is open and the VAD hears no speech.
    """
    vad_sess = session("vad", cfg["vad_backend"])
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, 64), np.float32)
    tm = turn_mini.Turn1Mini(session("mini", "cpu"))
    start_chunks = max(1, round(cfg["vad_start_secs"] * 1000 / CHUNK_MS))
    pre_len = max(1, math.ceil(cfg["st_pre_speech_ms"] / CHUNK_MS)) + start_chunks
    reset_every = round(5.0 * 1000 / CHUNK_MS)
    n_chunks = len(audio) // CHUNK
    speech_run, since, started, turn_start, done = 0, 0, False, 0, 0
    out = []
    for t in sorted(timestamps):
        target = min(n_chunks, int(math.floor(t * RATE + 1e-6)) // CHUNK)
        while done < target:
            chunk = audio[done * CHUNK:(done + 1) * CHUNK]
            x = np.concatenate((ctx, chunk.reshape(1, -1)), axis=1)
            o, state = vad_sess.run(None, {"input": x, "state": state})
            ctx = x[:, -64:]
            p = float(np.ravel(o)[0])
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            db = 20 * math.log10(rms) if rms > 1e-8 else -120.0
            if p >= cfg["vad_confidence"] and db >= cfg["vad_min_db"]:
                speech_run += 1
                since = 0
            else:
                speech_run = 0
                since += 1
            tm.push(chunk)
            if not started and speech_run >= start_chunks:
                started = True
                turn_start = max(0, done - pre_len + 1) * CHUNK
            done += 1
            if done % reset_every == 0:
                state = np.zeros((2, 1, 128), np.float32)
                ctx = np.zeros((1, 64), np.float32)
        end = done * CHUNK
        seg = audio[turn_start if started else 0:end]
        p_st = predict_endpoint(seg, cfg["st_backend"], cfg["st_threads"], cfg["st_max_duration_secs"],
                                cfg["fe_backend"])[0] if len(seg) else 0.0
        p_mini = tm.score()
        silent_s = since * CHUNK / RATE
        decidable = started and since > 0
        out.append({"t": t, "smart_turn": p_st, "mini": p_mini, "silence_s": silent_s, "started": started,
                    "decidable": decidable,
                    "blend": blend_prob(p_st, p_mini, silent_s) if decidable else 0.0})
    return out


@app.post("/api/bench")
async def bench(request: Request):
    """Body: JSON {"timestamps": [...]} in the X-Bench header-free form: raw float32 PCM (16 kHz mono) in the
    body, timestamps as a comma-separated `t` query parameter."""
    ts = [float(v) for v in request.query_params.get("t", "").split(",") if v]
    audio = np.frombuffer(await request.body(), dtype="<f4")
    cfg = dict(DEFAULTS)
    return {"config": {k: cfg[k] for k in ("vad_confidence", "vad_start_secs", "st_pre_speech_ms")},
            "blend_weights": BLEND, "points": await asyncio.to_thread(bench_turn, audio, ts, cfg)}


def clean_config(cfg, incoming):
    for k, v in incoming.items():
        if k not in DEFAULTS:
            continue
        if k == "st_backend":
            if v in ST_BACKENDS:
                cfg[k] = v
        elif k in ("vad_backend", "fe_backend"):
            if v in BACKENDS:
                cfg[k] = v
        elif k == "st_mode":
            if v in ("vad_stop", "streaming", "blend"):
                cfg[k] = v
        elif k in ("stream_every_chunks", "stream_confirm_chunks", "blend_confirm_chunks"):
            cfg[k] = max(1, min(int(v), 20))
        elif k == "st_threads":
            cfg[k] = max(1, min(int(v), 16))
        else:
            cfg[k] = float(v)
    return cfg


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cfg = dict(DEFAULTS)
    vad = SileroVAD(cfg["vad_backend"])

    speaking = False          # VAD state after start/stop debouncing
    speech_run = 0            # consecutive speech chunks
    silence_run = 0           # consecutive non-speech chunks
    in_turn = False
    turn = []                 # audio chunks of the current user turn
    pre = deque()             # rolling pre-speech buffer
    last_speech_seq = -1      # last chunk classified as speech
    silence_since_speech = 0  # chunks since last speech chunk, within a turn
    awaiting_timeout = False  # Smart Turn said "incomplete"; waiting for more speech or st_stop_secs
    turn_id = 0
    runs_in_turn = 0
    # streaming mode
    stream_i = 0              # chunks since the turn started
    pos_run = 0               # consecutive positive verdicts that counted (during silence)
    mid_pos = 0               # positive verdicts while the VAD still heard speech (ignored; reported)
    st_prob = None            # latest streaming probability
    # blend mode
    mini = None               # turn-1-mini runner (created when blend mode is on)
    mini_prob = bl_prob = None
    tent_seq = None           # chunk where the current tentative end of turn started
    tent_cancelled = 0        # tentatives cancelled by resumed speech in this turn

    async def send(obj):
        await ws.send_text(json.dumps(obj))

    def end_turn():
        nonlocal in_turn, turn, awaiting_timeout, st_prob, mini_prob, bl_prob, tent_seq
        st_prob = mini_prob = bl_prob = tent_seq = None
        in_turn = False
        awaiting_timeout = False
        turn = []

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "config":
                    clean_config(cfg, data.get("config", {}))
                    # load (or build) the models now so the first turn doesn't pay for it
                    vad.session = await asyncio.to_thread(load_models, cfg)
                    await send({"type": "config", "config": cfg})
                continue

            raw = msg.get("bytes")
            if not raw or len(raw) != 4 + CHUNK * 4:
                continue
            t_recv = time.perf_counter()
            seq = int.from_bytes(raw[:4], "little")
            chunk = np.frombuffer(raw, dtype="<f4", offset=4)

            # ---- VAD on this chunk
            t0 = time.perf_counter()
            p = vad.prob(chunk)
            vad_ms = (time.perf_counter() - t0) * 1000.0
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            db = 20 * math.log10(rms) if rms > 1e-8 else -120.0
            is_speech = p >= cfg["vad_confidence"] and db >= cfg["vad_min_db"]

            start_chunks = max(1, round(cfg["vad_start_secs"] * 1000 / CHUNK_MS))
            stop_chunks = max(1, round(cfg["vad_stop_secs"] * 1000 / CHUNK_MS))
            pre_len = max(1, math.ceil(cfg["st_pre_speech_ms"] / CHUNK_MS)) + start_chunks

            if is_speech:
                speech_run += 1
                silence_run = 0
                last_speech_seq = seq
                silence_since_speech = 0
            else:
                silence_run += 1
                speech_run = 0
                silence_since_speech += 1

            if in_turn:
                turn.append(chunk)
                max_keep = int(8 * RATE / CHUNK) + 1
                if len(turn) > max_keep:
                    del turn[: len(turn) - max_keep]
            else:
                pre.append(chunk)
                while len(pre) > pre_len:
                    pre.popleft()

            events = []
            blend = cfg["st_mode"] == "blend"
            streaming = blend or cfg["st_mode"] == "streaming"
            if blend:
                if mini is None:
                    mini = turn_mini.Turn1Mini(session("mini", "cpu"))
                mini.push(chunk)   # turn-1-mini hears everything, so its context is warm when a turn starts
            else:
                mini = None
            # ---- VAD start
            if not speaking and speech_run >= start_chunks:
                speaking = True
                if not in_turn:
                    in_turn = True
                    turn_id += 1
                    runs_in_turn = 0
                    stream_i = pos_run = mid_pos = 0
                    tent_seq, tent_cancelled = None, 0
                    turn = list(pre)
                    pre.clear()
                awaiting_timeout = False
                events.append({"type": "speech_start", "seq": seq, "turn": turn_id,
                               "resumed": runs_in_turn > 0})

            # ---- streaming mode: the VAD stop only updates the "speaking" display
            elif speaking and silence_run >= stop_chunks and streaming:
                speaking = False
                events.append({"type": "speech_stop", "seq": seq, "turn": turn_id,
                               "last_speech_seq": last_speech_seq})

            # ---- VAD stop -> run Smart Turn on the whole turn so far
            elif speaking and silence_run >= stop_chunks:
                speaking = False
                events.append({"type": "speech_stop", "seq": seq, "turn": turn_id,
                               "last_speech_seq": last_speech_seq})
                audio = np.concatenate(turn) if turn else np.zeros(CHUNK, dtype=np.float32)
                prob, fe_ms, inf_ms = await asyncio.to_thread(
                    predict_endpoint, audio, cfg["st_backend"], cfg["st_threads"],
                    cfg["st_max_duration_secs"], cfg["fe_backend"],
                )
                runs_in_turn += 1
                complete = prob > cfg["st_threshold"]
                events.append({
                    "type": "smart_turn", "seq": seq, "turn": turn_id, "run": runs_in_turn,
                    "last_speech_seq": last_speech_seq,
                    "probability": prob, "complete": complete,
                    "audio_secs": min(len(audio) / RATE, cfg["st_max_duration_secs"], 8.0),
                    "vad_wait_ms": (seq - last_speech_seq) * CHUNK_MS,
                    "feature_ms": fe_ms, "inference_ms": inf_ms,
                    "server_ms": (time.perf_counter() - t_recv) * 1000.0,
                    "st_backend": cfg["st_backend"], "fe_backend": cfg["fe_backend"],
                    "vad_backend": cfg["vad_backend"],
                })
                if complete:
                    events.append({"type": "turn_end", "seq": seq, "turn": turn_id, "reason": "smart_turn",
                                   "last_speech_seq": last_speech_seq})
                    end_turn()
                else:
                    awaiting_timeout = True

            # ---- streaming mode: score the turn so far on every chunk, from speech start
            if streaming and in_turn:
                stream_i += 1
                if stream_i % cfg["stream_every_chunks"] == 0:
                    prob, fe_ms, inf_ms = await asyncio.to_thread(
                        predict_endpoint, np.concatenate(turn), cfg["st_backend"], cfg["st_threads"],
                        cfg["st_max_duration_secs"], cfg["fe_backend"],
                    )
                    runs_in_turn += 1
                    st_prob = prob
                    silent_ms = silence_since_speech * CHUNK_MS
                    if prob > cfg["st_threshold"] and is_speech:
                        mid_pos += 1
                    if blend:
                        # decisions come from the blended probability, defined only while the VAD hears no speech
                        if is_speech:
                            mini_prob = bl_prob = None
                            positive = False
                            if tent_seq is not None:   # the user resumed: drop the tentative end of turn
                                tent_seq = None
                                tent_cancelled += 1
                                events.append({"type": "tentative_cancelled", "seq": seq, "turn": turn_id})
                        else:
                            mini_prob = mini.score()
                            bl_prob = blend_prob(prob, mini_prob, silent_ms / 1000)
                            if tent_seq is None and bl_prob > cfg["tent_threshold"] \
                                    and silent_ms >= cfg["tent_min_silence_ms"]:
                                tent_seq = seq
                                events.append({"type": "tentative", "seq": seq, "turn": turn_id,
                                               "last_speech_seq": last_speech_seq, "probability": bl_prob})
                            positive = bl_prob > cfg["blend_threshold"]
                        quiet = silent_ms >= cfg["blend_min_silence_ms"]
                        need = cfg["blend_confirm_chunks"]
                    else:
                        positive = prob > cfg["st_threshold"]
                        quiet = silent_ms >= cfg["stream_min_silence_ms"]
                        need = cfg["stream_confirm_chunks"]
                    pos_run = pos_run + 1 if (positive and quiet) else 0
                    if pos_run >= need:
                        events.append({
                            "type": "smart_turn", "seq": seq, "turn": turn_id, "run": runs_in_turn,
                            "streaming": True, "mid_speech_positives": mid_pos,
                            "blend": blend, "smart_turn_prob": prob, "mini_prob": mini_prob,
                            "tentative_lead_ms": (seq - tent_seq) * CHUNK_MS if tent_seq is not None else None,
                            "tentatives_cancelled": tent_cancelled,
                            "last_speech_seq": last_speech_seq,
                            "probability": bl_prob if blend else prob, "complete": True,
                            "audio_secs": min(len(turn) * CHUNK / RATE, cfg["st_max_duration_secs"], 8.0),
                            "vad_wait_ms": silence_since_speech * CHUNK_MS,
                            "feature_ms": fe_ms, "inference_ms": inf_ms,
                            "server_ms": (time.perf_counter() - t_recv) * 1000.0,
                            "st_backend": cfg["st_backend"], "fe_backend": cfg["fe_backend"],
                            "vad_backend": cfg["vad_backend"],
                        })
                        events.append({"type": "turn_end", "seq": seq, "turn": turn_id, "reason": "smart_turn",
                                       "last_speech_seq": last_speech_seq})
                        end_turn()

            # ---- no "complete" verdict and the user stayed silent: the incomplete timeout ends the turn
            if in_turn and ((awaiting_timeout and not speaking) or streaming) and \
                    silence_since_speech * CHUNK_MS >= cfg["st_stop_secs"] * 1000:
                events.append({"type": "turn_end", "seq": seq, "turn": turn_id, "reason": "timeout",
                               "last_speech_seq": last_speech_seq, "runs": runs_in_turn,
                               "mid_speech_positives": mid_pos, "tentatives_cancelled": tent_cancelled,
                               "probability": bl_prob if blend else st_prob})
                end_turn()

            await send({"type": "vad", "seq": seq, "p": round(p, 3), "db": round(db, 1),
                        "speaking": speaking, "in_turn": in_turn, "vad_ms": round(vad_ms, 3),
                        "sp": round(st_prob, 3) if (streaming and in_turn and st_prob is not None) else None,
                        "mp": round(mini_prob, 3) if (blend and in_turn and mini_prob is not None) else None,
                        "bp": round(bl_prob, 3) if (blend and in_turn and bl_prob is not None) else None})
            for e in events:
                await send(e)
    except (WebSocketDisconnect, RuntimeError):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print("Loading models (the first run downloads them, about 60 MB)...")
    load_models(DEFAULTS)
    print(f"CUDA: {GPU_OK} | TensorRT: {TRT_OK} | default backend: {DEFAULTS['st_backend']}")
    print(f"Open http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
