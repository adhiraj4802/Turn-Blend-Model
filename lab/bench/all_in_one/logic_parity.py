"""All-in-one ONNX with its default settings vs a plain Python port of the demo server's blend-mode loop
(server.py WebSocket handler), chunk by chunk, on multi-turn audio."""
import io, sys, math, wave, numpy as np, pyarrow.parquet as pq, soundfile as sf
from collections import deque
sys.path.insert(0, sys.argv[1])
import server as lab, mini as turn_mini
from turn_blend import TurnBlend
CH, MS = lab.CHUNK, lab.CHUNK_MS

def reference(a, cfg):
    vad_sess = lab.session("vad", "cpu"); state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros((1, 64), np.float32)
    tm = turn_mini.Turn1Mini(lab.session("mini", "cpu"))
    start_chunks = max(1, round(cfg["vad_start_secs"] * 1000 / MS)); pre_len = max(1, math.ceil(cfg["st_pre_speech_ms"] / MS)) + start_chunks
    speech_run = since = 0; in_turn = False; turn = []; pre = deque(); stream_i = pos_run = 0; tent = False; out = []
    for i in range(len(a) // CH):
        chunk = a[i * CH:(i + 1) * CH]
        x = np.concatenate((ctx, chunk.reshape(1, -1)), axis=1); o, state = vad_sess.run(None, {"input": x, "state": state}); ctx = x[:, -64:]
        if (i + 1) % round(5000 / MS) == 0: state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros((1, 64), np.float32)
        p = float(np.ravel(o)[0]); rms = float(np.sqrt(np.mean(chunk * chunk))); db = 20 * math.log10(rms) if rms > 1e-8 else -120.0
        is_speech = p >= cfg["vad_confidence"] and db >= cfg["vad_min_db"]
        if is_speech: speech_run += 1; since = 0
        else: speech_run = 0; since += 1
        if in_turn:
            turn.append(chunk); turn = turn[-250:]
        else:
            pre.append(chunk)
            while len(pre) > pre_len: pre.popleft()
        tm.push(chunk)
        if not in_turn and speech_run >= start_chunks:
            in_turn = True; stream_i = pos_run = 0; tent = False; turn = list(pre); pre.clear()
        blend = -1.0; end = 0
        if in_turn:
            stream_i += 1
            if stream_i % cfg["stream_every_chunks"] == 0:
                silent_ms = since * MS
                if is_speech:
                    positive = False; tent = False
                else:
                    p_st = lab.predict_endpoint(np.concatenate(turn), "cpu", 6, cfg["st_max_duration_secs"], "cpu")[0]
                    blend = lab.blend_prob(p_st, tm.score(), silent_ms / 1000)
                    if not tent and blend > cfg["tent_threshold"] and silent_ms >= cfg["tent_min_silence_ms"]: tent = True
                    positive = blend > cfg["blend_threshold"]
                pos_run = pos_run + 1 if (positive and silent_ms >= cfg["blend_min_silence_ms"]) else 0
                if pos_run >= cfg["blend_confirm_chunks"]: end = 1
            if not end and since * MS >= cfg["st_stop_secs"] * 1000: end = 2
        out.append((float(in_turn and not end), blend, float(tent), float(end)))
        if end: in_turn = False; turn = []; tent = False
    return out

rows = pq.read_table(sys.argv[2]).to_pylist()
def audio(r):
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    return a.mean(1) if a.ndim > 1 else a
w = wave.open(sys.argv[3]); clip = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
streams = {"lecture clip, 60 s": np.concatenate([np.zeros(8000, np.float32), clip[:16000 * 60], np.zeros(16000 * 4, np.float32)])}
for k in range(4):   # ten benchmark turns back to back, with 1.5 s of silence between them
    streams[f"10 benchmark turns in a row (set {k + 1})"] = np.concatenate([np.concatenate([audio(r), np.zeros(24000, np.float32)]) for r in rows[k * 40:k * 40 + 10]])
det = TurnBlend(sys.argv[1] + "/models/turn-blend.onnx"); total = bad = 0
for name, a in streams.items():
    ref = reference(a, dict(lab.DEFAULTS)); det.reset(); got = []
    for i in range(len(a) // CH):
        r = det(a[i * CH:(i + 1) * CH]); got.append((r["in_turn"], r["blend"], r["tentative"], r["end_of_turn"]))
    R, G = np.array(ref), np.array(got)
    mism = int((R[:, 0] != G[:, 0]).sum() + (R[:, 2] != G[:, 2]).sum() + (R[:, 3] != G[:, 3]).sum() + ((R[:, 1] < 0) != (G[:, 1] < 0)).sum())
    sc = R[:, 1] >= 0; total += len(R); bad += mism
    print(f"{name}: {len(R)} chunks, {int((R[:, 3] > 0).sum())} turn ends (commit {int((R[:, 3] == 1).sum())}, timeout {int((R[:, 3] == 2).sum())}), {int(sc.sum())} scored | flag mismatches {mism} | max blend diff {np.abs(R[sc, 1] - G[sc, 1]).max() if sc.any() and mism == 0 else float('nan'):.1e}")
print("TOTAL", total, "chunks,", bad, "mismatches")
