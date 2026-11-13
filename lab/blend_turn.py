"""
Our VAD and the turn-blend end-of-turn detector as Pipecat analyzers.

LabSileroVADAnalyzer  starts user turns and detects interruptions (the demo's Silero VAD).
BlendTurnAnalyzer     ends user turns.

Pipecat hands every incoming audio frame to BlendTurnAnalyzer.append_audio(). The audio is cut into 32 ms
chunks and scored on a worker thread with the same steps as the demo server (server.py): Silero VAD on
every chunk, and, while a turn is open, Smart Turn v3.2 + turn-1-mini + elapsed silence blended into one
probability. When the commit rule fires (or the silence timeout runs out) append_audio() reports COMPLETE.

The turn analyzer runs its own Silero VAD because the blend needs a speech / no-speech label for every
chunk, and Pipecat only passes its debounced "user is speaking" state.

Every setting is read from console.SETTINGS on each chunk, so edits made in the settings console apply
to the call in progress.
"""

import queue
import threading
import time
from collections import deque

import numpy as np
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.metrics.metrics import MetricsData, TurnMetricsData

import console
import mini as turn_mini
import server as lab

RATE, CHUNK, CHUNK_MS = lab.RATE, lab.CHUNK, lab.CHUNK_MS
S = console.SETTINGS
BACKEND, FE_BACKEND, THREADS = lab.DEFAULTS["st_backend"], lab.DEFAULTS["fe_backend"], lab.DEFAULTS["st_threads"]


def _vad_params():
    return VADParams(confidence=S["vad_confidence"], start_secs=S["vad_start_secs"],
                     stop_secs=S["vad_stop_secs"], min_volume=0.0)


class LabSileroVADAnalyzer(VADAnalyzer):
    """The demo's Silero VAD (static 16 kHz graph, 32 ms chunks) as Pipecat's VAD, with no volume gate."""

    def __init__(self):
        super().__init__(sample_rate=RATE, params=_vad_params())
        self._vad = lab.SileroVAD("cpu")
        console.register_vad(self)

    def apply_settings(self):
        if self.sample_rate:   # set once the pipeline has started
            self.set_params(_vad_params())
        else:
            self._params = _vad_params()

    def num_frames_required(self) -> int:
        return CHUNK

    def voice_confidence(self, buffer: bytes) -> float:
        chunk = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        return float(self._vad.prob(chunk))


class BlendTurnParams(BaseTurnParams):
    """What Pipecat is told about this analyzer. The live values are in console.SETTINGS."""

    stop_secs: float = lab.DEFAULTS["st_stop_secs"]


class BlendTurnAnalyzer(BaseTurnAnalyzer):
    def __init__(self):
        super().__init__(sample_rate=RATE)
        # load every model now, so the first turn doesn't pay for it
        lab.session("smart_turn", BACKEND, lab.model_threads(BACKEND, THREADS))
        lab.session("features", FE_BACKEND)
        self._vad = lab.SileroVAD("cpu")
        self._mini = turn_mini.Turn1Mini(lab.session("mini", "cpu"))

        self._pcm = np.zeros(0, np.float32)   # audio not yet cut into chunks
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._complete: TurnMetricsData | None = None   # set by the worker, consumed by Pipecat
        self.last_delay_ms: float | None = None         # last speech -> end-of-turn decision, latest turn
        self._in_turn = False
        self._warned_rate = False
        self._thread = threading.Thread(target=self._worker, name="blend-turn", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ Pipecat interface ---
    @property
    def speech_triggered(self) -> bool:
        return self._in_turn

    @property
    def params(self) -> BlendTurnParams:
        return BlendTurnParams(stop_secs=S["stop_secs"])

    def set_sample_rate(self, sample_rate: int):
        super().set_sample_rate(sample_rate)
        if sample_rate != RATE and not self._warned_rate:
            self._warned_rate = True
            logger.error(f"BlendTurnAnalyzer needs {RATE} Hz input audio, the pipeline delivers {sample_rate} Hz. "
                         f"Set PipelineParams(audio_in_sample_rate={RATE}).")

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        pcm = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        self._pcm = np.concatenate([self._pcm, pcm])
        while len(self._pcm) >= CHUNK:
            self._q.put((self._pcm[:CHUNK].copy(), time.time()))
            self._pcm = self._pcm[CHUNK:]
        return EndOfTurnState.COMPLETE if self._complete is not None else EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        with self._lock:
            done, self._complete = self._complete, None
        return (EndOfTurnState.COMPLETE, done) if done is not None else (EndOfTurnState.INCOMPLETE, None)

    def clear(self):
        self._q.put(("clear", time.time()))

    async def cleanup(self):
        self._q.put((None, 0.0))

    # ---------------------------------------------------------------- worker thread ---
    def _worker(self):
        pre, turn = deque(), []
        speech_run = silence = stream_i = pos_run = cancelled = 0
        tentative = None         # silence (ms) at which a tentative end was marked, if any

        def close_turn():
            nonlocal turn, pos_run, stream_i, tentative, cancelled
            self._in_turn = False
            turn, pos_run, stream_i, tentative, cancelled = [], 0, 0, None, 0

        while True:
            chunk, arrived = self._q.get()
            if chunk is None:
                return
            if isinstance(chunk, str):   # clear(): Pipecat ended the turn; keep it only if the user is mid-speech
                if self._in_turn and silence > 0:
                    close_turn()
                continue

            # settings are read fresh on every chunk, so console edits apply immediately
            start_chunks = max(1, round(S["vad_start_secs"] * 1000 / CHUNK_MS))
            pre_len = max(1, int(np.ceil(S["pre_speech_ms"] / CHUNK_MS))) + start_chunks
            max_keep = int(8 * RATE / CHUNK) + 1

            is_speech = self._vad.prob(chunk) >= S["vad_confidence"]
            self._mini.push(chunk)   # turn-1-mini hears everything, so its context is warm when a turn opens
            if is_speech:
                speech_run, silence = speech_run + 1, 0
            else:
                speech_run, silence = 0, silence + 1

            if not self._in_turn:
                pre.append(chunk)
                while len(pre) > pre_len:
                    pre.popleft()
                if speech_run >= start_chunks:
                    self._in_turn = True
                    turn = list(pre)
                    pre.clear()
                    with self._lock:
                        self._complete = None   # a verdict from before this speech is stale
                continue

            turn.append(chunk)
            if len(turn) > max_keep:
                del turn[: len(turn) - max_keep]
            silent_ms = silence * CHUNK_MS

            if is_speech:
                pos_run = 0
                if tentative is not None:
                    tentative, cancelled = None, cancelled + 1
                    logger.info("turn-blend tentative end cancelled: the user kept talking")
            stream_i += 1
            if not is_speech and stream_i % max(1, S["score_every_chunks"]) == 0:
                t0 = time.perf_counter()
                p_st = lab.predict_endpoint(np.concatenate(turn), BACKEND, THREADS, S["max_duration_secs"], FE_BACKEND)[0]
                p_mini = self._mini.score()
                blend = lab.blend_prob(p_st, p_mini, silent_ms / 1000)
                score_ms = (time.perf_counter() - t0) * 1000
                if tentative is None and blend > S["tent_threshold"] and silent_ms >= S["tent_min_silence_ms"]:
                    tentative = silent_ms
                ok = blend > S["blend_threshold"] and silent_ms >= S["blend_min_silence_ms"]
                pos_run = pos_run + 1 if ok else 0
                if pos_run >= S["blend_confirm_chunks"]:
                    self._finish("commit", arrived, silent_ms, blend, p_st, p_mini, score_ms, cancelled, tentative)
                    close_turn()
                    continue

            if silent_ms >= S["stop_secs"] * 1000:
                self._finish("timeout", arrived, silent_ms, None, None, None, 0.0, cancelled, tentative)
                close_turn()

    def _finish(self, reason, arrived, silent_ms, blend, p_st, p_mini, score_ms, cancelled, tentative):
        now = time.time()
        lag_ms = (now - arrived) * 1000
        delay_ms = silent_ms + lag_ms   # audio time since the last speech chunk + how far behind the worker runs

        def r3(v):
            return None if v is None else round(float(v), 3)

        result = dict(reason=reason, silence_ms=round(silent_ms), detect_ms=round(delay_ms), blend=r3(blend),
                      smart_turn=r3(p_st), mini=r3(p_mini), score_ms=round(score_ms, 1), worker_lag_ms=round(lag_ms, 1),
                      tentative_at_ms=None if tentative is None else round(tentative), tentatives_cancelled=cancelled)
        logger.info("turn-blend end of turn: " + ", ".join(f"{k}={v}" for k, v in result.items()))
        console.record("end_of_turn", **result)
        self.last_delay_ms = float(delay_ms)
        with self._lock:
            self._complete = TurnMetricsData(processor="BlendTurnAnalyzer", model="turn-blend", is_complete=True,
                                             probability=float(blend or 0.0), e2e_processing_time_ms=float(delay_ms))
