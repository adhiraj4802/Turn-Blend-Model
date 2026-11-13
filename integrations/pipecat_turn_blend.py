"""
turn-blend for Pipecat (works with the Daily transport or any other Pipecat transport).

Drop-in end-of-turn detection: a Pipecat turn analyzer backed by turn-blend.onnx.

    from pipecat_turn_blend import TurnBlendAnalyzer

    aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),                      # starts turns, detects interruptions
        user_turn_strategies=UserTurnStrategies(stop=[
            TurnAnalyzerUserTurnStopStrategy(turn_analyzer=TurnBlendAnalyzer("turn-blend.onnx")),
        ]),
    ))
    PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000))

Files needed next to your bot: this file, turn_blend.py and turn-blend.onnx.
Requires: pipecat-ai, numpy, onnxruntime. Written against pipecat-ai 1.16.

How it fits Pipecat: Pipecat's own VAD still opens the user turn and handles interruptions. This analyzer
receives every audio frame, runs the model on a worker thread, and reports the turn COMPLETE when the blend
commits (or the silence timeout runs out). Pipecat then waits for the final transcript and starts the LLM.
"""

import queue
import threading
import time
from collections.abc import Callable

import numpy as np
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.metrics.metrics import MetricsData, TurnMetricsData

from turn_blend import TurnBlend


class TurnBlendParams(BaseTurnParams):
    stop_secs: float = 3.0   # the silence timeout, reported to Pipecat


class TurnBlendAnalyzer(BaseTurnAnalyzer):
    def __init__(self, model_path: str = "turn-blend.onnx", *, threads: int = 4,
                 on_end_of_turn: Callable[[dict], None] | None = None, **config):
        """
        model_path:      path to turn-blend.onnx
        threads:         CPU threads for the model (2 or more for real-time use)
        on_end_of_turn:  optional callback with the result of each ended turn (called on the worker thread)
        config:          any turn-blend setting, e.g. blend_threshold=0.9, blend_min_silence_ms=256
        """
        super().__init__(sample_rate=16000)
        self.detector = TurnBlend(model_path, threads=threads, **config)
        self.last_result: dict | None = None      # details of the most recent end of turn
        self._on_end = on_end_of_turn
        self._pcm = np.zeros(0, np.float32)
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._complete: TurnMetricsData | None = None
        self._warned = False
        threading.Thread(target=self._worker, name="turn-blend", daemon=True).start()

    # ------------------------------------------------------------ Pipecat interface ---
    @property
    def speech_triggered(self) -> bool:
        return self.detector.in_turn

    @property
    def params(self) -> TurnBlendParams:
        return TurnBlendParams(stop_secs=self.detector.config["timeout_secs"])

    def set_sample_rate(self, sample_rate: int):
        super().set_sample_rate(sample_rate)
        if sample_rate != 16000 and not self._warned:
            self._warned = True
            logger.error(f"turn-blend needs 16 kHz input audio but the pipeline delivers {sample_rate} Hz. "
                         "Set PipelineParams(audio_in_sample_rate=16000).")

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        pcm = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        self._pcm = np.concatenate([self._pcm, pcm])
        n = self.detector.chunk_samples
        while len(self._pcm) >= n:
            self._q.put((self._pcm[:n].copy(), time.time()))
            self._pcm = self._pcm[n:]
        return EndOfTurnState.COMPLETE if self._complete is not None else EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        with self._lock:
            done, self._complete = self._complete, None
        return (EndOfTurnState.COMPLETE, done) if done is not None else (EndOfTurnState.INCOMPLETE, None)

    def clear(self):
        self._q.put(("clear", 0.0))

    async def cleanup(self):
        self._q.put((None, 0.0))

    # ---------------------------------------------------------------- worker thread ---
    def _worker(self):
        was_in_turn, last = False, None
        while True:
            chunk, arrived = self._q.get()
            if chunk is None:
                return
            if isinstance(chunk, str):
                # Pipecat ended the turn. If ours is still open and the user is not mid-speech, close it too.
                if self.detector.in_turn and last is not None and not last["speech"]:
                    self.detector.close_turn()
                    was_in_turn = False
                continue
            last = r = self.detector(chunk)
            if r["in_turn"] and not was_in_turn:
                with self._lock:
                    self._complete = None          # a verdict from before this speech is stale
            was_in_turn = bool(r["in_turn"])
            if r["end_of_turn"]:
                lag_ms = (time.time() - arrived) * 1000
                result = {
                    "reason": "commit" if r["end_of_turn"] == 1 else "timeout",
                    "silence_ms": round(r["silence_ms"]),
                    "detect_ms": round(r["silence_ms"] + lag_ms),   # last speech -> decision, including compute
                    "blend": round(r["blend"], 3), "smart_turn": round(r["smart_turn"], 3), "mini": round(r["mini"], 3),
                }
                self.last_result = result
                logger.debug(f"turn-blend end of turn: {result}")
                with self._lock:
                    self._complete = TurnMetricsData(processor="TurnBlendAnalyzer", model="turn-blend", is_complete=True,
                                                     probability=max(r["blend"], 0.0),
                                                     e2e_processing_time_ms=float(result["detect_ms"]))
                if self._on_end:
                    self._on_end(result)
