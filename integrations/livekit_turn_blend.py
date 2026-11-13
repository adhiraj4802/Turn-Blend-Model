"""
turn-blend for LiveKit Agents.

Drop-in voice activity + end-of-turn detection: a LiveKit VAD backed by turn-blend.onnx. It reports
START_OF_SPEECH when a user turn opens and END_OF_SPEECH only when turn-blend decides the turn is over
(not at every pause), so LiveKit's "vad" turn detection ends turns exactly where the model does.

    from livekit_turn_blend import TurnBlendVAD

    session = AgentSession(
        stt=..., llm=..., tts=...,
        vad=TurnBlendVAD("turn-blend.onnx"),
        turn_handling={
            "turn_detection": "vad",                                      # end turns on our END_OF_SPEECH
            "endpointing": {"mode": "fixed", "min_delay": 0.0, "max_delay": 3.0},   # no extra wait on top
        },
    )

Files needed next to your agent: this file, turn_blend.py and turn-blend.onnx.
Requires: livekit-agents, numpy, onnxruntime. Written against livekit-agents 1.8.

This does not use LiveKit's own turn-detector models.
"""

import asyncio
import time

import numpy as np
from livekit import rtc
from livekit.agents import vad

from turn_blend import TurnBlend

RATE = 16000
MAX_SPEECH_SECS = 60.0     # audio of the current turn kept for the END_OF_SPEECH event


class TurnBlendVAD(vad.VAD):
    def __init__(self, model_path: str = "turn-blend.onnx", *, threads: int = 4, **config):
        """
        model_path:  path to turn-blend.onnx
        threads:     CPU threads for the model (2 or more for real-time use)
        config:      any turn-blend setting, e.g. blend_threshold=0.9, blend_min_silence_ms=256
        """
        self._detector = TurnBlend(model_path, threads=threads, **config)
        super().__init__(capabilities=vad.VADCapabilities(
            update_interval=self._detector.chunk_samples / self._detector.sample_rate))

    @property
    def model(self) -> str:
        return "turn-blend"

    @property
    def provider(self) -> str:
        return "turn-blend"

    @property
    def detector(self) -> TurnBlend:
        """The loaded model; call .configure(...) on it to change settings for new streams."""
        return self._detector

    def stream(self) -> "TurnBlendVADStream":
        return TurnBlendVADStream(self, self._detector.new_stream())   # each stream has its own state


class TurnBlendVADStream(vad.VADStream):
    def __init__(self, owner: TurnBlendVAD, detector: TurnBlend) -> None:
        self._detector = detector
        super().__init__(owner)

    async def _main_task(self) -> None:
        det = self._detector
        n = det.chunk_samples
        window = n / RATE
        loop = asyncio.get_running_loop()
        resampler: rtc.AudioResampler | None = None
        input_rate = 0
        pending = np.zeros(0, np.int16)          # 16 kHz samples not yet scored
        speech = np.zeros(0, np.int16)           # audio of the current turn (with a little before it)
        speaking = False
        samples = 0
        speech_secs = silence_secs = raw_speech = 0.0

        def frame_of(data: np.ndarray) -> rtc.AudioFrame:
            return rtc.AudioFrame(data=data.tobytes(), sample_rate=RATE, num_channels=1, samples_per_channel=len(data))

        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):        # hard segment boundary: start fresh
                det.reset()
                resampler = None
                input_rate = 0
                pending, speech = np.zeros(0, np.int16), np.zeros(0, np.int16)
                speaking, samples, speech_secs, silence_secs, raw_speech = False, 0, 0.0, 0.0, 0.0
                continue
            if not isinstance(item, rtc.AudioFrame):
                continue

            if not input_rate:
                input_rate = item.sample_rate
                if input_rate != RATE:
                    resampler = rtc.AudioResampler(input_rate=input_rate, output_rate=RATE,
                                                   quality=rtc.AudioResamplerQuality.QUICK)
            frames = resampler.push(item) if resampler is not None else [item]
            for f in frames:
                data = np.frombuffer(f.data, dtype=np.int16)
                if f.num_channels > 1:
                    data = data.reshape(-1, f.num_channels).mean(axis=1).astype(np.int16)
                pending = np.concatenate([pending, data])

            while len(pending) >= n:
                chunk, pending = pending[:n], pending[n:]
                t0 = time.perf_counter()
                r = await loop.run_in_executor(None, det, chunk.astype(np.float32) / 32768.0)
                took = time.perf_counter() - t0
                samples += n

                keep = int((MAX_SPEECH_SECS if speaking else 0.7) * RATE)
                speech = np.concatenate([speech, chunk])[-keep:]
                raw_speech = raw_speech + window if r["speech"] else 0.0
                raw_silence = r["silence_ms"] / 1000.0

                def event(kind: vad.VADEventType, frames_out: list[rtc.AudioFrame]) -> vad.VADEvent:
                    return vad.VADEvent(
                        type=kind, samples_index=samples, timestamp=samples / RATE,
                        speech_duration=speech_secs, silence_duration=silence_secs, frames=frames_out,
                        probability=r["vad"], inference_duration=took, speaking=speaking,
                        raw_accumulated_silence=raw_silence, raw_accumulated_speech=raw_speech)

                if r["in_turn"] and not speaking:                       # a user turn opened
                    speaking = True
                    speech_secs, silence_secs = raw_speech, 0.0
                    self._event_ch.send_nowait(event(vad.VADEventType.START_OF_SPEECH, [frame_of(speech)]))
                elif speaking:
                    speech_secs += window
                else:
                    silence_secs += window

                self._event_ch.send_nowait(event(vad.VADEventType.INFERENCE_DONE, [frame_of(chunk)]))

                if r["end_of_turn"] and speaking:                       # turn-blend ended the turn
                    speaking = False
                    silence_secs = raw_silence                          # time since the last speech
                    self._event_ch.send_nowait(event(vad.VADEventType.END_OF_SPEECH, [frame_of(speech)]))
                    speech = speech[-int(0.7 * RATE):]
                    speech_secs = 0.0
