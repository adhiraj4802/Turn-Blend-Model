"""
Run turn-blend over a WAV file and print where each user turn ends.

    python example.py recording.wav

The file must be 16 kHz, 16-bit PCM (mono, or stereo which is mixed down).
Convert with:  ffmpeg -i input.mp3 -ar 16000 -ac 1 recording.wav
"""

import sys
import wave

import numpy as np

from turn_blend import TurnBlend


def main(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getsampwidth() != 2:
            sys.exit("Need a 16 kHz, 16-bit WAV file (see the ffmpeg line at the top of this script).")
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)

    detector = TurnBlend("turn-blend.onnx")
    n = detector.chunk_samples
    turn_started = None
    for i in range(len(audio) // n):
        was_in_turn = turn_started is not None
        r = detector(audio[i * n:(i + 1) * n])
        t = (i + 1) * n / detector.sample_rate
        if r["in_turn"] and not was_in_turn:
            turn_started = t
        if r["end_of_turn"]:
            how = "blend committed" if r["end_of_turn"] == 1 else "silence timeout"
            print(f"turn {turn_started:7.2f} s -> {t:7.2f} s   ended by {how} after {r['silence_ms']:.0f} ms of silence"
                  + (f"   (blend {r['blend']:.2f}, Smart Turn {r['smart_turn']:.2f}, turn-1-mini {r['mini']:.2f})"
                     if r["blend"] >= 0 else ""))
            turn_started = None


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
