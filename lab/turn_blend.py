"""
Runner for turn-blend.onnx, the all-in-one voice activity + end-of-turn model.

    from turn_blend import TurnBlend
    detector = TurnBlend("turn-blend.onnx")
    for chunk in chunks_of_512_samples:          # float32, 16 kHz mono, values in [-1, 1]
        r = detector(chunk)
        if r["end_of_turn"]:
            ...                                   # the speaker has finished

Needs only numpy and onnxruntime. The file holds every model and the turn logic; this class just carries the
state vector from one call to the next.
"""

import json

import numpy as np
import onnxruntime as ort


class TurnBlend:
    def __init__(self, path="turn-blend.onnx", threads=6, providers=("CPUExecutionProvider",), **config):
        """config: any of the names in `self.defaults` (for example blend_threshold=0.9)."""
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), so, providers=list(providers))
        meta = self.session.get_modelmeta().custom_metadata_map
        self.sample_rate = int(meta["sample_rate"])
        self.chunk_samples = int(meta["chunk_samples"])
        self.state_size = int(meta["state_size"])
        self.outputs = json.loads(meta["outputs"])
        spec = json.loads(meta["config"])
        self.names = [c["name"] for c in spec]
        self.defaults = {c["name"]: c["default"] for c in spec}
        self.components = json.loads(meta["components"])
        self._tail = json.loads(meta.get("state_tail", "[]"))
        self.config = dict(self.defaults)
        self.configure(**config)
        self.reset()

    def configure(self, **config):
        """Change settings; takes effect on the next chunk."""
        unknown = set(config) - set(self.names)
        if unknown:
            raise KeyError(f"unknown setting(s) {sorted(unknown)}; valid: {self.names}")
        self.config.update(config)
        self._config = np.array([self.config[n] for n in self.names], np.float32)

    def reset(self):
        """Start a new audio stream."""
        self.state = np.zeros(self.state_size, np.float32)

    def new_stream(self):
        """A detector for another audio stream that shares this one's loaded model (and its settings)."""
        other = object.__new__(TurnBlend)
        other.__dict__.update(self.__dict__)
        other.config = dict(self.config)
        other._config = self._config.copy()
        other.reset()
        return other

    @property
    def in_turn(self):
        """True while a user turn is open."""
        return bool(self._tail) and self.state[self._tail.index("in_turn") - len(self._tail)] > 0.5

    def close_turn(self):
        """End the open turn from outside (for example when your framework ended it for another reason).

        The models keep their audio context; only the turn bookkeeping is cleared, so the next speech opens a
        new turn."""
        for name in ("in_turn", "turn_chunks", "stream_i", "pos_run", "tentative"):
            self.state[self._tail.index(name) - len(self._tail)] = 0.0

    def __call__(self, chunk):
        """Feed one chunk of `chunk_samples` float32 samples. Returns a dict:

        vad          Silero speech probability for this chunk
        speech       1.0 if the chunk counts as speech
        in_turn      1.0 while a user turn is open (after this chunk)
        silence_ms   silence since the last speech chunk
        smart_turn   Smart Turn probability, or -1 if it was not scored on this chunk
        mini         turn-1-mini probability
        blend        blended end-of-turn probability, or -1 if not scored on this chunk
        tentative    1.0 while a tentative end of turn is pending
        end_of_turn  0 = no, 1 = the blend committed, 2 = silence timeout
        """
        chunk = np.asarray(chunk, np.float32).reshape(-1)
        if len(chunk) != self.chunk_samples:
            raise ValueError(f"expected {self.chunk_samples} samples, got {len(chunk)}")
        *values, self.state = self.session.run(None, {"audio": chunk, "state": self.state, "config": self._config})
        return {name: float(v[0]) for name, v in zip(self.outputs, values)}
