"""
Streaming runner for p99lab turn-1-mini v1 (official ONNX graph, Apache-2.0).
https://huggingface.co/p99lab/turn-1-mini  -  "turn-1-mini by p99lab"

The graph takes 160 ms of raw 16 kHz audio (its 4 kHz low-pass and log-mel are inside) plus its state
tensors and returns 8 values of P(end of turn), one per 20 ms frame. English only.
"""

import numpy as np

STEP = 2560  # 160 ms @ 16 kHz
REPO, FILE = "p99lab/turn-1-mini", "onnx/turn-1-mini.step.onnx"


class Turn1Mini:
    def __init__(self, session):
        self.sess = session
        self.names = [i.name for i in session.get_inputs() if i.name != "audio"]
        self.outputs = ["probs"] + ["new_" + n for n in self.names]
        self.reset()

    def reset(self):
        self.state = {i.name: np.zeros(i.shape, np.float32) for i in self.sess.get_inputs() if i.name != "audio"}
        self.pending = np.zeros(0, np.float32)
        self.last = 0.0

    def _run(self, audio):
        res = self.sess.run(self.outputs, {"audio": audio.reshape(1, STEP), **self.state})
        return res[0], dict(zip(self.names, res[1:]))

    def push(self, chunk):
        """Feed audio of any length; the model advances every time 160 ms have accumulated."""
        self.pending = np.concatenate([self.pending, np.asarray(chunk, np.float32)])
        while len(self.pending) >= STEP:
            probs, self.state = self._run(self.pending[:STEP])
            self.pending = self.pending[STEP:]
            self.last = float(probs[-1])

    def score(self):
        """P(end of turn) for the audio heard so far, between step boundaries: run the step on the pending
        audio padded with zeros, on a copy of the state (the model card's own recipe). Only frames made
        entirely of real audio are read."""
        n = len(self.pending)
        full = n // 320
        if full == 0:
            return self.last
        probs, _ = self._run(np.concatenate([self.pending, np.zeros(STEP - n, np.float32)]))
        return float(probs[full - 1])
