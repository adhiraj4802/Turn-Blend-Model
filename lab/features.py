"""
Whisper log-mel feature extraction as an ONNX graph, so it can run on the GPU
through onnxruntime's CUDA provider (no PyTorch needed).

Reproduces transformers.WhisperFeatureExtractor(chunk_length=8) with
do_normalize=True for a fixed 8 s / 16 kHz input:
  normalize -> reflect pad -> STFT (as a Conv1d with windowed DFT kernels)
  -> power -> mel filterbank -> log10 -> clamp to (max - 8) -> (x + 4) / 4
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N_FFT, HOP, N_SAMPLES = 400, 160, 8 * 16000


def build_feature_model(path, mel_filters):
    """mel_filters: (201, 80) array taken from WhisperFeatureExtractor.mel_filters."""
    n = np.arange(N_FFT)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)           # periodic Hann
    k = np.arange(N_FFT // 2 + 1)[:, None]
    ang = 2 * np.pi * k * n[None, :] / N_FFT
    kernels = np.concatenate([np.cos(ang) * window, -np.sin(ang) * window])  # (402, 400)
    nb = N_FFT // 2 + 1

    def const(name, arr, dtype=np.float32):
        return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name)

    inits = [
        const("kernels", kernels[:, None, :]),
        const("mel", mel_filters.T),                       # (80, 201)
        const("eps", 1e-7), const("floor", 1e-10), const("ln10", np.log(10.0)),
        const("eight", 8.0), const("four", 4.0),
        const("pads", [0, 0, N_FFT // 2, 0, 0, N_FFT // 2], np.int64),
        const("shape3", [1, 1, N_SAMPLES], np.int64),
        const("re_s", [0], np.int64), const("re_e", [nb], np.int64),
        const("im_s", [nb], np.int64), const("im_e", [2 * nb], np.int64),
        const("ax1", [1], np.int64), const("ax2", [2], np.int64),
        const("t_s", [0], np.int64), const("t_e", [N_SAMPLES // HOP], np.int64),
    ]
    N = helper.make_node
    nodes = [
        # zero-mean / unit-variance over the whole 8 s buffer
        N("ReduceMean", ["audio"], ["mean"], keepdims=1),
        N("Sub", ["audio", "mean"], ["centered"]),
        N("Mul", ["centered", "centered"], ["sq"]),
        N("ReduceMean", ["sq"], ["var"], keepdims=1),
        N("Add", ["var", "eps"], ["var_eps"]),
        N("Sqrt", ["var_eps"], ["std"]),
        N("Div", ["centered", "std"], ["normed"]),
        # STFT as a strided convolution
        N("Reshape", ["normed", "shape3"], ["x3"]),
        N("Pad", ["x3", "pads"], ["padded"], mode="reflect"),
        N("Conv", ["padded", "kernels"], ["stft"], strides=[HOP]),          # (1, 402, 801)
        N("Slice", ["stft", "re_s", "re_e", "ax1"], ["re"]),
        N("Slice", ["stft", "im_s", "im_e", "ax1"], ["im"]),
        N("Mul", ["re", "re"], ["re2"]),
        N("Mul", ["im", "im"], ["im2"]),
        N("Add", ["re2", "im2"], ["power"]),                                # (1, 201, 801)
        N("MatMul", ["mel", "power"], ["melspec"]),                         # (1, 80, 801)
        N("Slice", ["melspec", "t_s", "t_e", "ax2"], ["mel800"]),           # drop last frame
        N("Max", ["mel800", "floor"], ["floored"]),
        N("Log", ["floored"], ["ln"]),
        N("Div", ["ln", "ln10"], ["log10"]),
        N("ReduceMax", ["log10"], ["peak"], keepdims=1),
        N("Sub", ["peak", "eight"], ["lo"]),
        N("Max", ["log10", "lo"], ["clamped"]),
        N("Add", ["clamped", "four"], ["shifted"]),
        N("Div", ["shifted", "four"], ["input_features"]),
    ]
    graph = helper.make_graph(
        nodes, "whisper_features_8s",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, N_SAMPLES])],
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, 80, N_SAMPLES // HOP])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path
