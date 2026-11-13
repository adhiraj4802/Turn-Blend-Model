"""Smart Turn v3.2 with a shorter input window: keep the last N mel frames and the matching positional rows."""
import numpy as np, onnx
from onnx import numpy_helper

def make_short(src, dst, secs, keep="last"):
    frames = int(round(secs * 100)); assert frames % 2 == 0
    m = onnx.load(src); g = m.graph
    g.input[0].type.tensor_type.shape.dim[2].dim_value = frames
    for i in g.initializer:
        if i.name == "inner.encoder.embed_positions.weight":
            a = numpy_helper.to_array(i)
            a = a[-frames // 2:] if keep == "last" else a[:frames // 2]
            i.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a), i.name))
        if i.name == "val_21":
            i.CopyFrom(numpy_helper.from_array(np.array([frames // 2], np.int64), i.name))
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] == "val_21":
            for a in n.attribute:
                if a.name == "value": a.t.CopyFrom(numpy_helper.from_array(np.array([frames // 2], np.int64)))
    del g.value_info[:]
    onnx.save(m, dst)
    return dst
