"""turn-1-mini v1 (official ONNX) streaming scores, one per 20 ms frame, for every eot-bench English turn."""
import io, sys, pickle, time, numpy as np, pyarrow.parquet as pq, soundfile as sf, onnxruntime as ort
sys.path.insert(0, "archive")
import p99_onnx
so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
sess = ort.InferenceSession("models/p99/onnx/turn-1-mini.step.onnx", so, providers=["CPUExecutionProvider"])
out = {}; t0 = time.time()
for r in pq.read_table(sys.argv[1]).to_pylist():
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    need = int((r["words"][-1]["end"] + 3.3) * 16000) - len(a)
    if need > 0: a = np.concatenate([a, np.zeros(need, np.float32)])
    m = p99_onnx.Turn1MiniV1Onnx(sess)
    out[r["id"]] = m.push(a[: len(a) // 2560 * 2560])["eot"]
pickle.dump(out, open(sys.argv[2], "wb")); print("done", len(out), "turns in %.0f s" % (time.time() - t0))
