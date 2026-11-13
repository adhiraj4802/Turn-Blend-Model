import sys, time, io, numpy as np, onnxruntime as ort, wave
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import server as sv
from short_model import make_short
w = wave.open(sys.argv[3]); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
fe = sv.session("features", "cpu")
def feats(audio, secs):
    audio = audio[-int(secs * 16000):]
    buf = np.zeros(8 * 16000, np.float32); buf[-len(audio):] = audio
    return fe.run(None, {"audio": buf[None]})[0]
def sess(path, threads):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1; so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
full = "models/smart-turn-v3.2-gpu.onnx"
ends = [int(x * 16000) for x in (6.2, 9.5, 14.0, 21.3, 30.1, 38.0)]
print("window  frames | padded-8s score vs short-model score at 6 clip positions")
ref = sess(full, 6)
for secs in (8, 6, 4, 3, 2, 1.5, 1):
    path = full if secs == 8 else make_short(full, f"models/smart-turn-v3.2-{secs}s.onnx", secs)
    row = []
    for e in ends:
        f = feats(a[:e], secs)
        p_pad = ref.run(None, {"input_features": f})[0].item()
        p_short = sess(path, 6).run(None, {"input_features": f[:, :, -int(secs * 100):]})[0].item()
        row.append(f"{p_pad:.2f}/{p_short:.2f}")
    t = {}
    for th in (1, 2, 6):
        s = sess(path, th); x = f[:, :, -int(secs * 100):]
        for _ in range(5): s.run(None, {"input_features": x})
        t0 = time.perf_counter()
        for _ in range(40): s.run(None, {"input_features": x})
        t[th] = (time.perf_counter() - t0) / 40 * 1000
    print(f"{secs:>4}s  {int(secs*100):>4}  | " + "  ".join(row) + f" | ms/run 1 thread {t[1]:.1f}, 2 threads {t[2]:.1f}, 6 threads {t[6]:.1f}")
