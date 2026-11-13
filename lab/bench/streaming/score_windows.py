"""Smart Turn at every non-speech in-turn chunk of eot-bench en, for several window lengths.
short[W]: shortened model (W-second input). pad[W]: stock model, last W seconds zero-padded to 8 s."""
import io, sys, time, pickle, collections, numpy as np, pyarrow.parquet as pq, soundfile as sf, onnxruntime as ort
sys.path.insert(0, sys.argv[1])
import server as sv
rows = {r["id"]: r for r in pq.read_table(sys.argv[2]).to_pylist()}
D = pickle.load(open(sys.argv[3], "rb"))
SHORT = [6, 4, 3, 2, 1]; PAD = [4, 2]
fe = sv.session("features", "trt")
def sess(path):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"])
short = {w: sess(f"models/smart-turn-v3.2-{w}s.onnx") for w in SHORT}
full = sess("models/smart-turn-v3.2-gpu.onnx")
START, PRE = 6, 22; out = {}; t0 = time.time(); nrun = 0; chk = []
for n, d in enumerate(D):
    r = rows[d["id"]]
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    need = int((d["last_word_end"] + 3.3) * 16000) - len(a)
    if need > 0: a = np.concatenate([a, np.zeros(need, np.float32)])
    pv = d["vad"]; nch = len(pv)
    res = {("s", w): np.full(nch, np.nan, np.float32) for w in SHORT}; res.update({("p", w): np.full(nch, np.nan, np.float32) for w in PAD})
    res["full"] = np.full(nch, np.nan, np.float32)
    srun = 0; started = False; turn = []; pre = collections.deque(maxlen=PRE)
    for i in range(nch):
        c = a[i*512:(i+1)*512]
        srun = srun + 1 if pv[i] >= 0.7 else 0
        if started: turn.append(c); turn = turn[-251:]
        else: pre.append(c)
        if not started and srun >= START: started = True; turn = list(pre)
        if started and pv[i] < 0.7:
            audio = np.concatenate(turn)[-8 * 16000:]
            for w in sorted(set(SHORT + PAD + [8])):
                seg = audio[-w * 16000:]
                buf = np.zeros(8 * 16000, np.float32); buf[-len(seg):] = seg
                f = fe.run(None, {"audio": buf[None]})[0]
                if w == 8:
                    res["full"][i] = full.run(None, {"input_features": f})[0].item()
                if w in PAD: res[("p", w)][i] = full.run(None, {"input_features": f})[0].item()
                if w in SHORT: res[("s", w)][i] = short[w].run(None, {"input_features": np.ascontiguousarray(f[:, :, -w * 100:])})[0].item()
                nrun += 1
    ok = ~np.isnan(res["full"])
    if ok.any(): chk.append(np.abs(res["full"][ok] - d["st"][ok]).max())
    out[d["id"]] = res
    if n % 25 == 0: print(n, "turns", nrun, "feature runs", "%.0f s" % (time.time() - t0), "max diff vs cached 8 s score %.3f" % max(chk), flush=True)
pickle.dump(out, open(sys.argv[4], "wb"))
print("done", nrun, "%.0f s" % (time.time() - t0), "max diff vs cached %.3f" % max(chk))
