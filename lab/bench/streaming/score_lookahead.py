"""Silence look-ahead: in the first ~200 ms of a pause, append digital silence so Smart Turn sees the ~220 ms of
trailing silence it was trained on, instead of waiting for it. Scores every in-turn chunk with 1..6 chunks of silence."""
import io, sys, time, pickle, collections, numpy as np, pyarrow.parquet as pq, soundfile as sf
sys.path.insert(0, sys.argv[1])
import server as sv
rows = {r["id"]: r for r in pq.read_table(sys.argv[2]).to_pylist()}
D = pickle.load(open(sys.argv[3], "rb"))
START, PRE, TARGET = 6, 22, 7; out = {}; t0 = time.time(); nrun = 0
for n, d in enumerate(D):
    r = rows[d["id"]]
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    need = int((d["last_word_end"] + 3.3) * 16000) - len(a)
    if need > 0: a = np.concatenate([a, np.zeros(need, np.float32)])
    pv = d["vad"]; nch = len(pv); zero = np.full(nch, np.nan, np.float32); tone = np.full(nch, np.nan, np.float32)
    srun = 0; since = 0; started = False; turn = []; pre = collections.deque(maxlen=PRE); room = None
    for i in range(nch):
        c = a[i*512:(i+1)*512]
        srun = srun + 1 if pv[i] >= 0.7 else 0
        since = 0 if pv[i] >= 0.7 else since + 1
        if started: turn.append(c); turn = turn[-251:]
        else: pre.append(c)
        if not started and srun >= START:
            started = True; turn = list(pre)
            quiet = min(list(pre)[:-START] or list(pre), key=lambda x: float(np.abs(x).mean()))   # quietest pre-speech chunk = room tone
            room = quiet
        if started and 1 <= since < TARGET:
            k = TARGET - since
            zero[i] = sv.predict_endpoint(np.concatenate(turn + [np.zeros(512 * k, np.float32)]), "trt", 1, 8, "trt")[0]
            tone[i] = sv.predict_endpoint(np.concatenate(turn + [room] * k), "trt", 1, 8, "trt")[0]
            nrun += 2
    out[d["id"]] = {"zero": zero, "tone": tone}
    if n % 50 == 0: print(n, nrun, "%.0f s" % (time.time() - t0), flush=True)
pickle.dump(out, open(sys.argv[4], "wb")); print("done", nrun, "%.0f s" % (time.time() - t0))
