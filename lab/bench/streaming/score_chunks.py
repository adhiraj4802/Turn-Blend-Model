"""Per-chunk Silero + Smart Turn scores for every eot-bench English turn (streaming, from VAD speech start)."""
import io, sys, time, collections, pickle, numpy as np, pyarrow.parquet as pq, soundfile as sf
import server as sv
t = pq.read_table(sys.argv[1]).to_pylist()
START = 6; PRE = 22; out = []; t0 = time.time(); nrun = 0
for n, r in enumerate(t):
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    assert sr == 16000
    if a.ndim > 1: a = a.mean(1)
    last = r["words"][-1]["end"]
    need = int((last + 3.3) * 16000) - len(a)
    if need > 0: a = np.concatenate([a, np.zeros(need, np.float32)])     # room for a 3 s timeout
    vad = sv.SileroVAD("cpu"); vad.last_reset = 1e18                       # no periodic reset: deterministic
    nch = len(a) // 512
    pv = np.zeros(nch, np.float32); ps = np.full(nch, np.nan, np.float32)
    srun = 0; started = False; turn = []; pre = collections.deque(maxlen=PRE)
    for i in range(nch):
        c = a[i*512:(i+1)*512]; pv[i] = vad.prob(c)
        srun = srun + 1 if pv[i] >= 0.7 else 0
        if started: turn.append(c); turn = turn[-251:]
        else: pre.append(c)
        if not started and srun >= START: started = True; turn = list(pre)
        if started:
            ps[i] = sv.predict_endpoint(np.concatenate(turn), "trt", 1, 8, "trt")[0]; nrun += 1
    out.append({"id": r["id"], "vad": pv, "st": ps, "last_word_end": last, "first_word_start": r["words"][0]["start"],
                "spans": [(s["start"], s["end"]) for s in r["silence_spans"]], "orig_dur": r["duration"]})
    if n % 40 == 0: print(n, "turns", nrun, "runs", "%.0f s" % (time.time() - t0), flush=True)
pickle.dump(out, open(sys.argv[2], "wb"))
print("done", len(out), "turns", nrun, "smart turn runs in %.0f s" % (time.time() - t0))
