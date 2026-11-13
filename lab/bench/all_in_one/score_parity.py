"""All-in-one ONNX vs the reference Python pipeline (server.bench_turn) on benchmark turns."""
import io, sys, time, numpy as np, pyarrow.parquet as pq, soundfile as sf
sys.path.insert(0, sys.argv[1])
import server as lab
from turn_blend import TurnBlend
rows = pq.read_table(sys.argv[2]).to_pylist(); N = int(sys.argv[3])
det = TurnBlend(sys.argv[1] + "/models/turn-blend.onnx", score_every_chunks=1, blend_threshold=2.0, timeout_secs=1e9)
diffs = {"smart_turn": [], "mini": [], "blend": []}; flags = 0; npts = 0; t_ref = t_aio = 0.0; nchunks = 0
for r in rows[:: max(1, len(rows) // N)][:N]:
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    ts = [round(x, 1) for x in np.arange(0.3, len(a) / 16000, 0.1)]
    t0 = time.time(); ref = lab.bench_turn(a, ts, dict(lab.DEFAULTS)); t_ref += time.time() - t0
    det.reset(); out = []; t0 = time.time()
    for i in range(len(a) // 512): out.append(det(a[i * 512:(i + 1) * 512]))
    t_aio += time.time() - t0; nchunks += len(out)
    for p in ref:
        k = min(len(out), int(np.floor(p["t"] * 16000 + 1e-6)) // 512) - 1
        o = out[k] if k >= 0 else None
        started = bool(o and o["in_turn"]); decidable = bool(o and o["blend"] >= 0); npts += 1
        if started != p["started"] or decidable != p["decidable"]: flags += 1; continue
        diffs["mini"].append(abs(o["mini"] - p["mini"]))
        if decidable:
            diffs["smart_turn"].append(abs(o["smart_turn"] - p["smart_turn"])); diffs["blend"].append(abs(o["blend"] - p["blend"]))
print(f"{N} turns, {npts} timestamps, {nchunks} chunks")
print(f"started/decidable flags that differ: {flags}")
for k, v in diffs.items(): print(f"  {k:10s} compared {len(v):5d}  max abs diff {max(v):.2e}  mean {np.mean(v):.2e}")
print(f"all-in-one: {1000 * t_aio / nchunks:.2f} ms per chunk on average (reference took {t_ref:.0f} s, all-in-one {t_aio:.0f} s)")
