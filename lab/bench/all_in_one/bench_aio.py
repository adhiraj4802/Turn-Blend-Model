"""Scores every eot-bench English turn with the single all-in-one ONNX and writes the per-turn files the
eot-bench adapter reads (same format as the server-based run, same grid)."""
import io, sys, json, os, time, numpy as np, pyarrow.parquet as pq, soundfile as sf
repo, parquet, old_cache, new_cache, part, parts = sys.argv[1:7]; part, parts = int(part), int(parts)
sys.path.insert(0, repo)
from turn_blend import TurnBlend
det = TurnBlend(repo + "/models/turn-blend.onnx", threads=6, score_every_chunks=1, blend_threshold=2.0, timeout_secs=1e9)
weights = {k: det.defaults["w_" + k] for k in ("bias", "smart_turn", "mini", "log_silence")}
os.makedirs(new_cache, exist_ok=True); t0 = time.time(); n = 0
for idx, r in enumerate(pq.read_table(parquet).to_pylist()):
    if idx % parts != part: continue
    name = str(r["id"]).replace("/", "_") + ".json"
    grid = json.load(open(os.path.join(old_cache, name)))
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32"); assert sr == 16000
    if a.ndim > 1: a = a.mean(1)
    det.reset(); out = [det(a[i * 512:(i + 1) * 512]) for i in range(len(a) // 512)]
    for row in grid["rows"]:
        k = min(len(out), int(np.floor(row["timestamp"] * 16000 + 1e-6)) // 512) - 1
        o = out[k] if k >= 0 else None
        ok = bool(o and o["blend"] >= 0)
        row["point"] = {"t": row["timestamp"], "smart_turn": o["smart_turn"] if ok else 0.0, "mini": o["mini"] if o else 0.0,
                        "silence_s": o["silence_ms"] / 1000 if o else 0.0, "started": bool(o and o["in_turn"]),
                        "decidable": ok, "blend": o["blend"] if ok else 0.0}
    grid["weights"] = weights
    json.dump(grid, open(os.path.join(new_cache, name), "w")); n += 1
    if n % 20 == 0: print(part, n, "%.0f s" % (time.time() - t0), flush=True)
print("done", part, n, "%.0f s" % (time.time() - t0))
