"""
How fast is turn-blend's own decision rule?

Runs turn-blend.onnx with its default settings over every English turn of LiveKit's eot-bench and records
when it ends each turn relative to the true end of the last word (from the dataset's word timings).

    python benchmark/rule_latency.py path/to/en/validation-00000-of-00001.parquet [part parts]
    python benchmark/rule_latency.py --summary

Needs: numpy onnxruntime pyarrow soundfile. The parquet file comes from
https://huggingface.co/datasets/livekit/eot-bench-data (config "en", split "validation").
"""
import glob, io, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "results" / "rule_latency"


def run(parquet, part=0, parts=1):
    import pyarrow.parquet as pq
    import soundfile as sf
    from turn_blend import TurnBlend
    det = TurnBlend(str(HERE.parent / "turn-blend.onnx"), threads=6)
    res, t0 = [], time.time()
    for idx, r in enumerate(pq.read_table(parquet).to_pylist()):
        if idx % parts != part:
            continue
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000
        if a.ndim > 1:
            a = a.mean(1)
        last = r["words"][-1]["end"]
        need = int((last + 3.5) * 16000) - len(a)
        if need > 0:
            a = np.concatenate([a, np.zeros(need, np.float32)])      # room for the 3 s timeout
        det.reset()
        opened, tent_at, ends, heavy, light = False, None, [], [], []
        for i in range(len(a) // 512):
            t1 = time.perf_counter()
            o = det(a[i * 512:(i + 1) * 512])
            (heavy if o["smart_turn"] >= 0 else light).append((time.perf_counter() - t1) * 1000)
            t = (i + 1) * 0.032
            opened = opened or bool(o["in_turn"])
            if o["tentative"] and tent_at is None:
                tent_at = t
            if not o["tentative"] and not o["end_of_turn"]:
                tent_at = None
            if o["end_of_turn"]:
                ends.append({"t": round(t, 3), "how": int(o["end_of_turn"]), "tentative_t": tent_at and round(tent_at, 3),
                             "silence_ms": round(o["silence_ms"])})
                tent_at = None
                if t >= last - 0.03:
                    break
        res.append({"id": r["id"], "last_word_end": last, "opened": opened, "ends": ends,
                    "ms_scored": round(float(np.median(heavy)), 2) if heavy else None,
                    "ms_plain": round(float(np.median(light)), 2), "n_scored": len(heavy), "n_plain": len(light)})
        if len(res) % 40 == 0:
            print(part, len(res), "%.0f s" % (time.time() - t0), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(res, open(f"{OUT}.{part}.json", "w"))
    print("done", part, len(res), "%.0f s" % (time.time() - t0))


def summary():
    rows = [r for f in sorted(glob.glob(f"{OUT}.*.json")) for r in json.load(open(f))]
    n = len(rows)
    cut = [r for r in rows if any(e["t"] < r["last_word_end"] - 0.03 for e in r["ends"])]
    final = [(r, r["ends"][-1]) for r in rows if r["ends"] and r["ends"][-1]["t"] >= r["last_word_end"] - 0.03]
    delay = np.array([(e["t"] - r["last_word_end"]) * 1000 for r, e in final])
    clean = np.array([(e["t"] - r["last_word_end"]) * 1000 for r, e in final if r not in cut])
    tent = np.array([(e["tentative_t"] - r["last_word_end"]) * 1000 for r, e in final if e["tentative_t"]])
    lead = np.array([(e["t"] - e["tentative_t"]) * 1000 for r, e in final if e["tentative_t"]])
    timeouts = sum(e["how"] == 2 for _, e in final)
    q = lambda a, p: float(np.percentile(a, p))
    out = {
        "turns": n, "turn_opened": sum(r["opened"] for r in rows),
        "ended_early_pct": round(100 * len(cut) / n, 1),
        "final_end_by_timeout_pct": round(100 * timeouts / n, 1),
        "delay_ms_all": {"mean": round(delay.mean()), "median": round(q(delay, 50)), "p75": round(q(delay, 75)), "p90": round(q(delay, 90)), "p95": round(q(delay, 95))},
        "delay_ms_turns_not_cut": {"n": len(clean), "mean": round(clean.mean()), "median": round(q(clean, 50)), "p90": round(q(clean, 90))},
        "within_ms_pct": {str(b): round(100 * float((delay <= b).mean()), 1) for b in (400, 500, 600, 800, 1000, 1500)},
        "tentative": {"turns_with_tentative_pct": round(100 * len(tent) / n, 1), "tentative_delay_ms_median": round(q(tent, 50)),
                      "lead_over_commit_ms_median": round(q(lead, 50)), "lead_over_commit_ms_mean": round(lead.mean())},
        "compute_ms_per_chunk": {"plain_median": round(float(np.median([r["ms_plain"] for r in rows])), 2),
                                 "scored_median": round(float(np.median([r["ms_scored"] for r in rows if r["ms_scored"]])), 2),
                                 "share_of_chunks_scored_pct": round(100 * sum(r["n_scored"] for r in rows) / sum(r["n_scored"] + r["n_plain"] for r in rows), 1)},
    }
    json.dump(out, open(HERE / "results" / "rule_latency_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1], *(int(v) for v in sys.argv[2:4]))
