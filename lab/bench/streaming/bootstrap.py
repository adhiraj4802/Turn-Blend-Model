"""Paired bootstrap over turns: how much do the official metrics move between a baseline and other variants?"""
import sys, json, os, time, numpy as np, pandas as pd
from eot_harness.metrics import compute_metrics_from_predictions, get_pareto_operating_points, load_predictions
d, B = sys.argv[1], int(sys.argv[2]); wanted = sys.argv[3:]          # first name is the baseline
index = {v: k for k, v in json.load(open(os.path.join(d, "index.json"))).items()}
def find(prefix):
    sp = None
    if prefix.endswith("@0.2"): prefix, sp = prefix[:-4], 0.2
    hit = [v for v in index if v[2:].startswith(prefix)]
    assert len(hit) == 1, (prefix, hit)
    return hit[0][2:] + (" [single shot at 0.2 s]" if sp else ""), load_predictions(os.path.join(d, index[hit[0]])), sp
runs = [find(w) for w in wanted]
ids = sorted(runs[0][1]["id"].unique()); groups = [{i: g for i, g in df.groupby("id", sort=False)} for _, df, _ in runs]
def metrics(df, sp):
    tr, _ = compute_metrics_from_predictions(df, score_point_s=sp)
    lat = get_pareto_operating_points(tr, "mean_latency", [0.3, 0.6]); cut = get_pareto_operating_points(tr, "cutoff_rate", [0.05, 0.10])
    return np.array([100 * lat.value[0], 100 * lat.value[1], 1000 * cut.value[0], 1000 * cut.value[1]])
rng = np.random.default_rng(0); res = np.zeros((B, len(runs), 4)); t0 = time.time()
for b in range(B):
    pick = rng.choice(len(ids), len(ids))
    for r, (name, _, sp) in enumerate(runs):
        parts = []
        for j, k in enumerate(pick):
            g = groups[r][ids[k]].copy(); g["id"] = f"b{j}"; parts.append(g)
        res[b, r] = metrics(pd.concat(parts, ignore_index=True), sp)
    if b % 20 == 0: print(b, "%.0f s" % (time.time() - t0), flush=True)
np.save(os.path.join(d, "boot_" + str(abs(hash(tuple(wanted))) % 10**6) + ".npy"), res)
cols = ["cut@300 (pts)", "cut@600 (pts)", "lat@5% (ms)", "lat@10% (ms)"]
print("\nbaseline:", runs[0][0], "| bootstrap", B, "resamples of the 400 turns")
print("  90% interval of the baseline itself: " + "; ".join(f"{c} {np.nanpercentile(res[:, 0, i], 5):.1f}..{np.nanpercentile(res[:, 0, i], 95):.1f}" for i, c in enumerate(cols)))
for r in range(1, len(runs)):
    diff = res[:, r] - res[:, 0]
    print("vs", runs[r][0])
    for i, c in enumerate(cols):
        lo, hi = np.nanpercentile(diff[:, i], [5, 95])
        print(f"    {c:14s} mean change {np.nanmean(diff[:, i]):+7.1f}   90% interval {lo:+7.1f} .. {hi:+7.1f}   {'CLEAR' if lo > 0 or hi < 0 else 'not distinguishable from noise'}")
