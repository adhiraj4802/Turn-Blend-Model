"""Builds per-chunk probabilities for each variant (5-fold CV by turn where fitted) and writes them on the
eot-bench grid as predictions parquet files. Run in the main venv; metrics are computed by metrics_all.py."""
import pickle, sys, os, json, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
scores, minis, template, outdir = sys.argv[1:5]
windows = sys.argv[5] if len(sys.argv) > 5 and os.path.exists(sys.argv[5]) else None
D = pickle.load(open(scores, "rb")); M = pickle.load(open(minis, "rb")); W = pickle.load(open(windows, "rb")) if windows else {}
CH = 0.032
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
BASE = ["st", "mini", "log_sil"]
HIST = ["st_pre", "st_onset", "st_ema", "st_max", "st_slope", "st_prev_max", "st_prev_mean", "n_pauses", "st_turn_ema", "st_speech_mean", "log_speech_s"]
MHIST = ["mini_pre", "mini_ema", "mini_max", "mini_prev_max"]
NAMES = BASE + HIST + MHIST
for d in D:
    n = len(d["vad"]); sp = d["vad"] >= 0.7; t = (np.arange(n) + 1) * CH
    m = M[d["id"]]; mini = m[np.clip((t / 0.02).astype(int) - 1, 0, len(m) - 1)]
    started = ~np.isnan(d["st"]); st = np.where(started, d["st"], 0.0); ls, lm = lg(st), lg(mini)
    F = np.zeros((n, len(NAMES)), np.float32); S = np.zeros(n, int)
    since = 0; speech = 0; pauses = 0
    pre_st = pre_mi = -5.0; onset = -5.0; ema = ema_m = -5.0; mx = mx_m = -9.0; prev_max = prev_mean = prev_max_m = -5.0
    pause_sum = 0.0; turn_ema = -5.0; sp_sum = 0.0; sp_n = 0; recent = []; recent_m = []
    for i in range(n):
        if started[i]:
            turn_ema = ls[i] if turn_ema == -5.0 else 0.98 * turn_ema + 0.02 * ls[i]
        if sp[i]:
            if since >= 3:      # a real pause just ended and the speaker continued
                pauses += 1; prev_max = max(prev_max, mx) if pauses > 1 else mx; prev_mean = pause_sum / since
                prev_max_m = max(prev_max_m, mx_m) if pauses > 1 else mx_m
            since = 0; speech += 1
            if started[i]:
                sp_sum += ls[i]; sp_n += 1; recent.append(ls[i]); recent = recent[-10:]; recent_m.append(lm[i]); recent_m = recent_m[-10:]
        else:
            since += 1
            if since == 1:
                pre_st = float(np.mean(recent)) if recent else -5.0; pre_mi = float(np.mean(recent_m)) if recent_m else -5.0
                onset = ls[i]; ema = ls[i]; ema_m = lm[i]; mx = ls[i]; mx_m = lm[i]; pause_sum = 0.0
            else:
                ema = 0.7 * ema + 0.3 * ls[i]; ema_m = 0.7 * ema_m + 0.3 * lm[i]; mx = max(mx, ls[i]); mx_m = max(mx_m, lm[i])
            pause_sum += ls[i]
        S[i] = since; k = max(1, since)
        F[i] = [ls[i], lm[i], np.log(k * CH),
                pre_st, onset, ema, mx, ls[i] - ls[max(0, i - 3)], prev_max, prev_mean, min(pauses, 6), turn_ema,
                sp_sum / sp_n if sp_n else -5.0, np.log(max(speech, 1) * CH),
                pre_mi, ema_m, mx_m, prev_max_m]
    d.update(F=F, since=S, sp=sp, t=t, started=started, y=(t >= d["last_word_end"]).astype(int), ok=started & ~sp)
    if d["id"] in W:
        d["win"] = {k: lg(np.nan_to_num(v, nan=0.5)) for k, v in W[d["id"]].items() if k != "full"}
folds = np.arange(len(D)) % 5
ix = lambda names: [NAMES.index(x) for x in names]
LR = lambda: LogisticRegression(C=1.0, max_iter=1000)
GB = lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.06, l2_regularization=1.0, min_samples_leaf=200)

def X_of(d, names, swap=None):
    X = d["F"][:, ix(names)].copy()
    if swap is not None and "st" in names: X[:, names.index("st")] = d["win"][swap]
    return X
def cv(names, model=LR, swap=None, weights=False):
    out = {}; coefs = []
    for k in range(5):
        tr = [d for d, f in zip(D, folds) if f != k]
        mdl = model().fit(np.concatenate([X_of(d, names, swap)[d["ok"]] for d in tr]), np.concatenate([d["y"][d["ok"]] for d in tr]))
        if hasattr(mdl, "coef_"): coefs.append(np.r_[mdl.intercept_, mdl.coef_[0]])
        for d, f in zip(D, folds):
            if f == k:
                p = np.zeros(len(d["sp"])); idx = np.nonzero(d["ok"])[0]
                if len(idx): p[idx] = mdl.predict_proba(X_of(d, names, swap)[idx])[:, 1]
                out[d["id"]] = p
    if weights and coefs: print("   mean weights:", dict(zip(["bias"] + names, np.round(np.mean(coefs, 0), 3))))
    return out
def raw(key, swap=None):
    out = {}
    for d in D:
        v = d["F"][:, NAMES.index(key)] if swap is None else d["win"][swap]
        out[d["id"]] = np.where(d["ok"], 1 / (1 + np.exp(-v)), 0.0)
    return out

V = {}
V["a_blend3 (Smart Turn + mini + silence), CV"] = cv(BASE, weights=True)
V["b_Smart Turn streaming, raw score"] = raw("st")
V["b_turn-1-mini streaming, raw score"] = raw("mini")
V["c_Smart Turn + silence, CV"] = cv(["st", "log_sil"], weights=True)
V["c_Smart Turn + silence + ST history, CV (LR)"] = cv(["st", "log_sil"] + HIST, weights=True)
V["c_Smart Turn + silence + ST history, CV (GBM)"] = cv(["st", "log_sil"] + HIST, GB)
for h in HIST:
    V[f"d_Smart Turn + silence + {h}, CV"] = cv(["st", "log_sil", h])
V["e_blend3 + ST history, CV (LR)"] = cv(BASE + HIST, weights=True)
V["e_blend3 + ST history, CV (GBM)"] = cv(BASE + HIST, GB)
V["e_blend3 + ST and mini history, CV (LR)"] = cv(NAMES)
V["e_blend3 + ST and mini history, CV (GBM)"] = cv(NAMES, GB)
V["e_blend3, CV (GBM)"] = cv(BASE, GB)
if W:
    for kind, w in sorted(next(iter(D))["win"].keys(), key=lambda k: (k[0], -k[1])):
        tag = f"{w} s short model" if kind == "s" else f"{w} s padded to 8 s"
        V[f"w_Smart Turn {tag}, raw"] = raw("st", (kind, w))
        V[f"w_Smart Turn {tag} + silence, CV"] = cv(["st", "log_sil"], swap=(kind, w))
        V[f"w_blend3 with Smart Turn {tag}, CV"] = cv(BASE, swap=(kind, w))

tmpl = pq.read_table(template).to_pylist()
byid = {d["id"]: d for d in D}
os.makedirs(outdir, exist_ok=True); index = {}
for n, (name, P) in enumerate(V.items()):
    rows = []
    for r in tmpl:
        d = byid[r["id"]]; i = int(np.floor(r["timestamp"] * 16000 / 512 + 1e-6)) - 1
        p = float(P[r["id"]][i]) if 0 <= i < len(d["sp"]) else 0.0
        rows.append({**r, "p_eot": p})
    fn = f"v{n:03d}.parquet"; pq.write_table(pa.Table.from_pylist(rows), os.path.join(outdir, fn)); index[fn] = name
json.dump(index, open(os.path.join(outdir, "index.json"), "w"), indent=1)
print("wrote", len(index), "variants")
