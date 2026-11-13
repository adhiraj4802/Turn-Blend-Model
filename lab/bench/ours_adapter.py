"""eot-bench adapters for our turn-detection lab (Silero VAD + Smart Turn v3.2 + turn-1-mini v1 + blend).

Each adapter streams a whole benchmark turn to the lab server's /api/bench endpoint, which runs the same
pipeline as the live page and reports, for every grid timestamp, what that pipeline knows at that moment.
The four adapters differ only in which number they report as p_eot. Server responses are cached per turn.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np

from eot_harness.io import DEFAULT_MIN_SILENCE, _EPS, _time_grid, decode_audio

HERE = Path(__file__).parent
# OURS_CACHE: folder of per-turn score files. The default is filled by calling the lab server; the all-in-one
# model's folder is written by bench/all_in_one/bench_aio.py, in which case no server is needed.
# OURS_TAG: suffix for the run name, so results from different sources sit side by side.
CACHE = Path(os.environ.get("OURS_CACHE", HERE / "cache"))
TAG = os.environ.get("OURS_TAG", "")
LABEL = f" ({TAG.replace('-', ' ')})" if TAG else ""
SUFFIX = f"-{TAG}" if TAG else ""
SERVER = "http://127.0.0.1:8000/api/bench"
CV = json.loads((HERE / "blend_cv.json").read_text())


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _blend(w, pt):
    if not pt["decidable"]:
        return 0.0   # no open turn, or the VAD still hears speech: the live system cannot fire here
    z = w["bias"] + w["smart_turn"] * _logit(pt["smart_turn"]) + w["mini"] * _logit(pt["mini"]) \
        + w["log_silence"] * math.log(max(pt["silence_s"], 0.032))
    return 1 / (1 + math.exp(-z))


def _grid(row):
    rows = []
    spans = row["silence_spans"]
    for span_index, span in enumerate(spans):
        start, end = float(span["start"]), float(span["end"])
        if end - start < DEFAULT_MIN_SILENCE - _EPS:
            continue
        label = "eot" if span_index == len(spans) - 1 else "hold"
        for t in _time_grid(start, end, 0.1):
            rows.append({"id": row["id"], "span_index": span_index, "timestamp": t,
                         "silence_dur": round(t - start, 6), "label": label})
    return rows


def _score_turn(row, inference_interval):
    assert abs(inference_interval - 0.1) < 1e-9, "cache assumes the default 0.1 s grid"
    CACHE.mkdir(exist_ok=True)
    f = CACHE / (str(row["id"]).replace("/", "_") + ".json")
    if f.exists():
        return json.loads(f.read_text())
    audio, sr = decode_audio(row["audio"])
    assert sr == 16000, sr
    grid = _grid(row)
    ts = sorted({g["timestamp"] for g in grid})
    req = urllib.request.Request(SERVER + "?t=" + ",".join(repr(t) for t in ts),
                                 data=np.asarray(audio, "<f4").tobytes(), method="POST")
    res = json.load(urllib.request.urlopen(req, timeout=600))
    by_t = {round(p["t"], 6): p for p in res["points"]}
    out = {"audio_sec": len(audio) / sr, "weights": res["blend_weights"],
           "rows": [dict(g, point=by_t[round(g["timestamp"], 6)]) for g in grid]}
    f.write_text(json.dumps(out))
    return out


class _Base:
    concurrency = 1

    def supports_language(self, lang_code: str) -> bool:
        return str(lang_code).lower().startswith("en")   # turn-1-mini (and so the blend) is English only

    def p_eot(self, row, scored, point):
        raise NotImplementedError

    async def predict_turn(self, row, *, inference_interval):
        scored = await asyncio.to_thread(_score_turn, row, inference_interval)
        return {
            "id": row["id"], "audio_sec": scored["audio_sec"], "events": [],
            "prediction_rows": [
                {"id": r["id"], "span_index": r["span_index"], "timestamp": r["timestamp"],
                 "silence_dur": r["silence_dur"], "label": r["label"],
                 "p_eot": float(self.p_eot(row, scored, r["point"]))}
                for r in scored["rows"]
            ],
        }


class OursBlendCV(_Base):
    """Blend with weights fitted on the other four folds of turns (never on the turn being scored)."""
    adapter_id = "ours/blend-cv" + SUFFIX
    display_name = "Ours: blend, cross-validated" + LABEL

    def p_eot(self, row, scored, point):
        return _blend(CV["weights"][str(CV["folds"][row["id"]])], point)


class OursBlend(_Base):
    """Blend exactly as shipped in the live server. Its weights were fitted on these same 400 turns."""
    adapter_id = "ours/blend-insample" + SUFFIX
    display_name = "Ours: blend, as shipped (fitted on these turns)" + LABEL

    def p_eot(self, row, scored, point):
        return point["blend"]


class OursSmartTurn(_Base):
    """Sanity check: Smart Turn v3.2 alone through our pipeline, read at 0.2 s like the published row."""
    adapter_id = "ours/smart-turn-only"
    display_name = "Our pipeline: Smart Turn v3.2 only"
    score_point = 0.2

    def p_eot(self, row, scored, point):
        return point["smart_turn"]


class OursMini(_Base):
    """Sanity check: turn-1-mini v1 (official ONNX) alone through our pipeline."""
    adapter_id = "ours/turn-1-mini-only"
    display_name = "Our pipeline: turn-1-mini v1 only"

    def p_eot(self, row, scored, point):
        return point["mini"]
