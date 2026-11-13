"""
Live settings and results for the voice bot, with a small web console (static/console.html).

SETTINGS is the single source of truth: the VAD and the turn analyzer in blend_turn.py read it on every
audio chunk, so an edit in the console applies to the call in progress. Edits are saved to
bot_settings.json and reloaded on the next start. Turn results are kept in memory for the console and
appended to bot_turns.jsonl.
"""

import json
import threading
import time
import weakref
from collections import deque
from pathlib import Path

import server as lab

ROOT = Path(__file__).parent
SAVED = ROOT / "bot_settings.json"
LOG = ROOT / "bot_turns.jsonl"

# key, label, min, max, step, unit, group, what it does
SPEC = [
    ("vad_confidence", "Speech confidence", 0.3, 0.95, 0.05, "", "Voice activity (Silero VAD)",
     "A 32 ms chunk counts as speech when Silero's probability is at least this."),
    ("vad_start_secs", "Speech needed to start a turn", 0.064, 1.0, 0.032, "s", "Voice activity (Silero VAD)",
     "Continuous speech required before a user turn opens (and before the bot is interrupted)."),
    ("vad_stop_secs", "Silence before \"user stopped\"", 0.096, 1.504, 0.032, "s", "Voice activity (Silero VAD)",
     "Silence after which Pipecat reports the user stopped. This asks Deepgram to finalize the transcript, "
     "and a turn cannot end before that transcript arrives."),
    ("blend_threshold", "Commit threshold", 0.5, 0.99, 0.01, "", "End of turn (blend)",
     "The blended probability must be above this to end the turn."),
    ("blend_min_silence_ms", "Minimum silence to commit", 64, 1504, 32, "ms", "End of turn (blend)",
     "The turn cannot end before this much silence."),
    ("blend_confirm_chunks", "Scores in a row", 1, 5, 1, "", "End of turn (blend)",
     "Consecutive scores above the threshold needed to end the turn."),
    ("stop_secs", "Silence timeout", 1.0, 10.0, 0.5, "s", "End of turn (blend)",
     "Silence that ends the turn even if the blend never commits."),
    ("tent_threshold", "Tentative threshold", 0.2, 0.95, 0.05, "", "Tentative end (logged only)",
     "Blend above this marks a tentative end of turn. The bot does not act on it yet."),
    ("tent_min_silence_ms", "Minimum silence for tentative", 32, 1024, 32, "ms", "Tentative end (logged only)",
     "A tentative end cannot be marked before this much silence."),
    ("score_every_chunks", "Score every N chunks", 1, 6, 1, "", "Scoring",
     "1 scores every 32 ms. On the CPU Smart Turn takes about 25 ms, so 1 can fall behind real time."),
    ("pre_speech_ms", "Audio kept before speech", 0, 1504, 32, "ms", "Scoring",
     "Audio from before the turn opened that Smart Turn also hears."),
    ("max_duration_secs", "Smart Turn window", 2.0, 8.0, 0.5, "s", "Scoring",
     "How much of the turn Smart Turn hears (it was trained on 8 s)."),
    ("w_bias", "Bias", -5.0, 8.0, 0.05, "", "Blend weights",
     "Constant term. Higher makes every pause look more like an end of turn."),
    ("w_smart_turn", "Smart Turn weight", 0.0, 3.0, 0.01, "", "Blend weights", "Weight on logit(P_smart_turn)."),
    ("w_mini", "turn-1-mini weight", 0.0, 3.0, 0.01, "", "Blend weights", "Weight on logit(P_mini)."),
    ("w_log_silence", "Silence weight", 0.0, 4.0, 0.01, "", "Blend weights", "Weight on ln(silence in seconds)."),
]
INT_KEYS = {"blend_confirm_chunks", "score_every_chunks"}

DEFAULTS = {
    "vad_confidence": lab.DEFAULTS["vad_confidence"],
    "vad_start_secs": lab.DEFAULTS["vad_start_secs"],
    "vad_stop_secs": lab.DEFAULTS["vad_stop_secs"],
    "blend_threshold": lab.DEFAULTS["blend_threshold"],
    "blend_min_silence_ms": lab.DEFAULTS["blend_min_silence_ms"],
    "blend_confirm_chunks": lab.DEFAULTS["blend_confirm_chunks"],
    "stop_secs": lab.DEFAULTS["st_stop_secs"],
    "tent_threshold": lab.DEFAULTS["tent_threshold"],
    "tent_min_silence_ms": lab.DEFAULTS["tent_min_silence_ms"],
    "score_every_chunks": lab.DEFAULTS["stream_every_chunks"],
    "pre_speech_ms": lab.DEFAULTS["st_pre_speech_ms"],
    "max_duration_secs": lab.DEFAULTS["st_max_duration_secs"],
    "w_bias": lab.BLEND["bias"],
    "w_smart_turn": lab.BLEND["smart_turn"],
    "w_mini": lab.BLEND["mini"],
    "w_log_silence": lab.BLEND["log_silence"],
}
FIXED = {   # shown in the console, not editable while running
    "Smart Turn backend": lab.DEFAULTS["st_backend"], "Smart Turn CPU threads": lab.DEFAULTS["st_threads"],
    "Feature extraction backend": lab.DEFAULTS["fe_backend"], "VAD and turn-1-mini backend": "cpu",
}

SETTINGS = dict(DEFAULTS)
STATE = {"rev": 1, "applied_at": time.time(), "mode": "blend", "calls": 0}
_lock = threading.Lock()
_vads = weakref.WeakSet()       # live Pipecat VAD analyzers, re-configured on every edit
_turns = deque(maxlen=300)      # one row per user turn, newest last
_next_id = 0


def _apply(values):
    """Validate and apply edits. Returns the keys that were rejected."""
    bad = []
    spec = {s[0]: s for s in SPEC}
    for k, v in values.items():
        if k not in spec:
            bad.append(k)
            continue
        _, _, lo, hi, *_ = spec[k]
        try:
            v = float(v)
        except (TypeError, ValueError):
            bad.append(k)
            continue
        v = min(max(v, lo), hi)
        SETTINGS[k] = int(round(v)) if k in INT_KEYS else v
    # the blend formula lives in server.py and reads this dict on every call
    lab.BLEND.update(bias=SETTINGS["w_bias"], smart_turn=SETTINGS["w_smart_turn"],
                     mini=SETTINGS["w_mini"], log_silence=SETTINGS["w_log_silence"])
    for vad in list(_vads):
        vad.apply_settings()
    return bad


def update(values, save=True):
    with _lock:
        bad = _apply(values)
        STATE["rev"] += 1
        STATE["applied_at"] = time.time()
        if save:
            SAVED.write_text(json.dumps(SETTINGS, indent=2))
    return bad


def register_vad(vad):
    _vads.add(vad)


def load_saved():
    if SAVED.exists():
        try:
            _apply({k: v for k, v in json.loads(SAVED.read_text()).items() if k in DEFAULTS})
        except Exception as e:
            print(f"Ignoring {SAVED.name}: {e}")


# ------------------------------------------------------------------ results ---
def record(event, **kw):
    """Log one event and fold it into the per-turn rows the console shows."""
    global _next_id
    with _lock:
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "mode": STATE["mode"], "rev": STATE["rev"],
                                "event": event, **kw}) + "\n")
        last = _turns[-1] if _turns else None
        if event == "end_of_turn" or (event == "user_turn" and (last is None or "text" in last)):
            _next_id += 1
            last = {"id": _next_id, "t": time.time(), "rev": STATE["rev"], "mode": STATE["mode"]}
            _turns.append(last)
        if event == "end_of_turn":
            last.update(kw)
        elif event == "user_turn":
            last["text"] = kw.get("text") or ""
        elif event == "full_turn_latency" and last is not None and "full_ms" not in last:
            last["full_ms"] = kw.get("ms")
        elif event == "latency_breakdown" and last is not None and "breakdown" not in last:
            last["breakdown"] = kw.get("lines")


# ------------------------------------------------------------------ web console ---
def start(port=7861):
    """Serve the console on its own port, in a background thread of the bot process."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse

    app = FastAPI()

    @app.get("/")
    def page():
        return FileResponse(ROOT / "static" / "console.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/settings")
    def get_settings():
        return {"settings": SETTINGS, "defaults": DEFAULTS, "fixed": FIXED, "state": STATE,
                "spec": [dict(zip(("key", "label", "min", "max", "step", "unit", "group", "help"), s)) for s in SPEC]}

    @app.post("/api/settings")
    async def post_settings(request: Request):
        body = await request.json()
        values = DEFAULTS if body.get("reset") else body.get("values", {})
        bad = update(values)
        return {"settings": SETTINGS, "state": STATE, "rejected": bad}

    @app.get("/api/turns")
    def get_turns():
        with _lock:
            return {"turns": list(_turns), "state": STATE}

    @app.post("/api/turns/clear")
    def clear_turns():
        with _lock:
            _turns.clear()
        return {"ok": True}

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    threading.Thread(target=uvicorn.Server(config).run, name="console", daemon=True).start()
    print(f"   → Settings console: http://localhost:{port}", flush=True)
