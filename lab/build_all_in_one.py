"""
Builds turn-blend.onnx: the whole detector in one ONNX file.

One call = one 32 ms chunk of 16 kHz audio. The graph contains Silero VAD, the Whisper log-mel front end,
Smart Turn v3.2, turn-1-mini, the logistic blend and the turn logic (turn start, scoring cadence, tentative
end, commit rule, silence timeout). All memory lives in one flat `state` vector that the caller passes back
in on the next call; a new stream starts from zeros. It mirrors the blend mode of server.py.

    audio   float32 [512]      one chunk
    state   float32 [STATE]    zeros to start, then the previous call's new_state
    config  float32 [16]       optional; the defaults are stored in the file (see CONFIG below)

    vad, speech, in_turn, silence_ms, smart_turn, mini, blend, tentative, end_of_turn   float32 [1] each
    new_state                                                                           float32 [STATE]

smart_turn and blend are -1 on chunks where Smart Turn did not run (it runs only inside a turn, on a
non-speech chunk, on the scoring cadence; onnxruntime skips that branch otherwise). end_of_turn is 0, 1 when
the blend commits, 2 when the silence timeout ends the turn.

Third-party weights inside the file (see NOTICE.md): Silero VAD v6.2 (MIT), Smart Turn v3.2 by Pipecat
(BSD-2-Clause), turn-1-mini by p99lab (Apache-2.0).

Run:  .venv/bin/python build_all_in_one.py [output.onnx]
"""

import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper, version_converter

import mini as turn_mini
import server as lab

OPSET = 18
RATE, CHUNK = lab.RATE, lab.CHUNK
RING = 8 * RATE                      # last 8 s of audio, what Smart Turn can see
STEP = turn_mini.STEP                # turn-1-mini advances every 2560 samples = 5 chunks
VAD_RESET_CHUNKS = round(5.0 * 1000 / lab.CHUNK_MS)   # Silero state reset every 5 s, as upstream does

# name, default, meaning
CONFIG = [
    ("vad_confidence", lab.DEFAULTS["vad_confidence"], "a chunk is speech when Silero's probability is at least this"),
    ("vad_start_secs", lab.DEFAULTS["vad_start_secs"], "continuous speech needed to open a turn"),
    ("vad_min_db", lab.DEFAULTS["vad_min_db"], "chunks quieter than this (RMS dBFS) never count as speech"),
    ("pre_speech_ms", lab.DEFAULTS["st_pre_speech_ms"], "audio from before the turn opened that Smart Turn also hears"),
    ("smart_turn_max_secs", lab.DEFAULTS["st_max_duration_secs"], "how much of the turn Smart Turn hears (at most 8)"),
    ("score_every_chunks", lab.DEFAULTS["stream_every_chunks"], "score every N chunks while a turn is open"),
    ("blend_threshold", lab.DEFAULTS["blend_threshold"], "commit: blend above this ..."),
    ("blend_min_silence_ms", lab.DEFAULTS["blend_min_silence_ms"], "... after at least this much silence ..."),
    ("blend_confirm_scores", lab.DEFAULTS["blend_confirm_chunks"], "... on this many scores in a row"),
    ("tentative_threshold", lab.DEFAULTS["tent_threshold"], "tentative end of turn: blend above this ..."),
    ("tentative_min_silence_ms", lab.DEFAULTS["tent_min_silence_ms"], "... after at least this much silence"),
    ("timeout_secs", lab.DEFAULTS["st_stop_secs"], "silence that ends the turn even if the blend never commits"),
    ("w_bias", lab.BLEND["bias"], "blend: constant term"),
    ("w_smart_turn", lab.BLEND["smart_turn"], "blend: weight on logit(P_smart_turn)"),
    ("w_mini", lab.BLEND["mini"], "blend: weight on logit(P_turn_1_mini)"),
    ("w_log_silence", lab.BLEND["log_silence"], "blend: weight on ln(silence in seconds)"),
]
OUTPUTS = ["vad", "speech", "in_turn", "silence_ms", "smart_turn", "mini", "blend", "tentative", "end_of_turn"]
SCALARS = ["chunks", "speech_run", "since", "in_turn", "pre_count", "turn_chunks", "stream_i", "pos_run",
           "tentative", "mini_k"]


def upgraded(path, prefix):
    """Load a component, bring it to the common opset, and prefix every name so graphs cannot collide."""
    m = onnx.load(str(path))
    del m.opset_import[1:]                                   # only the default domain is used
    assert all(n.domain in ("", "ai.onnx") for n in m.graph.node), path
    if m.opset_import[0].version != OPSET:
        m = version_converter.convert_version(m, OPSET)
    m = compose.add_prefix(m, prefix)
    del m.graph.value_info[:]
    return m


class Builder:
    def __init__(self):
        self.nodes, self.inits, self.count, self.cache = [], [], 0, {}

    def name(self, hint):
        self.count += 1
        return f"tb/{hint}_{self.count}"

    def op(self, kind, inputs, hint=None, outs=1, **attrs):
        names = [self.name(hint or kind.lower()) for _ in range(outs)]
        self.nodes.append(helper.make_node(kind, list(inputs), names, **attrs))
        return names[0] if outs == 1 else names

    def const(self, value, dtype=np.float32):
        arr = np.asarray(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self.cache:
            self.cache[key] = self.name("c")
            self.inits.append(numpy_helper.from_array(arr, self.cache[key]))
        return self.cache[key]

    def f(self, v):
        return self.const([v])

    def i(self, *v):
        return self.const(list(v), np.int64)

    # arithmetic on float tensors
    def add(self, a, b): return self.op("Add", [a, b])
    def sub(self, a, b): return self.op("Sub", [a, b])
    def mul(self, a, b): return self.op("Mul", [a, b])
    def div(self, a, b): return self.op("Div", [a, b])
    def minimum(self, a, b): return self.op("Min", [a, b])
    def maximum(self, a, b): return self.op("Max", [a, b])
    # comparisons and logic (bool tensors)
    def ge(self, a, b): return self.op("GreaterOrEqual", [a, b])
    def gt(self, a, b): return self.op("Greater", [a, b])
    def eq(self, a, b): return self.op("Equal", [a, b])
    def and_(self, a, b): return self.op("And", [a, b])
    def or_(self, a, b): return self.op("Or", [a, b])
    def not_(self, a): return self.op("Not", [a])
    def where(self, c, a, b): return self.op("Where", [c, a, b])
    def tofloat(self, a): return self.op("Cast", [a], to=TensorProto.FLOAT)
    def toint(self, a): return self.op("Cast", [a], to=TensorProto.INT64)
    def slice(self, x, start, end): return self.op("Slice", [x, self.i(start), self.i(end), self.i(0)])
    def reshape(self, x, shape): return self.op("Reshape", [x, self.i(*shape)])
    def identity(self, x, name): self.nodes.append(helper.make_node("Identity", [x], [name]))


def toposort(nodes, available):
    """Order nodes so every input is produced before it is used (an If also waits for what its branches read)."""
    def needs(node):
        names = set(node.input)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                inner = {o for n in attr.g.node for o in n.output} | {i.name for i in attr.g.initializer}
                names |= {i for n in attr.g.node for i in n.input if i not in inner}
        return {n for n in names if n}

    have, todo, ordered = set(available), [(n, needs(n)) for n in nodes], []
    while todo:
        ready = [(n, d) for n, d in todo if d <= have]
        if not ready:
            missing = sorted(set().union(*[d for _, d in todo]) - have - {o for n, _ in todo for o in n.output})
            raise RuntimeError(f"graph has inputs nobody produces: {missing[:10]}")
        for n, _ in ready:
            ordered.append(n)
            have |= set(n.output)
        done = {id(n) for n, _ in ready}
        todo = [(n, d) for n, d in todo if id(n) not in done]
    return ordered


def build(path):
    silero = upgraded(lab.silero_path(), "silero/")
    feats = upgraded(lab.features_path(), "fe/")
    smart = upgraded(lab.hf_hub_download(lab.SMART_TURN_REPO, lab.SMART_TURN_FILE, local_dir=lab.MODELS), "st/")
    mini = upgraded(lab.hf_hub_download(turn_mini.REPO, turn_mini.FILE, local_dir=lab.MODELS / "p99"), "mini/")

    mini_states = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                   for i in mini.graph.input if i.name != "mini/audio"]
    layout, offset = {}, 0
    for name, shape in ([("ring", [RING]), ("vad_state", [2, 1, 128]), ("vad_ctx", [64]), ("mini_buf", [STEP])]
                        + mini_states + [(s, [1]) for s in SCALARS]):
        size = int(np.prod(shape))
        layout[name] = (offset, offset + size, shape)
        offset += size
    state_size = offset

    b = Builder()
    S = {k: b.slice("state", a, e) for k, (a, e, _) in layout.items()}
    cfg = {name: b.slice("config", n, n + 1) for n, (name, _, _) in enumerate(CONFIG)}
    one, zero = b.f(1.0), b.f(0.0)
    chunk_ms = b.f(lab.CHUNK_MS)

    # ---- Silero VAD on (64 samples of context + this chunk)
    x = b.op("Concat", [S["vad_ctx"], "audio"], axis=0)
    b.identity(b.reshape(x, [1, 64 + CHUNK]), "silero/input")
    b.identity(b.reshape(S["vad_state"], [2, 1, 128]), "silero/state")
    vad = b.reshape("silero/output", [1])
    chunks = b.add(S["chunks"], one)
    vad_reset = b.eq(b.op("Mod", [chunks, b.f(VAD_RESET_CHUNKS)], fmod=1), zero)
    keep = b.tofloat(b.not_(vad_reset))
    vad_state = b.mul(b.reshape("silero/stateN", [256]), keep)
    vad_ctx = b.mul(b.slice(x, CHUNK, CHUNK + 64), keep)
    rms = b.op("Sqrt", [b.op("ReduceMean", [b.mul("audio", "audio")], keepdims=1)])
    loud = b.ge(rms, b.op("Pow", [b.f(10.0), b.div(cfg["vad_min_db"], b.f(20.0))]))
    speech = b.and_(b.ge(vad, cfg["vad_confidence"]), loud)

    # ---- turn-1-mini: 5 chunks make one 160 ms step. Every chunk the step is run on the audio collected so far,
    # zero-padded; its state is only kept when the step is complete (the model card's recipe for scoring between steps).
    k = S["mini_k"]
    pads = b.toint(b.op("Concat", [b.mul(k, b.f(CHUNK)), b.mul(b.sub(b.f(STEP // CHUNK - 1), k), b.f(CHUNK))], axis=0))
    buf = b.add(S["mini_buf"], b.op("Pad", ["audio", pads]))
    b.identity(b.reshape(buf, [1, STEP]), "mini/audio")
    for name, shape in mini_states:
        b.identity(b.reshape(S[name], shape), name)
    k1 = b.add(k, one)
    frames = b.op("Floor", [b.div(b.mul(k1, b.f(CHUNK)), b.f(320.0))])        # complete 20 ms frames of real audio
    p_mini = b.op("Gather", ["mini/probs", b.toint(b.sub(frames, one))], axis=0)
    step_done = b.eq(k1, b.f(STEP // CHUNK))
    mini_new = {name: b.where(step_done, b.reshape("mini/new_" + name[len("mini/"):], [-1]), S[name])
                for name, _ in mini_states}
    mini_buf = b.mul(buf, b.tofloat(b.not_(step_done)))
    mini_k = b.where(step_done, zero, k1)

    # ---- audio memory and turn bookkeeping (same order of operations as the WebSocket loop in server.py)
    ring = b.op("Concat", [b.slice(S["ring"], CHUNK, RING), "audio"], axis=0)
    start_chunks = b.maximum(one, b.op("Round", [b.div(b.mul(cfg["vad_start_secs"], b.f(1000.0)), chunk_ms)]))
    pre_len = b.add(b.maximum(one, b.op("Ceil", [b.div(cfg["pre_speech_ms"], chunk_ms)])), start_chunks)
    speech_run = b.where(speech, b.add(S["speech_run"], one), zero)
    since = b.where(speech, zero, b.add(S["since"], one))
    silence_ms = b.mul(since, chunk_ms)
    was_in = b.gt(S["in_turn"], b.f(0.5))
    pre_count = b.where(was_in, S["pre_count"], b.minimum(b.add(S["pre_count"], one), pre_len))
    turn_chunks = b.where(was_in, b.minimum(b.add(S["turn_chunks"], one), b.f(RING // CHUNK)), S["turn_chunks"])
    opened = b.and_(b.not_(was_in), b.ge(speech_run, start_chunks))
    turn_chunks = b.where(opened, pre_count, turn_chunks)        # the turn starts with the pre-speech audio
    pre_count = b.where(opened, zero, pre_count)
    in_now = b.or_(was_in, opened)
    stream_i = b.where(in_now, b.add(b.where(opened, zero, S["stream_i"]), one), S["stream_i"])
    every = b.maximum(one, b.op("Round", [cfg["score_every_chunks"]]))
    scheduled = b.and_(in_now, b.eq(b.op("Mod", [stream_i, every], fmod=1), zero))
    run_smart_turn = b.and_(scheduled, b.not_(speech))

    # ---- Smart Turn on the turn so far, only when needed (the expensive part: inside an If)
    t = Builder()
    t.count = 100000
    window = t.op("Floor", [t.mul(t.minimum(t.maximum(cfg["smart_turn_max_secs"], t.f(0.5)), t.f(8.0)), t.f(RATE))])
    valid = t.minimum(t.mul(turn_chunks, t.f(CHUNK)), window)
    index = t.tofloat(t.op("Range", [t.const(0, np.int64), t.const(RING, np.int64), t.const(1, np.int64)]))
    heard = t.tofloat(t.ge(index, t.sub(t.f(RING), valid)))      # zero everything from before the turn
    t.identity(t.reshape(t.mul(ring, heard), [1, RING]), "fe/audio")
    t.nodes += list(feats.graph.node)
    t.identity("fe/input_features", "st/input_features")
    t.nodes += list(smart.graph.node)
    t.nodes.append(helper.make_node("Reshape", ["st/logits", t.i(1)], ["tb/smart_turn_scored"]))
    then_graph = helper.make_graph(t.nodes, "score_smart_turn", [],
                                   [helper.make_tensor_value_info("tb/smart_turn_scored", TensorProto.FLOAT, [1])], t.inits)
    else_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["tb/smart_turn_skipped"], value=numpy_helper.from_array(np.array([-1.0], np.float32)))],
        "skip_smart_turn", [], [helper.make_tensor_value_info("tb/smart_turn_skipped", TensorProto.FLOAT, [1])])
    p_st = b.op("If", [run_smart_turn], hint="smart_turn", then_branch=then_graph, else_branch=else_graph)

    # ---- blend of the two scores and the elapsed silence
    def logit(p):
        pc = b.op("Clip", [p, b.const(1e-4), b.const(1 - 1e-4)])
        return b.op("Log", [b.div(pc, b.sub(one, pc))])

    z = b.add(b.add(cfg["w_bias"], b.mul(cfg["w_smart_turn"], logit(p_st))),
              b.add(b.mul(cfg["w_mini"], logit(p_mini)),
                    b.mul(cfg["w_log_silence"], b.op("Log", [b.maximum(b.div(silence_ms, b.f(1000.0)), b.f(CHUNK / RATE))]))))
    blend = b.op("Sigmoid", [z])

    # ---- decisions
    tent_was = b.gt(b.where(opened, zero, S["tentative"]), b.f(0.5))
    tent_new = b.and_(run_smart_turn, b.and_(b.not_(tent_was), b.and_(b.gt(blend, cfg["tentative_threshold"]),
                                                                    b.ge(silence_ms, cfg["tentative_min_silence_ms"]))))
    tent_cancel = b.and_(scheduled, b.and_(speech, tent_was))
    tentative = b.where(tent_new, one, b.where(tent_cancel, zero, b.tofloat(tent_was)))
    positive = b.and_(run_smart_turn, b.and_(b.gt(blend, cfg["blend_threshold"]), b.ge(silence_ms, cfg["blend_min_silence_ms"])))
    pos_prev = b.where(opened, zero, S["pos_run"])
    pos_run = b.where(scheduled, b.where(positive, b.add(pos_prev, one), zero), pos_prev)
    commit = b.and_(scheduled, b.ge(pos_run, b.maximum(one, b.op("Round", [cfg["blend_confirm_scores"]]))))
    timeout = b.and_(b.and_(in_now, b.not_(commit)), b.ge(silence_ms, b.mul(cfg["timeout_secs"], b.f(1000.0))))
    ended = b.or_(commit, timeout)
    alive = b.tofloat(b.not_(ended))
    in_turn = b.mul(b.tofloat(in_now), alive)

    new = {"ring": ring, "vad_state": vad_state, "vad_ctx": vad_ctx, "mini_buf": mini_buf, **mini_new,
           "chunks": chunks, "speech_run": speech_run, "since": since, "in_turn": in_turn, "pre_count": pre_count,
           "turn_chunks": b.mul(turn_chunks, alive), "stream_i": b.mul(stream_i, alive), "pos_run": b.mul(pos_run, alive),
           "tentative": b.mul(tentative, alive), "mini_k": mini_k}
    b.nodes.append(helper.make_node("Concat", [new[k] for k in layout], ["new_state"], axis=0))
    out = {"vad": vad, "speech": b.tofloat(speech), "in_turn": in_turn, "silence_ms": silence_ms,
           "smart_turn": p_st, "mini": p_mini, "blend": b.where(run_smart_turn, blend, b.f(-1.0)), "tentative": tentative,
           "end_of_turn": b.where(commit, one, b.where(timeout, b.f(2.0), zero))}
    for name in OUTPUTS:
        b.identity(out[name], name)

    all_nodes = b.nodes + list(silero.graph.node) + list(mini.graph.node)
    inits = (b.inits + list(silero.graph.initializer) + list(mini.graph.initializer)
             + list(feats.graph.initializer) + list(smart.graph.initializer)
             + [numpy_helper.from_array(np.array([c[1] for c in CONFIG], np.float32), "config")])
    fl = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    nodes = toposort(all_nodes, {"audio", "state", "config"} | {i.name for i in inits})
    graph = helper.make_graph(
        nodes, "turn_blend",
        [fl("audio", [CHUNK]), fl("state", [state_size]), fl("config", [len(CONFIG)])],
        [fl(n, [1]) for n in OUTPUTS] + [fl("new_state", [state_size])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="turn-blend")
    model.ir_version = 10
    model.doc_string = ("turn-blend: voice activity + end-of-turn detection in one streaming graph. Contains Silero VAD v6.2 "
                        "(MIT, Silero Team), Smart Turn v3.2 (BSD-2-Clause, Pipecat / Daily; a fine-tuned OpenAI Whisper Tiny "
                        "encoder, MIT) and turn-1-mini by p99lab (Apache-2.0). The blend weights were fitted on LiveKit's "
                        "eot-bench data (CC BY 4.0). English only.")
    meta = {
        "sample_rate": RATE, "chunk_samples": CHUNK, "state_size": state_size,
        "config": json.dumps([{"name": n, "default": d, "meaning": m} for n, d, m in CONFIG]),
        "outputs": json.dumps(OUTPUTS),
        # the last len(state_tail) entries of the state vector are these counters, in this order
        "state_tail": json.dumps(SCALARS),
        "components": json.dumps([
            {"name": "Silero VAD v6.2", "by": "Silero Team", "license": "MIT", "source": "https://github.com/snakers4/silero-vad"},
            {"name": "Smart Turn v3.2", "by": "Pipecat / Daily", "license": "BSD-2-Clause",
             "source": "https://huggingface.co/pipecat-ai/smart-turn-v3"},
            {"name": "turn-1-mini", "by": "p99lab", "license": "Apache-2.0", "credit": "turn-1-mini by p99lab",
             "source": "https://huggingface.co/p99lab/turn-1-mini"},
        ]),
        "blend_fitted_on": "LiveKit eot-bench, English (CC BY 4.0) https://huggingface.co/datasets/livekit/eot-bench-data",
        "language": "en",
    }
    for key, value in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = key, str(value)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(path))
    return state_size


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(lab.MODELS / "turn-blend.onnx")
    size = build(out)
    print(f"wrote {out}  (state vector: {size} floats)")
