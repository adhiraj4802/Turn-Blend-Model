# turn-blend

End-of-turn detection for voice agents, built by combining three small open models, with a browser demo
you can test on your own microphone.

- **Silero VAD** (v6.2) decides whether each 32 ms chunk is speech.
- **Smart Turn v3.2** (Pipecat) scores the turn so far: does it sound finished?
- **turn-1-mini** (p99lab) gives a second, streaming end-of-turn score.
- A **logistic blend** of the two scores and the elapsed silence decides when the turn is over.

English only, because turn-1-mini is English only.

## Run

```bash
./run.sh
```

The first run creates `.venv`, installs the packages, and downloads the models (about 60 MB) into
`models/`. Then open http://localhost:8000, allow the microphone, and talk.

| Command | What it does |
|---|---|
| `./run.sh` | Everything on the CPU. Needs Python 3.10+ with `venv`. |
| `./run.sh gpu` | Also installs CUDA and TensorRT (NVIDIA GPU, several GB of packages). The page then offers `gpu`, `trt` and `trt-fp16` backends. |
| `PORT=8001 ./run.sh` | Serve on another port. |

## Voice bot (Pipecat)

`bot.py` puts the detector into a real call: WebRTC audio, Deepgram streaming STT, our Silero VAD and
turn-blend for the turn decisions, OpenAI `gpt-5.4-nano` with reasoning off, and Deepgram Aura-2 for speech.

```bash
# .env (not committed):  DEEPGRAM_API_KEY=...  OPENAI_API_KEY=...
./run_bot.sh
```

Open http://localhost:7860, press Connect, allow the microphone and talk. Nothing but the two API keys is
needed: the call runs over Pipecat's local WebRTC transport. Each turn's timings are logged and appended
to `bot_turns.jsonl`.

| Command | What it does |
|---|---|
| `./run_bot.sh` | Local WebRTC with our VAD and turn-blend. |
| `TURN_MODE=smart_turn ./run_bot.sh` | Same bot with Pipecat's stock Silero + Smart Turn v3, to compare. |
| `./run_bot.sh -t daily` | Daily room instead. Needs `DAILY_API_KEY` and a payment method on the Daily account. |

**Settings console:** http://localhost:7861 opens alongside the call page. Every VAD, end-of-turn and
blend-weight setting has a slider there; edits apply to the call in progress and are saved to
`bot_settings.json`. The same page lists each turn with its measured turn-detection time, full turn
latency (you stop speaking to the bot starts speaking), the three model scores and the STT, LLM and TTS
times, so you can see what a change did.

The pieces are in `blend_turn.py`: `LabSileroVADAnalyzer` (starts turns, detects interruptions) and
`BlendTurnAnalyzer` (ends turns). Both are standard Pipecat analyzers, so they drop into any pipeline.

## All-in-one model (one ONNX file)

`build_all_in_one.py` packs the whole detector into a single file, `turn-blend.onnx` (56 MB): Silero VAD,
the log-mel front end, Smart Turn v3.2, turn-1-mini, the blend and the turn logic. One call takes a 32 ms
chunk and a state vector and returns the scores and the end-of-turn decision. Smart Turn sits inside an ONNX
`If`, so it only runs on the chunks that need it.

```bash
.venv/bin/python build_all_in_one.py release/turn-blend.onnx
cd release && ../.venv/bin/python example.py some_16khz.wav
```

`release/` is the folder meant for publishing: the model card (`README.md`), the runner (`turn_blend.py`,
numpy + onnxruntime only), an example, and the licence and notice files for the three models inside. The
ONNX file itself is built, not committed.

It was verified against this repo's pipeline: scores equal to within 0.000001, identical turn decisions on
16,948 chunks, and the same eot-bench result (20.1% / 7.8% / 798 ms / 487 ms cross-validated, against
20.3% / 7.8% / 798 ms / 488 ms). The check scripts are in `bench/all_in_one/`.

## How a turn ends

```
mic (16 kHz) -> 32 ms chunks -> Silero VAD -> turn opens after 0.2 s of speech
                                  |
              while the turn is open, on every scored chunk:
                Smart Turn v3.2 (last 8 s of the turn)   -> P_smart_turn
                turn-1-mini (streaming, hears everything) -> P_mini
                                  |
   p = sigmoid(2.80 + 0.29*logit(P_smart_turn) + 0.49*logit(P_mini) + 1.29*ln(silence seconds))
                                  |
   tentative end:  p > 0.5 after 160 ms of silence   (an agent could start its LLM; cancelled if speech resumes)
   committed end:  p > 0.85 after 320 ms of silence, twice in a row
   timeout:        3 s of silence ends the turn regardless
```

All of these are sliders on the page. Two other modes are selectable for comparison: `vad_stop` (Smart
Turn once when the VAD reports silence, as Pipecat wires it) and `streaming` (Smart Turn alone on every
chunk).

## Speed

Measured on a 12-thread laptop CPU and an RTX 3050.

| Model | CPU | TensorRT |
|---|---|---|
| Silero VAD | 0.1 ms | 0.2 ms |
| Feature extraction (log-mel) | 3.5 ms | 0.7 ms |
| Smart Turn v3.2 | 21-26 ms (6 threads) | 2.7 ms fp32, 0.8 ms fp16 |
| turn-1-mini | about 2 ms | CPU only |

On the CPU, Smart Turn cannot be scored every 32 ms in real time, so the default scores every second
chunk. That delays decisions by 32-64 ms compared with TensorRT. If you switch Smart Turn to a GPU
backend, set "Run every N chunks" back to 1.

## Benchmark

On LiveKit's eot-bench (English, 400 turns), cross-validated:

| Model | False cut-offs @ 300 ms | Latency @ 5% cut-offs |
|---|---|---|
| **This blend** | **20.3%** | **798 ms** |
| turn-1-mini alone | 22.6% | 849 ms |
| LiveKit v1-mini | 27.8% | 1070 ms |
| Smart Turn v3.2 alone | 35.2% | 1051 ms |

The benchmark run used Smart Turn on TensorRT, scored every chunk. The blend weights were fitted on this
same dataset, so expect a smaller gain on other audio. Full results, method and limits are in
[bench/REPORT.md](bench/REPORT.md).

## Files

| Path | What it is |
|---|---|
| `server.py` | The server: models, VAD, blend, WebSocket loop, benchmark endpoint |
| `features.py` | Whisper log-mel feature extraction as an ONNX graph |
| `mini.py` | Streaming runner for turn-1-mini |
| `blend_turn.py` | Our VAD and the detector as Pipecat analyzers |
| `bot.py`, `run_bot.sh` | Pipecat voice bot that uses it (local WebRTC or Daily) |
| `console.py`, `static/console.html` | Live settings and per-turn results for the bot |
| `static/index.html` | The demo page |
| `bench/` | eot-bench adapter, cross-validation weights, results and report |
| `build_all_in_one.py`, `turn_blend.py` | Builds and runs the single-file model |
| `release/` | Publishable folder for the single-file model: model card, runner, licences |
| `NOTICE.md` | Licences and sources of the models and data |
