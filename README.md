# turn-blend

Voice activity detection and end-of-turn detection for voice agents in **one ONNX file**. Feed it 32 ms
chunks of microphone audio; it tells you when the speaker has started a turn and when they have finished.

turn-blend is not a newly trained network. It is three existing open models wired together in a single
graph, with a small fusion step fitted on top. All credit for the models belongs to their authors:

| Inside the file | By | Licence | What it does here |
|---|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) v6.2 | Silero Team | MIT | Labels each chunk speech or not speech |
| [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | Pipecat / Daily | BSD-2-Clause | Scores whether the turn so far sounds finished |
| [turn-1-mini](https://huggingface.co/p99lab/turn-1-mini) (turn-1-mini by p99lab) | p99lab | Apache-2.0 | A second, streaming end-of-turn score |

Smart Turn is itself a fine-tune of OpenAI's Whisper Tiny encoder (MIT). The weights of all three models are
unchanged. What turn-blend adds is the glue: a log-mel front end, a logistic blend of the two end-of-turn
scores with the elapsed silence, and the turn logic (turn start, tentative end, commit, timeout).

The graphs were joined, not averaged: nothing was retrained. This project is not affiliated with or endorsed
by Silero, Pipecat, Daily, p99lab, OpenAI or LiveKit.

| In this repository | |
|---|---|
| [`turn-blend.onnx`](turn-blend.onnx) | The model (56 MB, stored with Git LFS) |
| [`turn_blend.py`](turn_blend.py) | Runner: numpy + onnxruntime only |
| [`integrations/`](integrations) | Drop-in files for **LiveKit Agents** and **Pipecat / Daily** |
| [`examples/`](examples) | A complete voice agent for each framework |
| [`benchmark/`](benchmark) | eot-bench results, latency metrics, raw artifacts |
| [`configure_demo.py`](configure_demo.py) | Try any setting on a WAV file from the command line |
| [`lab/`](lab) | The development project: browser demo, research reports, and the script that builds the model |

## Quick start

```bash
git lfs install                     # the model file is stored with Git LFS
git clone https://github.com/adhiraj4802/Turn-Blend-Model.git
cd Turn-Blend-Model
pip install numpy onnxruntime
```

If `turn-blend.onnx` is only a few hundred bytes after cloning, Git LFS was not installed; run
`git lfs install && git lfs pull`.

```python
from turn_blend import TurnBlend

detector = TurnBlend("turn-blend.onnx")
for chunk in chunks:                       # 512 float32 samples each, 16 kHz mono, values in [-1, 1]
    r = detector(chunk)
    if r["end_of_turn"]:
        print("the speaker has finished")
```

`python example.py recording.wav` prints the turns found in a 16 kHz WAV file.

Without the helper class it is three inputs and a loop:

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("turn-blend.onnx")
state = np.zeros(331606, np.float32)       # zeros start a new stream
*out, state = sess.run(None, {"audio": chunk, "state": state})
```

## Use it in a LiveKit agent

Copy `turn_blend.py`, `turn-blend.onnx` and [`integrations/livekit_turn_blend.py`](integrations/livekit_turn_blend.py)
next to your agent. turn-blend plugs in as the session's VAD and reports "end of speech" only when it decides
the turn is over, so LiveKit's `vad` turn detection ends turns exactly where the model does.

```python
from livekit_turn_blend import TurnBlendVAD

session = AgentSession(
    stt=..., llm=..., tts=...,
    vad=TurnBlendVAD("turn-blend.onnx"),
    turn_handling={
        "turn_detection": "vad",
        "endpointing": {"mode": "fixed", "min_delay": 0.0, "max_delay": 3.0},
    },
)
```

A complete agent (Deepgram STT and TTS, OpenAI LLM) is in [`examples/livekit_agent.py`](examples/livekit_agent.py).
Written against livekit-agents 1.8. It does not use LiveKit's own turn-detector models.

## Use it in a Pipecat agent (Daily or any other transport)

Copy `turn_blend.py`, `turn-blend.onnx` and [`integrations/pipecat_turn_blend.py`](integrations/pipecat_turn_blend.py)
next to your bot. turn-blend plugs in as the turn analyzer; Pipecat's own VAD still starts the turn and
handles interruptions.

```python
from pipecat_turn_blend import TurnBlendAnalyzer

aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
    vad_analyzer=SileroVADAnalyzer(),
    user_turn_strategies=UserTurnStrategies(stop=[
        TurnAnalyzerUserTurnStopStrategy(turn_analyzer=TurnBlendAnalyzer("turn-blend.onnx")),
    ]),
))
worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000))   # 16 kHz input is required
```

A complete bot is in [`examples/daily_bot.py`](examples/daily_bot.py): `-t daily` joins a Daily room, `-t webrtc`
uses Pipecat's local WebRTC with no account. Written against pipecat-ai 1.16.

How the two integrations were tested is in [Integration status](#integration-status) below.

## Inputs and outputs

| Input | Shape | Meaning |
|---|---|---|
| `audio` | float32 [512] | One 32 ms chunk at 16 kHz |
| `state` | float32 [331606] | Zeros to start, then the previous call's `new_state` |
| `config` | float32 [16] | Optional. The defaults are stored in the file |

| Output | Meaning |
|---|---|
| `vad` | Silero's speech probability for this chunk |
| `speech` | 1 if the chunk counts as speech |
| `in_turn` | 1 while a user turn is open |
| `silence_ms` | Silence since the last speech chunk |
| `smart_turn` | Smart Turn's probability, or -1 if it was not scored on this chunk |
| `mini` | turn-1-mini's probability |
| `blend` | Blended end-of-turn probability, or -1 if not scored on this chunk |
| `tentative` | 1 while a tentative end of turn is pending (an agent can start its LLM and hold the output) |
| `end_of_turn` | 0 = no, 1 = the blend committed, 2 = the silence timeout ended the turn |
| `new_state` | Pass back in as `state` |

Smart Turn is the expensive part, so it sits inside an ONNX `If` and runs only while a turn is open, on a
non-speech chunk, on the scoring cadence. Every other chunk skips it.

## How a turn ends

```
p = sigmoid(2.80 + 0.29*logit(P_smart_turn) + 0.49*logit(P_turn_1_mini) + 1.29*ln(silence seconds))

tentative end:  p > 0.5  after 160 ms of silence      (cancelled if speech resumes)
committed end:  p > 0.85 after 320 ms of silence, on two scores in a row
timeout:        3 s of silence ends the turn regardless
```

## Settings

Pass any of these to `TurnBlend(...)`, `TurnBlendVAD(...)`, `TurnBlendAnalyzer(...)` or `detector.configure(...)`.
They can also come from a JSON file or from `TURN_BLEND_<SETTING>` environment variables (`load_settings()`),
and `python configure_demo.py --list` prints them all. They are the `config` input, in this order.

| Setting | Default | Meaning |
|---|---|---|
| `vad_confidence` | 0.7 | A chunk is speech when Silero's probability is at least this |
| `vad_start_secs` | 0.2 | Continuous speech needed to open a turn |
| `vad_min_db` | -80 | Chunks quieter than this (RMS dBFS) never count as speech |
| `pre_speech_ms` | 500 | Audio from before the turn opened that Smart Turn also hears |
| `smart_turn_max_secs` | 8 | How much of the turn Smart Turn hears (at most 8) |
| `score_every_chunks` | 2 | Score every N chunks while a turn is open |
| `blend_threshold` | 0.85 | Commit: blend above this ... |
| `blend_min_silence_ms` | 320 | ... after at least this much silence ... |
| `blend_confirm_scores` | 2 | ... on this many scores in a row |
| `tentative_threshold` | 0.5 | Tentative end: blend above this ... |
| `tentative_min_silence_ms` | 160 | ... after at least this much silence |
| `timeout_secs` | 3 | Silence that ends the turn even if the blend never commits |
| `w_bias`, `w_smart_turn`, `w_mini`, `w_log_silence` | 2.8007, 0.2939, 0.4873, 1.2880 | The blend weights |

## Benchmark

Full results, the latency metrics and the raw harness artifacts are in [`benchmark/`](benchmark).

[LiveKit eot-bench](https://github.com/livekit/eot-bench), English, 400 turns, scored with the official
harness. Lower is better.

| Model | False cut-offs @ 300 ms | @ 600 ms | Latency @ 5% cut-offs | @ 10% |
|---|---:|---:|---:|---:|
| **turn-blend, this file, cross-validated** | **20.1%** | **7.8%** | **798 ms** | **487 ms** |
| turn-blend, this file, weights as shipped | 19.1% | 7.1% | 722 ms | 495 ms |
| turn-1-mini alone (published row) | 22.6% | 9.6% | 849 ms | 580 ms |
| LiveKit Turn Detector v1-mini (published row) | 27.8% | 12.1% | 1070 ms | 698 ms |
| Smart Turn v3.2 alone (published row) | 35.2% | 14.8% | 1051 ms | 739 ms |
| VAD only, silence timer (published row) | 55.6% | 21.7% | 1600 ms | 1000 ms |

Read these with care:

- **The blend weights were fitted on this benchmark's own 400 turns.** The "as shipped" row is therefore
  scored on its training data. The cross-validated row refits the four weights on 320 turns and scores the
  other 80, five times, and is the fair figure. Both rows use the scores this file produces.
- **The benchmark is small.** A bootstrap over the turns puts the 90% interval of the cross-validated
  result at roughly 16 to 23% (300 ms) and 630 to 940 ms (5% budget).
- **The benchmark sweeps its own decision rule** over the `blend` output. It does not test the built-in
  commit rule, the tentative end or the timeout.
- Larger or hosted models score better on this benchmark (the best published entries are near 10% at 300 ms).

## Latency

**The built-in decision rule** (default settings) on the same 400 turns, measured from the true end of the
last word to the moment the model ends the turn:

| | |
|---|---:|
| Median delay | 512 ms |
| Mean delay | 630 ms |
| 90th percentile | 1168 ms |
| Turns ended within 600 ms / 1000 ms | 62% / 84% |
| Turns it ended too early at least once | 14.5% |
| Turns that needed the 3 s timeout | 0.5% |
| Tentative end, median delay | 224 ms (a median 256 ms before the commit) |

**In a live call** (Pipecat, local WebRTC, Deepgram STT, `gpt-5.4-nano`, Deepgram Aura-2, benchmark speech
played as the microphone): turn-blend decided 426 to 803 ms after the speaker stopped, and the bot started
speaking 1.6 to 2.2 s after the speaker stopped. Most of that total is the LLM and speech synthesis.

## Does the single file behave like the original pipeline?

It was checked against the Python pipeline it was built from (separate ONNX files and Python turn logic):

- **Scores:** on benchmark turns, Smart Turn, turn-1-mini and blend outputs differ by at most 0.000001.
- **Turn logic:** on 16,948 chunks of multi-turn audio containing 47 turn ends, every `in_turn`,
  `tentative` and `end_of_turn` value was identical.
- **Benchmark:** the original pipeline scored 20.3% / 7.8% / 798 ms / 488 ms cross-validated; this file
  scores 20.1% / 7.8% / 798 ms / 487 ms.

## Speed

Measured on a 12-thread laptop CPU with onnxruntime 1.31, 60 s of speech.

| Threads | Ordinary chunk | Chunk where Smart Turn runs | 60 s of audio processed in |
|---|---:|---:|---:|
| 6 | 2.3 ms | 25.6 ms | 8.4 s |
| 2 | 2.6 ms | 45.1 ms | 12.1 s |
| 1 | 3.2 ms | 79.9 ms | 18.4 s |

Smart Turn runs on about 1 chunk in 12. With the default cadence a scored chunk has 64 ms before the next
one, so use at least 2 threads for real-time use.

## Limits

- **English only**, because turn-1-mini is English only.
- **16 kHz mono**, exactly 512 samples per call. One `state` per audio stream.
- **CPU is the tested path.** The `If` node keeps Smart Turn conditional on the CPU provider. GPU providers
  were not tested with this file.
- **Fitted and evaluated on one dataset** of clean, single-speaker turns. Telephony audio, other languages,
  heavy noise and overlapping speakers are untested.
- **Audio only.** It does not read a transcript.
- File size 56 MB. SHA-256: `f755abf26761688d22e0cd5f05ced041d5d2b155da50e54eb7330a64ef29474c`

## Integration status

| Integration | What was tested | What was not |
|---|---|---|
| Pipecat, local WebRTC | A real WebRTC call from a headless browser with recorded speech as the microphone: 6 user turns, all ended by turn-blend, bot replied each time | A human on a real microphone |
| Pipecat, Daily | The same bot code path; only the transport differs | A Daily room: the test account could not join rooms (no payment method), so this is untested end to end |
| LiveKit Agents | The VAD event stream against the model run directly (identical turn starts and ends, at 16 and 48 kHz input); `AgentSession` accepts the options; the example agent loads | A live LiveKit room or console session: no LiveKit credentials were available |

## Licences and credit

This file redistributes third-party model weights. Each component stays under its own licence; the full
texts are in `LICENSE` and `licenses/`, and `NOTICE` carries the notices that must travel with the file.

- Silero VAD: MIT, Copyright (c) 2020-present Silero Team.
- Smart Turn v3.2: BSD-2-Clause, Copyright (c) 2024-2025, Daily.
- turn-1-mini: Apache-2.0, Copyright 2026 p99lab. Credit: "turn-1-mini by p99lab",
  https://huggingface.co/p99lab/turn-1-mini. The names "p99lab" and "turn-1-mini" are used here only to
  identify the component.
- Whisper Tiny (upstream of Smart Turn): MIT, Copyright (c) 2022 OpenAI.
- The blend weights were fitted on [livekit/eot-bench-data](https://huggingface.co/datasets/livekit/eot-bench-data)
  (CC BY 4.0 per its dataset card).
- The fusion weights, turn logic, graph assembly and helper code added by this project: Apache-2.0.

The same credits are embedded in the ONNX file's metadata (`components`, `blend_fitted_on`).
