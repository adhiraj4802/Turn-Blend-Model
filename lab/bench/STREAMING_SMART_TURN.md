# Can Smart Turn v3.2 be made streaming, and does its score history help?

Date: 12 November 2026. Benchmark: LiveKit eot-bench, English, 400 turns, the same 1,250 pauses and 11,191
decision points as `REPORT.md`. All fitted variants are cross-validated (5 folds by turn).

## Answer

1. **Compute: yes, by shortening the input window, with the existing weights.** A Smart Turn model cut to a
   4 s window runs in 8.8 ms on the CPU instead of 23.8 ms, and the blend scores the same on the benchmark
   (19.3% vs 20.0% false cut-offs at 300 ms; the difference is inside the noise). That is fast enough to
   score every 32 ms chunk on the CPU, which the 8 s model could not do.
2. **Score history as a bias: no measurable gain.** Fifteen history features (Smart Turn's score just before
   the pause, its running average, its peak in earlier pauses, and so on) did not improve the blend. One of
   four metrics got slightly worse, the rest did not move beyond noise.
3. **Smart Turn already gives its verdict at the first silent chunk.** Appending 190 ms of digital silence
   reproduces the score it would give 190 ms later (correlation 0.99), and the raw score at 32 ms of silence
   separates true ends from pauses as well as the score at 224 ms. Waiting buys Smart Turn nothing; the
   silence itself is the extra evidence.
4. **True incremental streaming needs retraining.** No published method reuses a bidirectional Whisper
   encoder's earlier computation without a causal fine-tune.

So the turn model's *compute* latency can drop by about 60% today. Its *decision* latency cannot be
improved by re-reading Smart Turn's own history; the benchmark says the remaining error is in what Smart
Turn knows, not in how its scores are combined.

## Results on eot-bench

Lower is better in every column.

### Streaming Smart Turn on its own

| Variant | False cut-offs @ 300 ms | @ 600 ms | Latency @ 5% cut-offs | @ 10% |
|---|---:|---:|---:|---:|
| Single shot at 0.2 s of silence (how the published row is scored) | 36.5% | 14.9% | 1025 ms | 739 ms |
| Re-scored through the pause, raw score | 32.1% | 13.9% | 1036 ms | 725 ms |
| Re-scored + elapsed silence | 29.9% | 12.8% | 1060 ms | 718 ms |
| Re-scored + elapsed silence + 11 history features | 30.2% | 12.8% | 942 ms | 697 ms |

- Streaming plus silence beats single shot by 5.3 points at 300 ms and 1.9 points at 600 ms. Both are
  clear of noise. The two latency columns are not.
- History moved one column (latency at 5% cut-offs, 86 ms better, 90% interval 9 to 176 ms) and nothing
  else. No single history feature moved any column by more than about a point.

### History features inside the blend

| Variant | False cut-offs @ 300 ms | @ 600 ms | Latency @ 5% cut-offs | @ 10% |
|---|---:|---:|---:|---:|
| Blend (Smart Turn + turn-1-mini + silence) | 20.0% | 8.1% | 807 ms | 489 ms |
| Blend + Smart Turn history | 22.0% | 7.8% | 776 ms | 483 ms |
| Blend + Smart Turn and turn-1-mini history | 21.0% | 7.9% | 746 ms | 499 ms |
| Blend, gradient-boosted trees instead of logistic | 18.7% | 8.1% | 768 ms | 494 ms |

- With Smart Turn history the 300 ms column is 2.0 points worse (clear of noise). Every other change in this
  table is indistinguishable from noise.
- The baseline row reproduces the server-based run in `REPORT.md` (20.3% / 7.8% / 798 ms / 488 ms).

### Shorter input window

| Smart Turn window | CPU time per run (6 threads) | Blend: false cut-offs @ 300 ms | @ 600 ms | Latency @ 5% | @ 10% |
|---|---:|---:|---:|---:|---:|
| 8 s (stock) | 23.8 ms | 20.0% | 8.1% | 807 ms | 489 ms |
| 6 s | 12.6 ms | 20.3% | 7.9% | 794 ms | 497 ms |
| **4 s** | **8.8 ms** | **19.3%** | **7.2%** | **777 ms** | **484 ms** |
| 3 s | 6.4 ms | 20.3% | 7.8% | 776 ms | 502 ms |
| 2 s | 4.7 ms | 21.8% | 7.7% | 845 ms | 517 ms |
| 1 s | 3.3 ms | 25.1% | 8.2% | 880 ms | 523 ms |

- 4 s and 3 s are indistinguishable from the 8 s model on all four metrics.
- 2 s trends worse (2.2 points at 300 ms, 82 ms at 5%) without clearing noise. 1 s is clearly worse (5.0
  points at 300 ms).
- On one CPU thread the 4 s model takes 33.6 ms against 77.5 ms; on two threads 18.6 ms against 49.3 ms.
- Smart Turn alone follows the same pattern: 4 s is equal to 8 s, 2 s and 1 s are worse.

How the short model is made: the input is cut to the last N mel frames, the positional embedding is cut to
its last N/2 rows, and one shape constant is changed. No weights are retrained. A 4 s short model gives
nearly the same scores as the stock model fed 4 s of audio padded to 8 s, so the saving comes from dropping
padding the model was not using.

### Silence look-ahead

In the first 200 ms of a pause, digital silence was appended so Smart Turn saw about 224 ms of trailing
silence, which is what it was trained on.

| Real silence so far | AUC, raw score | AUC, with look-ahead padding | Correlation of padded score with the real score at 224 ms |
|---|---:|---:|---:|
| 32 ms | 0.810 | 0.801 | 0.99 |
| 96 ms | 0.825 | 0.821 | 1.00 |
| 160 ms | 0.832 | 0.829 | 1.00 |
| 224 ms | 0.839 | 0.839 | 1.00 |

- Padding predicts the later score almost exactly, so the "200 ms verdict" is available at 32 ms.
- It does not separate ends from pauses any better than the raw early score. It only shifts the scale (mean
  score 0.30 raw, 0.38 padded, 0.37 at a real 224 ms).
- AUC is measured per chunk against "the last word has ended". eot-bench cannot test decisions before
  200 ms of silence, so this part is not an official benchmark number.

## How much to trust these numbers

A paired bootstrap over the 400 turns (150 resamples) gives the 90% interval of the blend baseline itself as
15.6 to 23.1% at 300 ms and 632 to 938 ms at the 5% budget. Differences of a point or two, or of tens of
milliseconds, between variants cannot be told apart on this dataset. "Clear" above means the 90% interval
of the paired difference excludes zero; with this many comparisons, an isolated "clear" result such as the
86 ms history gain is weak evidence.

## What the published work says

Sourced web research; items I could not open directly are marked.

**Smart Turn itself**
- Whisper Tiny encoder plus a small head, about 8M parameters, 8 s input padded at the start. The README
  says to run it once on the whole turn when the VAD reports silence and that it is not designed for very
  short segments. https://github.com/pipecat-ai/smart-turn , https://huggingface.co/pipecat-ai/smart-turn-v3
- v3.2 added short-utterance and noise data and fixed a training padding bug.
  https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/
- No streaming, incremental or shorter-window plan was found in the README roadmap (the issue tracker was
  not browsed).

**Streaming Whisper encoders**
- CarelessWhisper makes the encoder block-causal with a LoRA fine-tune and shows that a bidirectional
  encoder's earlier outputs change when new frames arrive, so caching them is only valid after such a
  fine-tune. https://arxiv.org/html/2508.12301v1
- U2-Whisper also fine-tunes with causal masks. https://arxiv.org/abs/2506.12154
- Simul-Whisper and Whisper-Streaming need no training but recompute the whole encoder each step.
  https://arxiv.org/abs/2406.10052 , https://arxiv.org/abs/2307.14743 (second link from memory)
- Shortened context without retraining is reported to hurt Whisper ASR badly (Moonshine, whisper.cpp
  `audio_ctx`, FUTO whisper-acft), mostly through decoder hallucination. Nobody had published a number for
  an encoder-plus-classifier; the table above is that number for Smart Turn.
  https://arxiv.org/abs/2410.15608 , https://github.com/futo-org/whisper-acft (found through mirrors)

**Incremental encoders in general**
- Emformer, cache-aware FastConformer and streaming Zipformer all get their caching from training with
  limited context. None applies to a bidirectional encoder as is.
  https://arxiv.org/abs/2010.10759 , https://arxiv.org/abs/2312.17279

**Using score history**
- LiveKit adapts its endpointing delay from the end-of-turn probability and from a running average of the
  session's pauses. https://docs.livekit.io/agents/logic/turns/tuning.md (from search snippets)
- Deepgram Flux uses a tentative/commit design (eager end of turn, cancelled if the user resumes), which is
  what our tentative end already mirrors. https://developers.deepgram.com/docs/flux/voice-agent-eager-eot
- No published evidence was found, for or against, on feeding a model's own earlier in-turn scores back in
  as features. The tables above are our evidence: it does not help here.

## What this means for our system

- **Switch the blend to the 4 s Smart Turn.** Per chunk on the CPU that is about 3.5 ms features + 8.8 ms
  Smart Turn + 2.2 ms turn-1-mini, roughly 15 ms, inside the 32 ms budget. The live system could then score
  every chunk instead of every second one, which removes the 32 to 64 ms decision delay the all-CPU setup
  currently pays. I have not run this live yet; the 15 ms figure is the sum of measured parts.
- **Do not add history features.** They cost complexity and showed no gain.
- **Do not wait for silence on Smart Turn's account.** Its verdict is formed by the first silent chunk. The
  minimum-silence wait is justified only by the silence term and by turn-1-mini.
- **To go further, the model has to change, not the wrapper.** The two routes are a causal fine-tune of
  Smart Turn (CarelessWhisper style), which needs its training data and pipeline, or leaning on a model that
  is already streaming, which is the role turn-1-mini plays in the blend.

## Method and limits

- **Data path.** Variants were built from cached per-chunk traces of every turn (Silero VAD, Smart Turn at
  every in-turn chunk, turn-1-mini) and written on the benchmark's own grid, then scored with the harness's
  metric functions (`compute_metrics_from_predictions`, default sweep). They were not run through the live
  server adapter; the baseline reproducing the server run is the check that the two paths agree.
- **Window scoring.** 343,176 Smart Turn runs: every non-speech in-turn chunk, seven window settings. Scores
  with the stock 8 s model matched the cached ones to within 0.007.
- **Features for short windows** still come from the 8 s feature graph (audio padded to 8 s, last N frames
  kept), so the 3.5 ms feature cost is unchanged. A shorter feature graph would save more but changes the
  normalisation and was not tested.
- **One dataset, English, clean 16 kHz.** The 4 s result should be rechecked on your own recordings,
  especially long, slow turns where the final clause may depend on context more than 4 s back.
- **Short-model timing** was measured on this laptop (12 threads); the accuracy runs used the GPU, the
  timings used the CPU.
- Nothing was submitted to the public leaderboard.

## Files (under `bench/streaming/`)

| File | What it does |
|---|---|
| `short_model.py` | Builds a Smart Turn ONNX with a shorter input window |
| `time_short.py` | Times the short models on the CPU |
| `score_chunks.py`, `score_mini.py` | Per-chunk Smart Turn and turn-1-mini traces for every benchmark turn |
| `score_windows.py` | Smart Turn at every silent chunk for each window length |
| `score_lookahead.py` | The silence look-ahead test |
| `build_variants.py` | History features, cross-validated fits, predictions on the benchmark grid |
| `metrics_all.py` | Official metrics for every variant |
| `bootstrap.py` | Paired bootstrap over turns |
| `metrics.json`, `bootstrap_*.txt` | The numbers behind this report |

The scripts take their inputs as arguments and expect the lab's `server.py`, the benchmark parquet and, for
`score_mini.py`, the turn-1-mini ONNX wrapper from the original lab folder. They are research scripts, not
part of the demo.
