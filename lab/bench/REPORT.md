# eot-bench results for our turn-detection architecture

Run date: 12 November 2026. Benchmark: LiveKit eot-bench, English, 400 real user turns
(`livekit/eot-bench-data`, revision `ca9d98a`, harness commit `0b5098f`).

## Result

Our blend (Silero VAD + Smart Turn v3.2 + turn-1-mini v1 + elapsed silence) scores **20.3% false cut-offs
at a 300 ms latency budget and 798 ms mean latency at a 5% cut-off budget**, cross-validated.

That is better than either model it is built from, and better than every other open model that runs
without a large GPU. It is still well behind the top three entries (LiveKit's hosted v1, p99lab turn-1,
JoinIn Baton). Deepgram Flux is ahead at the 300 ms budget but slower at the 5% cut-off budget.

| Model | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% cut-offs | Latency @ 10% cut-offs |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1 (hosted) | 9.9% | 4.5% | 543 ms | 295 ms |
| p99lab turn-1 (20 GB GPU) | 10.6% | 4.5% | 553 ms | 321 ms |
| JoinIn AI Baton (hosted) | 12.3% | 4.8% | 577 ms | 350 ms |
| Deepgram Flux (hosted) | 12.9% | 9.9% | 1151 ms | 548 ms |
| **Ours: blend, cross-validated** | **20.3%** | **7.8%** | **798 ms** | **488 ms** |
| Ours: blend, as shipped (fitted on these turns) | 19.1% | 7.1% | 723 ms | 475 ms |
| p99lab turn-1-mini | 22.6% | 9.6% | 849 ms | 580 ms |
| ultraVAD (8B LLM) | 27.7% | 11.9% | 899 ms | 663 ms |
| LiveKit Turn Detector v1-mini | 27.8% | 12.1% | 1070 ms | 698 ms |
| Smart Turn v3.2 | 35.2% | 14.8% | 1051 ms | 739 ms |
| VAD baseline (silence only) | 55.6% | 21.7% | 1600 ms | 1000 ms |

Lower is better everywhere. Rows other than ours are the results committed in the eot-bench repository.
The full table with all 15 published entries is in `results/comparison/report.md`.

Position among published entries: 5th of 12 at the 300 ms budget, 5th of 15 on latency at 5% cut-offs.

### What the blend adds over its parts

| | False cut-offs @ 300 ms | @ 600 ms | Latency @ 5% | Latency @ 10% | AUC |
|---|---:|---:|---:|---:|---:|
| Smart Turn v3.2 alone | 35.2% | 14.8% | 1051 ms | 739 ms | 0.845 |
| turn-1-mini alone | 22.6% | 9.6% | 849 ms | 580 ms | 0.970 |
| Blend, cross-validated | 20.3% | 7.8% | 798 ms | 488 ms | 0.985 |

Against Smart Turn alone the gain is large. Against turn-1-mini alone it is modest: 2.3 points at 300 ms,
1.8 points at 600 ms, 51 ms at the 5% budget and 92 ms at the 10% budget. Most of the blend's strength
comes from turn-1-mini; Smart Turn and the silence term add a smaller improvement on top.

## Which number to trust

- **Cross-validated (headline).** The blend's four weights were refitted five times, each time on 320
  turns, and each turn was scored with weights that never saw it. This is the honest figure.
- **As shipped.** The weights in `server.py` were fitted on all 400 of these turns, so this row is scored
  on its own training data. It is 1.2 points better at 300 ms and 75 ms better at the 5% budget, which is
  the size of the optimism.
- Like every row on the board, the policy (threshold, minimum silence, timeout) is chosen by the harness
  on the same 400 turns it scores.

## Does our pipeline reproduce the published numbers?

Yes. Running each model alone through our server gives nearly the same result as the published rows, so
the adapter and pipeline are faithful.

| | False cut-offs @ 300 ms | @ 600 ms | Latency @ 5% | Latency @ 10% |
|---|---:|---:|---:|---:|
| Smart Turn v3.2, published | 35.2% | 14.8% | 1051 ms | 739 ms |
| Smart Turn v3.2, our pipeline | 35.0% | 14.8% | 1022 ms | 736 ms |
| turn-1-mini, published | 22.6% | 9.6% | 849 ms | 580 ms |
| turn-1-mini, our pipeline | 23.7% | 9.8% | 862 ms | 592 ms |

turn-1-mini is about one point worse in our pipeline. Two differences explain it: we run the official
streaming ONNX graph (causal low-pass, continuous state) where the published row scores each audio prefix
from scratch in PyTorch, and we answer from the last complete 32 ms chunk, so our score can be up to 32 ms
older than the timestamp.

## How the benchmark was run

- **What was scored.** The harness asks, at every 0.1 s inside each labelled pause, "what is your
  probability that the turn is over?" It then sweeps threshold, minimum silence and timeout itself and
  reports the best trade-offs. 400 turns, 1,250 pauses, 11,191 decision points.
- **Our answer at each point** comes from a new server endpoint (`/api/bench`) that runs the same steps
  as the live page over the whole turn: Silero VAD on every 32 ms chunk, turn start after 0.2 s of speech
  with the pre-speech buffer, turn-1-mini fed every chunk, Smart Turn on the turn so far (last 8 s), then
  the blend. Smart Turn and feature extraction ran on TensorRT, Silero and turn-1-mini on CPU, as in the
  live defaults.
- **VAD gating is included.** The blend is only defined while a turn is open and our VAD hears no speech.
  Where that isn't true we report probability 0, because the live system could not fire there. This
  affected 1.7% of end-of-turn points and 6.7% of mid-turn points. Our VAD opened a turn in all 400 turns.
- **Same grid as everyone else.** Our runs use the identical span set and the identical 11,191
  (turn, pause, timestamp) points as the published runs; I verified both.
- **Speed.** 400 turns (84 minutes of audio) scored in 2 minutes 35 seconds, 32 times real time.

## What the benchmark does not measure

- **Our fixed commit rule.** The live page ends a turn when the blend is above 0.85 for 2 chunks after
  320 ms of silence. The benchmark ignores that and sweeps its own policy. Its sweep prefers a higher
  threshold with less silence for our score: 0.93 after 200 ms at the 5% budget, 0.84 after 200 ms at 10%.
  That suggests the live defaults could move to a higher threshold and a shorter minimum silence.
- **The tentative/commit split.** Starting the LLM early and cancelling on resumed speech is not part of
  the benchmark.
- **Our earlier per-turn figures.** Before this run I measured the live rule at 14.5% of turns cut off and
  605 ms mean delay on 200 of these turns. That counts a turn as cut off if any pause in it fires, and is
  not comparable with the benchmark's per-pause rates.
- **Anything but English at 16 kHz.** The blend is English-only because turn-1-mini is. Telephony audio,
  other languages, and your own callers are untested.

## Limits of this result

- The blend was designed, fitted and tuned on this one dataset. Cross-validation guards against fitting
  individual turns, not against fitting this dataset's speakers, microphones and task style.
- The blend's silence term overlaps with the minimum-silence knob the harness sweeps, so part of what the
  blend learned is something the benchmark already gives every model.
- This is a local run. Nothing was submitted to the public leaderboard.

## What I would do next

1. **Retune the live commit rule** towards what the sweep found (threshold about 0.9, minimum silence
   about 200 ms) and re-measure on the live path.
2. **Re-fit and re-score on your own recordings.** That is the only way to know whether the gain over
   turn-1-mini alone survives outside this dataset.
3. **Decide whether Smart Turn earns its place in the English path.** It adds about 2 points and 50–90 ms
   on top of turn-1-mini, at the cost of a GPU model run on every chunk. For other languages Smart Turn
   remains the only option here.

## Files

| Path (under `bench/`) | What it is |
|---|---|
| `REPORT.md` | This report |
| `results/comparison/report.md` | The harness's full comparison report, all models |
| `results/comparison/*.png` | Pareto frontier and operating-point charts |
| `results/ours_*/` | Our four runs: predictions, manifest, metrics |
| `ours_adapter.py` | The eot-bench adapters for our server |
| `blend_cv.json` | Fold assignment and per-fold blend weights |

Not in this repository: the harness itself (clone `github.com/livekit/eot-bench` into `bench/eot-bench`),
its Python environment (`bench/.venv`), and the per-turn response cache (`bench/cache/`, created on the
first run). Server side, `server.py` has the `/api/bench` endpoint and `bench_turn()`.

## Disk used

1.4 GB in `bench/` (1.3 GB of that is the harness environment), plus 155 MB for the English dataset in
the Hugging Face cache. Earlier estimate was 1.5–2 GB.

## Reproducing

With the lab server running (`./run.sh` in the project root):

```bash
cd bench/eot-bench
for a in OursBlendCV OursBlend OursSmartTurn OursMini; do
  PYTHONPATH=.. ../.venv/bin/eot-harness predict-streaming \
    --path livekit/eot-bench-data --name en --split validation \
    --revision ca9d98a9686b920a2d8c9eb984224ba9be74e4dd \
    --adapter ours_adapter:$a --output-dir output
done
```

Then copy the four `ours_*` run folders next to the published runs, run `eot-harness compute-metrics` on
each (`--score-point 0.2` for the Smart Turn-only run), and `eot-harness compare-models` on the folder.
Delete `bench/cache/` first to force the server to re-score. The harness environment needs the versions
from its lock file (`pyarrow==24.0.0`, `pandas==2.3.3`, `datasets==5.0.1`); a plain install pulls a
pyarrow that conflicts with its `numpy<2` pin.
