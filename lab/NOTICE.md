# Third-party models and data

This repository contains no model weights. They are downloaded on first run from their publishers.

| Component | Publisher | Licence | Source |
|---|---|---|---|
| Smart Turn v3.2 | Pipecat | BSD-2-Clause | https://huggingface.co/pipecat-ai/smart-turn-v3 |
| turn-1-mini (turn-1-mini by p99lab) | p99lab | Apache-2.0 | https://huggingface.co/p99lab/turn-1-mini |
| Silero VAD v6.2 | Silero Team | MIT | https://github.com/snakers4/silero-vad |
| eot-bench harness | LiveKit | Apache-2.0 | https://github.com/livekit/eot-bench |
| eot-bench dataset | LiveKit | CC BY 4.0 | https://huggingface.co/datasets/livekit/eot-bench-data |

The blend weights in `server.py` and `bench/blend_cv.json` were fitted on the eot-bench English set.
`bench/results/comparison/` includes rows for other models; those figures are the results published in
the eot-bench repository, reproduced for comparison.
