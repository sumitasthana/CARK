# Full-sequence run: source summary

Transcribed from `ops-docs/notes/run-log.html`, a local record dated 21 September
2026 naming commit `c6d89ce`. The original runtime JSON and console output were
not inspected for this archive. These are reported measurements, not a new run.

The record names Tiny ImageNet sequence 1, seed 0, an A100 GPU, a ResNet50 target,
200 chunks, five epochs, batch size 64, Adam learning rate 0.001, beta 0.01,
gamma 0.01, and ten noise samples. It reports 18 learn and 12 forget requests.

| Measurement | Reported value |
| --- | --- |
| Retained accuracy | 10.00% |
| Forgotten accuracy | 10.00% |
| Mean spill | 30.72 |
| Mean relapse | 0.00 |
| Wall time | 55.9 minutes |
| Peak GPU memory | 7.72 GiB |
| Generated parameters | 23,520,842 |
| Hypernetwork parameters | 134,015,898 |
| Chunks: weights / residual / BatchNorm | 174 / 24 / 2 |

The underlying cost artifact is named `costs_seq1_resnet50_seed0.json`; its
current location and availability have not been established. Cost-model fits
and projections in the local note are not additional completed experiments.
