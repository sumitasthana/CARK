# Full sequence 1 on Tiny ImageNet

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does the implementation retain tasks while processing the full sequence?

## Record

| Field | Value |
| --- | --- |
| Identifier | full-sequence-20260921 |
| Category | reproduction |
| Date | 2026-09-21 |
| Date evidence | Date stated in the archived local run log. |
| Reported source commit | c6d89ce |
| Hardware | NVIDIA A100, Colab (reported) |
| Starting state | Not recorded |


## Settings

| Setting | Value |
| --- | --- |
| dataset | Tiny ImageNet |
| backbone | ResNet50 |
| sequence | 1 |
| requests | 30 |
| learn requests | 18 |
| forget requests | 12 |
| seed | 0 |
| partition seed | 42 |
| chunks | 200 |
| epochs | 5 |
| batch size | 64 |
| learning rate | 0.001 |
| beta | 0.01 |
| Noise scale (gamma) | 0.01 |
| Noise samples | 10 |
| initial forget steps | 100 |
| burn in decay | 0.9 |
| minimum forget steps | 20 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| retained accuracy pct | 10.0 |
| forgotten accuracy pct | 10.0 |
| mean spill | 30.72 |
| mean relapse | 0.0 |
| reported wall minutes | 55.9 |
| reported peak gpu gib | 7.72 |
| generated parameters | 23520842 |
| hypernetwork parameters | 134015898 |


## Decision

Reproduction failed

## Interpretation

Retained accuracy is at chance. Chance forgotten accuracy and zero relapse do not establish successful selective forgetting.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

Use the short learn-learn-forget diagnostic to investigate retention failure.

## Evidence

- [full_sequence_20260921.md](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/full_sequence_20260921.md): transcribed local run note. Summarises ops-docs/notes/run-log.html; underlying runtime JSON not inspected.
- [PLAN.md](https://github.com/sumitasthana/CARK/blob/main/docs/PLAN.md): tracked record

## Reported artifact locations

- `costs_seq1_resnet50_seed0.json`: Named in the local run note; not located or inspected.
