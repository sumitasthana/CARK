# Full sequence 1 on Tiny ImageNet

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
| learn_requests | 18 |
| forget_requests | 12 |
| seed | 0 |
| partition_seed | 42 |
| chunks | 200 |
| epochs | 5 |
| batch_size | 64 |
| learning_rate | 0.001 |
| beta | 0.01 |
| gamma | 0.01 |
| noise_samples | 10 |
| initial_forget_steps | 100 |
| burn_in_decay | 0.9 |
| minimum_forget_steps | 20 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| retained_accuracy_pct | 10.0 |
| forgotten_accuracy_pct | 10.0 |
| mean_spill | 30.72 |
| mean_relapse | 0.0 |
| reported_wall_minutes | 55.9 |
| reported_peak_gpu_gib | 7.72 |
| generated_parameters | 23520842 |
| hypernetwork_parameters | 134015898 |


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
