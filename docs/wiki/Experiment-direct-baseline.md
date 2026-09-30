# Direct ResNet50 baseline on task 3

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Can the target architecture learn task 3 without the hypernetwork?

## Record

| Field | Value |
| --- | --- |
| Identifier | direct-baseline |
| Category | learning baseline |
| Date | Not recorded |
| Date evidence | Not recorded |
| Reported source commit | Not recorded |
| Hardware | Not recorded |
| Starting state | Not recorded |


## Settings

| Setting | Value |
| --- | --- |
| dataset | Tiny ImageNet |
| backbone | ResNet50 |
| task | 3 |
| epochs | 5 |
| learning rate | 0.001 |
| hypernetwork | False |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| validation accuracy pct by epoch | [38.4, 24.2, 55.6, 47.0, 53.8] |


## Decision

Learning demonstrated

## Interpretation

Direct training reached 53.8% final validation accuracy. This does not identify which hypernetwork choice caused weaker learning.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

Check the hypernetwork learning path.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
