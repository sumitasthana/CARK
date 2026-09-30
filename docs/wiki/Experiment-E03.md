# E03: learn tasks 3 and 0

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Do both tasks remain above the learning screen before forgetting?

## Record

| Field | Value |
| --- | --- |
| Identifier | E03 |
| Category | learning |
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
| requests | ["L3", "L0"] |
| learning rate | 0.0001 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3 after own learning pct | 32.0 |
| task3 after task0 learning pct | 30.4 |
| Task 0 at end (%) | 43.8 |


## Decision

Both learning screens passed

## Interpretation

Both tasks exceeded 25%. No forget request was run.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

Test forgetting task 3 while protecting task 0.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
