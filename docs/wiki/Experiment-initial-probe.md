# Initial learn-learn-forget probe

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Was task 3 learned well enough to evaluate forgetting?

## Record

| Field | Value |
| --- | --- |
| Identifier | initial-probe |
| Category | learning and forgetting |
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
| shared learning rate | 0.001 |
| Noise scale (gamma) | 0.01 |
| forget steps | 100 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3 after own learning pct | 14.4 |
| task3 before forgetting pct | 12.2 |
| task3 after forgetting pct | 10.0 |
| task0 before forgetting pct | 36.6 |
| task0 after forgetting pct | 34.8 |


## Decision

Invalid starting point

## Interpretation

Task 3 was below the 25% learning screen before forgetting.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

Check direct classifier learning and hypernetwork learning separately.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
