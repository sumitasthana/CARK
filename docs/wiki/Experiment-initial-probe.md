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
| shared_learning_rate | 0.001 |
| gamma | 0.01 |
| forget_steps | 100 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3_after_own_learning_pct | 14.4 |
| task3_before_forgetting_pct | 12.2 |
| task3_after_forgetting_pct | 10.0 |
| task0_before_forgetting_pct | 36.6 |
| task0_after_forgetting_pct | 34.8 |


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
