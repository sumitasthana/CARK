# E06: fresh-training forgetting probe

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Can this forgetting setting reduce task 3 accuracy while preserving task 0?

## Record

| Field | Value |
| --- | --- |
| Identifier | E06 |
| Category | learning and forgetting |
| Date | Not recorded |
| Date evidence | Not recorded |
| Reported source commit | Not recorded |
| Hardware | Not recorded |
| Starting state | Freshly trained for this run; not the later E08 checkpoint. |


## Settings

| Setting | Value |
| --- | --- |
| dataset | Tiny ImageNet |
| backbone | ResNet50 |
| shared learning rate | 0.0001 |
| Noise scale (gamma) | 0.0001 |
| forget steps | 100 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3 after own learning pct | 42.6 |
| task3 before forgetting pct | 35.2 |
| task3 after forgetting pct | 10.0 |
| task0 before forgetting pct | 37.0 |
| task0 after forgetting pct | 10.0 |
| reported spill pp | 27.0 |
| final forgetting loss | 6864.8228 |
| reported forgetting seconds | 22.6 |


## Decision

Retention failed

## Interpretation

Both tasks ended at chance. Fresh training produced different starting states across these runs.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.
- The output directory reused an E04 label; gamma is user-confirmed, not inferred from that directory name.

## Next step

Restore a single saved starting model for subsequent comparisons.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
