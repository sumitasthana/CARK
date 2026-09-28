# E07: fresh-training forgetting probe

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Can this forgetting setting reduce task 3 accuracy while preserving task 0?

## Record

| Field | Value |
| --- | --- |
| Identifier | E07 |
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
| shared_learning_rate | 0.0001 |
| gamma | 0.0001 |
| forget_steps | 10 |
| minimum_forget_steps | 10 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3_after_own_learning_pct | 37.0 |
| task3_before_forgetting_pct | 39.6 |
| task3_after_forgetting_pct | 10.0 |
| task0_before_forgetting_pct | 48.8 |
| task0_after_forgetting_pct | 10.0 |
| reported_spill_pp | 38.8 |
| final_forgetting_loss | 36058.9922 |
| reported_forgetting_seconds | 3.1 |


## Decision

Retention failed

## Interpretation

Both tasks ended at chance. Fresh training produced different starting states across these runs.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

Restore a single saved starting model for subsequent comparisons.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
