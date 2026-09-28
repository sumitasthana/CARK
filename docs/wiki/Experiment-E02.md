# E02: task 3 learning at a lower rate

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does a lower learning rate let the hypernetwork learn task 3?

## Record

| Field | Value |
| --- | --- |
| Identifier | E02 |
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
| task | 3 |
| learning_rate | 0.0001 |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3_final_accuracy_pct | 32.0 |
| final_learning_loss | 1.574293 |


## Decision

Learning screen passed

## Interpretation

Task 3 exceeded 25% in this run. This is not a multi-seed hyperparameter comparison.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

Learn task 0 and recheck task 3.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
