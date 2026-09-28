# Preparation of the persistent E08 checkpoint

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Can subsequent diagnostics share one trained starting state?

## Record

| Field | Value |
| --- | --- |
| Identifier | checkpoint-preparation |
| Category | checkpoint preparation |
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

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3_after_own_learning_pct | 36.4 |
| task3_saved_accuracy_pct | 26.0 |
| task0_saved_accuracy_pct | 44.6 |


## Decision

Checkpoint saved and restored

## Interpretation

The reported restored predictions matched the saved accuracies. This is preparation, not a successful unlearning result.

## Limitations

- Original runtime artifacts were not independently inspected for this archive.

## Next step

No separate next step was recorded.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/before_forgetting.pt`: Reported on Drive; restored in later user runs, including E15; not opened from this machine.
