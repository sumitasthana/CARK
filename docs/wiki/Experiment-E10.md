# E10: checkpoint-based forgetting

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E10 |
| Category | forgetting diagnostic |
| Date | 2026-09-23 |
| Date evidence | Historical log or reported artifact filename. |
| Reported source commit | Not recorded |
| Hardware | Tesla T4 (explicitly user-reported) |
| Starting state | Restore the E08 pre-forgetting checkpoint: task 3 = 26.0%, task 0 = 44.6%. |


## Settings

| Setting | Value |
| --- | --- |
| forgetting_lr | 1e-05 |
| gamma | 1e-05 |
| steps | 10 |

original configuration JSON not independently inspected

## Observations

| Measurement | Value |
| --- | --- |
| task3_initial_accuracy_pct | 26.0 |
| task0_initial_accuracy_pct | 44.6 |
| task3_final_accuracy_pct | 16.0 |
| task0_final_accuracy_pct | 44.8 |
| minimum_task3_accuracy_pct | 16.0 |
| maximum_task0_absolute_drift_pp | 2.0 |
| first_retention_failure_step | None in supplied trace |
| passing_steps | [] |
| accuracy_observations | 11 |


## Accuracy trace

| Step | Task 3 % | Task 0 % | Absolute drift (pp) | Passes screen | Raw norm before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | False | Not supplied |
| 1 | 23.0 | 45.0 | 0.4 | False | Not supplied |
| 2 | 21.6 | 46.6 | 2.0 | False | Not supplied |
| 3 | 19.4 | 46.0 | 1.4 | False | Not supplied |
| 4 | 18.4 | 44.8 | 0.2 | False | Not supplied |
| 5 | 17.4 | 44.8 | 0.2 | False | Not supplied |
| 6 | 16.6 | 44.6 | 0.0 | False | Not supplied |
| 7 | 16.0 | 45.0 | 0.4 | False | Not supplied |
| 8 | 16.2 | 44.6 | 0.0 | False | Not supplied |
| 9 | 16.0 | 44.8 | 0.2 | False | Not supplied |
| 10 | 16.0 | 44.8 | 0.2 | False | Not supplied |


## Decision

No passing step

## Interpretation

Retention stayed within the limit for ten updates, but target accuracy remained above 12%.

## Limitations

- One reported checkpoint and run per setting; not multi-seed evidence.
- Original runtime JSON was not inspected for this archive.
- Comparison with E09 also changes the reported hardware and diagnostic implementation; gamma is not the only difference.

## Next step

See the next recorded experiment; this run did not complete the reproduction.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
- [forgetting_traces.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/forgetting_traces.csv): transcribed measurements
- [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json): tracked record

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_142742_335991_88cd96f2.screen.json`: Reported location; original file not independently opened.
