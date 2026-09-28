# E11: checkpoint-based forgetting

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E11 |
| Category | forgetting diagnostic |
| Date | 2026-09-23 |
| Date evidence | Historical log or reported artifact filename. |
| Reported source commit | Not recorded |
| Hardware | Not recorded |
| Starting state | Restore the E08 pre-forgetting checkpoint: task 3 = 26.0%, task 0 = 44.6%. |


## Settings

| Setting | Value |
| --- | --- |
| forgetting_lr | 1e-05 |
| gamma | 1e-05 |
| steps | 30 |

original configuration JSON not independently inspected

## Observations

| Measurement | Value |
| --- | --- |
| task3_initial_accuracy_pct | 26.0 |
| task0_initial_accuracy_pct | 44.6 |
| task3_final_accuracy_pct | 13.4 |
| task0_final_accuracy_pct | 36.4 |
| minimum_task3_accuracy_pct | 12.6 |
| maximum_task0_absolute_drift_pp | 8.2 |
| first_retention_failure_step | 23 |
| passing_steps | [] |
| accuracy_observations | 31 |


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
| 11 | 16.4 | 44.6 | 0.0 | False | Not supplied |
| 12 | 16.8 | 44.2 | 0.4 | False | Not supplied |
| 13 | 16.4 | 43.6 | 1.0 | False | Not supplied |
| 14 | 16.4 | 43.4 | 1.2 | False | Not supplied |
| 15 | 16.2 | 43.2 | 1.4 | False | Not supplied |
| 16 | 15.8 | 42.6 | 2.0 | False | Not supplied |
| 17 | 15.0 | 42.0 | 2.6 | False | Not supplied |
| 18 | 15.0 | 41.4 | 3.2 | False | Not supplied |
| 19 | 14.2 | 41.6 | 3.0 | False | Not supplied |
| 20 | 13.4 | 40.6 | 4.0 | False | Not supplied |
| 21 | 13.6 | 40.2 | 4.4 | False | Not supplied |
| 22 | 13.4 | 39.8 | 4.8 | False | Not supplied |
| 23 | 12.8 | 39.4 | 5.2 | False | Not supplied |
| 24 | 12.6 | 38.6 | 6.0 | False | Not supplied |
| 25 | 12.8 | 38.6 | 6.0 | False | Not supplied |
| 26 | 12.8 | 38.6 | 6.0 | False | Not supplied |
| 27 | 13.2 | 38.2 | 6.4 | False | Not supplied |
| 28 | 13.8 | 37.8 | 6.8 | False | Not supplied |
| 29 | 13.6 | 37.6 | 7.0 | False | Not supplied |
| 30 | 13.4 | 36.4 | 8.2 | False | Not supplied |


## Decision

No passing step

## Interpretation

Longer forgetting reduced target accuracy further, but retention first failed at step 23. The minimum target accuracy was still above 12%.

## Limitations

- One reported checkpoint and run per setting; not multi-seed evidence.
- Original runtime JSON was not inspected for this archive.
- The same T4 runtime was requested; a separate per-run hardware identity was not supplied.

## Next step

See the next recorded experiment; this run did not complete the reproduction.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
- [forgetting_traces.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/forgetting_traces.csv): transcribed measurements
- [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json): tracked record

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143249_572038_f52e5457.screen.json`: Reported location; original file not independently opened.
