# E12: checkpoint-based forgetting

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E12 |
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
| gamma | 3e-06 |
| steps | 30 |

original configuration JSON not independently inspected

## Observations

| Measurement | Value |
| --- | --- |
| task3_initial_accuracy_pct | 26.0 |
| task0_initial_accuracy_pct | 44.6 |
| task3_final_accuracy_pct | 13.0 |
| task0_final_accuracy_pct | 42.8 |
| minimum_task3_accuracy_pct | 13.0 |
| maximum_task0_absolute_drift_pp | 1.8 |
| first_retention_failure_step | None in supplied trace |
| passing_steps | [] |
| accuracy_observations | 31 |


## Accuracy trace

| Step | Task 3 % | Task 0 % | Absolute drift (pp) | Passes screen | Raw norm before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | False | Not supplied |
| 1 | 23.0 | 45.0 | 0.4 | False | Not supplied |
| 2 | 22.6 | 45.2 | 0.6 | False | Not supplied |
| 3 | 21.8 | 45.4 | 0.8 | False | Not supplied |
| 4 | 21.0 | 45.8 | 1.2 | False | Not supplied |
| 5 | 19.2 | 45.4 | 0.8 | False | Not supplied |
| 6 | 18.6 | 45.8 | 1.2 | False | Not supplied |
| 7 | 17.4 | 46.2 | 1.6 | False | Not supplied |
| 8 | 16.4 | 46.0 | 1.4 | False | Not supplied |
| 9 | 15.8 | 44.6 | 0.0 | False | Not supplied |
| 10 | 16.0 | 44.4 | 0.2 | False | Not supplied |
| 11 | 16.2 | 44.2 | 0.4 | False | Not supplied |
| 12 | 16.0 | 44.6 | 0.0 | False | Not supplied |
| 13 | 16.2 | 44.4 | 0.2 | False | Not supplied |
| 14 | 16.8 | 44.6 | 0.0 | False | Not supplied |
| 15 | 16.8 | 44.6 | 0.0 | False | Not supplied |
| 16 | 16.4 | 44.8 | 0.2 | False | Not supplied |
| 17 | 16.2 | 44.2 | 0.4 | False | Not supplied |
| 18 | 16.2 | 44.2 | 0.4 | False | Not supplied |
| 19 | 15.8 | 43.8 | 0.8 | False | Not supplied |
| 20 | 15.4 | 43.8 | 0.8 | False | Not supplied |
| 21 | 15.2 | 43.4 | 1.2 | False | Not supplied |
| 22 | 15.0 | 43.4 | 1.2 | False | Not supplied |
| 23 | 14.8 | 43.4 | 1.2 | False | Not supplied |
| 24 | 14.2 | 43.4 | 1.2 | False | Not supplied |
| 25 | 14.0 | 43.4 | 1.2 | False | Not supplied |
| 26 | 14.4 | 43.2 | 1.4 | False | Not supplied |
| 27 | 14.0 | 43.2 | 1.4 | False | Not supplied |
| 28 | 13.6 | 42.8 | 1.8 | False | Not supplied |
| 29 | 13.0 | 42.8 | 1.8 | False | Not supplied |
| 30 | 13.0 | 42.8 | 1.8 | False | Not supplied |


## Decision

No passing step

## Interpretation

Retention passed throughout, but target accuracy did not reach 12%.

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

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143653_273412_d31bb28c.screen.json`: Reported location; original file not independently opened.
