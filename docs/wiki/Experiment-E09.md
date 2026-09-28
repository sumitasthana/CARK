# E09: checkpoint-based forgetting

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E09 |
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
| gamma | 0.0001 |
| steps | 10 |

original configuration JSON not independently inspected

## Observations

| Measurement | Value |
| --- | --- |
| task3_initial_accuracy_pct | 26.0 |
| task0_initial_accuracy_pct | 44.6 |
| task3_final_accuracy_pct | 15.8 |
| task0_final_accuracy_pct | 30.0 |
| minimum_task3_accuracy_pct | 15.8 |
| maximum_task0_absolute_drift_pp | 14.6 |
| first_retention_failure_step | 6 |
| passing_steps | [] |
| accuracy_observations | 11 |


## Accuracy trace

| Step | Task 3 % | Task 0 % | Absolute drift (pp) | Passes screen | Raw norm before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | False | Not supplied |
| 1 | 23.0 | 45.0 | 0.4 | False | Not supplied |
| 2 | 21.2 | 46.6 | 2.0 | False | Not supplied |
| 3 | 18.8 | 44.8 | 0.2 | False | Not supplied |
| 4 | 17.4 | 43.4 | 1.2 | False | Not supplied |
| 5 | 16.0 | 41.8 | 2.8 | False | Not supplied |
| 6 | 16.2 | 39.2 | 5.4 | False | Not supplied |
| 7 | 16.6 | 37.0 | 7.6 | False | Not supplied |
| 8 | 16.8 | 35.2 | 9.4 | False | Not supplied |
| 9 | 16.4 | 32.6 | 12.0 | False | Not supplied |
| 10 | 15.8 | 30.0 | 14.6 | False | Not supplied |


## Supplied losses before updates

| Step | Weighted noise | Preservation | Total |
| --- | --- | --- | --- |
| 1 | 38765.68 | 0.0 | Not supplied |
| 2 | 38546.75 | 7.25 | Not supplied |
| 3 | 38331.81 | 27.22 | Not supplied |
| 4 | 38116.39 | 57.94 | Not supplied |
| 5 | 37906.2 | 97.46 | Not supplied |
| 6 | 37700.46 | 144.03 | Not supplied |
| 7 | 37498.27 | 196.03 | Not supplied |
| 8 | 37302.28 | 252.03 | Not supplied |
| 9 | 37112.92 | 310.69 | Not supplied |
| 10 | 36926.21 | 370.77 | Not supplied |


## Decision

No passing step

## Interpretation

The smaller forgetting learning rate delayed retention failure to step 6; no step passed both criteria.

## Limitations

- One reported checkpoint and run per setting; not multi-seed evidence.
- Original runtime JSON was not inspected for this archive.

## Next step

See the next recorded experiment; this run did not complete the reproduction.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
- [forgetting_traces.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/forgetting_traces.csv): transcribed measurements
- [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json): tracked record

## Reported artifact locations

- `E09_forget_lr_1e-5_20260923_022053_354220.json`: Reported location; original file not independently opened.
