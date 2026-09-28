# E13: checkpoint-based forgetting

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E13 |
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
| steps | 50 |
| noise_samples | 10 |

Corroborated by saved notebook output; original runtime report JSON not inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3_initial_accuracy_pct | 26.0 |
| task0_initial_accuracy_pct | 44.6 |
| task3_final_accuracy_pct | 13.2 |
| task0_final_accuracy_pct | 43.8 |
| minimum_task3_accuracy_pct | 13.0 |
| maximum_task0_absolute_drift_pp | 1.8 |
| first_retention_failure_step | None in supplied trace |
| passing_steps | [] |
| accuracy_observations | 51 |


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
| 31 | 13.0 | 42.8 | 1.8 | False | Not supplied |
| 32 | 13.2 | 42.8 | 1.8 | False | Not supplied |
| 33 | 13.2 | 42.8 | 1.8 | False | Not supplied |
| 34 | 13.4 | 42.8 | 1.8 | False | Not supplied |
| 35 | 13.6 | 42.8 | 1.8 | False | Not supplied |
| 36 | 13.8 | 42.8 | 1.8 | False | Not supplied |
| 37 | 13.8 | 42.8 | 1.8 | False | Not supplied |
| 38 | 14.0 | 42.8 | 1.8 | False | Not supplied |
| 39 | 13.8 | 42.8 | 1.8 | False | Not supplied |
| 40 | 13.2 | 43.0 | 1.6 | False | Not supplied |
| 41 | 13.4 | 43.0 | 1.6 | False | Not supplied |
| 42 | 13.8 | 43.0 | 1.6 | False | Not supplied |
| 43 | 13.0 | 43.2 | 1.4 | False | Not supplied |
| 44 | 13.4 | 43.2 | 1.4 | False | Not supplied |
| 45 | 13.4 | 43.4 | 1.2 | False | Not supplied |
| 46 | 13.4 | 43.4 | 1.2 | False | Not supplied |
| 47 | 13.4 | 43.4 | 1.2 | False | Not supplied |
| 48 | 13.2 | 43.4 | 1.2 | False | Not supplied |
| 49 | 13.0 | 43.4 | 1.2 | False | Not supplied |
| 50 | 13.2 | 43.8 | 0.8 | False | Not supplied |


## Supplied losses before updates

| Step | Weighted noise | Preservation | Total |
| --- | --- | --- | --- |
| 1 | 1163.01 | 0.0 | 1163.01 |
| 10 | 1140.43 | 3.66 | 1144.09 |
| 20 | 1117.63 | 4.15 | 1121.78 |
| 30 | 1095.24 | 5.29 | 1100.53 |
| 40 | 1073.38 | 6.5 | 1079.88 |
| 50 | 1052.21 | 7.66 | 1059.87 |


## Decision

No passing step

## Interpretation

Extending the E12 setting to 50 steps preserved retention but did not bring target accuracy to the threshold.

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
- [notebook_saved_outputs.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/notebook_saved_outputs.txt): saved notebook output. Confirms settings; setup hardware and commit are session evidence, not per-run metadata.
- [e13_sampled_losses.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/e13_sampled_losses.csv): sampled measurements

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143926_526429_aa83a19b.screen.json`: Reported location; original file not independently opened.
