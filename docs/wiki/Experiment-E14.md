# E14: checkpoint-based forgetting

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E14 |
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
| gamma | 5e-06 |
| steps | 50 |
| noise_samples | 10 |

Corroborated by saved notebook output; original runtime report JSON not inspected.

## Observations

| Measurement | Value |
| --- | --- |
| task3_initial_accuracy_pct | 26.0 |
| task0_initial_accuracy_pct | 44.6 |
| task3_final_accuracy_pct | 12.6 |
| task0_final_accuracy_pct | 37.4 |
| minimum_task3_accuracy_pct | 12.4 |
| maximum_task0_absolute_drift_pp | 7.2 |
| first_retention_failure_step | 36 |
| passing_steps | [] |
| accuracy_observations | 51 |


## Accuracy trace

| Step | Task 3 % | Task 0 % | Absolute drift (pp) | Passes screen | Raw norm before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | False | Not supplied |
| 1 | 23.0 | 45.0 | 0.4 | False | Not supplied |
| 2 | 22.0 | 45.6 | 1.0 | False | Not supplied |
| 3 | 21.0 | 46.0 | 1.4 | False | Not supplied |
| 4 | 19.4 | 46.2 | 1.6 | False | Not supplied |
| 5 | 18.6 | 46.4 | 1.8 | False | Not supplied |
| 6 | 18.0 | 46.6 | 2.0 | False | Not supplied |
| 7 | 17.0 | 46.2 | 1.6 | False | Not supplied |
| 8 | 16.2 | 45.8 | 1.2 | False | Not supplied |
| 9 | 16.0 | 44.6 | 0.0 | False | Not supplied |
| 10 | 16.2 | 44.6 | 0.0 | False | Not supplied |
| 11 | 16.2 | 44.8 | 0.2 | False | Not supplied |
| 12 | 16.4 | 44.6 | 0.0 | False | Not supplied |
| 13 | 17.0 | 44.6 | 0.0 | False | Not supplied |
| 14 | 16.4 | 44.6 | 0.0 | False | Not supplied |
| 15 | 16.4 | 44.4 | 0.2 | False | Not supplied |
| 16 | 16.4 | 43.8 | 0.8 | False | Not supplied |
| 17 | 15.6 | 43.4 | 1.2 | False | Not supplied |
| 18 | 15.4 | 43.2 | 1.4 | False | Not supplied |
| 19 | 15.0 | 42.4 | 2.2 | False | Not supplied |
| 20 | 14.8 | 42.8 | 1.8 | False | Not supplied |
| 21 | 14.2 | 42.8 | 1.8 | False | Not supplied |
| 22 | 13.8 | 42.4 | 2.2 | False | Not supplied |
| 23 | 13.6 | 41.8 | 2.8 | False | Not supplied |
| 24 | 13.6 | 41.4 | 3.2 | False | Not supplied |
| 25 | 13.0 | 41.2 | 3.4 | False | Not supplied |
| 26 | 13.0 | 41.4 | 3.2 | False | Not supplied |
| 27 | 12.6 | 41.2 | 3.4 | False | Not supplied |
| 28 | 12.8 | 40.8 | 3.8 | False | Not supplied |
| 29 | 13.2 | 39.8 | 4.8 | False | Not supplied |
| 30 | 13.4 | 40.2 | 4.4 | False | Not supplied |
| 31 | 14.0 | 40.2 | 4.4 | False | Not supplied |
| 32 | 14.0 | 40.0 | 4.6 | False | Not supplied |
| 33 | 13.8 | 40.0 | 4.6 | False | Not supplied |
| 34 | 13.6 | 39.8 | 4.8 | False | Not supplied |
| 35 | 13.0 | 39.8 | 4.8 | False | Not supplied |
| 36 | 13.2 | 39.2 | 5.4 | False | Not supplied |
| 37 | 13.4 | 39.0 | 5.6 | False | Not supplied |
| 38 | 13.0 | 39.2 | 5.4 | False | Not supplied |
| 39 | 13.0 | 39.0 | 5.6 | False | Not supplied |
| 40 | 12.8 | 38.6 | 6.0 | False | Not supplied |
| 41 | 13.0 | 38.0 | 6.6 | False | Not supplied |
| 42 | 13.0 | 38.0 | 6.6 | False | Not supplied |
| 43 | 12.6 | 38.0 | 6.6 | False | Not supplied |
| 44 | 12.4 | 38.0 | 6.6 | False | Not supplied |
| 45 | 12.6 | 38.2 | 6.4 | False | Not supplied |
| 46 | 12.6 | 38.0 | 6.6 | False | Not supplied |
| 47 | 12.6 | 38.0 | 6.6 | False | Not supplied |
| 48 | 12.6 | 37.8 | 6.8 | False | Not supplied |
| 49 | 12.6 | 37.8 | 6.8 | False | Not supplied |
| 50 | 12.6 | 37.4 | 7.2 | False | Not supplied |


## Supplied losses before updates

| Step | Weighted noise | Preservation | Total |
| --- | --- | --- | --- |
| 1 | 1938.36 | 0.00 | Not supplied |
| 2 | 1927.39 | 7.25 | Not supplied |
| 3 | 1921.92 | 8.82 | Not supplied |
| 4 | 1918.78 | 7.22 | Not supplied |
| 5 | 1916.59 | 5.26 | Not supplied |
| 6 | 1914.14 | 3.70 | Not supplied |
| 7 | 1910.84 | 2.73 | Not supplied |
| 8 | 1906.76 | 2.48 | Not supplied |
| 9 | 1902.21 | 2.94 | Not supplied |
| 10 | 1897.02 | 3.91 | Not supplied |
| 11 | 1891.83 | 4.95 | Not supplied |
| 12 | 1886.97 | 5.68 | Not supplied |
| 13 | 1882.44 | 5.96 | Not supplied |
| 14 | 1878.28 | 5.91 | Not supplied |
| 15 | 1874.42 | 5.72 | Not supplied |
| 16 | 1870.54 | 5.54 | Not supplied |
| 17 | 1866.52 | 5.46 | Not supplied |
| 18 | 1862.40 | 5.53 | Not supplied |
| 19 | 1857.97 | 5.78 | Not supplied |
| 20 | 1853.50 | 6.21 | Not supplied |
| 21 | 1848.98 | 6.77 | Not supplied |
| 22 | 1844.47 | 7.34 | Not supplied |
| 23 | 1839.96 | 7.80 | Not supplied |
| 24 | 1835.72 | 8.07 | Not supplied |
| 25 | 1831.51 | 8.18 | Not supplied |
| 26 | 1827.64 | 8.21 | Not supplied |
| 27 | 1823.51 | 8.25 | Not supplied |
| 28 | 1819.55 | 8.36 | Not supplied |
| 29 | 1815.45 | 8.57 | Not supplied |
| 30 | 1811.30 | 8.88 | Not supplied |
| 31 | 1806.93 | 9.26 | Not supplied |
| 32 | 1802.72 | 9.69 | Not supplied |
| 33 | 1798.42 | 10.09 | Not supplied |
| 34 | 1794.36 | 10.41 | Not supplied |
| 35 | 1790.24 | 10.63 | Not supplied |
| 36 | 1786.33 | 10.76 | Not supplied |
| 37 | 1782.49 | 10.85 | Not supplied |
| 38 | 1778.51 | 10.97 | Not supplied |
| 39 | 1774.61 | 11.15 | Not supplied |
| 40 | 1770.65 | 11.39 | Not supplied |
| 41 | 1766.61 | 11.69 | Not supplied |
| 42 | 1762.53 | 12.03 | Not supplied |
| 43 | 1758.52 | 12.36 | Not supplied |
| 44 | 1754.57 | 12.65 | Not supplied |
| 45 | 1750.76 | 12.87 | Not supplied |
| 46 | 1746.82 | 13.04 | Not supplied |
| 47 | 1743.09 | 13.17 | Not supplied |
| 48 | 1739.28 | 13.32 | Not supplied |
| 49 | 1735.46 | 13.49 | Not supplied |
| 50 | 1731.68 | 13.70 | Not supplied |


## Decision

No passing step

## Interpretation

Target accuracy approached the threshold without reaching it. Retention first failed at step 36.

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

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_151049_593433_8a010c9b.screen.json`: Reported location; original file not independently opened.
