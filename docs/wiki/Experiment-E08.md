# E08: checkpoint-based forgetting

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Does this setting meet the forgetting and retention screen from the shared checkpoint?

## Record

| Field | Value |
| --- | --- |
| Identifier | E08 |
| Category | forgetting diagnostic |
| Date | 2026-09-23 |
| Date evidence | Historical log or reported artifact filename. |
| Reported source commit | Not recorded |
| Hardware | Not recorded |
| Starting state | Restore the E08 pre-forgetting checkpoint: task 3 = 26.0%, task 0 = 44.6%. |


## Settings

| Setting | Value |
| --- | --- |
| Forgetting learning rate | 0.0001 |
| Noise scale (gamma) | 0.0001 |
| Updates | 10 |

original configuration JSON not independently inspected

## Observations

| Measurement | Value |
| --- | --- |
| Task 3 at start (%) | 26.0 |
| Task 0 at start (%) | 44.6 |
| Task 3 at end (%) | 10.0 |
| Task 0 at end (%) | 10.0 |
| Lowest task 3 accuracy (%) | 10.0 |
| Largest task 0 change (points) | 34.6 |
| First update outside task 0 limit | 1 |
| Updates that met both targets | [] |
| Accuracy measurements | 11 |


## Accuracy trace

| Step | Task 3 % | Task 0 % | Absolute drift (pp) | Passes screen | Raw norm before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | False | Not supplied |
| 1 | 16.0 | 23.4 | 21.2 | False | Not supplied |
| 2 | 13.0 | 14.6 | 30.0 | False | Not supplied |
| 3 | 12.6 | 13.6 | 31.0 | False | Not supplied |
| 4 | 12.8 | 13.6 | 31.0 | False | Not supplied |
| 5 | 13.8 | 14.2 | 30.4 | False | Not supplied |
| 6 | 14.4 | 12.4 | 32.2 | False | Not supplied |
| 7 | 14.4 | 11.2 | 33.4 | False | Not supplied |
| 8 | 11.2 | 10.8 | 33.8 | False | Not supplied |
| 9 | 10.2 | 10.2 | 34.4 | False | Not supplied |
| 10 | 10.0 | 10.0 | 34.6 | False | Not supplied |


## Supplied losses before updates

| Step | Weighted noise | Preservation | Total |
| --- | --- | --- | --- |
| 1 | 38765.68 | 0.0 | Not supplied |
| 2 | 36631.19 | 711.19 | Not supplied |
| 3 | 34988.89 | 1559.62 | Not supplied |
| 4 | 33794.32 | 2083.34 | Not supplied |
| 5 | 32940.63 | 2225.69 | Not supplied |
| 6 | 32308.67 | 2088.67 | Not supplied |
| 7 | 31802.68 | 1809.4 | Not supplied |
| 8 | 31353.69 | 1499.43 | Not supplied |
| 9 | 30910.52 | 1225.5 | Not supplied |
| 10 | 30435.68 | 1015.46 | Not supplied |


## Decision

No passing step

## Interpretation

The first update reduced retained accuracy by 21.2 points. Later target accuracy reached chance with retention already lost.

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

- `forget_trace_20260923_020958_555174.json`: Reported location; original file not independently opened.
