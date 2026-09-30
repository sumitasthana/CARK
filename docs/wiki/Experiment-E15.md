# E15: gradient diagnostic

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

How do noise and preservation gradients compare across parameter groups?

## Record

| Field | Value |
| --- | --- |
| Identifier | E15 |
| Category | gradient diagnostic |
| Date | 2026-09-28 |
| Date evidence | UTC date encoded in the reported JSON filename; not an independently read runtime timestamp. |
| Reported source commit | e3087f0 |
| Hardware | Not recorded |
| Starting state | Restore the E08 pre-forgetting checkpoint: task 3 = 26.0%, task 0 = 44.6%. |


## Settings

| Setting | Value |
| --- | --- |
| Forgetting learning rate | 1e-05 |
| Noise scale (gamma) | 5e-06 |
| Updates | 10 |
| Noise samples | 10 |

Settings, commit, complete status and valid start were included in the pasted output. Original runtime JSON has not been inspected.

## Observations

| Measurement | Value |
| --- | --- |
| Task 3 at start (%) | 26.0 |
| Task 0 at start (%) | 44.6 |
| Task 3 at end (%) | 16.2 |
| Task 0 at end (%) | 44.6 |
| Lowest task 3 accuracy (%) | 16.0 |
| Largest task 0 change (points) | 1.8 |
| First update outside task 0 limit | None in supplied trace |
| Updates that met both targets | [] |
| Accuracy measurements | 11 |
| Reported run status | complete |
| Valid starting accuracy reported | True |
| Updates with reported gradients | [1, 2, 5, 10] |
| Raw output size before update 1 | 19080 |
| raw norm before step10 | 18860 |


## Accuracy trace

| Step | Task 3 % | Task 0 % | Absolute drift (pp) | Passes screen | Raw norm before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | False | Not supplied |
| 1 | 23.0 | 45.0 | 0.4 | False | 1.908e+04 |
| 2 | 22.0 | 45.6 | 1.0 | False | 1.903e+04 |
| 3 | 21.0 | 46.0 | 1.4 | False | 1.9e+04 |
| 4 | 19.2 | 46.2 | 1.6 | False | 1.898e+04 |
| 5 | 18.6 | 46.4 | 1.8 | False | 1.897e+04 |
| 6 | 18.0 | 46.4 | 1.8 | False | 1.896e+04 |
| 7 | 17.0 | 46.0 | 1.4 | False | 1.894e+04 |
| 8 | 16.4 | 45.8 | 1.2 | False | 1.892e+04 |
| 9 | 16.0 | 44.6 | 0.0 | False | 1.889e+04 |
| 10 | 16.2 | 44.6 | 0.0 | False | 1.886e+04 |


## Supplied gradient norms before updates

Noise includes gamma. Values retain printed precision.

| Step | Group | Weighted noise gradient | Preservation gradient |
| --- | --- | --- | --- |
| 1 | heads.batchnorm | 2.959 | 0 |
| 1 | heads.residual | 16.5 | 0 |
| 1 | heads.weights | 128.2 | 0 |
| 1 | trunk | 2786 | 0 |
| 2 | heads.batchnorm | 2.95 | 0.1124 |
| 2 | heads.residual | 16.42 | 30.51 |
| 2 | heads.weights | 127.6 | 153.1 |
| 2 | trunk | 2771 | 3493 |
| 5 | heads.batchnorm | 2.932 | 0.2842 |
| 5 | heads.residual | 16.37 | 16.87 |
| 5 | heads.weights | 127 | 134.6 |
| 5 | trunk | 2756 | 2928 |
| 10 | heads.batchnorm | 2.904 | 0.5428 |
| 10 | heads.residual | 16.22 | 15.57 |
| 10 | heads.weights | 125.9 | 102.8 |
| 10 | trunk | 2728 | 2309 |


## Decision

No passing step

## Interpretation

Preservation gradients are zero before update 1 and substantial by update 2. Their magnitudes vary by parameter group. Norms alone do not establish cancellation or Adam update contributions.

## Limitations

- One reported checkpoint and run per setting; not multi-seed evidence.
- Original runtime JSON was not inspected for this archive.
- GPU and runtime were not supplied.
- Gradient and raw-output norms have the precision printed in the supplied table.
- Loss components and full gradient vectors were not supplied.
- Commit 4c08f54 changed forgetting-noise generation; E15 is not an exact replay of E14.

## Next step

Measure gradient alignment, combined gradients, and actual update norms before attributing the failure to cancellation.

## Evidence

- [EXPERIMENT_LOG.md](https://github.com/sumitasthana/CARK/blob/main/docs/EXPERIMENT_LOG.md): reported GPU results. Historical user reports transcribed in the experiment log; not rerun for this archive.
- [forgetting_traces.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/forgetting_traces.csv): transcribed measurements
- [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json): tracked record
- [e15_reported_output.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/e15_reported_output.txt): user-pasted notebook output
- [gradient_norms.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/gradient_norms.csv): transcribed measurements
- [raw_output_norms.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/raw_output_norms.csv): transcribed measurements

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/E15_gradients/forget_20260928_011538_895545_c81f33b2.json`: Reported location; original file not independently opened.
