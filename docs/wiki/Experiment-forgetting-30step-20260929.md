# 30-step forgetting diagnostic from the saved checkpoint

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

Task 3 fell from 26.0% to 12.6% by update 27. That is close, but the target is 12% or lower. Task 0 changed by 3.6 percentage points at that update, within its 5-point limit. No update met both targets. The gradient table shows signal sizes, not whether the signals point in opposite directions.

## Question

Does extending the 10-step run to 30 steps reach the joint forgetting and retention screen?

## Record

| Field | Value |
| --- | --- |
| Identifier | forgetting-30step-20260929 |
| Category | gradient diagnostic |
| Date | 2026-09-29 |
| Date evidence | UTC date in the reported JSON filename; original file not opened. |
| Reported source commit | fcdc1ff |
| Hardware | Not recorded |
| Starting state | Restore the E08 pre-forgetting checkpoint: task 3 = 26.0%, task 0 = 44.6%. |


## Settings

| Setting | Value |
| --- | --- |
| Forgetting learning rate | 1e-05 |
| Noise scale (gamma) | 5e-06 |
| Updates | 30 |
| Noise samples | 10 |

Printed settings and status in user-supplied output; original report JSON not inspected.

## Observations

| Measurement | Value |
| --- | --- |
| Task 3 at start (%) | 26.0 |
| Task 0 at start (%) | 44.6 |
| Task 3 at end (%) | 13.4 |
| Task 0 at end (%) | 40.0 |
| Lowest task 3 accuracy (%) | 12.6 |
| Largest task 0 change (points) | 4.8 |
| First update outside task 0 limit | None in supplied trace |
| Updates that met both targets | [] |
| Accuracy measurements | 31 |
| Reported run status | complete |
| Valid starting accuracy reported | True |
| Updates with reported gradients | [1, 2, 5, 10, 20, 25, 27, 30] |
| Raw output size before update 1 | 19080 |
| Raw output size before update 30 | 18400 |


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
| 11 | 16.2 | 44.8 | 0.2 | False | 1.884e+04 |
| 12 | 16.4 | 44.6 | 0.0 | False | 1.881e+04 |
| 13 | 17.0 | 44.4 | 0.2 | False | 1.879e+04 |
| 14 | 16.4 | 44.2 | 0.4 | False | 1.877e+04 |
| 15 | 16.4 | 44.4 | 0.2 | False | 1.874e+04 |
| 16 | 16.4 | 43.8 | 0.8 | False | 1.872e+04 |
| 17 | 15.8 | 43.6 | 1.0 | False | 1.87e+04 |
| 18 | 15.4 | 43.4 | 1.2 | False | 1.868e+04 |
| 19 | 15.0 | 42.4 | 2.2 | False | 1.866e+04 |
| 20 | 14.8 | 42.8 | 1.8 | False | 1.863e+04 |
| 21 | 14.2 | 42.8 | 1.8 | False | 1.861e+04 |
| 22 | 13.8 | 42.2 | 2.4 | False | 1.858e+04 |
| 23 | 13.6 | 41.8 | 2.8 | False | 1.856e+04 |
| 24 | 13.6 | 41.4 | 3.2 | False | 1.854e+04 |
| 25 | 13.0 | 41.2 | 3.4 | False | 1.851e+04 |
| 26 | 12.8 | 41.4 | 3.2 | False | 1.849e+04 |
| 27 | 12.6 | 41.0 | 3.6 | False | 1.847e+04 |
| 28 | 12.8 | 40.6 | 4.0 | False | 1.845e+04 |
| 29 | 13.2 | 39.8 | 4.8 | False | 1.843e+04 |
| 30 | 13.4 | 40.0 | 4.6 | False | 1.84e+04 |


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
| 20 | heads.batchnorm | 2.851 | 0.9567 |
| 20 | heads.residual | 15.92 | 12.37 |
| 20 | heads.weights | 123.5 | 102 |
| 20 | trunk | 2667 | 2252 |
| 25 | heads.batchnorm | 2.824 | 1.116 |
| 25 | heads.residual | 15.76 | 16.08 |
| 25 | heads.weights | 122.2 | 111 |
| 25 | trunk | 2636 | 2510 |
| 27 | heads.batchnorm | 2.815 | 1.167 |
| 27 | heads.residual | 15.71 | 14.4 |
| 27 | heads.weights | 121.8 | 103.3 |
| 27 | trunk | 2625 | 2301 |
| 30 | heads.batchnorm | 2.801 | 1.24 |
| 30 | heads.residual | 15.63 | 13.26 |
| 30 | heads.weights | 121.1 | 100.7 |
| 30 | trunk | 2608 | 2204 |


## Decision

No passing step; best target accuracy was 12.6% at step 27 while retained drift was 3.6 points.

## Interpretation

More updates lowered target accuracy after step 10, but no update reached 12%. Retained accuracy fell later in the run. Gradient norms near step 27 show both terms are active but do not give their directions.

## Limitations

- One reported checkpoint and one run; not multi-seed evidence.
- Original runtime JSON was not inspected from this workspace.
- GPU, runtime, loss components, and full gradient vectors were not supplied.
- Gradient and raw-output norms retain the printed precision only.
- The code that measures gradient cosine and Adam update size was added after this run; those measurements cannot be reconstructed from norms.

## Next step

Run the same setup with gradient cosine, actual Adam update norms, and a component-state audit. Pair recovery against matched controls before making residual-knowledge claims.

## Evidence

- [forgetting_30step_20260929.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/forgetting_30step_20260929.txt): user-pasted notebook output. Transcribed accuracies, raw norms, and selected gradient norms.
- [forgetting_traces.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/forgetting_traces.csv): transcribed accuracy measurements
- [gradient_norms.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/gradient_norms.csv): transcribed gradient measurements
- [raw_output_norms.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/raw_output_norms.csv): transcribed raw-output measurements
- [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json): tracked settings and provenance

## Reported artifact locations

- `/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/E15_gradients/forget_20260929_021642_430512_33bd5249.json`: Reported on Drive; original JSON not opened from this workspace.
