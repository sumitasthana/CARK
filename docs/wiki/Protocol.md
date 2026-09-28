# Protocol and interpretation

The short diagnostic is `L3 L0 U3`: learn task 3, learn task 0, then forget task 3
while protecting task 0. Tiny ImageNet accuracy uses its labelled validation split.
Each task has ten classes. The common intended model is ResNet50 generated in
200 chunks, with hidden widths 128/256/512 and 32-dimensional task/chunk codes.
Common defaults are intended settings, not verified metadata for every run.

The E08 checkpoint has reported starting accuracies of 26.0% for task 3 and 44.6%
for task 0. E08-E15 restore that starting state for each diagnostic. Within a
forget request, keep one continuous Adam optimiser and one frozen reference.
Calling one-step forget requests repeatedly changes the experiment.

## Diagnostic screen

- Both initial accuracies must be at least 25%.
- After an update, target-task accuracy must be at most 12%.
- Absolute retained-task accuracy drift must be strictly below five percentage points.
- Step zero is a starting observation, never a passing update.

The full reproduction has separate criteria in [PLAN.md](https://github.com/sumitasthana/CARK/blob/main/docs/PLAN.md). A short
screen pass would identify a candidate for validation, not establish deletion.
Learning-only and numerical checks are not judged by the forgetting screen.

## Timing and units

Accuracies are percentages; drift is in percentage points. Losses, raw-output
norms, and gradient norms are measured before their named update. Accuracy is
measured after it. Step 10's raw norm therefore describes the model after nine
updates. E15 noise-gradient norms already include gamma.

## Comparison limits

E04-E07 start from separately trained models. E10 also changes the reported GPU
and helper relative to E09. E11-E14 requested the same T4 runtime, but lack
separate per-run hardware evidence. Commit `4c08f54` introduces separate task-code
and forgetting-noise streams; E15 reports commit `e3087f0`. Matching E14's gamma
and learning rate therefore does not make E15 an exact replay of its trajectory.

The preservation penalty constrains generated weights, not accuracy directly.
At the initial snapshot its gradient is zero. Gradient magnitudes do not reveal
their relative directions or their contributions to an Adam update. Measurements
for the BatchNorm head concern generated scale and offset parameters; per-task
running means and variances are separate stored buffers.

## Next work, not yet a result

Measure gradient alignment, combined gradients, actual parameter updates, and
raw/scaled generated-weight changes from the same checkpoint. These are proposed
diagnostics. No gradient-cancellation finding, relearning advantage, or component
storage attribution has been established by the archived runs.
