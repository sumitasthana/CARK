# Rules for reading results

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

The short test is `L3 L0 U3`: learn task 3, learn task 0, then try to forget
task 3 while keeping task 0. Accuracy comes from Tiny ImageNet validation images.
Each task has ten classes. The intended model is ResNet50 with a hypernetwork
that generates weights in 200 chunks. These are intended settings; older run
records do not confirm every setting.

The E08 checkpoint starts at 26.0% on task 3 and 44.6% on task 0. Later
checkpoint tests restore it before forgetting. Each test makes one continuous
forget request. Restarting the request after every step would change the test.

## What counts as a pass?

- Both tasks must start at 25% accuracy or higher.
- After at least one update, task 3 must be at 12% or lower.
- Task 0 must stay within 5 percentage points of its starting accuracy.

For example, task 0 starts at 44.6%. A result at 41.0% has changed by 3.6
points and is within the limit. A result at 39.6% has changed by 5 points and
fails. Passing this short test would only make a run worth studying further.
It would not prove that task 3's information is gone. The full study has
separate rules in [PLAN.md](https://github.com/sumitasthana/CARK/blob/main/docs/PLAN.md).

## When are measurements taken?

The loss, raw weight size, and gradient size at a step are measured before its
update. Accuracy is measured after the update. In the printed tables, the raw
size at step 10 describes the model after nine updates. The noise gradient
numbers already include gamma.

## Which runs can we compare?

E04-E07 used separately trained models. E10 also changed the reported GPU and
helper relative to E09. E11-E14 requested the same T4 runtime, but we lack
separate hardware records for each run. E15 used code with separate random
streams for task codes and forgetting noise. Even with the same learning rate
and gamma, E14 and E15 are not exact replays.

The preservation term tries to hold generated weights near saved values. It
does not hold accuracy directly. Its gradient is zero at the saved starting
state. Gradient size alone cannot tell us whether two gradients oppose each
other or how Adam will update the weights. BatchNorm running statistics are
stored separately from generated BatchNorm parameters.

## What remains unknown?

New code can inspect weights, gradients, and diagonal Fisher scores. No GPU
result from that complete inspection has been reported. These measurements
do not, by themselves, show that task information was removed or locate where
it is stored. See [Model diagnostics](https://github.com/sumitasthana/CARK/wiki/Model-diagnostics).
