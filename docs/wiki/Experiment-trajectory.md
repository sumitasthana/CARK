# Experiment trajectory

Updated 6 October 2026. This page is the experiment record. It contains completed results, their limits, and the next controlled comparison. Files linked as evidence are measurements or original transcriptions, not separate experiment reports.

## Current finding and next experiment

Learning a new task can lower older-task scores even without an unlearning request. In the latest continuation, task 17 reached 53.4%, while the four older tasks lost 6.25 percentage points on average. Task 9 fell from 48.4% to 32.6%. The protection term contributed to the loss, but that does not establish its gradient strength or identify the cause of the decline.

Next, use notebook 14 to compare beta values `0`, `0.001`, `0.01`, `0.1`, and `1` in one A100 session. Every value starts from the same confirmed checkpoint before task 17. The completed beta-0.01 result is reused. The other four trials run sequentially and save separate outputs. This tests protection strength for one lesson and one source model; it is not a multi-seed or full-sequence result. No new trial from this batch has been reported yet.

The earlier cross-session replay problem remains unexplained. Its first recorded mismatch was at training update 1, after matching recorded inputs. A single new beta trial should therefore be treated as diagnostic evidence rather than a settled ranking.

## Contents

- [Session replay](#session-replay-5-october)
- [Learning without unlearning](#learning-without-unlearning-5-and-6-october)
- [Hyperfan learning runs](#hyperfan-learning-runs-6-october)
- [Task-17 continuation](#task-17-continuation-beta-001)
- [History through 4 October](#history-through-4-october)
- [Historical run notes through 29 September](#historical-run-notes-through-29-september)
- [Learning audit and its qualification](#learning-audit-and-its-qualification)
- [Hyperfan implementation and CPU checks](#hyperfan-implementation-and-cpu-checks)
- [Paper recheck and diagnostic validation](#paper-recheck-and-diagnostic-validation)

## Session replay, 5 October

The user reported two fresh-session L9 replays from the same selected post-U3 model:

| Run | Final task 0 (%) | Final task 9 (%) |
| --- | --- | --- |
| A | 29.2 | 43.8 |
| B | 36.4 | 42.2 |

The comparator's first recorded difference was at training step 1: model buffers, loss, and parameters differed. It did not list image order or input values among those first differences. Environment or setup metadata also differed. This narrows the first observed mismatch but does not identify its cause. The supplied output does not contain the full environment comparison or source fingerprint.

Reported comparison: `/content/drive/MyDrive/uncle/E08_forgetting_trace/L9_session_replay/20261005_L9_replay_01/comparison.json`.

For a later pair named `20261005_L9_same_gpu_01`, run A's session ended after reporting saved training results. Run B subsequently reported 395 saved training steps and Drive flushing. No final A/B comparison for that pair was supplied, so it is not recorded as a reproducibility success. Deterministic settings were discussed; no supplied controlled result establishes that they resolve the earlier mismatch.

## Learning without unlearning, 5 and 6 October

Two reported learning-only runs started from task 3 = 26.0% and task 0 = 44.6%, then learned 9, 5, and 17. The printed heading called these test scores; Tiny ImageNet evaluation in this project uses validation images.

| Beta | Stage | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | After learning 9 | 19.0 | 40.6 | 41.6 | Not learned | Not learned |
| 0.01 | After learning 5 | 16.2 | 33.0 | 34.8 | 42.4 | Not learned |
| 0.01 | After learning 17 | 22.4 | 29.4 | 29.8 | 44.4 | 41.2 |
| 0.1 | After learning 9 | 15.4 | 40.0 | 39.0 | Not learned | Not learned |
| 0.1 | After learning 5 | 14.2 | 34.4 | 31.0 | 36.4 | Not learned |
| 0.1 | After learning 17 | 18.8 | 35.8 | 29.4 | 29.8 | 34.0 |

Both runs show old-task decline during learning. Higher beta helped task 0 in this pair but lowered several other final scores. This pair alone does not establish a reliable beta effect. These models use the historical initialization and are not the source for the later Hyperfan continuation.

## Hyperfan learning runs, 6 October

The revised initialization was tested with learning-only sequences. The following tables are separate runs; the four-task and five-task outputs have different scores from their first lessons. They cannot be combined into a single uninterrupted trajectory. Continuation from an explicitly selected checkpoint was subsequently implemented in notebook 14.

| Run | Stage | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Three tasks | After learning 3 | 46.0 | Not learned | Not learned | Not learned | Not learned |
| Three tasks | After learning 0 | 42.2 | 40.6 | Not learned | Not learned | Not learned |
| Three tasks | After learning 9 | 39.0 | 44.9 | 51.2 | Not learned | Not learned |
| Four tasks | After learning 3 | 46.8 | Not learned | Not learned | Not learned | Not learned |
| Four tasks | After learning 0 | 43.4 | 47.0 | Not learned | Not learned | Not learned |
| Four tasks | After learning 9 | 34.8 | 34.0 | 47.8 | Not learned | Not learned |
| Four tasks | After learning 5 | 27.0 | 31.2 | 48.4 | 54.8 | Not learned |
| Five tasks | After learning 3 | 43.2 | Not learned | Not learned | Not learned | Not learned |
| Five tasks | After learning 0 | 43.8 | 37.0 | Not learned | Not learned | Not learned |
| Five tasks | After learning 9 | 39.0 | 20.4 | 46.8 | Not learned | Not learned |
| Five tasks | After learning 5 | 37.6 | 26.0 | 48.8 | 50.8 | Not learned |
| Five tasks | After learning 17 | 35.6 | 18.6 | 37.4 | 49.2 | 52.6 |

The four-task source was inspected in Colab. It contains completed tasks `['3', '0', '9', '5']`, scores `{'3': 27.0, '0': 31.2, '9': 48.4, '5': 54.8}`, initialization `hyperfan_in`, and learning rate `0.0001`.

Confirmed source: `/content/drive/MyDrive/uncle/learning_initialization/20261006_hyperfan_L3_L0_L9_seed0_02/checkpoint_seq1_resnet50_seed0.pt`.

Hyperfan learning scores do not establish that the initialization change alone improves retention. These are one-seed runs, and the comparison does not isolate each initialization component.

## Task-17 continuation, beta 0.01

The user supplied notebook 14's five-epoch output for teaching task 17 from the selected checkpoint after tasks 3, 0, 9, and 5. Source scores below are from the user's earlier checkpoint inspection. The Drive report, source hash, and gradient samples have not been inspected locally. The notebook's final file-check and GPU-release cell was still pending in the supplied output.

Source: `/content/drive/MyDrive/uncle/learning_initialization/20261006_hyperfan_L3_L0_L9_seed0_02/checkpoint_seq1_resnet50_seed0.pt`.

Result: `/content/drive/MyDrive/uncle/learning_loss_diagnostic/20261006_learning_loss_01/beta_0_01/checkpoint.pt`.

| Task | Saved starting score (%) | After learning 17 (%) | Change (percentage points) |
| --- | --- | --- | --- |
| 3 | 27.0 | 24.6 | -2.4 |
| 0 | 31.2 | 28.0 | -3.2 |
| 9 | 48.4 | 32.6 | -15.8 |
| 5 | 54.8 | 51.2 | -3.6 |
| 17 | Not supplied | 53.4 | Not calculated |

| Epoch | New-task loss | Beta times protection | Task 3 (%) | Task 0 (%) | Task 9 (%) | Task 5 (%) | Task 17 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.8112 | 0.8174 | 19.2 | 26.6 | 50.0 | 50.0 | 35.8 |
| 2 | 1.4661 | 0.4982 | 26.2 | 40.8 | 43.8 | 49.2 | 43.8 |
| 3 | 1.3163 | 0.4087 | 24.8 | 29.6 | 37.8 | 50.6 | 45.4 |
| 4 | 1.1408 | 0.3998 | 25.6 | 27.0 | 43.8 | 47.0 | 50.8 |
| 5 | 1.0235 | 0.4102 | 24.6 | 28.0 | 32.6 | 51.2 | 53.4 |

The mean old-task score fell from 40.35% to 34.10%, a loss of 6.25 percentage points. The notebook displayed this as -6.2 points. Task 9 had the largest final drop. Old-task scores fluctuated across epochs, so the decline was not monotonic.

The positive protection term shows that protection contributed to the objective. Its scalar value does not establish the strength or direction of its gradient, or prove that increasing beta will improve retention. This run confirms older-task degradation during the new lesson under these settings; it does not identify the full cause or establish a result across seeds.

For the batch comparison, finish notebook 14's final cell, then use a fresh A100 session with `MODE = "RUN"`, `EXPERIMENT_ID = "20261006_learning_loss_01"`, and `BETAS = (0.0, 0.001, 0.01, 0.1, 1.0)`. The saved selection is reused; no new selection or retraining of the source tasks is needed. Compare new-task accuracy, mean old-task change, and the largest old-task drop. The beta batch changes only the weight of protection during task 17, keeping the checkpoint, code, seed, images, learning rate, and training budget fixed.

## History through 4 October

This is the historical report through 4 October. Its status and proposed steps describe that date; the current decision is at the top of this page.

**Status report.** Tiny ImageNet, ResNet50 backbone, Colab T4 and
A100. All figures are drawn from the evidence bundle `asof_20261004`, exported at
commit `489c9e8`.

This report traces twenty-one recorded runs in the order they were performed. For
each stage it states the question asked, the measurement obtained, and the decision
that followed.

### Contents of the historical report

- [Summary](#summary)
- [Paper settings and our first full run](#paper-settings-and-our-first-full-run)
- [Definitions and evaluation criteria](#definitions-and-evaluation-criteria)
- [Stage 1. Initial full-sequence reproduction](#stage-1-initial-full-sequence-reproduction)
- [Stage 2. Isolating the learning failure](#stage-2-isolating-the-learning-failure)
- [Stage 3. Unlearning from independently trained models](#stage-3-unlearning-from-independently-trained-models)
- [Stage 4. Parameter study from a fixed checkpoint](#stage-4-parameter-study-from-a-fixed-checkpoint)
- [Stage 5. Full sequence with the selected setting](#stage-5-full-sequence-with-the-selected-setting)
- [Stage 6. Paired task-9 comparison and replay variance](#stage-6-paired-task-9-comparison-and-replay-variance)
- [Stage 7. Component-level weight audit](#stage-7-component-level-weight-audit)
- [Numerical checks](#numerical-checks)
- [Current status](#current-status)
- [Provenance and limitations](#provenance-and-limitations)

### Summary

- **Goal.** Remove one task from a trained model without degrading the tasks we
  keep.

- **Result after 21 runs.** No configuration has met the criteria. The best
  attempt reached 12.6% on the target task, against a criterion of 12% or lower,
  while the protected task moved 3.6 points.

- **Main finding.** In the full sequence, most tasks had already fallen to chance
  accuracy of 10% *before* their unlearn request ran. The model loses earlier
  tasks while learning later ones. Unlearning is therefore being tested on tasks
  that hold almost nothing. This is a continual-learning failure, not an
  unlearning failure, and it has to be fixed first.

- **Measurement is blocked.** Reruns from the same checkpoint, with identical
  settings, differ by 11 points across sessions. Any effect smaller than that
  cannot be measured today.

- **Best lead.** The loss term that protects retained tasks is roughly 0.5% of
  the total, 8.2 against 1823.6. It is too small to steer the update. A related
  numerical check suggests the unlearning term may drive outputs toward zero
  rather than toward noise.

- **What works.** The experimental pipeline. Checkpoints restore exactly, file
  hashes match, and every run records its settings, code commit, and runtime.

- **Next.** Rebalance the two loss terms and repeat the 30-step diagnostic. In
  parallel, find the cause of the cross-session variance, and measure continual
  learning on its own with no unlearn requests.

### Paper settings and our first full run

This compares the paper's Tiny ImageNet Sequence 1 with our first full run on
21 September. The paper's results average three seeds. Our result is one
reported run with seed 0.

| Setting or result | Paper | Our first full run |
| --- | ---: | ---: |
| Backbone | ResNet50 | ResNet50 |
| Generated-weight chunks | 200 | 200 |
| Learning rate | 0.001 | 0.001 |
| Learning regularization (beta) | 0.01 | 0.01 |
| Unlearning weight (gamma) | 0.01 | 0.01 |
| Noise samples per update | 10 | 10 |
| Unlearning updates | 100 initially; then 10% fewer per unlearn request, minimum 20 | 100 initially; 10% decay and minimum 20 were intended but not checked in the original config |
| Request sequence | Sequence 1, 30 requests | Sequence 1, 30 requests |
| Retained-task accuracy | 55.24% | 10.00% |
| Forgotten-task accuracy | 10.00% | 10.00% |
| Mean spill per unlearn request | 0.722 points | 30.72 points |

The matching settings above are *reported settings*, not proof that every
implementation detail matched. Our original run configuration and result JSON
were not inspected for this archive. At 10% retained accuracy, our 10% forgotten
accuracy does not show selective forgetting. Paper values come from its
[implementation, Appendix C and Tables 1 and 2](https://arxiv.org/html/2509.17530).
Our values come from the
[transcribed run note](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/full_sequence_20260921.md).

### Definitions and evaluation criteria

| Term | Definition |
| --- | --- |
| task | A group of 10 image classes. The dataset is divided into 20 tasks. Random guessing yields **10%**. |
| L3 | **Learn** task 3. The model is trained on that task. |
| U3 | **Unlearn** task 3. The model is updated to remove that task. |
| target task | The task being removed. Its accuracy should fall. |
| protected task | A task being kept. Its accuracy should remain stable. |
| drift | Movement of a protected task, in either direction, in percentage points. |
| spill | Total accuracy change across all other tasks during one unlearn request. |
| gamma | Strength of the unlearning term. Larger values remove more and damage more. |

A run is recorded as passing only if all three criteria hold at the **same** update
step.

| Criterion | Threshold | Reason |
| --- | --- | --- |
| Valid start | both tasks ≥ 25% | Unlearning cannot be evaluated on a task that was never learned. |
| Removal | target ≤ 12% | The target must approach chance accuracy of 10%. |
| Retention | drift < 5 points | The protected task must remain close to its starting accuracy. |

These are screening criteria, not proof of deletion. Passing them identifies a
configuration worth studying further. Low accuracy can still conceal knowledge that
later recovers.

### Stage 1. Initial full-sequence reproduction

On 21 September the complete sequence was run end to end: 30 requests, 18 learn and
12 unlearn, on one A100 in 56 minutes. The initial unlearning budget was 100
updates, with learning rate 0.001 and gamma 0.01.

| Measure | Result | Intended | Reading |
| --- | ---: | ---: | --- |
| Accuracy on retained tasks | 10.0% | high | chance accuracy |
| Accuracy on unlearned tasks | 10.0% | ≈ 10% | uninformative here |
| Mean spill per unlearn request | 30.7 pts | < 3 | severe degradation |

Every task finished at chance. When the retained tasks are also at chance, a low
score on the unlearned task carries no information. This raised a question that had
not yet been tested in isolation: can the model learn a single task at all?

### Stage 2. Isolating the learning failure

The next four records tested learning alone, with no unlearning. A ResNet50 trained
directly reached 53.8%. The hypernetwork at the same learning rate reached 14.4%.
Reducing the learning rate by a factor of ten resolved this.

| Record | Setup | Learning rate | Task 3 | Task 0 | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| initial-probe | hypernetwork | 0.001 | 14.4% | not run | below the 25% criterion |
| direct-baseline | plain ResNet50 | 0.001 | 53.8% | not run | learns normally |
| E02 | hypernetwork | 0.0001 | 32.0% | not run | passed |
| E03 | hypernetwork, L3 then L0 | 0.0001 | 30.4% | 43.8% | both passed |

E03 provides the first valid starting state in the project: two tasks learned, both
above the 25% criterion, no unlearning attempted. The hypernetwork reaches roughly
32% where a direct network reaches 54%. That gap remains unexplained, and the two
runs are not a controlled comparison.

### Stage 3. Unlearning from independently trained models

Experiments E04 to E07 each trained a new model and then unlearned task 3 at a
different gamma. All four finished with both tasks at exactly 10%.

| Record | Gamma | Updates | Task 3 before | Task 3 after | Task 0 before | Task 0 after | Spill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E04 | 0.01 | 100 | 26.4% | 10.0% | 41.6% | 10.0% | 31.6 |
| E05 | 0.001 | 100 | 35.8% | 10.0% | 34.8% | 10.0% | 24.8 |
| E06 | 0.0001 | 100 | 35.2% | 10.0% | 37.0% | 10.0% | 27.0 |
| E07 | 0.0001 | 10 | 39.6% | 10.0% | 48.8% | 10.0% | 38.8 |

Reducing gamma by a factor of 100 produced no change. Reducing the update count
from 100 to 10 produced no change. These four runs are not comparable with one
another, because each began from a separately trained model: task 0 started at
34.8% in one run and 48.8% in another.

This produced the most consequential procedural decision in the project. One model
was trained, evaluated, and saved to disk at task 3 = 26.0% and task 0 = 44.6%. All
subsequent experiments restore that file, so differences between runs reflect
settings rather than variation during training.

### Stage 4. Parameter study from a fixed checkpoint

With a fixed starting state, the unlearning step could be studied systematically.
Nine runs varied the unlearning learning rate, gamma, and update count.

![E08 accuracy trace over ten unlearning updates](figures/2026-10-04/c1a.svg)

E08 used a diagnostic learning rate of 0.0001 and gamma 1e-4, rather than the
paper's 0.001 and 0.01. Task 0 fell from 44.6% to 23.4% on the first
update. By update 10 both tasks were at 10%.

![30-step run accuracy trace](figures/2026-10-04/c1b.svg)

The 30-step run is the best result obtained. Task 0 held near 44% for 15 updates.
Task 3 reached 12.6% at update 27, then rose again.

In both figures the blue line is task 3 (being unlearned), the green line is task 0
(protected), the shaded band marks the 5-point retention limit, and the dotted line
marks chance accuracy at 10%.

| Run | Gamma | Updates | Task 3 at end | Task 0 at end | Worst drift | Retention lost at |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E08 (lr 0.0001) | 1e-4 | 10 | 10.0% | 10.0% | 34.6 | update 1 |
| E09 | 1e-4 | 10 | 15.8% | 30.0% | 14.6 | update 6 |
| E10 | 1e-5 | 10 | 16.0% | 44.8% | 2.0 | never |
| E11 | 1e-5 | 30 | 13.4% | 36.4% | 8.2 | update 23 |
| E12 | 3e-6 | 30 | 13.0% | 42.8% | 1.8 | never |
| E13 | 3e-6 | 50 | 13.2% | 43.8% | 1.8 | never |
| E14 | 5e-6 | 50 | 12.6% | 37.4% | 7.2 | update 36 |
| E15 | 5e-6 | 10 | 16.2% | 44.6% | 1.8 | never |
| **30-step run** | 5e-6 | 30 | 13.4% | 40.0% | 4.8 | never |

All runs start from task 3 = 26.0% and task 0 = 44.6%. The unlearning learning rate
is 0.00001 except in E08.

A consistent trade-off runs through the table. Configurations that protect task 0
leave task 3 near 16%. Configurations that reduce task 3 further degrade task 0. The
figure below applies both criteria jointly: for each run it reports the lowest task
3 accuracy reached at any update where task 0 remained within five points.

![Lowest task 3 accuracy per run within the retention limit](figures/2026-10-04/c2.svg)

The shaded region on the left is the passing range. No run entered it. The two
closest runs, E14 and the 30-step run, both stopped at 12.6%.

**Loss-term imbalance.** The 30-step run recorded both components of the objective.
At update 27 the unlearning term was 1823.6 and the preservation term was 8.2. The
preservation term is approximately 0.5% of the total, roughly 220 times smaller. The
component intended to protect task 0 has little influence on the update direction.

### Stage 5. Full sequence with the selected setting

The selected configuration, 27 updates at gamma 5e-6, was applied to the full
thirty-request sequence, resuming from the saved checkpoint. The first unlearn
request behaved as expected: task 3 fell from 26.0% to 12.6% with 3.6 points of
spill. Performance degraded substantially thereafter.

| Measure | Result | Intended |
| --- | ---: | ---: |
| Accuracy on retained tasks | 22.4% | above 40% |
| Mean spill per unlearn request | 19.0 pts | under 3 |
| Accuracy on unlearned tasks | 10.6% | ≈ 10% |

The third figure does not indicate successful removal. The request history records
each task's accuracy before every request, and shows that the tasks had already
decayed.

![Task accuracy after learning and immediately before its unlearn request](figures/2026-10-04/c3.svg)

Green bars show each task's accuracy immediately after it was learned. Red bars show
the same task immediately before its unlearn request. Nine of the twelve tasks were
already at or near chance accuracy.

Task 0 is the clearest case. It was learned to 44.6%. By request 21, when its
unlearn request ran, it had fallen to 11.8% with no unlearning applied to it.
Learning other tasks alone caused this. Task 1 fell from 41.8% to 10.0%, task 12
from 39.0% to 10.0%, and task 6 from 44.2% to 10.0%.

This changes what the experiment measures. An unlearn request applied to a task
already at 10% cannot be evaluated. It also indicates that the continual-learning
component is failing before the unlearning component is tested. Adjusting the
unlearning step alone will not correct the sequence.

![Spill at each unlearn request](figures/2026-10-04/c4.svg)

Degradation extends well beyond the target task. Three individual unlearn requests
cost more than 30 points across the other tasks. The later requests appear small
only because little accuracy remained to lose.

### Stage 6. Paired task-9 comparison and replay variance

A separate question was posed: does unlearning task 3 leave the model more
susceptible to later interference? To test this, two copies of the model learned
task 9, one starting before the unlearn request and one after. If the unlearned copy
loses more accuracy on task 0, the unlearn request left a measurable effect.

| Attempt | Loss without U3 | Loss after U3 | Difference |
| --- | ---: | ---: | ---: |
| First attempt | 10.2 pts | 17.4 pts | 7.2 pts |
| Second attempt, single session | 6.0 pts | 10.2 pts | 4.2 pts |

Both attempts point in the same direction, but the magnitude moved by 3 points
between them. The measurement noise proved larger than the effect.

![Task 0 accuracy after learning task 9, across repeated runs](figures/2026-10-04/c5.svg)

All five runs restored the identical checkpoint, with matching file hashes, matching
settings, and the same recorded GPU and library versions. Across sessions the
results spread by 11 points. Within a single session they agreed to 0.4 points.

The audit file records the appropriate conclusion: the paired effect is inconclusive
until the replay difference is explained. The contrast between across-session spread
and within-session agreement suggests the cause lies in session setup rather than in
the model file.

A provenance gap is also recorded. The first two arms were resumed after completion
under earlier logging code, which overwrote their environment files. Their original
runtime details can no longer be verified.

### Stage 7. Component-level weight audit

The final diagnostic compared the two copies directly. It reconstructed task 0's
generated weights from each saved model and measured how far they moved while
learning task 9. If the unlearned copy were more susceptible, its weights should
move differently.

| Weight group | Without U3 | After U3 | Difference |
| --- | ---: | ---: | ---: |
| Batch-norm parameters | 6.89% | 7.03% | 0.14 |
| Residual parameters | 0.770% | 0.803% | 0.03 |
| Main weights | 0.632% | 0.642% | 0.01 |
| Task code | unchanged | unchanged | none |
| Normalisation buffers | unchanged | unchanged | none |

Relative change is measured as L2 distance against the starting magnitude of each
group.

The two copies moved by nearly the same total amount, yet one lost 6.0 points of
accuracy and the other lost 10.2. If the accuracy difference is real, it is not
explained by a larger magnitude of movement. It would have to arise from *which*
weights moved, which a total-distance measure cannot resolve.

### Numerical checks

Two records in the archive are calculation checks rather than model runs. They
test whether the code computes what its definition says. Neither measures
whether the model forgets a task.

| Check | What was tested | Result |
| --- | --- | --- |
| noise-objective-check | Whether fresh Gaussian targets drive outputs toward noise or toward zero. 2000 values, 3000 updates. | Averaging 10 fresh draws left a spread of 0.04, one fresh draw 0.07, one fixed draw 0.98. |
| fisher-sanity-check | Gradients against finite differences, and two definitions of the Fisher information, on a small linear model. | Autograd and finite-difference gradients agreed to 2.4e-09. Squaring before averaging differed from averaging before squaring by a factor of 26.27. |

The first result is consistent with the squared-error objective being minimised
at zero when the targets are fresh and zero-mean. In other words, the term that
is supposed to push a task toward noise may instead be pushing its outputs
toward zero. That is worth testing directly against the loss-term imbalance
described in Stage 4.

The second confirms the gradient and Fisher code matches its definitions on a
constructed example. It does not validate Fisher attribution on the
hypernetwork, which has not been implemented.

Both records carry limitations. For the noise check, only summary values
survive: the definition of spread, the seed, and the optimiser were not
archived. For the Fisher check, no date or source commit was recorded.

### Current status

#### Established results

- The experimental pipeline is sound. Checkpoints save and restore correctly, file
  hashes match, and every run records its settings, code commit, and runtime.
- Learning succeeds at learning rate 0.0001, reaching approximately 32% where a
  direct network reaches 54%.
- No unlearning configuration has met the evaluation criteria. The best result is
  12.6% against a 12% criterion, with 3.6 points of drift.
- Gamma governs a trade-off rather than removing it. Protecting the retained task
  consistently leaves the target task above the criterion.

#### Open problems

1. **Retention failure during learning.** Tasks fall to chance accuracy after a few
   subsequent tasks are learned. This invalidates the full sequence before
   unlearning is tested, and should be addressed first.
2. **Cross-session variance.** An 11-point spread from identical inputs makes any
   effect smaller than 11 points unmeasurable.
3. **Loss-term imbalance.** At 8.2 against 1823.6, the preservation term cannot
   influence the update direction. This is a measured quantity and is inexpensive
   to test.

#### Proposed next steps

1. Rebalance the two loss terms, or normalise the unlearning term, and repeat the
   30-step diagnostic. This is a single run and tests the imbalance directly.
2. Identify the source of the cross-session variance before conducting further
   paired comparisons. Candidate causes include data ordering, download-time
   differences, and non-deterministic kernels.
3. Measure continual learning in isolation, with no unlearn requests: learn ten
   tasks in sequence and record the decay of each. This separates the retention
   failure from unlearning.

### Provenance and limitations

All figures derive from the evidence bundle `asof_20261004`, exported at commit
`489c9e8` with a clean working tree. The bundle contains 105 tracked repository
files and 69 saved run files. All 174 manifest entries were re-checked against the
files on disk: every file is present and every SHA-256 hash matches.

Records before E08 are transcriptions of reported results rather than files re-read
from the original runs. Every run uses a single seed and a single model, so no
result here constitutes multi-seed evidence. Ten checkpoints, approximately 5.4 GB,
were excluded from the bundle, so results cannot be recomputed from the weights. The
three criteria are screening filters for deciding what to investigate next; meeting
them would not establish that a task was genuinely removed.

#### Source files

| Content | Path |
| --- | --- |
| Experiment archive | `docs/experiments/registry.json` |
| Per-step accuracy traces | `docs/experiments/forgetting_traces.csv` |
| Full-sequence request history | `ops-docs/experiment_evidence/asof_20261004/drive/full_sequence_27step_candidate/` |
| Paired task-9 runs and audits | `ops-docs/experiment_evidence/asof_20261004/drive/paired_L9_probe/` |
| Figure sources | `docs/figures/2026-10-04/` |

## Historical run notes through 29 September

These earlier notes retain detailed measurements and evidence qualifications. Their proposed next steps have been superseded.

## UnCLe experiment log: learning and selective-forgetting diagnostics

This record separates user-reported GPU experiments from local software checks.
The GPU experiments were run in Colab; their complete artifacts remain on the
user's Drive or in earlier runtime-local directories. The tables below transcribe
results supplied in the conversation. They are not newly reproduced measurements.

Status of the short-run archive through 2026-09-29: **no recorded diagnostic
passed both original criteria. The 30-step run reached 12.6% target accuracy
at step 27 while retained drift was 3.6 points. Gradient directions and actual
Adam updates were not measured in that run.** Later full-sequence and paired L9
artifacts are on Drive and are not part of this earlier narrative log.

Choose a current workflow from the [notebook guide](https://github.com/sumitasthana/CARK/blob/main/notebooks/README.md).
The [gradient diagnostic](https://github.com/sumitasthana/CARK/blob/main/notebooks/04_gradient_diagnostics.ipynb) restores
the existing checkpoint when another forgetting run is needed.

### Reading this record

Last updated: 2026-09-29. This is the narrative record of all results supplied
in this conversation, not an archive of every original Colab output file.
[Structured observations](https://github.com/sumitasthana/CARK/wiki/Experiment-trajectory#historical-run-notes-through-29-september) provide CSV traces for E08-E15 and the 30-step run,
sampled E13 losses, E15 gradient and raw-output norms, and reported artifact paths.
The [registry](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/registry.json) also includes the earlier full-sequence
run and numerical checks. The [wiki](https://github.com/sumitasthana/CARK/wiki)
presents one structured page per record, generated from these files.
Earlier learning results and request summaries appear below; missing
measurements are not reconstructed.

Run identifiers are the mentoring sequence. Some early directory labels were
reused, so a label alone does not verify a configuration. Settings described
as intended or instructed must be checked against original report JSON before
using them as independently verified experimental metadata.

E15 reports a successful restore of the existing Drive checkpoint. Expected
starting accuracies remain task 3 = 26.0%, task 0 = 44.6%. The original report
and checkpoint have not been opened from this machine. Notebook 04 contains
the current setup and gradient diagnostic; notebook 03 retains historical cells.

### Question and protocol

Can the shared hypernetwork forget task 3 while preserving task 0?

The diagnostic sequence is `L3 L0 U3`: learn task 3, learn task 0, forget task 3.
Tiny ImageNet tasks use the saved seed-42 class partition, ten classes per task,
and task-local labels. Accuracy is validation accuracy, not labeled test accuracy.
Full task splits contain 5,000 training and 500 validation images.

The notebook configuration specifies ResNet50, 200 chunks, hidden widths
128/256/512, 32-dimensional task and chunk codes, five learning epochs, batch
size 64, evaluation batch size 256, beta 0.01, ten noise samples, and seed 0.
These are the intended common settings; the repository has not independently
inspected every earlier run's environment JSON. Where included in the pasted
logs, the environment was NVIDIA L4, 22.0 GiB, PyTorch 2.11.0+cu128, commit
`4ff99e8`. That commit predates the diagnostic-helper refactor.

Screening criteria, fixed before interpreting the short runs:

- Both task accuracies must be at least 25% immediately before forgetting.
- Task 3 must finish at or below 12%.
- Task 0's absolute accuracy change must be less than five percentage points.

The 25% threshold is a practical learning screen chosen during mentoring. It is
not a paper reproduction target or a statistical test. The 12% and five-point
criteria follow the short-probe plan. A pass would identify a candidate requiring
full-sequence validation, not prove information deletion.

### Run summary

Here, LR means learning rate. Before/after refers to the forget request unless
the row explicitly describes learning only. Each early run trained a fresh model;
starting accuracies varied despite the intended fixed seed. The cause of that
variation was not established.

| Run | Change and budget | Task 3 | Task 0 | Decision |
| --- | --- | --- | --- | --- |
| Initial probe | Shared LR 0.001; gamma 0.01; 100 forget steps | 14.4% after first learning; 12.2% before forgetting; 10.0% after | 36.6% to 34.8% | Invalid: target failed the learning screen. |
| Direct baseline | Train task 3 directly with ResNet50; five epochs, LR 0.001 | Validation by epoch: 38.4, 24.2, 55.6, 47.0, 53.8% | Not trained | Direct training can learn task 3 within this budget. |
| E02 | Hypernetwork, task 3 only; learning LR 0.0001 | 32.0%; final epoch loss 1.574293 | Not trained | Learning screen passed. |
| E03 | Learn tasks 3 and 0; learning LR 0.0001 | 32.0% after own learning, 30.4% after task 0 | 43.8% | Both tasks passed; no forget request. |
| E04 | Shared LR 0.0001; gamma 0.01; 100 steps | 31.0% after first learning; 26.4% to 10.0% at forgetting | 41.6% to 10.0% | Fail; spill 31.6 points. |
| E05 | Same budget; gamma 0.001 | 38.2% after first learning; 35.8% to 10.0% | 34.8% to 10.0% | Fail; spill 24.8 points. |
| E06 | Same budget; gamma 0.0001 | 42.6% after first learning; 35.2% to 10.0% | 37.0% to 10.0% | Fail; spill 27.0 points. |
| E07 | Gamma 0.0001; shared LR 0.0001; ten steps, floor set to ten | 37.0% after first learning; 39.6% to 10.0% | 48.8% to 10.0% | Fail; spill 38.8 points. Damage occurs within ten updates. |
| E08 | Saved pre-forgetting model; gamma 0.0001; forgetting LR 0.0001; ten measured updates | 26.0% to 10.0% | 44.6% to 10.0% | Fail; first update already damages task 0 by 21.2 points. |
| E09 | Restore E08 model; gamma 0.0001; forgetting-only LR 0.00001; ten measured updates | 26.0% to 15.8% | 44.6% to 30.0% | Fail; smaller updates delay damage, but no measured step passes both criteria. |
| E10 | Same saved model; gamma 0.00001; forgetting-only LR 0.00001; ten steps; Tesla T4 | 26.0% to 16.0% | 44.6% to 44.8% | Fail: retention passed, target remained above 12%. |
| E11 | Same checkpoint, LR 0.00001 and gamma 0.00001; 30 updates | 26.0% to 13.4%; minimum 12.6% | 44.6% to 36.4% | Fail: retention first breached at step 23; no target pass. |
| E12 | Same checkpoint, LR 0.00001 and 30-step budget; gamma 0.000003 | 26.0% to 13.0% | 44.6% to 42.8% | Fail: target above 12%; retention passed throughout. |
| E13 | Same checkpoint, LR 0.00001 and gamma 0.000003; 50 updates | 26.0% to 13.2%; minimum 13.0% | 44.6% to 43.8% | Fail: retention passes throughout; extra updates do not reach 12%. |
| E14 | Same checkpoint, intended LR 0.00001 and gamma 0.000005; 50 updates | 26.0% to 12.6%; minimum 12.4% | 44.6% to 37.4% | Fail: retention first breached at step 36; target never passed. |
| E15 | Same checkpoint; LR 0.00001, gamma 0.000005, ten updates; commit e3087f0; gradient measurement enabled | 26.0% to 16.2%; minimum 16.0% | 44.6% to 44.6%; maximum absolute drift 1.8 points | Fail: retention passed throughout, target remained above 12%. |
| 2026-09-29 30-step run | Same checkpoint; LR 0.00001, gamma 0.000005, 30 updates; commit fcdc1ff | 26.0% to 13.4%; minimum 12.6% at step 27 | 44.6% to 40.0%; maximum absolute drift 4.8 points | Fail: retention passed throughout, target remained above 12%. |

E06 reused a directory whose label began `E04_first_forget`; the user clarified
that the hyperparameters changed while the label was reused. We record the gamma
as user-confirmed, not inferred from that directory name or the loss value.

Selected reported costs and losses:

| Run | Final forgetting loss | Forget time | Other observations |
| --- | --- | --- | --- |
| E04 | 565622.8750 | 22.6 s / 100 steps | Learning requests took about 95 and 140 s. |
| E05 | 53909.2578 | 22.6 s / 100 steps | Lower loss after changing gamma is not evidence of better forgetting. |
| E06 | 6864.8228 | 22.6 s / 100 steps | Both tasks still ended at chance. |
| E07 | 36058.9922 | 3.1 s / ten steps | Total run about four minutes; most time was repeated learning. |

These timings do not predict the new diagnostic's runtime: per-step evaluation
adds work. The short runs reported peak allocated GPU memory around 4.64 GiB.

### Persistent starting checkpoint

An earlier Colab session ended, losing its in-memory trainer and runtime-local
files. E04 through E07 had disabled checkpoint saving, so their JSON logs could
not restore their model states. The two learning requests were run again and
saved persistently:

```text
/content/drive/MyDrive/uncle/E08_forgetting_trace/before_forgetting.pt
```

This model reached 36.4% on task 3 after its first learn. After learning task 0,
the saved accuracies were task 3 = 26.0% and task 0 = 44.6%. Those are the starting
values for E08 and E09. Restored predictions were checked and matched both values.

E08 and E09 use one continuous Adam optimizer and one frozen reference within
each forget request. Evaluation preserves PyTorch RNG state and restores model
modes. Calling `forget(burn_in=1)` repeatedly would reset the optimizer and
reference, and would be a different experiment.

### E08: trace at forgetting LR 0.0001, gamma 0.0001

Loss components are measured **before** the update. Accuracies and drift are
measured **after** the update. Drift is task 0's absolute change from 44.6%.

| Step | Task 3 % | Task 0 % | Drift, points | Weighted noise before | Preservation before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | N/A | N/A |
| 1 | 16.0 | 23.4 | 21.2 | 38765.68 | 0.00 |
| 2 | 13.0 | 14.6 | 30.0 | 36631.19 | 711.19 |
| 3 | 12.6 | 13.6 | 31.0 | 34988.89 | 1559.62 |
| 4 | 12.8 | 13.6 | 31.0 | 33794.32 | 2083.34 |
| 5 | 13.8 | 14.2 | 30.4 | 32940.63 | 2225.69 |
| 6 | 14.4 | 12.4 | 32.2 | 32308.67 | 2088.67 |
| 7 | 14.4 | 11.2 | 33.4 | 31802.68 | 1809.40 |
| 8 | 11.2 | 10.8 | 33.8 | 31353.69 | 1499.43 |
| 9 | 10.2 | 10.2 | 34.4 | 30910.52 | 1225.50 |
| 10 | 10.0 | 10.0 | 34.6 | 30435.68 | 1015.46 |

No step passed. Reported artifact, relative to the persistent checkpoint directory:
`forget_trace_20260923_020958_555174.json`.

### E09: trace at forgetting LR 0.00001, gamma 0.0001

| Step | Task 3 % | Task 0 % | Drift, points | Weighted noise before | Preservation before |
| --- | --- | --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 | N/A | N/A |
| 1 | 23.0 | 45.0 | 0.4 | 38765.68 | 0.00 |
| 2 | 21.2 | 46.6 | 2.0 | 38546.75 | 7.25 |
| 3 | 18.8 | 44.8 | 0.2 | 38331.81 | 27.22 |
| 4 | 17.4 | 43.4 | 1.2 | 38116.39 | 57.94 |
| 5 | 16.0 | 41.8 | 2.8 | 37906.20 | 97.46 |
| 6 | 16.2 | 39.2 | 5.4 | 37700.46 | 144.03 |
| 7 | 16.6 | 37.0 | 7.6 | 37498.27 | 196.03 |
| 8 | 16.8 | 35.2 | 9.4 | 37302.28 | 252.03 |
| 9 | 16.4 | 32.6 | 12.0 | 37112.92 | 310.69 |
| 10 | 15.8 | 30.0 | 14.6 | 36926.21 | 370.77 |

No step passed. Reported artifact:
`E09_forget_lr_1e-5_20260923_022053_354220.json`.
The filenames carry UTC timestamps; this document preserves the supplied names.

### E10: retention holds over ten steps, target remains above threshold

User-reported on 2026-09-23, using Tesla T4. Settings are those prepared in
the E10 notebook: forgetting LR 0.00001, gamma 0.00001, ten steps. The supplied
screening table does not include the raw report settings or loss components;
those have not been independently inspected.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 21.6 | 46.6 | 2.0 |
| 3 | 19.4 | 46.0 | 1.4 |
| 4 | 18.4 | 44.8 | 0.2 |
| 5 | 17.4 | 44.8 | 0.2 |
| 6 | 16.6 | 44.6 | 0.0 |
| 7 | 16.0 | 45.0 | 0.4 |
| 8 | 16.2 | 44.6 | 0.0 |
| 9 | 16.0 | 44.8 | 0.2 |
| 10 | 16.0 | 44.8 | 0.2 |

Passing steps: none. Task 0 stayed within the five-point drift limit at every
measured step. Task 3 fell ten points but remained four points above the target
threshold. The last four observations suggest a short plateau; ten updates do
not establish whether it persists.

Screening artifact supplied by the user:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_142742_335991_88cd96f2.screen.json`.

Compared with E09, the retained-task endpoint is better (44.8% versus 30.0%).
This is consistent with improved preservation at lower gamma, but the GPU also
changed from the earlier L4 environment to T4 and the diagnostic implementation
changed. Equal starting accuracies do not establish identical numerical update
trajectories. Do not attribute the whole difference to gamma without a matched
control. For the next budget comparison, keep this T4 runtime and helper fixed.

### E11: longer run exposes the retention tradeoff

User-reported on 2026-09-23, following the E11 instructions. The supplied table
covers 30 updates. LR 0.00001 and gamma 0.00001 are the instructed settings;
the raw configuration JSON has not been inspected. E11 was requested in the
same T4 runtime as E10; the user did not supply a separate GPU line for E11.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 21.6 | 46.6 | 2.0 |
| 3 | 19.4 | 46.0 | 1.4 |
| 4 | 18.4 | 44.8 | 0.2 |
| 5 | 17.4 | 44.8 | 0.2 |
| 6 | 16.6 | 44.6 | 0.0 |
| 7 | 16.0 | 45.0 | 0.4 |
| 8 | 16.2 | 44.6 | 0.0 |
| 9 | 16.0 | 44.8 | 0.2 |
| 10 | 16.0 | 44.8 | 0.2 |
| 11 | 16.4 | 44.6 | 0.0 |
| 12 | 16.8 | 44.2 | 0.4 |
| 13 | 16.4 | 43.6 | 1.0 |
| 14 | 16.4 | 43.4 | 1.2 |
| 15 | 16.2 | 43.2 | 1.4 |
| 16 | 15.8 | 42.6 | 2.0 |
| 17 | 15.0 | 42.0 | 2.6 |
| 18 | 15.0 | 41.4 | 3.2 |
| 19 | 14.2 | 41.6 | 3.0 |
| 20 | 13.4 | 40.6 | 4.0 |
| 21 | 13.6 | 40.2 | 4.4 |
| 22 | 13.4 | 39.8 | 4.8 |
| 23 | 12.8 | 39.4 | 5.2 |
| 24 | 12.6 | 38.6 | 6.0 |
| 25 | 12.8 | 38.6 | 6.0 |
| 26 | 12.8 | 38.6 | 6.0 |
| 27 | 13.2 | 38.2 | 6.4 |
| 28 | 13.8 | 37.8 | 6.8 |
| 29 | 13.6 | 37.6 | 7.0 |
| 30 | 13.4 | 36.4 | 8.2 |

All first-ten-step accuracies match E10 exactly at the reported precision.
The apparent plateau near 16% was temporary. At step 22, task 3 was 13.4%
and task 0 was 39.8%, still within the retention limit. At step 23, retained
drift reached 5.2 points, failing the screen. The lowest target accuracy was
12.6% at step 24, with six points of retained drift. No target observation was
at or below 12%, and no step passed both criteria. By step 30 retained drift
was 8.2 points. Extending this run did not solve selective forgetting.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143249_572038_f52e5457.screen.json`.

Code inspection confirms the forgetting loss is gamma times the noise loss
plus the preservation loss. Beta scales preservation during learning only;
changing beta would not change this forgetting objective. E12 therefore tests
a lower gamma, without changing the learning configuration. This may improve
retention, but may also leave more target accuracy. It is a hypothesis, not an
expected pass. Do not relax the screen because E11 came close.

### E12: retention passes throughout, target ends one point above threshold

User-reported on 2026-09-23 after the E12 instructions. The instructed settings
were forgetting LR 0.00001, gamma 0.000003, and 30 updates. Raw configuration
and loss components were not supplied for inspection. The same T4 runtime was
requested; no separate GPU identification was included in this result.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 22.6 | 45.2 | 0.6 |
| 3 | 21.8 | 45.4 | 0.8 |
| 4 | 21.0 | 45.8 | 1.2 |
| 5 | 19.2 | 45.4 | 0.8 |
| 6 | 18.6 | 45.8 | 1.2 |
| 7 | 17.4 | 46.2 | 1.6 |
| 8 | 16.4 | 46.0 | 1.4 |
| 9 | 15.8 | 44.6 | 0.0 |
| 10 | 16.0 | 44.4 | 0.2 |
| 11 | 16.2 | 44.2 | 0.4 |
| 12 | 16.0 | 44.6 | 0.0 |
| 13 | 16.2 | 44.4 | 0.2 |
| 14 | 16.8 | 44.6 | 0.0 |
| 15 | 16.8 | 44.6 | 0.0 |
| 16 | 16.4 | 44.8 | 0.2 |
| 17 | 16.2 | 44.2 | 0.4 |
| 18 | 16.2 | 44.2 | 0.4 |
| 19 | 15.8 | 43.8 | 0.8 |
| 20 | 15.4 | 43.8 | 0.8 |
| 21 | 15.2 | 43.4 | 1.2 |
| 22 | 15.0 | 43.4 | 1.2 |
| 23 | 14.8 | 43.4 | 1.2 |
| 24 | 14.2 | 43.4 | 1.2 |
| 25 | 14.0 | 43.4 | 1.2 |
| 26 | 14.4 | 43.2 | 1.4 |
| 27 | 14.0 | 43.2 | 1.4 |
| 28 | 13.6 | 42.8 | 1.8 |
| 29 | 13.0 | 42.8 | 1.8 |
| 30 | 13.0 | 42.8 | 1.8 |

No step passed both criteria. All steps passed retention; target accuracy
never reached 12%. Task 3 ended at 13.0%, one percentage point above the target,
while task 0 drift was 1.8 points. Compared with E11 at step 30, the endpoints
were 13.0% versus 13.4% for task 3 and 42.8% versus 36.4% for task 0. This is
an improved retention result on this checkpoint, not a validated optimum.

Task 3 declined from 15.4% at step 20 to 13.0% at step 30. That trend and the
remaining retention margin motivate a bounded extension before changing gamma
again. They do not guarantee continued improvement or a passing step. The
loss components remain uninspected; no gradient-dominance claim follows.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143653_273412_d31bb28c.screen.json`.

### E13: retention holds, but extra steps do not reach the target

User-reported on 2026-09-23 after the E13 instructions. Intended settings were
forgetting LR 0.00001, gamma 0.000003 and 50 updates. Raw settings and losses
have not yet been inspected. The same T4 runtime was requested; no separate
GPU identification was supplied with this result.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 22.6 | 45.2 | 0.6 |
| 3 | 21.8 | 45.4 | 0.8 |
| 4 | 21.0 | 45.8 | 1.2 |
| 5 | 19.2 | 45.4 | 0.8 |
| 6 | 18.6 | 45.8 | 1.2 |
| 7 | 17.4 | 46.2 | 1.6 |
| 8 | 16.4 | 46.0 | 1.4 |
| 9 | 15.8 | 44.6 | 0.0 |
| 10 | 16.0 | 44.4 | 0.2 |
| 11 | 16.2 | 44.2 | 0.4 |
| 12 | 16.0 | 44.6 | 0.0 |
| 13 | 16.2 | 44.4 | 0.2 |
| 14 | 16.8 | 44.6 | 0.0 |
| 15 | 16.8 | 44.6 | 0.0 |
| 16 | 16.4 | 44.8 | 0.2 |
| 17 | 16.2 | 44.2 | 0.4 |
| 18 | 16.2 | 44.2 | 0.4 |
| 19 | 15.8 | 43.8 | 0.8 |
| 20 | 15.4 | 43.8 | 0.8 |
| 21 | 15.2 | 43.4 | 1.2 |
| 22 | 15.0 | 43.4 | 1.2 |
| 23 | 14.8 | 43.4 | 1.2 |
| 24 | 14.2 | 43.4 | 1.2 |
| 25 | 14.0 | 43.4 | 1.2 |
| 26 | 14.4 | 43.2 | 1.4 |
| 27 | 14.0 | 43.2 | 1.4 |
| 28 | 13.6 | 42.8 | 1.8 |
| 29 | 13.0 | 42.8 | 1.8 |
| 30 | 13.0 | 42.8 | 1.8 |
| 31 | 13.0 | 42.8 | 1.8 |
| 32 | 13.2 | 42.8 | 1.8 |
| 33 | 13.2 | 42.8 | 1.8 |
| 34 | 13.4 | 42.8 | 1.8 |
| 35 | 13.6 | 42.8 | 1.8 |
| 36 | 13.8 | 42.8 | 1.8 |
| 37 | 13.8 | 42.8 | 1.8 |
| 38 | 14.0 | 42.8 | 1.8 |
| 39 | 13.8 | 42.8 | 1.8 |
| 40 | 13.2 | 43.0 | 1.6 |
| 41 | 13.4 | 43.0 | 1.6 |
| 42 | 13.8 | 43.0 | 1.6 |
| 43 | 13.0 | 43.2 | 1.4 |
| 44 | 13.4 | 43.2 | 1.4 |
| 45 | 13.4 | 43.4 | 1.2 |
| 46 | 13.4 | 43.4 | 1.2 |
| 47 | 13.4 | 43.4 | 1.2 |
| 48 | 13.2 | 43.4 | 1.2 |
| 49 | 13.0 | 43.4 | 1.2 |
| 50 | 13.2 | 43.8 | 0.8 |

The first 30 accuracy pairs match E12 at the reported precision. After step
30, task 3 stays between 13.0% and 14.0%. Its minimum across the run is 13.0%,
so no step reaches the 12% threshold. Task 0 meets retention at every step
and finishes only 0.8 points below its starting accuracy. The 20 extra updates
did not improve the lowest observed target accuracy.

This is an accuracy plateau over the measured window, not evidence that the
loss or parameters converged. The two tasks' accuracies need not change
monotonically. Keep the existing thresholds; do not redefine a pass because
the target came close. No claim about erasure follows from these measurements.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_143926_526429_aa83a19b.screen.json`.

### E13 loss follow-up: objective decreases despite the accuracy plateau

User supplied these sampled values after reading E13's report. All losses
are measured before the corresponding update. The requested settings and
status output were not included; exact run settings remain based on the
experiment instructions rather than an independently inspected configuration.

| Step | Weighted noise | Preservation | Total loss |
| --- | --- | --- | --- |
| 1 | 1163.01 | 0.00 | 1163.01 |
| 10 | 1140.43 | 3.66 | 1144.09 |
| 20 | 1117.63 | 4.15 | 1121.78 |
| 30 | 1095.24 | 5.29 | 1100.53 |
| 40 | 1073.38 | 6.50 | 1079.88 |
| 50 | 1052.21 | 7.66 | 1059.87 |

The sampled total falls by 103.14, about 8.9%, from step 1 to step 50.
Weighted noise decreases while preservation increases. This is consistent
with reducing the noise objective while generated protected weights move
away from their reference. It does not establish optimizer convergence,
relative gradient strength, or the accuracy effect of further updates.
Fresh noise samples also contribute to differences between loss observations.
The objective acts on generated weights, not directly on validation accuracy;
therefore continued loss reduction need not cross the accuracy screen.

Recommendation: test gamma 0.000005 with the E13 learning rate and 50-step
budget. This is between 0.000003, which preserved task 0 but missed target
accuracy, and 0.00001, which damaged retention in E11's 30-step run. These
outcomes motivate an intermediate test but do not guarantee monotonic behavior
or a feasible gamma. No thresholds are changed.

### E14: intermediate gamma still misses the joint screen

User-reported after the E14 instructions: intended gamma 0.000005, forgetting
LR 0.00001, and 50 updates from the same pre-forgetting checkpoint. The supplied
screen does not independently confirm the configuration or GPU.

| Step | Task 3 % | Task 0 % | Absolute task 0 drift, points |
| --- | --- | --- | --- |
| 0 | 26.0 | 44.6 | 0.0 |
| 1 | 23.0 | 45.0 | 0.4 |
| 2 | 22.0 | 45.6 | 1.0 |
| 3 | 21.0 | 46.0 | 1.4 |
| 4 | 19.4 | 46.2 | 1.6 |
| 5 | 18.6 | 46.4 | 1.8 |
| 6 | 18.0 | 46.6 | 2.0 |
| 7 | 17.0 | 46.2 | 1.6 |
| 8 | 16.2 | 45.8 | 1.2 |
| 9 | 16.0 | 44.6 | 0.0 |
| 10 | 16.2 | 44.6 | 0.0 |
| 11 | 16.2 | 44.8 | 0.2 |
| 12 | 16.4 | 44.6 | 0.0 |
| 13 | 17.0 | 44.6 | 0.0 |
| 14 | 16.4 | 44.6 | 0.0 |
| 15 | 16.4 | 44.4 | 0.2 |
| 16 | 16.4 | 43.8 | 0.8 |
| 17 | 15.6 | 43.4 | 1.2 |
| 18 | 15.4 | 43.2 | 1.4 |
| 19 | 15.0 | 42.4 | 2.2 |
| 20 | 14.8 | 42.8 | 1.8 |
| 21 | 14.2 | 42.8 | 1.8 |
| 22 | 13.8 | 42.4 | 2.2 |
| 23 | 13.6 | 41.8 | 2.8 |
| 24 | 13.6 | 41.4 | 3.2 |
| 25 | 13.0 | 41.2 | 3.4 |
| 26 | 13.0 | 41.4 | 3.2 |
| 27 | 12.6 | 41.2 | 3.4 |
| 28 | 12.8 | 40.8 | 3.8 |
| 29 | 13.2 | 39.8 | 4.8 |
| 30 | 13.4 | 40.2 | 4.4 |
| 31 | 14.0 | 40.2 | 4.4 |
| 32 | 14.0 | 40.0 | 4.6 |
| 33 | 13.8 | 40.0 | 4.6 |
| 34 | 13.6 | 39.8 | 4.8 |
| 35 | 13.0 | 39.8 | 4.8 |
| 36 | 13.2 | 39.2 | 5.4 |
| 37 | 13.4 | 39.0 | 5.6 |
| 38 | 13.0 | 39.2 | 5.4 |
| 39 | 13.0 | 39.0 | 5.6 |
| 40 | 12.8 | 38.6 | 6.0 |
| 41 | 13.0 | 38.0 | 6.6 |
| 42 | 13.0 | 38.0 | 6.6 |
| 43 | 12.6 | 38.0 | 6.6 |
| 44 | 12.4 | 38.0 | 6.6 |
| 45 | 12.6 | 38.2 | 6.4 |
| 46 | 12.6 | 38.0 | 6.6 |
| 47 | 12.6 | 38.0 | 6.6 |
| 48 | 12.6 | 37.8 | 6.8 |
| 49 | 12.6 | 37.8 | 6.8 |
| 50 | 12.6 | 37.4 | 7.2 |

At step 27, task 3 reached 12.6% while task 0 retained 41.2% accuracy
(3.4 points of drift). Retention first failed at step 36, with 5.4 points of
drift. The lowest task 3 accuracy was 12.4% at step 44, when retention drift
was already 6.6 points. Final values were 12.6% and 37.4%, with 7.2 points of
retained drift. No step passed. Keep the 12% threshold unchanged.

Reported screening artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/forget_20260923_151049_593433_8a010c9b.screen.json`.

The measured gamma values expose a retention/forgetting tradeoff on this
checkpoint. This finite search does not prove that no useful gamma exists or
that a structural defect is the cause. It does justify switching from further
nearby scalar guesses to the planned objective audit.

### E15: gradient measurements from the saved checkpoint

The user supplied a completed diagnostic table reporting commit `e3087f0`,
forgetting LR 0.00001, gamma 0.000005, ten updates, ten noise samples, and a
valid starting point. The filename carries the UTC date 2026-09-28. GPU identity,
runtime, loss components, and the original JSON were not supplied for inspection.

Task 3 fell from 26.0% to 16.2%, reaching a minimum of 16.0% at step 9.
Task 0 ended at its starting 44.6%; its maximum absolute drift was 1.8 points.
No step passed the joint screen. The printed raw-output norm fell from
`1.908e+04` before update 1 to `1.886e+04` before update 10. Those norms describe
the model before the named updates, not the final model after ten updates.

At the first update, preservation gradients were zero in every measured group.
Before update 2, the shared-layer preservation gradient norm was 3,493 against
2,771 for the weighted noise term. Preservation is therefore not negligible in
every group. The BatchNorm-head noise gradient remained larger than its
preservation gradient at the supplied steps. These are generated BatchNorm scale
and offset parameters, not running statistics.

Gradient norms alone do not establish cancellation, storage, or each term's
contribution to Adam's update. Commit `4c08f54` changed the forgetting-noise
stream, so E15 is not an exact replay of the first ten E14 steps.

The [transcribed output](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/e15_reported_output.txt) preserves the
supplied values. Accuracy traces, raw norms, and the 16 supplied gradient rows
are in the [structured files](https://github.com/sumitasthana/CARK/wiki/Experiment-trajectory#historical-run-notes-through-29-september). Full gradient vectors and
unreported steps' gradient norms are not reconstructed.

Reported artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/E15_gradients/forget_20260928_011538_895545_c81f33b2.json`.

### 2026-09-29: 30-step forgetting diagnostic

The user reported a completed run from the same saved starting checkpoint,
with commit `fcdc1ff`, forgetting LR 0.00001, gamma 0.000005, 30 updates, and
ten noise samples. The printed start was task 3 = 26.0% and task 0 = 44.6%.
Task 3 reached 12.6% at step 27 with task 0 at 41.0%, an absolute drift of
3.6 points. At step 30 they were 13.4% and 40.0%. Task 0 remained inside the
five-point retention screen throughout; its largest drift was 4.8 points at
step 29. No step reached the 12% target threshold.

The supplied late gradient norms show both forgetting and preservation terms
active near step 27. They do not show whether the gradients oppose each other.
The original runtime JSON was not opened from this workspace. The [transcribed
output](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/forgetting_30step_20260929.txt) preserves all 31
accuracy observations, 30 raw-output norms, and the eight selected steps of
gradient norms supplied by the user. Its descriptive archive ID is
`forgetting-30step-20260929`; the user did not assign an E-number.

Reported artifact:
`/content/drive/MyDrive/uncle/E08_forgetting_trace/diagnostics/E15_gradients/forget_20260929_021642_430512_33bd5249.json`.

The [research diagnostic](https://github.com/sumitasthana/CARK/blob/main/docs/RESEARCH_DIAGNOSTIC.md) now screens all nine tracked
checkpoint trajectories together and can inspect the original JSON when it is
available. New code can audit component changes and gradient directions, but
those measurements have not yet been reported from a GPU run.

#### Initial code audit: the two terms use different parameter scales

`UnCLe.forget` uses `hypernet.raw_for(task)` for its noise term, before layer
scaling. `UnCLe.preserve` compares `weights_from_code`, after layer scaling.
`HyperNetwork` assigns scale 0.01 to one-dimensional parameters, 1/fan_in to
classifier matrices, and sqrt(2/fan_in) to other matrices.

For a protected parameter with raw change delta and scale s, its contribution
to preservation is s squared times delta squared. Thus a global gamma cannot
remove all differences in relative weighting across layers. This is a code
property, not evidence by itself that these scales caused E14's failure.
Neither loss magnitude nor this observation establishes gradient dominance.
Changing to a scaled noise objective would change the experiment and require
an explicit choice of noise scale; it must not be silently treated as a fix.

### What we learned, and what remains uncertain

1. **Initial learning was a prerequisite failure.** The original target accuracy
   was weak before any deletion. Its subsequent chance accuracy could not
   demonstrate successful removal of a learned ability.
2. **The direct classifier learned task 3.** Its 53.8% final accuracy directs
   investigation toward the hypernetwork training path and optimization. It
   does not identify a particular faulty component because initialization and
   parameterization also differ.
3. **A lower training LR helped in the reported runs.** Moving from 0.001 to
   0.0001 enabled the target to pass our learning screen. This is one-seed
   diagnostic evidence, not a robust hyperparameter optimum.
4. **Chance accuracy on both tasks is a failure of selective forgetting.** A
   smaller spill across fresh runs can simply reflect a lower starting accuracy.
   Likewise, smaller losses after changing gamma have different scales.
5. **The damaging motion starts immediately.** At the initial snapshot, squared
   weight-difference preservation has zero loss and zero gradient. It responds
   after drift occurs; the first E08 update already reduced retained accuracy by
   21.2 points. This is a property of this penalty, not proof of an implementation
   defect by itself.
6. **Smaller forgetting updates delay, but have not prevented, the tradeoff.** E09
   keeps task 0 within the drift limit through step 5, while task 3 remains above
   the forgetting threshold. Continuing to step 10 damages retention.
7. **Loss magnitudes do not establish gradient dominance.** E15 adds per-group
   gradient norms, but their directions and the actual Adam update still need
   measurement. Low weight drift does not directly guarantee low accuracy drift.
8. **Chance accuracy does not prove erasure.** No recovery experiment has yet
   established whether residual task information remains.
9. **Zero relapse here is not a stability result.** These traces have no later
   learning after deletion.
10. **Preserve the trained starting state.** Checkpoint comparisons avoid spending
    roughly four minutes relearning the same pair and reduce confounding from
    variation between fresh training attempts.

### Software changes supporting the next session

- `Config.forgetting_learning_rate` separates forgetting from learning. Its
  default, `None`, retains the previous shared-rate behavior.
- `UnCLe.forget(..., on_step=...)` reports step zero and each update, isolating
  callback PyTorch RNG consumption and restoring model modes.
- `diagnose_forgetting(...)` restores a checkpoint, calls the existing forgetting
  loop, evaluates each step, and saves a unique report. It never retrains or
  overwrites the source checkpoint.
- Checkpoint loading now keeps tensors on CPU until restoration. CUDA RNG states
  are validated and supplied as CPU byte tensors; entries are never silently
  skipped. This addresses the user's GPU RNG restoration error. Invalid GPU
  counts or states produce explicit errors.
- Older checkpoints without `forgetting_learning_rate` remain readable.

Local verification of this refactor: 15 core checks, eight task checks, 18
experiment checks, and eight diagnostic checks passed. The additional CUDA RNG
regression was skipped because the local machine has no CUDA device. All 19
executable Colab-guide examples passed after supplying an isolated local IPython
dependency. These checks establish software behavior on the tested environment;
they do not reproduce the GPU research outcomes above.

#### Evidence added from the saved notebook on 2026-09-24

The newer notebook commit `4407978` contains saved outputs that corroborate
E13 settings (LR 0.00001, gamma 0.000003, 50 steps, ten noise samples) and E14
settings (LR 0.00001, gamma 0.000005, 50 steps, ten noise samples). This updates
the earlier notes saying those settings had not been supplied. The E13 cell
reads its older saved report; it is not another E14 result.

Setup outputs report Tesla T4 and source commit `04c31b3`. E14's 50 printed
accuracy pairs match the conversation trace. All 50 weighted-noise and
preservation pairs are now in the CSV. Weighted noise falls from 1938.36 to
1731.68 while preservation rises from zero to 13.70. These sampled changes
do not establish relative gradient strength or explain the failure by themselves.
The original report JSON remains on Drive and was not read from this machine.

[Archived notebook outputs](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/notebook_saved_outputs.txt) retain this
evidence with its source commit and cell indices. The notebook has historical
E10 prose alongside E14 code and outputs; follow the current status here.

### Next priority: objective diagnostics (T0.2)

Pause further gamma and step sweeps. Preserve the current checkpoint and all
reports; do not retrain or relax the acceptance thresholds. The next useful
diagnostic should measure target and retained generated-weight changes by
parameter group, including their raw and scaled norms, and relate those to
accuracy. If measuring gradient contributions, distinguish them from losses.

E15 and the 30-step run supply selected gradient magnitudes. The next GPU
diagnostic can now measure gradient alignment, actual parameter-update norms,
and before/after component changes while preserving optimizer continuity and
observational RNG behavior. Combined-gradient norms remain unmeasured. The scaling
audit remains a hypothesis to test, not a confirmed root cause. This archive
update transcribes existing results; it does not run a new GPU experiment.


## Learning audit and its qualification

These notes describe the review and software checks preceding the reported continuation. Proposed steps in this historical section are superseded by the current batch decision above.

The protection penalty passed a direct equation and gradient check. The clearest implementation difference is initialization: our code does not implement the paper's Hyperfan method. This is a candidate explanation for poor learning and retention, not a demonstrated cause.

This audit checked training revision `83e1c32773fc59886c8b43a0ed159ccf6424637a`, which notebook 12 selects. The local pooling and determinism edits are not part of that revision. No GPU training was run and no training code was changed.

### What the two completed experiments show

Both runs started from the saved model after learning tasks 3 and 0, then learned tasks 9, 5, and 17 without any unlearning. The user supplied these final test scores:

| Task | Beta 0.01 | Beta 0.1 | Difference in percentage points |
| --- | ---: | ---: | ---: |
| 3 | 22.4% | 18.8% | -3.6 |
| 0 | 29.4% | 35.8% | +6.4 |
| 9 | 29.8% | 29.4% | -0.4 |
| 5 | 44.4% | 29.8% | -14.6 |
| 17 | 41.2% | 34.0% | -7.2 |

Increasing beta helped task 0 in this pair, but did not improve every older task and produced lower scores on newly learned tasks. One run per setting does not establish a reliable beta effect. Learning alone can reduce older task scores, so unlearning is not required for the decline observed here.

### Protection penalty

Equation 2 compares the generated model weights with weights from a frozen copy of the hypernetwork taken before learning the next task. It sums squared differences over model parameters, then averages across previous tasks. Our `UnCLe.preserve()` follows that formula for the supplied protected tasks. It uses the same scaled weights as prediction. Dividing again by the number of model parameters would change this objective. [Paper, Learning](https://arxiv.org/html/2509.17530#S3.SS1.SSS1)

A CPU check used a small two-layer target, two protected tasks, and the pinned hypernetwork code. The checked trainer file was identical to the pinned trainer file. Results with PyTorch `2.12.0+cpu`:

- Penalty before changing the generator: `0.0`.
- Penalty after adding `0.03` to the output-head weights: `0.1256491243839264`.
- Independently calculated squared-distance penalty: `0.1256491243839264`.
- Largest gradient difference between the two calculations: `0.0`.
- The snapshot had no trainable parameters.

This checks the penalty's arithmetic and gradient on a small model. It does not prove that protection is strong enough during ResNet50 training.

### Initialization

Initialization means choosing the model's numbers before its first lesson. The paper specifies Hyperfan initialization to produce Kaiming He initialized main-network parameters. Our code instead initializes the generator's hidden layers with Kaiming normal, its output heads with another normal distribution, then applies hand-selected output scales. Those steps are not an implementation of Hyperfan. [Paper, Learning](https://arxiv.org/html/2509.17530#S3.SS1.SSS1)

A separate CPU measurement used the actual Tiny ImageNet ResNet50 configuration, hidden widths 128, 256, and 512, 200 chunks, seed 0, and task code 3. It measured a fresh model, not the saved E08 checkpoint:

| Generated parameter | Mean | Standard deviation | Applied scale |
| --- | ---: | ---: | ---: |
| `conv1.weight` | -0.004057 | 0.194999 | 0.272166 |
| `bn1.weight` | -0.000770 | 0.007965 | 0.01 |
| `bn1.bias` | -0.000326 | 0.008961 | 0.01 |
| `fc.weight` | 0.000000589 | 0.000448 | 0.000488 |

The final classifier uses a scale of `1 / 2048`. A fan-in He reference for that shape has standard deviation `sqrt(2 / 2048) = 0.03125`. The measured classifier spread is approximately 70 times smaller than that reference. The paper does not specify every classifier and BatchNorm initialization detail, so this comparison does not establish the authors' exact tensors.

`bn1.weight` controls the size of normalized activations. Our generated values start close to zero; the target template's BatchNorm weights start at one. These differences justify checking initialization before another beta sweep. Their effect on trained accuracy remains untested.

### Other confirmed difference

The paper states Adam with learning rate `0.001` and a scheduler. Our learning loop has Adam but no scheduler. E08 used the diagnostic learning rate `0.0001`; notebook 12 inherits the starting checkpoint's configuration. The cited appendix does not supply an identifiable scheduler recipe, so an exact schedule should not be invented. [Paper, Implementation](https://arxiv.org/html/2509.17530#S4)

### Next target and steps

**Target: make the initialization match a documented Hyperfan method, and verify the generated starting weights before spending more GPU time.**

1. Prepare the initialization change from the primary Hyperfan reference. Document choices the UnCLe paper leaves unspecified, especially BatchNorm and the final classifier.
2. Check generated weight statistics and a finite forward/backward pass on CPU. Keep the protection equation unchanged.
3. After those checks pass, run a short fresh sequence: learn 3, then 0, then 9. Use beta `0.01`, keep the diagnostic learning rate fixed for this pilot, and record each task's score after each lesson. Judge whether learning and retention warrant a longer experiment.

A fresh start is necessary to test initialization. Loading the old checkpoint overwrites newly initialized generator weights. Also, the output scales are not saved in the state dictionary: changing those scales would change the old checkpoint's generated model even if loading succeeds. Preserve the old code and checkpoint together.

The next pilot would assess the revised implementation. It would not establish generalization across seeds or reproduce the full paper.

### Qualification after reading the Hyperfan reference

The original convolution-weight recipe is already algebraically consistent with the weight-only Hyperfan-in variance formula: a raw head with weight variance `1 / hidden_width`, followed by multiplication by `sqrt(2 / fan_in)`, has the required effective variance when the last hidden activations have unit second moment. The audit's claim that the entire initialization recipe differs from Hyperfan was too broad. The classifier and one-dimensional parameter policies are the substantive differences. The measured spread of one freshly generated tensor is not, by itself, evidence that this variance formula is wrong.

The new option and its choices are documented in [Hyperfan pilot](#hyperfan-implementation-and-cpu-checks). The historical measurements above remain measurements of the original recipe.


## Hyperfan implementation and CPU checks

These notes describe the review and software checks preceding the reported continuation. Proposed steps in this historical section are superseded by the current batch decision above.

The new `Config(initialization="hyperfan_in")` option changes the starting recipe for a fresh model. The default remains `legacy` so historical runs and checkpoints keep their original generated weights. The protection objective is unchanged.

### Formula and implementation choices

Hyperfan-in sets the effective output-head weight variance to `gain_squared / (bias_factor * target_fan_in * hidden_width * embedding_variance)`. Here embedding variance is one, `bias_factor` is two when generating an affine layer's bias, and gain squared is two for ReLU layers or one for linear layers. Generated bias variance is `gain_squared / 2`. See Table 1 and section 4.1 of [Chang, Flokas, and Lipson](https://arxiv.org/html/2312.08399#S4.SS1).

Our shared chunk heads generate parameters with different fan-in values. A head initialized with variance `1 / hidden_width`, followed by a separate scale for each generated tensor, implements the effective variance formula. The hidden ReLU trunk keeps its Kaiming fan-in initialization. Neither embeddings nor chunks are normalized using observed samples.

The original bias-free convolution recipe was already consistent with this formula. The new recipe changes these policies:

- Convolution weights use gain squared two. If a convolution has a generated bias, half the variance budget goes to that bias.
- The final classifier is linear, with gain squared one. Its weight variance is `1 / (2 * fan_in)` and its generated bias variance is `1 / 2`.
- BatchNorm starts with gamma one and beta zero. Its output head starts at zero and remains trainable. An additive offset supplies gamma one. Output scale is one for both parameters.

The UnCLe paper does not specify these classifier and BatchNorm details or which Hyperfan variant it used. These are documented reproduction choices, not recovered author code. The new option has been checked for learning; this pilot does not assess its forgetting behavior.

### CPU evidence

[Saved check](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/hyperfan_cpu_check_20261006.json): full Tiny ImageNet ResNet50 shapes, 200 chunks, hidden widths 128/256/512, seed 0, PyTorch 2.12.0+cpu. Original adaptive average pooling was used, matching the published pilot.

- Generated classifier weight standard deviation: `0.0143448` (formula reference `0.015625`). The old recipe measured `0.0004483` for the same seed and architecture.
- BatchNorm gamma was exactly one and beta exactly zero.
- A synthetic two-image forward/backward pass had finite generated values, predictions, loss, and gradients. Cross-entropy was `2.8180091`.
- BatchNorm-head gradients were nonzero. The smaller automated check also verified that an optimizer update moves gamma.
- Across 24 independent initializations, generated affine weight and bias second moments were within 20% of the formula references.
- Legacy generation matched the pinned historical implementation exactly.
- Protection values and gradients matched a direct squared-distance calculation with the new offsets.

Synthetic images do not measure dataset accuracy. A single tensor's empirical spread need not equal the formula's expectation across random initializations.

### Checkpoints and notebook

Checkpoint files record the initialization option. Files predating this field are interpreted as `legacy`. Restoring a checkpoint into a model with a different initialization option is refused before loading model state. The offsets and scales are derived from that option.

Notebook [13](https://github.com/sumitasthana/CARK/blob/main/notebooks/13_hyperfan_learning_pilot.ipynb) starts a new model and learns tasks 3, 0, and 9. Settings: seed 0, beta 0.01, learning rate 0.0001, five epochs per task, batch size 64, ResNet50, and the existing fixed class partition. No scheduler or deterministic-mode change is introduced. Learning rate and epoch count remain diagnostic settings, so this is not a complete paper reproduction.

The notebook saves the code version, settings, class-partition hash, task scores, environment, and a checkpoint after each completed task. It can resume its own checkpoint after a disconnected session. It does not load E08 or any other historical starting model. Its final cell checks the files, flushes Drive, then releases the GPU.

Judge the pilot by each task's learning score and the change in earlier scores after the next lesson. Do not use this one run to claim a seed-independent improvement or identify one of the changed initialization components as the cause. Stop after task 9 and review the saved table before extending the sequence.

### Validation and code version

Training code is pinned to `b1797e0a93644f501508dfb856d38ad921bb6db8`. The published version retains the original ResNet pooling.

Forty-one automated checks passed across Hyperfan statistics, historical output compatibility, experiments, replay, and notebook helpers. The additional pilot notebook check passed using real CPU training and checkpoint files with a small stand-in target. It simulated a disconnect after saving task 3, resumed tasks 0 and 9, reopened the completed run without training again, and verified that Drive flushing precedes runtime release. Colab and Drive services were mocked; no remote GPU run was performed during validation.


## Paper recheck and diagnostic validation

These notes describe the review and software checks preceding the reported continuation. Proposed steps in this historical section are superseded by the current batch decision above.

The new-task objective and protected-task implementation agree with the paper's stated learning equation. The review did not find a missing replay buffer, a missing old-task loss on images, or a missing extra coefficient. It did find remaining reproduction gaps and a confirmed difference in the forgetting path. The current learning-only results cannot establish complete agreement with the paper.

### Paper and code comparison

Reviewed sources: the [arXiv HTML](https://arxiv.org/html/2509.17530) and [17-page PDF](https://arxiv.org/pdf/2509.17530), version 1. A search also surfaced an OpenReview submission, but access returned a browser-verification page; its full contents were not verified. No author implementation was established from the searches performed.

| Item | Finding in our code |
| --- | --- |
| Learning objective, Equation 2 | Cross-entropy plus beta times generated-weight preservation. Squared differences are summed over target parameters and averaged across protected tasks. No additional averaging across parameters is applied. |
| Frozen reference | Snapshot before adding the new task; cached reference weights come from that snapshot. Cache is reset for each lesson. |
| Trainable parameters | Shared generator and new task code. Old task codes are frozen. Chunk codes are trained only for the first task and frozen afterwards. |
| Protected tasks | All previously learned tasks except the task being taught. No forgetting occurs in the pilot or new diagnostic. |
| Architecture | ResNet50, hidden widths 128/256/512, task and chunk codes of size 32, 200 chunks, separate parameter-type heads. |
| Hyperfan | Opt-in Hyperfan-in variance recipe is available. The original convolution scaling was already compatible with its weight-only formula. Final-classifier, generated-bias, and BatchNorm choices are documented implementation choices. |
| Optimizer | Adam, newly constructed for each lesson. The paper does not state whether Adam state persists across lessons. |
| Learning rate and schedule | Pilot uses constant 0.0001. Paper states 0.001 and a scheduler, but an identifiable schedule was not supplied in the reviewed appendix. This gap remains open. |
| Training length and batch size | Five epochs and batch size 64 are our diagnostic settings. The reviewed appendix does not identify the promised values. |
| Target stem and data | Our ResNet uses a 3x3 stride-one stem. Images use ToTensor without augmentation or normalization, with a fixed class partition. These details are not fully specified by the paper. |
| BatchNorm running statistics | Stored separately for each task; old buffers are not updated during later learning. This handling is not specified by the paper. |
| Forgetting representation | Noise alignment uses raw generator outputs, while preservation uses scaled target weights. Equation 3 uses the same H for both terms. The two code paths therefore require resolution before claiming a faithful unlearning implementation. |
| Benchmark scope | The five-task learning-only pilot is a diagnostic. It is not the paper's mixed 30-request Tiny ImageNet sequence or a multi-seed result. |

The learning equation, architecture, freezing rule, optimizer statement, and beta search are given in the [Learning, Implementation, and appendices](https://arxiv.org/html/2509.17530). Unknown details above are not assumed to be implementation bugs. The noise-representation difference cannot explain accuracy decline in a run with no forgetting.

### Confirmed CPU checks

The new instrumentation passed a comparison against training without instrumentation. Generated parameters, BatchNorm buffers, and returned epoch losses were identical. The test evaluated validation scores and consumed random numbers inside callbacks, confirming that callback mode and PyTorch RNG isolation work.

The same check verified that old task codes and chunk codes are frozen and unchanged, and that old BatchNorm buffers stay unchanged. Protection was zero before the first update and positive after the generator moved. A zero initial protection gradient is expected for squared distance to an identical snapshot; it is not evidence of a disabled term.

Continuation tests restored actual CPU checkpoint files, trained exactly one additional task, preserved the source file hash and earlier history, and saved a new model. Two different beta branches had the same source hash, starting scores, and first-batch task loss. Completed runs were not retrained. Interrupted runs kept their partial epoch report and required a separate attempt folder.

The notebook test used its actual selection, training, and finishing cells with a small CPU target and mocked Colab services. SELECT mode did not train. RUN mode added exactly one task. File checks preceded Drive flushing and GPU release. These are machinery checks, not new dataset accuracy results.

### What to run next

Notebook [14](https://github.com/sumitasthana/CARK/blob/main/notebooks/14_learning_loss_diagnostic.ipynb) makes source selection explicit:

1. Use a CPU session in SELECT mode. It lists saved model files on Drive. Select a file and inspect its actual saved tasks and scores.
2. Prefer a source after L3, L0, L9, and L5, so the next lesson is L17. If that checkpoint no longer exists, use a retained source after L3, L0, and L9, then diagnose L5. Do not use the final model after L17 to recreate a model before L17.
3. Save the selection. In a fresh A100 session, use RUN mode with beta 0.01. Source scores must match before training starts. Only the next task is taught.
4. Record cross-entropy, unweighted protection, beta times protection, and combined loss for every update. Save validation scores and batch-mean loss components after every epoch.
5. Sample shared-generator gradient norms and their cosine similarity on the first batch of each epoch. The gradient of beta times protection is compared with the task gradient. The new task code is excluded from this comparison because protection does not act on it.
6. Verify the final history and save the new checkpoint in a separate folder. Keep the source file unchanged. Flush Drive and release the GPU.

Loss and gradient samples describe the model before an update. Epoch accuracies describe it after that epoch. Raw gradient norms are useful diagnostics, but Adam's adaptive scaling means they are not a direct measurement of the resulting parameter movement. Negative gradient cosine indicates conflict on that sampled group; sparse samples do not prove behavior at every update.

The new notebook preserves the source model's initialization, epochs, batch size, seed, and learning rate. It does not silently introduce a scheduler or change the learning objective. For an interrupted lesson, epoch measurements survive, but the current optimizer is not saved mid-lesson; a new attempt restarts that lesson from the original source.

### Search decision

The paper searches beta in `{0.001, 0.01, 0.1, 1}` and selects 0.01 for Tiny ImageNet. [Appendix C, Table 5](https://arxiv.org/html/2509.17530)

There is one relative loss weight in our learning objective: beta. A grid over two independent loss coefficients is not needed for this objective. A beta-by-learning-rate grid is a separate, larger experiment and is not scheduled now.

The immediate plan is one measured beta-0.01 lesson. If its losses and gradients show that protection is active and correct, an optional four-value beta sweep is already supported by changing BETAS in notebook 14. Every value starts from the same selected checkpoint, with the same source RNG state, code, seed, task images, and training settings. Each value has its own result folder. The completed baseline can be reused.

Compare both new-task accuracy and old-task score changes. The comparison report includes the mean old-task change, worst old-task drop, and individual changes; it does not automatically pick a winner by retention alone. A setting that prevents all old-task loss but fails to learn the new task is not sufficient.

This continuation sweep tests beta only for the new lesson. It cannot reproduce the paper's beta search throughout an entire sequence or establish generalization across seeds. If beta does not improve the learning/retention tradeoff, the next controlled question is the documented optimizer/schedule gap, not an unlimited expansion of the search.

### Validation and publication

Training revision: `6a59648a5512724f94917ed9b97a12807c68809b`. The targeted CPU regression set passed 29 tests; one CUDA RNG test was skipped because this machine has no CUDA runtime. Notebook code cells compile, and the written prose was checked for the repository style constraint. No Colab GPU training or inspection of the user's actual Drive checkpoint files was performed locally.



## Evidence and maintenance

Original checkpoints and GPU report files remain at their reported Drive paths; they were not downloaded or independently inspected here. Screenshots and pasted outputs are labelled as user-reported results. Missing gradients, environment details, timings, and fingerprints are not reconstructed.

The tracked measurements are in [registry.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/registry.json), [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json), and the CSV and text evidence under [docs/experiments](https://github.com/sumitasthana/CARK/tree/main/docs/experiments). Figures retain their dated source data. The registry covers the older archived runs; the newly reported results above have not been added to that structured registry.

Update this page for subsequent experiments. Keep the question, starting model, actual settings, observed results, limitations, and next decision together. Keep measurements in their evidence files and link them here. Git history retains previous versions of the page.
