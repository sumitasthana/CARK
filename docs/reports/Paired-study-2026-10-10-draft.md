# Paired continual-unlearning study: draft report

Draft dated 10 October 2026. Study: `paired_generalization_v2`. This report covers the selected order-01 study, not the full original matrix. Review snapshot saved at 2026-10-10T20:16:26.108265+00:00. Results may be added after this cutoff.

## Current finding

Four of six selected comparisons are complete, covering seeds 0 and 1. In all four, the forgotten task falls from above chance to 10% validation accuracy after unlearning and remains at 10% after learning task 15. No evaluated L15 epoch exceeds the fixed 12% tolerance after the unlearning gate is met. This is evidence of suppressed classification performance during one subsequent lesson, not proof of information erasure or resistance to recovery.

Task-15 effects depend on the seed. U3/L15 changes task-15 accuracy by 0.0 points for seed 0 and +3.2 points for seed 1. U14/L15 changes it by -2.6 and +3.8 points. Both branches have negative average retained-task differences, with much larger losses on some individual tasks. Seed-2 outcomes and final three-seed summaries remain pending.

## Question and design

We test whether unlearning a previously learned task changes later learning and harms other tasks. The source sequence is:

**L3 → L0 → L9 → L5 → L17 → L1 → L7 → L14**

For each seed, the same completed source starts three independent continuations:

| Continuation | Requests | Role |
| --- | --- | --- |
| Control | L15 | Learn the new task without unlearning |
| Early-task branch | U3 → L15 | Unlearn task 3, which was learned first, then learn 15 |
| Recent-task branch | U14 → L15 | Unlearn task 14, which was learned last, then learn 15 |

Each branch is compared with its matching seed's L15-only control. The two branches share that control and are correlated. The selected study has 3 seeds, 12 jobs, 6 paired comparisons, and 39 requests. Other orders and U5 branches in the immutable 45-job plan are deferred. A job completion is distinct from a successful forgetting gate.

## Actual settings

| Setting | Selected diagnostic study |
| --- | --- |
| Dataset | Tiny ImageNet; fixed partition seed 42; 10 classes per task |
| Images per task | 5,000 training; 500 validation |
| Input and preprocessing | 64×64 RGB; tensor values in [0, 1]; no augmentation or additional normalization |
| Classifier | ResNet-50 with a 3×3, stride-1 stem; 10 outputs; no pretrained weights |
| Hypernetwork | 64-input shared MLP with hidden widths 128, 256, 512 and ReLU; three linear output heads |
| Codes and chunks | 32-value task code and 32-value chunk code; 200 generation chunks |
| Initialization | Hyperfan-in scaling of generated parameters |
| Optimizer | Adam; learning and unlearning learning rate 0.0001 |
| Learning | 5 epochs per request; batch size 64; validation batch size 256 |
| Unlearning | 100 steps; 10 Gaussian-noise samples per step |
| Loss weights | Learning protection beta 0.1; unlearning noise gamma 0.01 |
| Chance and forgetting gate | Chance 10%; fixed gate at or below 12% |
| GPU | A100 in the user-reported training sessions |
| Fixed training code | `65e2e7c850fd6f1852a99d9e81780c27f799c926` |

The hypernetwork generates the classifier's weights. Learning updates its shared generator and the new task code; chunk codes are trained only for the first learned task. Unlearning updates the shared generator while keeping codes fixed. Protection compares generated parameters with the snapshot taken before the request. Per-task batch-normalization buffers are retained. The study uses the generated, scaled parameters in both protection and unlearning noise losses.

This is an extension of the earlier experiment-06 question, not a paper reproduction. The selected beta is 0.1; the repository's paper-based Tiny ImageNet default is 0.01. The source is freshly trained with uniform settings and differs from the older mixed-setting checkpoint chain. Historical experiment-06 results are not pooled into these averages. Implementation: [fixed hypernetwork](https://github.com/sumitasthana/CARK/blob/65e2e7c850fd6f1852a99d9e81780c27f799c926/uncle/hypernet.py), [study training](https://github.com/sumitasthana/CARK/blob/65e2e7c850fd6f1852a99d9e81780c27f799c926/uncle/study_training.py), and [pair review](https://github.com/sumitasthana/CARK/blob/65e2e7c850fd6f1852a99d9e81780c27f799c926/uncle/paired_study.py).

## Paired results and placeholders

Absolute accuracy columns are percentages. Difference and drop columns are percentage points. Differences are branch minus control; positive values mean higher branch accuracy. Retained means exclude the forgotten target. Drop magnitudes are positive numbers describing a loss.

| Seed | Branch | Status | Target before U | Target after U | Target after L15 | L15 control | L15 branch | L15 difference | Retained mean difference | Largest final retained drop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | U3 → L15 | Complete | 35.60 | 10.00 | 10.00 | 50.80 | 50.80 | 0.00 | -1.09 | 11.00 |
| 0 | U14 → L15 | Complete | 42.60 | 10.00 | 10.00 | 50.80 | 48.20 | -2.60 | -1.60 | 14.20 |
| 1 | U3 → L15 | Complete | 40.40 | 10.00 | 10.00 | 44.00 | 47.20 | 3.20 | -1.57 | 11.60 |
| 1 | U14 → L15 | Complete | 40.40 | 10.00 | 10.00 | 44.00 | 47.80 | 3.80 | -1.74 | 14.00 |
| 2 | U3 → L15 | **PENDING: resumable** | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 2 | U14 → L15 | **PENDING: pending** | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

All recorded pairing checks pass for the four completed comparisons: matching source and starting model, configuration, task-15 initialization and random streams, frozen state, image values, batch order, and update count. The review labels these pairs complete. This validates the recorded pairing evidence; it does not establish cross-session determinism for every future runtime.

## Provisional averages

These means and sample standard deviations use only seeds 0 and 1. They are descriptive partial summaries, not the final planned three-seed results. Two seeds provide little evidence about the distribution of possible outcomes. No significance test or confidence interval is reported.

| Branch | Completed seeds | L15 difference: mean ± sample SD | Retained mean difference: mean ± sample SD | Final three-seed result |
| --- | --- | ---: | ---: | --- |
| U3 → L15 | 0, 1 | +1.60 ± 2.26 | -1.33 ± 0.34 | **Pending seed 2** |
| U14 → L15 | 0, 1 | +0.60 ± 4.53 | -1.67 ± 0.10 | **Pending seed 2** |

![Completed-seed paired effects; seed 2 pending](figures/paired-study-2026-10-10.png)

Missing results are displayed as pending, not as zero. Do not combine the two forget targets into six independent observations; their controls are shared and their retained-task sets differ.

## Individual retained-task changes

| Task | Seed 0: U3/L15 | Seed 0: U14/L15 | Seed 1: U3/L15 | Seed 1: U14/L15 |
| --- | ---: | ---: | ---: | ---: |
| 3 | Forgotten; excluded | -2.2 | Forgotten; excluded | +4.0 |
| 0 | +10.6 | +18.6 | -5.4 | -14.0 |
| 9 | -11.0 | -7.0 | +9.8 | +4.2 |
| 5 | +4.8 | -2.4 | +4.2 | +4.6 |
| 17 | -8.0 | -14.2 | +2.6 | -0.6 |
| 1 | -6.2 | -1.0 | -11.6 | -4.0 |
| 7 | +2.2 | -3.0 | -1.4 | -6.4 |
| 14 | +0.0 | Forgotten; excluded | -9.2 | Forgotten; excluded |

The average losses hide offsetting changes. For example, seed-0 U14/L15 improves task 0 by 18.6 points but reduces task 17 by 14.2 points. Seed-1 U14/L15 reduces task 0 by 14.0 points. The largest final retained loss ranges from 11.0 to 14.2 points across the completed branches. The largest recorded temporary drop during unlearning ranges from 18.2 to 21.6 points, relative to the starting model. Temporary and final drops use different baselines and should not be compared as the same metric.

## Runtime and interruptions

The CPU review contains 9 saved session logs totaling 46118.438499 seconds, rounded to **12 h 48 m 38 s**. This includes the interrupted R2 upload session. It is runner wall time, covering in-run dataset preparation, training, evaluation, and transfers. Earlier notebook setup, gaps, active sessions without a final log, and disconnected sessions whose log was not uploaded are excluded. It is not a complete measure of billed GPU consumption.

Learning saves at epoch boundaries and unlearning at 20-step boundaries. Checkpoints and reports are content-verified before the progress pointer advances. Interrupted work resumes with optimizer and random state; work after the previous verified save may repeat. Completed branch models are discarded after their verification receipt and report are durable. Source models, unfinished state, and reports remain. CPU review reads reports and checkpoint metadata without downloading models.

## Limits on the conclusion

- Only one learning order is tested in this narrowed study. Task identity and original learning position change together, so these results do not isolate a causal position effect.
- Task 15 is the only incoming task. There is no evidence here for other new-task requests or long future sequences.
- A 10% validation score does not prove removal of information, membership privacy, or inability to relearn. No reserved-image recovery or privacy test is part of this study.
- The reported persistence check covers the saved evaluations during five epochs of one new lesson, not every future model state.
- Positive and negative task-15 effects vary across the first two seeds. A reliable general direction should not be claimed before seed 2 is available.

## Remaining placeholders and finalization

| Item | Current status | Required update |
| --- | --- | --- |
| Seed 2 U3/L15 | Resumable | Target gate, relapse checks, task-15 and retained-task results |
| Seed 2 U14/L15 | Pending | Same result fields |
| Three-seed means and SD | Pending | Average paired effects separately for U3/L15 and U14/L15 |
| Final runtime | Partial snapshot | Add the last session logs |
| Final conclusion | Draft | Reassess direction, variation, forgetting persistence, and retained-task losses |

Keep SEED 2 in notebook 19 until both branches complete, then run notebook 20 on CPU and save verified R2 exports. No extra training on deferred orders is needed to complete this report's selected scope. Update this draft from the final reports; preserve the partial snapshot for provenance.

## Evidence

The draft is based on verified R2 reports under `uncle/paired_generalization/paired_generalization_v2/` and the matching CPU review. A compact [evidence snapshot](evidence/paired-study-2026-10-10.json) contains the immutable plan, comparison checks/results, unfinished-job list, and session records. The [dated experiment record](../wiki/Experiment-trajectory-2026-10-10.md) documents user-supplied progress and subsequent verification. No GPU training was performed to create this report.
