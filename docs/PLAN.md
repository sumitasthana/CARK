# UnCLe: reproduction and diagnostic implementation plan

This plan implements [PROPOSAL.md](../ops-docs/PROPOSAL.md). The proposal takes
precedence on experimental details, scope, and interpretation. This document
tracks the code, dependencies, checks, and runs needed to carry it out.
The proposal is currently in the ignored `ops-docs/` directory; the protocol
below records its requirements for readers of the tracked repository.

Last updated: 2026-09-27.

[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) records the reported GPU experiments
through E14 and the checkpoint-based diagnostic refactor. No reported
trajectory has passed both the short-run forgetting and retention criteria.
These are existing observations, not measurements made during this plan update.

## 1. Study question and scope

Does UnCLe retain information that helps it relearn a forgotten task? If so,
which model components contribute to that recovery?

| Component | Current behavior during forgetting | Planned test |
| --- | --- | --- |
| Task embedding | Unchanged | Adapt alone; replace with a random embedding |
| Shared layers | Updated | Adapt alone and jointly |
| Output heads | Updated | Adapt alone and jointly |
| Chunk embeddings | Frozen after the first task | Test jointly with shared layers and heads; no independent attribution |
| Per-task BatchNorm buffers | Unchanged in this implementation | Compare keeping and resetting statistics under an explicit adaptation policy |

The study evaluates unlearning and records implementation choices. Developing
an algorithm to remove information from identified components is outside scope.
Faster recovery is evidence to examine with the controls below. A negative
result is informative only when the pre-deletion positive control recovers.
Neither outcome alone establishes complete information deletion or storage in
a particular component.

## 2. Protocol from the proposal

### Dataset and model settings

- Tiny ImageNet: 20 tasks of 10 classes, ResNet50 target network, no pretrained
  weights, three request sequences, seeds 0, 1 and 2, class partition seed 42.
- Reserve 100 of the 500 training images per class before study training. Use
  the other 400 for initial learning and the reserved images for adaptation.
  Evaluate on the validation split. Apply the reservation to every study task.
- Use identical adaptation examples and evaluation examples for paired models.
  Save the split indices rather than reconstructing them from an unstored seed.
- Permuted MNIST with ResNet18 is the likely pilot and debugging setting.
  Results from that setting do not establish results on Tiny ImageNet.
- **[to discuss]** The 20-by-10 Tiny ImageNet partition uses all 200 classes.
  Resolve where the never-seen control task Y comes from before study training.
  Keep the existing partition for historical comparisons; record any diagnostic
  partition or sequence changes separately. Do not silently drop a task.
- **[to discuss]** Define the adaptation reservation for a Permuted MNIST pilot;
  the Tiny ImageNet counts do not specify that dataset's split.

The 400-image study training split differs from training on all 500 images per
class. State that difference when comparing against the published accuracy.

### Four model conditions

For each probed task X, match seed, architecture, data splits, and the order of
all common requests. Record the exact sequence for every condition.

| Condition | Training history | Purpose |
| --- | --- | --- |
| Unlearned | Learn X, then forget X in the full sequence | Model under test |
| Never-learned-X | Remove X's learn and forget requests | Paired learning reference |
| Pre-deletion | Stop the full run immediately before forgetting X | Positive control for probe sensitivity |
| Never-forgot-X | Keep X's learn request; remove only its forget request | End-of-sequence comparison |

The pre-deletion condition can reuse a saved checkpoint. Never-forgot-X is
specific to X; a sequence with every forget request removed is not equivalent.
Save aligned reference checkpoints for both immediate and endpoint probes.

Use separate recorded random streams for initialisation, task-code creation,
data order, forget noise, and adaptation where needed. Removing a request must
not silently change random draws for unrelated operations. The changed training
history still changes the model, protected-task sets, and potentially the
burn-in schedule. Record those differences rather than claiming identical
training. In particular, if X was the first task, removing it changes which
task trains the chunk embeddings; account for that when selecting probe tasks.

### Relearning and component interventions

Start every probe from a fresh checkpoint copy and a fresh optimiser. Disable
the preservation penalty during adaptation. Evaluate at fixed intervals,
including step zero, on adaptation images, held-out X images, and retained tasks.
The never-learned-X reference needs a newly initialised task embedding and
buffers. The unlearned model reuses its saved state unless the intervention
explicitly changes it. Apply the same initialisation policy to Y in both models.

| Intervention | Parameters allowed to adapt or state changed |
| --- | --- |
| Full adaptation | Task embedding, shared layers, and heads |
| Embedding only | X's task embedding |
| Shared layers only | Shared layers |
| Heads only | Output heads |
| Random embedding | Replace X's embedding; record whether it is then frozen or adapted |
| BatchNorm keep versus reset | Keep saved statistics or reset to the clean template; match other settings |
| Joint chunk test | Change or adapt chunk embeddings together with shared layers and heads |
| Combined interventions | Apply selected resets or swaps together and compare with their separate effects |

Every restricted arm needs its own never-learned-X reference with the same
training restriction. Chunk embeddings remain frozen in the ordinary arms;
the joint chunk test is the explicit exception. It does not isolate their
independent contribution.

**[to discuss]** Specify whether BatchNorm statistics update during adaptation,
separately from whether they are initially kept or reset. Define the random
embedding arm's trainable parameters, swap donors, and combined interventions
before the main runs. Recalibrating buffers without gradient updates can be a
supplementary diagnostic, but does not replace the keep-versus-reset comparison.

Earlier planning choices remain candidates where the proposal is silent:

| Setting | Candidate to confirm before the main runs |
| --- | --- |
| Sequence-1 probe tasks | 3, 17, 9, 0, 12; recheck against the final Y design and sequence |
| Main budgets | 1, 5, 10, 25, 50, 100 images per class, as nested subsets |
| Adaptation optimiser | Adam; set and record an adaptation learning rate, separately from any forgetting-only rate |
| Batch size | `min(64, adaptation pool size)` |
| Update budget | 200 steps per budget; measurements at 0, 20, 40, ..., 200 |
| Probe timing | Immediately after forgetting and at the sequence endpoint; pilot uses the immediate probe |

A fixed update count matches the number of optimiser steps across budgets.
It does not guarantee identical runtime when batch sizes or trainable
components differ. Measure cost separately. These candidate values are not
requirements stated by the proposal.

### Metrics

Compute paired metrics at matching adaptation budgets and measurement steps.
Keep raw values for each task, seed, sequence, condition, and intervention.

| Metric | Definition | Purpose |
| --- | --- | --- |
| Recovery advantage `A(b)` | Unlearned accuracy minus never-learned-X accuracy after adaptation on `b` images per class | Primary recovery comparison |
| Likelihood gain `G(b)` | Unlearned minus reference mean correct-class log-probability on held-out images | Detect differences when accuracy is near chance |
| Specificity ratio `S(b)` | Recovery advantage on X divided by the corresponding advantage on never-seen Y | Check generic transfer |
| Component share `p(c)` | Drop in recovery advantage after resetting or swapping component `c`, divided by the full advantage | Measure dependence on the intervention |
| Interaction residual `D` | Effect of a combined intervention minus the sum of separate effects | Check whether separate component effects explain the combined result |
| Maintenance cost `M(b)` | Change in mean retained-task accuracy during adaptation | Report recovery's effect on retained tasks |

Also report retained accuracy, forgotten-task accuracy, spill, and relapse.
For `M(b)`, record the retained-task set and use each probe's step-zero value
as the baseline. State the sign convention. For `p(c)` and `D`, record the exact
reset or swap and the paired reference treatment. Restricted-adaptation speed
alone is not the reset-or-swap measurement that defines `p(c)`.

Report numerator and denominator alongside `S(b)` and `p(c)`. Mark ratios
undefined when their denominator is zero. **[to discuss]** Set a policy for
near-zero denominators before analysis; do not silently clip them or interpret
unstable ratios as strong effects.

### Controls and interpretation

| Concern | Required control or measurement |
| --- | --- |
| Generic transfer | Apply the same probe to Y, which was absent from every training sequence |
| Memorisation of adaptation images | Report adaptation accuracy alongside held-out accuracy |
| Damage to retained tasks | Measure retained accuracy at every interval and compute `M(b)` |
| An insensitive probe | Include the pre-deletion positive control in every experiment; a failed control makes a null result inconclusive |
| Different optimisation difficulty | Match each restricted arm to a reference under the same restriction |
| Recovery attributed to the embedding | Replace the stored embedding with a freshly sampled vector |
| Indirect preservation of X | Repeat one forget request without the preservation penalty and report retained-task damage |
| Component interactions | Compare combined interventions with the sum of individual effects |

Removing protection during forgetting is a separate experiment from disabling
it during relearning. Implement a separate preservation switch or coefficient;
setting the current `gamma` to zero would remove the noise term instead.

Compare within seed and sequence. Five tasks across three seeds give fifteen
paired comparisons, but tasks within a run share a model and are not independent.
Report per-pair results and uncertainty that respects that dependence.
**[to discuss]** Fix the decision rule, uncertainty method, and smallest effect
of interest before the main runs. The earlier mean-gap-versus-seed-spread rule
is not a settled statistical criterion.

Membership inference is needed if the final paper makes the proposal's
conditional claim that recovery occurs while accuracy, relapse, and membership
inference indicate success. Specify and run that evaluation before making the
claim. Without it, report comparisons only against the metrics actually tested.

## 3. Current implementation status

Existing code provides configuration, task streams, model construction, learning,
forgetting, four original metrics, request runners, telemetry, and checkpoints.
`uncle/diagnostics.py` restores a saved model and records a forgetting trajectory.
Existing tests cover parts of these paths; no tests were rerun for this document
update, and software checks do not establish a successful reproduction.

The reproduction remains unresolved. The earlier full Tiny ImageNet run reported
10% retained accuracy against 55.24% in the paper, with mean spill 30.72.
The later short diagnostics through E14 have not met both screening criteria.
See the experiment log for measured settings and limitations.

The dedicated relearning path, four-condition reference builder, reserved probe
splits, Y control, component intervention framework, Fisher screening, and six
diagnostic metrics are not yet implemented as a complete study workflow.
Membership inference is also missing. CIFAR-100, 5-Tasks, alternative noising
strategies, and the seven comparison methods are outside the required build
unless the study scope changes.

## 4. Implementation decisions to retain

- Cite the seven comparison methods without reimplementing them, as the proposal
  currently plans. **[to discuss]** Confirm the final comparison scope.
- Keep the seed-42 class partition for existing runs. Resolve Y explicitly for
  the diagnostic study and record any departure from the reproduction setting.
- Use Tiny ImageNet validation labels for evaluation; its public test split
  has no labels. Keep `uncle/data.py` routed through the existing task loaders.
- Retain the current `1/fan_in` classifier output scale as an implementation
  choice. The paper specifies Hyperfan, which is not implemented here. Record
  this difference; do not assume its impact is settled.
- Preserve the planned 100-image adaptation reservation before study training.
  That is 20% of the 500 training images per class, leaving 400 for learning.
  Older checkpoints trained on all images cannot serve as split-compliant study
  checkpoints; they can still support reproduction debugging.
- Add a dedicated adaptation entry point. Keep ordinary continual-learning
  checks against duplicate learn requests; the probe should not need to bypass
  them or call `add_task` on an existing embedding.
- Preserve explicit generator restrictions, no protection during adaptation,
  and equal update counts within each paired comparison. Use a fresh optimiser.
- Record unspecified choices such as epochs, batch size, parameter scaling,
  head allocation, and buffer policies. Separate code behavior from claims
  about what the original method intended.

## 5. Operational constraints

- Give every study run a distinct identifier covering its configuration,
  condition, split, checkpoint, and intervention. Existing sequence/backbone/seed
  filenames are insufficient for larger sweeps.
- Keep datasets on local storage in GPU runtimes and persist results separately.
  Pass the selected classes to the Tiny ImageNet reader to avoid indexing
  unused classes.
- Clear `UnCLe._reference` whenever the snapshot or protected task codes change.
  Reusing stale reference weights changes the preservation objective.
- The normal request runner reuses the previous request's accuracy as the next
  request's starting value. Intervention branches must evaluate their actual
  starting state after any reset or swap.
- `run_experiment` returns a dictionary. Keep notebook examples consistent with
  the callable API and run the guide checker when changing executable examples.
- Keep evaluation observational: it must not update stored buffers, parameters,
  or the random stream used by training.

## 6. Modules to build or extend

Module IDs below identify functional responsibilities, not a requirement to
create one file per module. Existing paths are starting points. Each module
needs recorded inputs and outputs so another run can reproduce its result.

### M01. Study configuration

Extend `uncle/config.py` or add a separate probe configuration. Record conditions,
X and Y, budgets, checkpoint stages, random streams, interventions, evaluation
intervals, and analysis choices. This keeps paired comparisons consistent.
Depends on the protocol decisions in section 9. Accept when incomplete study
configurations fail validation and saved configurations reproduce the run matrix.

### M02. Dataset splits and control task

Extend `uncle/data.py`, `uncle/tasks.py`, and `uncle/streams.py`. Build persistent
initial-training, adaptation, and evaluation indices, nested budgets, and the
never-seen Y task. This prevents overlap and supplies the specificity control.
Depends on M01 and the Y decision. Accept when split counts and disjointness are
checked, Y is absent from all training sequences, and paired runs use identical
adaptation examples and labels.

### M03. Model component access

Extend `uncle/hypernet.py` and the trainer's buffer handling with explicit groups
for embeddings, shared layers, heads, chunk embeddings, and BatchNorm buffers.
This makes interventions precise. Depends on M01. Accept when groups match the
actual model, parameters and buffers are distinguished, and interventions change
only the requested state.

### M04. Learning and forgetting

Use `uncle/trainer.py` and `uncle/experiment.py`; resolve the reproduction failure
and add an explicit way to remove preservation for the requested control.
This is the method the study evaluates. Depends on M01-M03. Accept when the
proposal's applicable reproduction or recovery-entry criteria in section 7 are
measured, forgetting uses no X data, and implementation departures are recorded.

### M05. Reproduction diagnostics

Extend `uncle/diagnostics.py`. Restore the same checkpoint for each comparison;
inspect raw-versus-scaled objectives, parameter-group contributions, and retained
accuracy trajectories. This addresses the current blocking issue before the
study runs. Depends on M03, M04, and existing checkpoint support. Accept when
reports identify their starting checkpoint and settings, preserve source state,
and distinguish observed results from candidate explanations.

### M06. Matched reference construction

Extend the experiment runner to build all four model conditions, aligned stages,
and recorded random streams. This supplies the recovery baseline and controls.
Depends on M01, M02, M04, and M07. Accept when request-list differences are exactly
those specified, initial shared state matches, and the manifest records changes
to protection sets, burn-in, and first-task chunk training.

### M07. Checkpoint collection and restoration

Extend `uncle/checkpoint.py` and the runner to retain pre-deletion,
immediate-post-deletion, and endpoint checkpoints without overwriting them.
Include all model parameters, task buffers, random state, and split identifiers.
This isolates probes and allows aligned comparisons. Depends on M01-M03.
Accept when restoration reproduces step-zero outputs and probing cannot mutate
the source or another branch. Save optimiser and sampler state too if resuming
inside an adaptation run; otherwise restart that probe from its source.

### M08. Relearning probe

Add a dedicated adaptation API alongside `UnCLe.learn`. Support existing task
state and creation of missing state for reference models and Y, fresh optimisers,
fixed update counts, and protection disabled. This implements the primary test.
Depends on M02, M03, and M07. Accept when paired runs use matching examples and
update budgets, record step zero, and learn in a controlled positive example
without changing frozen parameters.

### M09. Component interventions

Implement the arms in section 2, including resets, swaps, joint chunk tests,
and combined changes. This tests contributions to recovery rather than only
reporting a model-wide gap. Depends on M03, M07, and M08. Accept when trainable
parameter sets and buffer policies are verified for every arm, each arm has a
matched reference, and swap donors and incompatibility checks are recorded.

### M10. Experimental controls

Implement all eight controls in section 2, including Y, adaptation-versus-held-out
accuracy, the positive control, and removal of preservation during forgetting.
These check alternative explanations for a recovery advantage. Depends on M02,
M04, M06, M08, and M12; component controls also depend on M09. Accept when every
reported result links to its required controls and failed controls are visible.

### M11. Fisher screening

Add diagonal empirical and model-predicted Fisher calculations at the
pre-deletion checkpoint. Square per-example gradients before averaging; compute
the model-predicted expectation exactly over classes, then normalise component
scores per parameter. This orders component tests without claiming storage.
Depends on M03 and M07. Accept when a small exact calculation agrees with the
implementation, input examples are recorded, model state stays unchanged, and
BatchNorm buffers are explicitly excluded from the ranking.

### M12. Evaluation and metrics

Extend `uncle/metrics.py` and evaluation to implement all six diagnostic metrics,
held-out log-probabilities, adaptation accuracy, and retained-task measurements.
This turns raw probe outputs into the proposed comparisons. Depends on M01 and
M08; attribution metrics also use M09. Accept when hand-calculated cases agree,
paired records cannot be mismatched, and zero-denominator ratios are reported
as undefined. Preserve the original four unlearning metrics.

### M13. Statistical analysis

Add paired analysis by task, seed, and sequence with uncertainty that respects
shared models. Record the decision rule and smallest effect of interest before
main experiments. This supports interpretation of positive and null results.
Depends on M12 and the section 9 decisions. Accept when per-pair results remain
available, shared-model dependence is retained, and conclusions follow the
recorded rule rather than a rule chosen after seeing the main results.

### M14. Experiment orchestration

Extend `uncle/experiments.py` with separate pilot and main-study run matrices,
condition dependencies, continuation criteria, unique output paths, and resume
behavior. This avoids missing or duplicated comparisons. Depends on M01,
M06-M08, M10, and M12; full attribution also needs M09 and M11. Accept when a
small run exercises the complete pilot matrix and an interruption neither mixes
configurations nor silently repeats completed probes.

### M15. Logging and report generation

Extend `uncle/telemetry.py` and add analysis outputs for recovery curves,
likelihood, X-versus-Y comparisons, component effects, interactions, and retained
accuracy. Record code version, configuration, split and checkpoint identities,
runtime, GPU memory, and raw measurements. This makes findings traceable.
Depends on M12-M14 for final reports; log support starts with M01. Accept when
every plotted value can be traced to a raw run record and figures can be rebuilt
without retraining.

### M16. Experiment verification

Extend `tests/` with checks for split separation, matched histories, restoration,
probe isolation, freeze masks, BatchNorm behavior, Fisher, metrics, and run
resumption. These detect errors that could create or hide recovery advantages.
Build checks alongside the relevant modules. Accept when they include positive
examples and deliberately invalid inputs, and a small end-to-end run exercises
the study path before expensive experiments.

### Conditional module: membership inference

Implement a separate evaluator if retaining the membership-inference comparison.
Specify attack access, original member and nonmember sets, held-out attack
evaluation, and success criteria; record any difference from the original paper's
protocol. Depends on M02, M06, M07, and M15. Accept when the attack is checked on
appropriate controls and evaluated on the same study checkpoints. Do not infer
membership-inference success from chance-level forgotten-task accuracy.

## 7. Execution order, schedule, and budget

### Weeks 1 and 2: reproduction and study foundations

Resolve the objective and scaling issue using the existing checkpoint diagnostic.
The next reproduction task remains **T0.2**, so references in the experiment log
still point to the objective investigation. Pause further scalar sweeps until
that investigation identifies a reason for the next change. Use
[the diagnostic notebook](../notebooks/03_forgetting_diagnostics.ipynb) with the
experiment log's restore instructions; do not automatically repeat saved runs.

The existing short screen is a debugging check: before forgetting, both tasks
must have at least 25% accuracy; a candidate step must bring X to at most 12%
while changing retained-task accuracy by less than five percentage points.
Passing it still requires full-sequence validation.

In parallel with that code work, implement M01-M03 and M07 and resolve the data
split. Historical unsplit checkpoints remain diagnostic-only.

The proposal distinguishes two thresholds for the Tiny ImageNet setting:

- Recovery work may begin once retained accuracy exceeds 40% and spill is below 3.
- The reproduction target is retained accuracy within 10 percentage points of
  55.24%, forgotten-task accuracy within one point of 10%, and spill below 2.

Record forgotten-task accuracy at the recovery-entry check too. Do not report
that meeting the entry threshold completes the reproduction. Run the proposed
three request sequences and three seeds for the main reproduction. If using
Permuted MNIST as the fallback or pilot, define suitable criteria for that
setting before interpreting its results; Tiny ImageNet thresholds are not its
published targets.

### Week 3: probe implementation and pilot

Complete the minimum working M06, M08, M10, M12, M14-M16 path. The pilot uses
one task, three seeds, three small budgets, the immediate-post-deletion probe,
the never-learned reference, the pre-deletion positive control, and Y. The
never-forgot-X condition is required for the later endpoint study.

**[to discuss]** Choose the pilot dataset, task, budgets, and runtime estimate.
Proceed to component tests only if the pre-deletion control shows a recovery
advantage and the measured recovery advantage has a consistent sign across
seeds. Otherwise stop component work and report what the pilot establishes.
This continuation rule does not require an assumed positive finding and does
not replace the final statistical decision rule.

### Week 4: reference models and attribution support

After the pilot decision, build the main four-condition references for selected
X tasks and seeds, preserving both immediate and endpoint stages. Complete
M09, M11, and M13. Use Fisher scores to order the parameter-component tests;
BatchNorm tests do not depend on those scores. Finalise all analysis and
intervention choices before the main recovery comparisons.

### Week 5: recovery comparisons and controls

Run the main recovery budgets, component interventions, joint chunk tests,
combined interventions, and all eight controls. Include pre-deletion controls
in every experiment and matched restrictions in every arm. Run the conditional
membership-inference evaluation if making that comparison.

### Week 6: analysis and writing

Complete M13 and M15 outputs. Report uncertainty, failed controls, retained-task
costs, implementation choices, and the limits of attribution. Do not require
BatchNorm or embeddings to rank first for the experiment to be valid. A null
result with a successful positive control is a reportable outcome under the
proposal; it is not proof of complete deletion.

The proposal budgets about **50 A100-hours**, to be revised after the pilot.
Measure actual reference, adaptation, Fisher, and evaluation costs before
committing the remaining matrix. Earlier per-phase estimates and an assumed
2x H100 speedup do not establish the cost of this revised design.

Broad beta/gamma grids, alternative noising strategies, a learning-only
saturation study, extra datasets, and a buffers-only recalibration arm are
optional follow-ups. They do not precede or replace the proposal's pilot.

## 8. Verification and deliverables

Run checks appropriate to each code change and record their results. Existing
entry points are:

```bash
python tests/test_uncle.py
python tests/test_tasks.py
python tests/test_experiments.py
python -m unittest discover -s tests -p test_diagnostics.py
python scripts/check_guide.py
```

Some checks require the Tiny ImageNet data or the local Colab guide. Record
missing prerequisites or skipped checks explicitly. All applicable checks must
pass before committing implementation changes. Update this list as the new
study modules receive tests; avoid fixed test counts that become stale.

Before a main GPU run, verify the resolved configuration, split identities,
condition sequences, checkpoint stages, freeze masks, BatchNorm policy, and
required controls. Verify on small cases that evaluation does not affect the
next training update and that restoring a checkpoint reproduces its outputs.

The study deliverables are the reproduction record, relearning entry point,
four-condition checkpoint manifest, raw probe and control measurements, six
metric calculations, component and interaction results, analysis protocol,
rebuildable figures and tables, and runtime report. A successful experiment
produces interpretable evidence; it does not require the preferred hypothesis
to be confirmed.

## 9. Open decisions before dependent work

- **Y and splits:** choose the Tiny ImageNet control-task design and record its
  effect on task count, class partition, and request sequences before study
  training. Define the Permuted MNIST reservation if used.
- **Pilot:** choose dataset, task, three budgets, training and adaptation
  settings, and runtime estimate. State applicable reproduction criteria for
  a pilot outside Tiny ImageNet.
- **Main matrix:** confirm selected X tasks and sequences, adaptation budgets,
  step count, evaluation intervals, and learning rate. The candidate table in
  section 2 is not a substitute for these decisions.
- **Interventions:** fix BatchNorm update behavior, random-embedding policy,
  swap donors, joint chunk treatment, combined interventions, and the forget
  request used for the no-preservation control.
- **Analysis:** fix the decision rule, smallest effect of interest, uncertainty
  procedure, and handling of near-zero ratio denominators before main runs.
- **Comparison scope:** confirm whether membership inference is evaluated and
  retain only claims supported by the completed measurements. Confirm the plan
  to cite rather than implement the seven comparison methods.
- **Novelty:** complete the literature review before claiming this setting or
  attribution protocol has not been tested before.

Resolve each item before its dependent run. Continue independent implementation
and reproduction debugging while those choices remain open.
