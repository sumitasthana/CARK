# Research diagnostic: current code

The study asks whether a model can relearn a forgotten task faster than a
matched model that never learned it. It also asks which model parts matter.
The [proposal](../ops-docs/PROPOSAL.md) defines that test.

The diagnostic means looking inside the model. See the plain-language
[model diagnostics guide](MODEL_DIAGNOSTICS.md) for weights, gradients, Adam
updates, and Fisher. The run archive is only one output of this work.

## What works now

`uncle/research_diagnostic.py` evaluates each step of a forgetting trace. It
accepts the tracked historical tables or a full `diagnose_forgetting` JSON
report. It keeps every step, the exact source type, settings, checkpoint path
when known, the earliest passing steps, and the best target accuracy while
retention stays within the screen. Transcribed tables remain marked as such;
their missing original JSON metadata is not filled in.

Run from the repository root:

```bash
python scripts/research_diagnostic.py --archive
python scripts/research_diagnostic.py --archive --report /path/to/forget.json
python scripts/research_diagnostic.py --report /path/to/forget.json --output /path/to/assessment.json
```

The first command includes the nine tracked checkpoint trajectories, including
the reported 30-step run from 2026-09-29. The second can add the original JSON
from Drive without replacing its source. The output file includes the full
assessed trajectory. The source JSON remains the primary record.

The 30-step Colab notebook now requests a before/after component audit. The
audit captures named tensors and measures their changes for these UnCLe roles:
task embedding, chunk embeddings, shared layers, output heads, and the task's
normalization buffers. The comparison itself accepts any mapping of semantic
roles to named tensors. A role without tensors is marked unavailable. An
unchanged role is a state observation, not proof that it contains or lacks
task information.

The same component mapping supports two intervention primitives.
`select_trainable_components` freezes all model parameters except the chosen
roles and returns those parameters for a fresh optimizer.
`replace_components` copies named donor tensors after checking their shapes
and types, then audits every role for unintended changes. These primitives do
not choose donors or define the study arms.

`inspect_components` reads current weight and gradient sizes by named tensor.
`diagonal_fisher` computes empirical or model-predicted diagonal Fisher from
individual examples. It returns values per weight and mean scores per
component. The caller supplies the model, examples, and a logits function;
there is no E08 checkpoint runner for it yet. It uses evaluation mode and
restores the model's prior training flags. It does not alter stored gradients.

The notebook still saves accuracy and loss measurements at every forgetting
step. It now also records gradient cosine and actual Adam update size by
parameter group. Cosine describes alignment between the two objective terms;
update size describes the optimizer's resulting parameter movement. Neither
alone explains the effect on accuracy.

## How future probes fit

`paired_recovery` accepts model-neutral observations keyed by pair, task,
seed, sequence, split identity, adaptation budget, step, and intervention. It
rejects missing unlearned or never-learned conditions, duplicate conditions,
and missing step-zero baselines. It computes paired recovery advantage `A`,
correct-class log-probability gain `G`, and retained-task maintenance cost `M`
at matching points. It exposes whether pre-deletion and never-forgotten controls
are present and the pre-deletion recovery gap. It leaves probe sensitivity
undecided until an explicit decision rule is set.

The separate `specificity_ratio`, `component_share`, and
`interaction_residual` functions implement `S`, `p(c)`, and `D` once matched
X/Y and component interventions exist. Zero denominators produce an undefined
ratio with both operands retained. The near-zero policy remains open in the
plan and should be fixed before the main analysis.

An adapter for another model needs to expose semantic state roles and produce
the same probe observations. The analysis does not assume ResNet, BatchNorm,
three output heads, or a 200-chunk generator. The UnCLe adapter is the first
implementation, not a requirement of the study protocol.

## What is still required for the proposed result

The current E08 checkpoint was trained before the study's 100-image-per-class
adaptation reserve existed. Its forgetting reports have no never-learned-X
reference, pre-deletion recovery probe, Y control, or component intervention.
They are reproduction diagnostics only.

Next, build the reserved split and Y control, then a separate adaptation entry
point with a fresh optimizer and matched reference checkpoints. Wire the
component primitives into restricted adaptation, embedding replacement,
BatchNorm keep/reset, and combined study arms. Only matched probe results can
support a recovery or component claim. Fisher ranking and statistical analysis
remain later modules. The
protocol's combination is a proposed contribution whose novelty requires the
literature review noted in the proposal; this software does not establish it.
