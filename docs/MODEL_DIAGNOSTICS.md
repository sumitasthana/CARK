# Model diagnostics

A model diagnostic lets us look inside a model while it learns or forgets.
An experiment report records the results; it is not the diagnostic itself.

## What can we inspect now?

| Measure | Plain meaning | Current code |
| --- | --- | --- |
| Weights | The model's adjustable numbers. Which groups moved, and by how much? | `inspect_components` reads each named tensor; the before/after audit measures changes. |
| Gradients | The signal asking each weight to move. Which term asks for a larger move, and do the terms point in the same direction? | `inspect_components` reads live gradients. The forgetting run records gradient sizes and alignment. |
| Adam updates | The moves the optimizer actually made. | The forgetting run records update size by parameter group. |
| Diagonal Fisher | For each weight, how sensitive is a chosen prediction score to that weight on chosen examples? | `diagonal_fisher` computes empirical or model-predicted values from individual examples. |
| Accuracy | Did behavior change on the task we want to forget and the tasks we want to keep? | The run records it after every step. |

The Fisher function computes the **diagonal**, one value per weight. It does
not compute the full weight-by-weight matrix. For empirical Fisher it uses the
true class of each example. For model-predicted Fisher it sums over every class,
weighted by the model's predicted probability. In both cases it squares each
example's gradient before averaging. BatchNorm running statistics are buffers,
so they are not ranked as trainable weights.

These measurements answer different questions. A large gradient is a request
to move; an Adam update is the move that happened. A Fisher score measures
sensitivity on the examples used. None of these alone proves that forgotten
information is gone or tells us where it is stored.

## What is missing?

The pieces are reusable across models, but there is not yet one notebook or
command that loads the E08 checkpoint and runs all the weight, gradient, and
Fisher views together. We also need to choose and record the examples used for
Fisher. The proposal calls for Fisher at the pre-deletion checkpoint, followed
by matched relearning tests. Past GPU runs did not produce these Fisher values.

For the existing runs, use `python scripts/research_diagnostic.py --archive` to
read the accuracy record. In a new forgetting run, the
[`04_gradient_diagnostics.ipynb`](../notebooks/04_gradient_diagnostics.ipynb)
notebook records gradients, Adam update sizes, and a before/after component
audit. The lower-level inspection functions are in
[`uncle/research_diagnostic.py`](../uncle/research_diagnostic.py).
