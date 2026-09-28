# Numerical checks of Fisher calculations

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Do the gradient and Fisher calculations match their definitions on a small model?

## Record

| Field | Value |
| --- | --- |
| Identifier | fisher-sanity-check |
| Category | numerical check |
| Date | Not recorded |
| Date evidence | Not recorded |
| Reported source commit | Not recorded |
| Hardware | Windows CPU; Python 3.14.6 and torch 2.12.0+cpu (reported) |
| Starting state | Not recorded |


## Settings

| Setting | Value |
| --- | --- |
| input_features | 4 |
| classes | 2 |
| model | linear |
| class_expectation | exact |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| autograd_gradient | 2.810392324786 |
| finite_difference_gradient | 2.810392322417 |
| reported_absolute_difference | 2.369e-09 |
| empirical_to_model_fisher_ratio_uniform | 1.0 |
| ratio_confident_correct | 0.01 |
| ratio_confident_wrong | 99.0 |
| square_then_average | 0.3681804 |
| average_then_square | 0.0140179 |
| reported_ordering_ratio | 26.27 |


## Decision

Recorded numerical checks passed

## Interpretation

The saved output checks gradients, differences between Fisher definitions, and averaging order on this constructed example.

## Limitations

- Date and source commit were not recorded. This check was not rerun for the archive.
- No evidence here establishes that an unlearned classifier must be confidently wrong. Chance accuracy does not determine confidence.
- This does not implement or validate Fisher attribution on the hypernetwork.

## Next step

No separate next step was recorded.

## Evidence

- [fisher_sanity_output.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/fisher_sanity_output.txt): saved numerical output. Copied from the fenced output in ops-docs/airlift/04_sanity_output.md.
