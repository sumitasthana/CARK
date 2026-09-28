# Numerical check of the noise objective

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

[Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index)

## Question

Do fresh Gaussian targets produce noise-shaped or near-zero outputs?

## Record

| Field | Value |
| --- | --- |
| Identifier | noise-objective-check |
| Category | numerical check |
| Date | Not recorded |
| Date evidence | Not recorded |
| Reported source commit | Not recorded |
| Hardware | Not recorded |
| Starting state | Not recorded |


## Settings

| Setting | Value |
| --- | --- |
| vector_values | 2000 |
| updates | 3000 |
| variants | ["average of 10 fresh draws", "one fresh draw", "one fixed draw"] |

Reported or intended settings in the cited record; original configuration JSON not independently inspected.

## Observations

| Measurement | Value |
| --- | --- |
| reported_final_spread_average_10_fresh | 0.04 |
| reported_final_spread_one_fresh | 0.07 |
| reported_final_spread_fixed_draw | 0.98 |


## Decision

Numerical behavior reported

## Interpretation

The recorded toy result is consistent with the expected squared-error objective being minimised at zero for fresh zero-mean targets.

## Limitations

- Only summary values are available. The definition of spread, seed, optimiser, runtime, and original output were not archived here.
- This is a vector optimisation check, not a model unlearning experiment.

## Next step

No separate next step was recorded.

## Evidence

- [README.md](https://github.com/sumitasthana/CARK/blob/main/README.md): reported numerical result. Section: What the noise objective actually does.
