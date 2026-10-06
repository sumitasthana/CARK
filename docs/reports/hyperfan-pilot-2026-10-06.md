# Hyperfan learning pilot, 6 October 2026

The new `Config(initialization="hyperfan_in")` option changes the starting recipe for a fresh model. The default remains `legacy` so historical runs and checkpoints keep their original generated weights. The protection objective is unchanged.

## Formula and implementation choices

Hyperfan-in sets the effective output-head weight variance to `gain_squared / (bias_factor * target_fan_in * hidden_width * embedding_variance)`. Here embedding variance is one, `bias_factor` is two when generating an affine layer's bias, and gain squared is two for ReLU layers or one for linear layers. Generated bias variance is `gain_squared / 2`. See Table 1 and section 4.1 of [Chang, Flokas, and Lipson](https://arxiv.org/html/2312.08399#S4.SS1).

Our shared chunk heads generate parameters with different fan-in values. A head initialized with variance `1 / hidden_width`, followed by a separate scale for each generated tensor, implements the effective variance formula. The hidden ReLU trunk keeps its Kaiming fan-in initialization. Neither embeddings nor chunks are normalized using observed samples.

The original bias-free convolution recipe was already consistent with this formula. The new recipe changes these policies:

- Convolution weights use gain squared two. If a convolution has a generated bias, half the variance budget goes to that bias.
- The final classifier is linear, with gain squared one. Its weight variance is `1 / (2 * fan_in)` and its generated bias variance is `1 / 2`.
- BatchNorm starts with gamma one and beta zero. Its output head starts at zero and remains trainable. An additive offset supplies gamma one. Output scale is one for both parameters.

The UnCLe paper does not specify these classifier and BatchNorm details or which Hyperfan variant it used. These are documented reproduction choices, not recovered author code. The new option has been checked for learning; this pilot does not assess its forgetting behavior.

## CPU evidence

[Saved check](../experiments/hyperfan_cpu_check_20261006.json): full Tiny ImageNet ResNet50 shapes, 200 chunks, hidden widths 128/256/512, seed 0, PyTorch 2.12.0+cpu. Original adaptive average pooling was used, matching the published pilot.

- Generated classifier weight standard deviation: `0.0143448` (formula reference `0.015625`). The old recipe measured `0.0004483` for the same seed and architecture.
- BatchNorm gamma was exactly one and beta exactly zero.
- A synthetic two-image forward/backward pass had finite generated values, predictions, loss, and gradients. Cross-entropy was `2.8180091`.
- BatchNorm-head gradients were nonzero. The smaller automated check also verified that an optimizer update moves gamma.
- Across 24 independent initializations, generated affine weight and bias second moments were within 20% of the formula references.
- Legacy generation matched the pinned historical implementation exactly.
- Protection values and gradients matched a direct squared-distance calculation with the new offsets.

Synthetic images do not measure dataset accuracy. A single tensor's empirical spread need not equal the formula's expectation across random initializations.

## Checkpoints and notebook

Checkpoint files record the initialization option. Files predating this field are interpreted as `legacy`. Restoring a checkpoint into a model with a different initialization option is refused before loading model state. The offsets and scales are derived from that option.

Notebook [13](../../notebooks/13_hyperfan_learning_pilot.ipynb) starts a new model and learns tasks 3, 0, and 9. Settings: seed 0, beta 0.01, learning rate 0.0001, five epochs per task, batch size 64, ResNet50, and the existing fixed class partition. No scheduler or deterministic-mode change is introduced. Learning rate and epoch count remain diagnostic settings, so this is not a complete paper reproduction.

The notebook saves the code version, settings, class-partition hash, task scores, environment, and a checkpoint after each completed task. It can resume its own checkpoint after a disconnected session. It does not load E08 or any other historical starting model. Its final cell checks the files, flushes Drive, then releases the GPU.

Judge the pilot by each task's learning score and the change in earlier scores after the next lesson. Do not use this one run to claim a seed-independent improvement or identify one of the changed initialization components as the cause. Stop after task 9 and review the saved table before extending the sequence.

## Validation and code version

Training code is pinned to `b1797e0a93644f501508dfb856d38ad921bb6db8`. The published version retains the original ResNet pooling.

Forty-one automated checks passed across Hyperfan statistics, historical output compatibility, experiments, replay, and notebook helpers. The additional pilot notebook check passed using real CPU training and checkpoint files with a small stand-in target. It simulated a disconnect after saving task 3, resumed tasks 0 and 9, reopened the completed run without training again, and verified that Drive flushing precedes runtime release. Colab and Drive services were mocked; no remote GPU run was performed during validation.
