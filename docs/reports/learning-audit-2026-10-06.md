# Learning code audit, 6 October 2026

The protection penalty passed a direct equation and gradient check. The clearest implementation difference is initialization: our code does not implement the paper's Hyperfan method. This is a candidate explanation for poor learning and retention, not a demonstrated cause.

This audit checked training revision `83e1c32773fc59886c8b43a0ed159ccf6424637a`, which notebook 12 selects. The local pooling and determinism edits are not part of that revision. No GPU training was run and no training code was changed.

## What the two completed experiments show

Both runs started from the saved model after learning tasks 3 and 0, then learned tasks 9, 5, and 17 without any unlearning. The user supplied these final test scores:

| Task | Beta 0.01 | Beta 0.1 | Difference in percentage points |
| --- | ---: | ---: | ---: |
| 3 | 22.4% | 18.8% | -3.6 |
| 0 | 29.4% | 35.8% | +6.4 |
| 9 | 29.8% | 29.4% | -0.4 |
| 5 | 44.4% | 29.8% | -14.6 |
| 17 | 41.2% | 34.0% | -7.2 |

Increasing beta helped task 0 in this pair, but did not improve every older task and produced lower scores on newly learned tasks. One run per setting does not establish a reliable beta effect. Learning alone can reduce older task scores, so unlearning is not required for the decline observed here.

## Protection penalty

Equation 2 compares the generated model weights with weights from a frozen copy of the hypernetwork taken before learning the next task. It sums squared differences over model parameters, then averages across previous tasks. Our `UnCLe.preserve()` follows that formula for the supplied protected tasks. It uses the same scaled weights as prediction. Dividing again by the number of model parameters would change this objective. [Paper, Learning](https://arxiv.org/html/2509.17530#S3.SS1.SSS1)

A CPU check used a small two-layer target, two protected tasks, and the pinned hypernetwork code. The checked trainer file was identical to the pinned trainer file. Results with PyTorch `2.12.0+cpu`:

- Penalty before changing the generator: `0.0`.
- Penalty after adding `0.03` to the output-head weights: `0.1256491243839264`.
- Independently calculated squared-distance penalty: `0.1256491243839264`.
- Largest gradient difference between the two calculations: `0.0`.
- The snapshot had no trainable parameters.

This checks the penalty's arithmetic and gradient on a small model. It does not prove that protection is strong enough during ResNet50 training.

## Initialization

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

## Other confirmed difference

The paper states Adam with learning rate `0.001` and a scheduler. Our learning loop has Adam but no scheduler. E08 used the diagnostic learning rate `0.0001`; notebook 12 inherits the starting checkpoint's configuration. The cited appendix does not supply an identifiable scheduler recipe, so an exact schedule should not be invented. [Paper, Implementation](https://arxiv.org/html/2509.17530#S4)

## Next target and steps

**Target: make the initialization match a documented Hyperfan method, and verify the generated starting weights before spending more GPU time.**

1. Prepare the initialization change from the primary Hyperfan reference. Document choices the UnCLe paper leaves unspecified, especially BatchNorm and the final classifier.
2. Check generated weight statistics and a finite forward/backward pass on CPU. Keep the protection equation unchanged.
3. After those checks pass, run a short fresh sequence: learn 3, then 0, then 9. Use beta `0.01`, keep the diagnostic learning rate fixed for this pilot, and record each task's score after each lesson. Judge whether learning and retention warrant a longer experiment.

A fresh start is necessary to test initialization. Loading the old checkpoint overwrites newly initialized generator weights. Also, the output scales are not saved in the state dictionary: changing those scales would change the old checkpoint's generated model even if loading succeeds. Preserve the old code and checkpoint together.

The next pilot would assess the revised implementation. It would not establish generalization across seeds or reproduce the full paper.

## Qualification after reading the Hyperfan reference

The original convolution-weight recipe is already algebraically consistent with the weight-only Hyperfan-in variance formula: a raw head with weight variance `1 / hidden_width`, followed by multiplication by `sqrt(2 / fan_in)`, has the required effective variance when the last hidden activations have unit second moment. The audit's claim that the entire initialization recipe differs from Hyperfan was too broad. The classifier and one-dimensional parameter policies are the substantive differences. The measured spread of one freshly generated tensor is not, by itself, evidence that this variance formula is wrong.

The new option and its choices are documented in [Hyperfan pilot](hyperfan-pilot-2026-10-06.md). The historical measurements above remain measurements of the original recipe.
