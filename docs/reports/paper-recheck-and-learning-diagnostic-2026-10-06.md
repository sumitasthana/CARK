# Paper recheck and next learning diagnostic, 6 October 2026

The new-task objective and protected-task implementation agree with the paper's stated learning equation. The review did not find a missing replay buffer, a missing old-task loss on images, or a missing extra coefficient. It did find remaining reproduction gaps and a confirmed difference in the forgetting path. The current learning-only results cannot establish complete agreement with the paper.

## Paper and code comparison

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

## Confirmed CPU checks

The new instrumentation passed a comparison against training without instrumentation. Generated parameters, BatchNorm buffers, and returned epoch losses were identical. The test evaluated validation scores and consumed random numbers inside callbacks, confirming that callback mode and PyTorch RNG isolation work.

The same check verified that old task codes and chunk codes are frozen and unchanged, and that old BatchNorm buffers stay unchanged. Protection was zero before the first update and positive after the generator moved. A zero initial protection gradient is expected for squared distance to an identical snapshot; it is not evidence of a disabled term.

Continuation tests restored actual CPU checkpoint files, trained exactly one additional task, preserved the source file hash and earlier history, and saved a new model. Two different beta branches had the same source hash, starting scores, and first-batch task loss. Completed runs were not retrained. Interrupted runs kept their partial epoch report and required a separate attempt folder.

The notebook test used its actual selection, training, and finishing cells with a small CPU target and mocked Colab services. SELECT mode did not train. RUN mode added exactly one task. File checks preceded Drive flushing and GPU release. These are machinery checks, not new dataset accuracy results.

## What to run next

Notebook [14](../../notebooks/14_learning_loss_diagnostic.ipynb) makes source selection explicit:

1. Use a CPU session in SELECT mode. It lists saved model files on Drive. Select a file and inspect its actual saved tasks and scores.
2. Prefer a source after L3, L0, L9, and L5, so the next lesson is L17. If that checkpoint no longer exists, use a retained source after L3, L0, and L9, then diagnose L5. Do not use the final model after L17 to recreate a model before L17.
3. Save the selection. In a fresh A100 session, use RUN mode with beta 0.01. Source scores must match before training starts. Only the next task is taught.
4. Record cross-entropy, unweighted protection, beta times protection, and combined loss for every update. Save validation scores and batch-mean loss components after every epoch.
5. Sample shared-generator gradient norms and their cosine similarity on the first batch of each epoch. The gradient of beta times protection is compared with the task gradient. The new task code is excluded from this comparison because protection does not act on it.
6. Verify the final history and save the new checkpoint in a separate folder. Keep the source file unchanged. Flush Drive and release the GPU.

Loss and gradient samples describe the model before an update. Epoch accuracies describe it after that epoch. Raw gradient norms are useful diagnostics, but Adam's adaptive scaling means they are not a direct measurement of the resulting parameter movement. Negative gradient cosine indicates conflict on that sampled group; sparse samples do not prove behavior at every update.

The new notebook preserves the source model's initialization, epochs, batch size, seed, and learning rate. It does not silently introduce a scheduler or change the learning objective. For an interrupted lesson, epoch measurements survive, but the current optimizer is not saved mid-lesson; a new attempt restarts that lesson from the original source.

## Search decision

The paper searches beta in `{0.001, 0.01, 0.1, 1}` and selects 0.01 for Tiny ImageNet. [Appendix C, Table 5](https://arxiv.org/html/2509.17530)

There is one relative loss weight in our learning objective: beta. A grid over two independent loss coefficients is not needed for this objective. A beta-by-learning-rate grid is a separate, larger experiment and is not scheduled now.

The immediate plan is one measured beta-0.01 lesson. If its losses and gradients show that protection is active and correct, an optional four-value beta sweep is already supported by changing BETAS in notebook 14. Every value starts from the same selected checkpoint, with the same source RNG state, code, seed, task images, and training settings. Each value has its own result folder. The completed baseline can be reused.

Compare both new-task accuracy and old-task score changes. The comparison report includes the mean old-task change, worst old-task drop, and individual changes; it does not automatically pick a winner by retention alone. A setting that prevents all old-task loss but fails to learn the new task is not sufficient.

This continuation sweep tests beta only for the new lesson. It cannot reproduce the paper's beta search throughout an entire sequence or establish generalization across seeds. If beta does not improve the learning/retention tradeoff, the next controlled question is the documented optimizer/schedule gap, not an unlimited expansion of the search.

## Validation and publication

Training revision: `6a59648a5512724f94917ed9b97a12807c68809b`. The targeted CPU regression set passed 29 tests; one CUDA RNG test was skipped because this machine has no CUDA runtime. Notebook code cells compile, and the written prose was checked for the repository style constraint. No Colab GPU training or inspection of the user's actual Drive checkpoint files was performed locally.
