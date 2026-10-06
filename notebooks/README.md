# Notebook guide

Start here when sharing the repository or opening it in Colab. Notebook numbers show when a notebook was added, not a sequence to run from 01 to 14. Choose the notebook for the job. Historical notebooks stay at their original paths because experiment records and earlier links cite them.

Open the main workflows directly in Colab: [04: gradient diagnostic](https://colab.research.google.com/github/sumitasthana/CARK/blob/main/notebooks/04_gradient_diagnostics.ipynb), [07: paired L9 probe](https://colab.research.google.com/github/sumitasthana/CARK/blob/main/notebooks/07_paired_L9_probe.ipynb), [08: paired report](https://colab.research.google.com/github/sumitasthana/CARK/blob/main/notebooks/08_paired_L9_reporting.ipynb), [09: experiment overview](https://colab.research.google.com/github/sumitasthana/CARK/blob/main/notebooks/09_experiment_overview.ipynb), or [10: evidence export](https://colab.research.google.com/github/sumitasthana/CARK/blob/main/notebooks/10_export_experiment_evidence.ipynb).

## Current workflows

| Task | Notebook | Runtime | What it reads and saves |
| --- | --- | --- | --- |
| Inspect Tiny ImageNet files and labels | [01: dataset walkthrough](01_tiny_imagenet.ipynb) | CPU | Reads the dataset; no experiment result. |
| Inspect one task and its batches | [02: task walkthrough](02_task1.ipynb) | CPU | Reads the dataset and fixed class partition; no experiment result. |
| Continue a saved model and measure protection | [14: learning-loss diagnostic](14_learning_loss_diagnostic.ipynb) | CPU selection, then one A100 session | Selects a checkpoint after learning 9 or 5; teaches only the next task; saves separate losses, sampled gradients, and scores after each epoch. Optional beta trials each restart from the same source. |
| Test revised initialization on a fresh model | [13: Hyperfan learning pilot](13_hyperfan_learning_pilot.ipynb) | One A100 session | Starts fresh; learns 3, 0, and 9 with Hyperfan-in, beta 0.01, and learning rate 0.0001; saves scores and a checkpoint after each task. |
| Test whether learning damages older tasks | [12: learning-only retention](12_learning_only_retention.ipynb) | One A100 session | Starts before U3; learns 9, 5, and 17 without forgetting; saves an accuracy table and can continue after completed tasks. |
| Diagnose L9 differences across fresh sessions | [11: L9 session replay](11_L9_session_replay.ipynb) | Two fresh GPU sessions, then CPU | Freezes A's checkpoint, code, settings, and data for B; saves per-update traces and the first recorded mismatch. |
| Measure one forgetting run from a saved checkpoint | [04: gradient diagnostic](04_gradient_diagnostics.ipynb) | GPU | Edit the configuration cell. Saves a per-step JSON report, checks it, then flushes Drive and releases the GPU. Its default is the E08 checkpoint and task 3. |
| Compare L9 with and without U3, then inspect task 0 weights | [07: paired L9 probe](07_paired_L9_probe.ipynb) | GPU for a new pair; CPU for saved-pair diagnostics | Set `RUN_TRAINING` and `PAIR_ID` in `SETUP-02`. Saves both arms, a comparison, and component changes. This is a specific control experiment, not a general sequence runner. |
| Check saved paired L9 runs in detail | [08: paired L9 report](08_paired_L9_reporting.ipynb) | CPU | Reads each arm's saved JSON and component diagnostic. Writes a JSON and Markdown report. |
| See the full experiment history in charts | [09: experiment overview](09_experiment_overview.ipynb) | CPU | Reads the tracked archive and available Drive JSON. Plots early runs, forgetting traces, every task in the full sequence, and L9 pairs. Saves PNG charts. |
| Share the evidence and its original files | [10: evidence export](10_export_experiment_evidence.ipynb) | CPU | Creates a ZIP with tracked files, saved Drive records, request timelines, and hashes. Checkpoints are optional. |

For a new full request sequence, use the validated [`run_experiment`](../uncle/experiments.py) entry point with a new output directory and saved configuration. [05](05_full_sequence_candidate.ipynb) shows the exact completed 27-update candidate; it is not a blank template. A proposed study run with reserved images and multiple seeds still needs its own validated setup. Do not treat these historical all-images checkpoints as study checkpoints.

## Historical notebooks

These remain readable for provenance. Do not run them as instructions for a new experiment.

| Notebook | Why it remains |
| --- | --- |
| [03: forgetting diagnostics](03_forgetting_diagnostics.ipynb) | Contains saved early output, but its E-run labels, settings, and instructions no longer agree. Use 04 for new diagnostics. |
| [05: full-sequence candidate](05_full_sequence_candidate.ipynb) | Records the completed E08-based, 27-update run and its checkpoint policy. Its output path belongs to that run. |
| [06: full-sequence review](06_review_full_sequence_candidate.ipynb) | Reviews only the completed candidate. Use 09 for the current cross-run and all-task view. |
| [scalable_hypernetworks_colab.ipynb](scalable_hypernetworks_colab.ipynb) | Older self-contained Permuted MNIST prototype with saved exploratory output. It predates the current `uncle` package. |

## Use and share results

GPU experiment notebooks save their results to Drive before releasing the runtime. CPU notebooks read those files in a fresh session; do not expect Python variables to survive a Colab disconnect. The archive under [`docs/experiments`](../docs/experiments/README.md) contains earlier reported or transcribed results. Original Drive files are preferable when available.

Share the ZIP from notebook 10 when another person needs the evidence. It includes the tracked code version and file hashes, but it cannot recreate unsaved Colab output. Start with notebook 09 when the person only needs an explained visual overview. Every chart should be read with its source and limits, especially when comparing one seed or one paired run.
