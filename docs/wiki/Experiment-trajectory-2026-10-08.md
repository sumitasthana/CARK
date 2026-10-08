# Experiment trajectory: 8 October 2026

[Back to the experiment index](Experiment-trajectory)

## Recovery study preparation

The user requested explained notebooks for the reserved-image setup and the next model-preparation step. The preparation notes below precede the reported shared-stage completion at the end of this page. No recovery result is available. The earlier unlearning findings remain those on the 7 October page.

Notebook [16](https://github.com/sumitasthana/CARK/blob/main/notebooks/16_recovery_data_setup.ipynb) creates CPU-only task-3 lists: 400 initial-training images and 100 reserved images per class, with nested recovery budgets of 1, 5, and 20 images per class. Validation stays separate. The notebook exports the lists and their fingerprint, without moving images or loading a model. Existing all-images checkpoints cannot satisfy this reserved-data design retrospectively. Synthetic-file checks verified counts, repeatability, overlap rejection, and refusal to overwrite an incompatible split. No real dataset was downloaded during these checks.

Notebook [17](https://github.com/sumitasthana/CARK/blob/main/notebooks/17_recovery_study_models.ipynb) prepares fresh study models, not recovery adaptation. It defaults to CPU CHECK mode with training disabled. CPU PREPARE saves the protocol and image manifest to R2. Explicit GPU TRAIN runs one seed and one stage per session.

The selected setup uses task 3 as the target and task 2 as the unseen control. Both branches first learn task 0 from one shared checkpoint per seed. This gives identical learned chunk embeddings before branching; they stay frozen thereafter. The unlearned branch follows L0, L3, L9, L5, L17, L1, L7, L14, U3. The never-learned branch follows L0, L9, L5, L17, L1, L7, L14. Task 2 is excluded from both. Moving L0 before L3 is a deliberate diagnostic change, not an exact reproduction of the paper or a continuation of the previous pair.

Task 3 uses the 4,000-image initial-training list. Retained tasks use 5,000 images each. Settings are Hyperfan-in, ResNet50, learning beta 0.1, learning and unlearning LR 0.0001, five epochs per lesson, batch size 64, evaluation batch size 256, unlearning gamma 0.01, ten noise samples, and 100 unlearning updates. U3 uses actual generated target weights, including scales and offsets. The notebook preserves shared, never-learned final, before-unlearning, and after-unlearning checkpoints. It saves epoch reports and request-boundary progress to R2 and releases a successful Colab GPU session only after verification.

The gate keeps the proposal's retained mean above 40% and summed absolute spill below 3 points. It also checks target accuracy before unlearning at least 25% and afterward at most 12%. These are practical entry criteria. The previous pair's 10.8-point summed spill fails this gate; its 1.54-point mean retained loss is a different measure. A failed new gate is saved without an automatic sweep or recovery run.

Four small CPU tests passed for branch isolation, shared-state reuse, frozen-state preservation, request-boundary resume, configuration mismatch refusal, spill arithmetic, and partition mismatch rejection. These tests do not establish study-model accuracy or GPU runtime. The notebook code cells compile. Cloud credentials, dataset download, R2 transfers, and A100 training were not executed in this review.

Next decision: run CPU image preparation and protocol checks first. When training is explicitly resumed, prepare and review seed 0 before committing further GPU sessions. Recovery adaptation and component tests still require a later runner. Main-repository records are updated now; wiki publication stays on its Wednesday/Saturday schedule.

## Manifest persistence after a disconnected session

The user reported that the notebook-16 session had disconnected and asked where its manifest was saved. The original notebook saved local JSON files and exported a ZIP; it did not upload to R2. A read-only bucket check found no object at `uncle/recovery_study/recovery_task3_reserved_v1/manifest.json`. Whether the user downloaded the ZIP or whether the disconnected runtime retains local files is unknown.

Notebook 16 now uploads the manifest and preparation metadata to R2 by default, with content verification, before offering the ZIP export. It uploads no images or checkpoints. Notebook 17 now reads the R2 manifest in CPU CHECK and PREPARE modes as well as TRAIN. LOCAL remains an explicit alternative. A missing remote manifest stops with instructions to run notebook 16's CPU preparation and R2 save cell. The same split seed and class partition regenerate the same selection for the same extracted dataset layout.

Both notebooks' code cells compile. Mocked-storage execution checked notebook 16's JSON uploads, notebook 17's CPU CHECK and PREPARE paths, and the missing-object message. No new manifest was uploaded to the user's bucket by this work. No dataset or model training was performed.

## Seed 0 shared stage completed

The user supplied notebook 17 output reporting `Completed: shared Seed: 0` and task-0 validation accuracy of 51.4%. The notebook reported verification of these R2 objects:

- `uncle/recovery_study/recovery_task3_reserved_v1/seed_0/shared/checkpoint.pt`
- `uncle/recovery_study/recovery_task3_reserved_v1/seed_0/shared/report.json`

This is user-reported output. The remote report and checkpoint have not been independently read for this update. Actual configuration, source fingerprints, epoch measurements, GPU type, runtime, and GPU release are not established by the supplied excerpt. The intended shared stage is fresh L0 with notebook 17's fixed settings, followed by reuse of that checkpoint in both sequences. No task-3 recovery, unlearning gate, or retained-task comparison is available yet.

Next: keep STUDY_ID `recovery_task3_reserved_v1` and SEED 0, select TRAIN with STAGE `never_learned`, and restore the shared checkpoint. Continue L9, L5, L17, L1, L7, L14 without learning tasks 3 or 2. After that sequence completes, prepare the separate unlearned sequence from the same shared checkpoint, not from the never-learned final model.
