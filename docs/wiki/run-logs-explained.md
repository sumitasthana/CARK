# Run logs explained

Updated 2026-09-29. This section records completed experiments and numerical
checks. It separates reported measurements, interpretation, and proposed next work.

**Current result:** the full-sequence reproduction failed retention, and none of
the recorded checkpoint diagnostics passed the joint forgetting/retention screen.
The latest 30-step run reached 12.6% target accuracy at step 27 while retained
accuracy had drifted 3.6 points. The target threshold is 12%.

- [Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index): all 21 archived records.
- [Protocol and decision rules](https://github.com/sumitasthana/CARK/wiki/Protocol): settings, thresholds, and comparability.
- [Latest 30-step result](https://github.com/sumitasthana/CARK/wiki/Experiment-forgetting-30step-20260929): accuracy and gradient measurements.
- [Evidence and downloadable files](https://github.com/sumitasthana/CARK/wiki/Evidence-and-files): provenance and source data.
- [Software validation](https://github.com/sumitasthana/CARK/wiki/Software-validation): checks kept separate from research runs.
- [Record template](https://github.com/sumitasthana/CARK/wiki/Record-template): how to log the next experiment.
- [Anchor paper](https://github.com/sumitasthana/CARK/wiki/An-Unlearning-Framework-for-Continual-Learning): the reproduction reference.

Original Drive artifacts are not mirrored here. Measurements transcribed from
tables retain their reported precision; missing values remain missing. The
archive includes the full-sequence run, early learning and forgetting probes,
checkpoint preparation, E08-E15, the latest 30-step run, and two numerical checks. It does not present
planned Fisher attribution or relearning experiments as completed work.

The canonical structured record is [registry.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/registry.json).
Pages are generated with `python scripts/build_experiment_wiki.py`.

The [RESEARCH_DIAGNOSTIC.md](https://github.com/sumitasthana/CARK/blob/main/docs/RESEARCH_DIAGNOSTIC.md) keeps current forgetting screens
separate from future matched recovery and component tests.
