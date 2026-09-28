# CARK experiment history

Updated 2026-09-28. This wiki records completed experiments and numerical
checks. It separates reported measurements, interpretation, and proposed next work.

**Current result:** the full-sequence reproduction failed retention, and none of
the checkpoint diagnostics E08-E15 passed the joint forgetting/retention screen.
E15 retained task 0 accuracy but ended at 16.2% on task 3, above the 12% threshold.

- [Experiment index](https://github.com/sumitasthana/CARK/wiki/Experiment-index): all 20 archived records.
- [Protocol and decision rules](https://github.com/sumitasthana/CARK/wiki/Protocol): settings, thresholds, and comparability.
- [Latest result: E15](https://github.com/sumitasthana/CARK/wiki/Experiment-E15): accuracy and gradient measurements.
- [Evidence and downloadable files](https://github.com/sumitasthana/CARK/wiki/Evidence-and-files): provenance and source data.
- [Software validation](https://github.com/sumitasthana/CARK/wiki/Software-validation): checks kept separate from research runs.
- [Record template](https://github.com/sumitasthana/CARK/wiki/Record-template): how to log the next experiment.

Original Drive artifacts are not mirrored here. Measurements transcribed from
tables retain their reported precision; missing values remain missing. The
archive includes the full-sequence run, early learning and forgetting probes,
checkpoint preparation, E08-E15, and two numerical checks. It does not present
planned Fisher attribution or relearning experiments as completed work.

The canonical structured record is [registry.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/registry.json).
Pages are generated with `python scripts/build_experiment_wiki.py`.
