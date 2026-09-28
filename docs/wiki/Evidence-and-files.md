# Evidence and files

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

- [registry.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/registry.json)
- [manifest.json](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/manifest.json)
- [forgetting_traces.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/forgetting_traces.csv)
- [gradient_norms.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/gradient_norms.csv)
- [raw_output_norms.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/raw_output_norms.csv)
- [e13_sampled_losses.csv](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/e13_sampled_losses.csv)
- [e15_reported_output.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/e15_reported_output.txt)
- [notebook_saved_outputs.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/notebook_saved_outputs.txt)
- [full_sequence_20260921.md](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/full_sequence_20260921.md)
- [fisher_sanity_output.txt](https://github.com/sumitasthana/CARK/blob/main/docs/experiments/sources/fisher_sanity_output.txt)

The main registry includes every archived experiment. The older manifest covers
checkpoint diagnostics E08-E15 and records their artifact paths. Together the
trace files contain 208 accuracy observations, 16 supplied E15 gradient rows,
and ten supplied E15 raw-output norms. Only steps 1, 2, 5, and 10 have supplied
gradient measurements. Missing losses are blank, not inferred from norms.

`e15_reported_output.txt` is a transcription of the user's pasted table, not the
original runtime JSON. The runtime reported completion and a valid starting
point. GPU identity, runtime, and loss components were not supplied for E15.
The E15 date comes from its report filename; the original file was not opened.

Some earlier settings are only intended settings from instructions. Saved
notebook outputs corroborate E13/E14 settings, but the notebook mixes several
runs. Its setup cell is not proof of every run's device or source commit.

The full-sequence summary and Fisher output come from local historical notes.
They are archived as reported evidence. The Fisher toy check does not imply
that a forgotten classifier is confidently wrong or that Fisher locates storage.

Drive paths are reported artifact locations, not public download links or a
guarantee of current availability. No checkpoint weights are published here.
