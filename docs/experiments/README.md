# Reported experiment observations

The [Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained) section presents this archive
as an index, a shared protocol, and one page per completed record. The canonical
structured source is `registry.json`; generated Markdown is in `docs/wiki/`.

The wiki's general FAQ is maintained in `docs/CONCEPTS.md`. The existing
anchor-paper link is preserved in `docs/ANCHOR_PAPER.md`. The generator publishes
both alongside the archive and maintains the home page and section navigation.

Read [the experiment log](../EXPERIMENT_LOG.md) for the complete narrative,
earlier learning runs, decisions, limitations, and code findings.

| File | Contents |
| --- | --- |
| [registry.json](registry.json) | All 20 located experiment records: the full sequence, initial probe, direct baseline, E02-E15, checkpoint preparation, and two numerical checks. |
| [forgetting_traces.csv](forgetting_traces.csv) | All 208 supplied accuracy observations for E08-E15, including each starting row. E08/E09 include supplied losses; E14 includes all 50 loss pairs from saved notebook output. |
| [gradient_norms.csv](gradient_norms.csv) | The 16 supplied E15 gradient observations: four parameter groups at steps 1, 2, 5, and 10. Noise gradients include gamma. |
| [raw_output_norms.csv](raw_output_norms.csv) | Ten E15 raw-output norms, measured before updates and retained at printed precision. |
| [e15_reported_output.txt](e15_reported_output.txt) | Transcription of the user's E15 table, settings, source commit, status, and report path. |
| [e13_sampled_losses.csv](e13_sampled_losses.csv) | Six separately supplied E13 loss observations. |
| [notebook_saved_outputs.txt](notebook_saved_outputs.txt) | Text outputs preserved from notebook commit `4407978`, including E14 loss trace, environment setup and E13 sampled losses. |
| [manifest.json](manifest.json) | Intended settings, evidence limits, reported artifact locations, and derived screening summaries. |
| [sources/full_sequence_20260921.md](sources/full_sequence_20260921.md) | Reported measurements from the earlier local full-sequence run note. |
| [sources/fisher_sanity_output.txt](sources/fisher_sanity_output.txt) | Saved numerical output from the local Fisher check; not a hypernetwork experiment. |

These files transcribe the conversation tables and the saved notebook outputs.
The latter corroborate E13/E14 settings and add E14 loss components. The
notebook contains outputs from multiple runs; do not assign every cell to E14. Original Colab
JSON files and model checkpoints are not included or verified here. Earlier
runtime-local files may have been lost when sessions ended. Drive paths record
reported locations, not a guarantee that those files currently exist.

Accuracy is in percent. Drift is an absolute change in percentage points from
the initial task 0 accuracy of 44.6%. Loss values are measured before an update;
accuracies are measured after it. Step 0 is the pre-forgetting model. Blank CSV
cells mean the measurement was not supplied, not zero. Values retain the
precision of the supplied tables. Totals are not reconstructed from rounded
components; separately rounded terms can differ from reported totals by 0.01.

`passes_screen` is derived from both initial accuracies being at least 25%,
target accuracy at most 12%, retained absolute drift strictly below five points,
and step greater than zero. All eight traces start at 26.0% and 44.6%; none
passes the joint screen. This is a diagnostic screen, not proof of erasure.

The manifest distinguishes intended settings from independently verified
metadata. It does not invent missing commit hashes, device details, timings,
losses or seeds. Compare original runtime JSON before making stronger claims.
The existing local HTML mentoring guide remains under ignored `ops-docs/`.

E15 reports commit `e3087f0` and uses the independent forgetting-noise stream
introduced in `4c08f54`. Its GPU and runtime were not supplied. Do not treat it
as an exact numerical continuation or replay of E14.

## Maintaining the archive

Keep intended settings, observed measurements, interpretations, and next steps
separate. Unknown metadata is null in JSON and blank in CSV. Existing E-numbers
are preserved; descriptive IDs for older records are archive labels. Planned
experiments do not appear as completed runs.

After updating the registry and observations, generate and verify the wiki:

```bash
python scripts/build_experiment_wiki.py
python scripts/build_experiment_wiki.py --check
```

Commit the source data and generated `docs/wiki/*.md` pages. Publish those pages
to the wiki's separate Git repository. The generated record template describes
the fields required for the next run. Original runtime JSON remains preferable
to a transcript when it is available.
