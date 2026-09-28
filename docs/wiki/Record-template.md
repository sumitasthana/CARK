# Adding an experiment

[Run logs explained](https://github.com/sumitasthana/CARK/wiki/run-logs-explained)

Create the record when a run starts and fill its observed results when it ends.
Preserve failed and interrupted runs. Distinguish intended settings from the
settings printed or stored by the actual run. Use null for unavailable metadata.

1. Add an entry to `docs/experiments/registry.json` with a stable identifier,
   question, category, date evidence, commit, hardware, settings, initial state,
   observations, decision, interpretation, limitations, evidence, and artifacts.
2. Preserve the original JSON report when available. If only a pasted table is
   available, archive a labelled transcription and retain its printed precision.
3. Add observations to the relevant CSV. Record whether measurements are before
   or after an update. Do not reconstruct missing gradients or losses.
4. For a checkpoint diagnostic, update `manifest.json` and derive the screen
   from the original thresholds. Do not change thresholds to make a run pass.
5. Run `python scripts/build_experiment_wiki.py`, then
   `python scripts/build_experiment_wiki.py --check`. Update the narrative log
   with the result and next question. Commit source data and generated pages.
6. Copy the generated `docs/wiki/*.md` files into the wiki checkout and push its
   default branch. Preserve any existing pages outside the generated set.

## Entry outline

- Question and the one change being tested.
- Starting checkpoint and initial measured accuracies.
- Actual configuration, code commit, GPU, seed, data split, and timing.
- Results, including failures and the full measured trajectory.
- Decision under the predeclared criterion.
- Interpretation, alternative explanations, and missing evidence.
- Next experiment or stop decision.
- Source reports and their availability.

Code tests belong on the software-validation page. Proposed experiments belong
in the plan until they have an actual run record.
