# Command scripts

`uncle/` contains code that experiments and notebooks import. This folder
contains commands and dataset walkthrough helpers. Keep both folders: they
have different jobs.

| File | Use |
| --- | --- |
| `colab_setup.py` | Prepare a clean checkout in Colab. |
| `research_diagnostic.py` | Assess saved forgetting traces. |
| `build_experiment_wiki.py` | Build or check the tracked experiment archive. |
| `export_experiment_bundle.py` | Package saved evidence for sharing. |
| `explore.py`, `task1.py` | Helpers for notebooks 01 and 02. |
| `run.py`, `baseline.py` | Older command wrappers. Use the installed `uncle-run` and `uncle-baseline` commands for new runs. |

An early gamma probe is retained under
[`reference/historical_scripts`](../reference/historical_scripts/README.md).
The old Colab HTML guide and its checker were removed because the guide is
not part of the repository. Shared notebook code cells are now checked in
`tests/test_tasks.py`.
