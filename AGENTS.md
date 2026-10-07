# Experiment records and publication

Keep all experiment narratives in `docs/wiki/Experiment-trajectory.md`. Record results as the user shares them, including failed and interrupted runs, intended and actual settings, starting checkpoints, missing evidence, interpretations, and the next decision. Commit these working records to the main repository. Do not create separate experiment reports or setup-help wiki pages.

Publish accumulated updates to the GitHub wiki's Experiment trajectory page on Wednesdays and Saturdays at 6 p.m. America/New_York. Other-day publication requires an explicit user request. The local Windows task `CARK-Experiment-Trajectory-Publish` runs `scripts/publish_experiment_trajectory.py --publish`; it publishes the committed main-repository record and its figures. It does not read private chat history or discover new experiment results. Keep the working record current during the conversation.

Use the user's configured Git identity. Add no assistant, tool, or coauthor attribution. Preserve unrelated working changes. Write plain English for a non-native freshman reader. Do not invent missing results, metadata, citations, or conclusions. Do not use U+2014 in newly written prose.
