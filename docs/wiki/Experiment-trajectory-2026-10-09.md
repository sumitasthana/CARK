# Experiment trajectory: 9 October 2026

## Paired generalization progress in R2

A read-only R2 check at 11:37 a.m. America/New_York found 2 completed jobs, 1 resumable job, and 42 pending jobs in `paired_generalization_v2`. The check read the saved study plan, progress pointers, and completed reports, and validated the referenced artifact sizes and recorded hashes in checkpoint metadata. It downloaded no models and changed no R2 objects. Local evidence is in `outputs/r2-review/progress-2026-10-09/progress.json`.

The completed jobs are `sources/order_01/seed_0`, with all eight source lessons complete, and `controls/order_01/seed_0/learn_15`, with the learning-only task-15 lesson complete. The resumable job is `branches/order_01/seed_0/forget_3/learn_15`. Its saved active request is forgetting task 3 at step 0, with the eight source requests complete. This pointer does not establish any saved unlearning updates. Work in a running session after this boundary is not reflected in this snapshot.

One of nine sources and one of nine controls are complete. None of the 27 unlearn-then-learn branches is complete, so none of the 27 paired comparisons is ready. The 2-of-45 job count is 4.4%, but jobs require different amounts of work, so it is not a GPU-time completion estimate. No accuracy or averaged effect was assessed in this check.

Next: continue notebook 19 with the same study ID, fixed revision, and settings on A100. Keep only one writer active. Notebook 20 can inspect partial reports, but paired averages require completed branches and matching controls. Wiki publication remains on its scheduled days.

## Deadline queue and saved step-60 progress

The user requested a smaller run because the study must be wrapped up today, authorized reuse of existing progress, and confirmed that the notebook was stopped. A fresh read-only R2 check at 11:51 a.m. America/New_York confirmed that the source and control remain complete and that the task-3 branch now has a verified resumable boundary at unlearning step 60. No branch has completed yet. The local progress audit now contains this latest snapshot.

Notebook 19 now selects only the order-01, seed-0 source, its L15 control, U3 followed by L15, and U14 followed by L15. The source and control are reused. U3/L15 runs before U14/L15. All other 41 jobs remain in the saved study but are excluded from execution. The study plan, job contracts, fixed training revision, and checkpoints are unchanged. Selection is applied only to the session scheduler, so the existing step-60 state remains resumable. MAX_JOBS is now two; the 120-minute soft budget and 15-minute save reserve remain. An unfinished branch stops the session and resumes in a later session.

Five mocked notebook checks passed, including exact selected-job execution, source/control reuse, unchanged plan and original scheduler, missing-source refusal, multipart retry, and existing CPU flows. No GPU training was performed by this change. The resulting two comparisons, if completed, are a single-seed pilot. Notebook 20 still reports the full study as partial and does not treat this subset as the completed multi-seed matrix. The next step is to reopen the updated notebook 19 on A100, enable training, and run all cells with the same study ID. Do not run PREPARE again.
