# Experiment trajectory: 9 October 2026

## Paired generalization progress in R2

A read-only R2 check at 11:37 a.m. America/New_York found 2 completed jobs, 1 resumable job, and 42 pending jobs in `paired_generalization_v2`. The check read the saved study plan, progress pointers, and completed reports, and validated the referenced artifact sizes and recorded hashes in checkpoint metadata. It downloaded no models and changed no R2 objects. Local evidence is in `outputs/r2-review/progress-2026-10-09/progress.json`.

The completed jobs are `sources/order_01/seed_0`, with all eight source lessons complete, and `controls/order_01/seed_0/learn_15`, with the learning-only task-15 lesson complete. The resumable job is `branches/order_01/seed_0/forget_3/learn_15`. Its saved active request is forgetting task 3 at step 0, with the eight source requests complete. This pointer does not establish any saved unlearning updates. Work in a running session after this boundary is not reflected in this snapshot.

One of nine sources and one of nine controls are complete. None of the 27 unlearn-then-learn branches is complete, so none of the 27 paired comparisons is ready. The 2-of-45 job count is 4.4%, but jobs require different amounts of work, so it is not a GPU-time completion estimate. No accuracy or averaged effect was assessed in this check.

Next: continue notebook 19 with the same study ID, fixed revision, and settings on A100. Keep only one writer active. Notebook 20 can inspect partial reports, but paired averages require completed branches and matching controls. Wiki publication remains on its scheduled days.
