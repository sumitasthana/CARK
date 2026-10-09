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

## User-reported results for the two selected comparisons

The user supplied notebook-20 output for both order-01, seed-0 comparisons. The review output reports one valid seed for each setting. These results were supplied by the user and have not been independently reread from R2 for this update. The supplied local export path is `/content/cark-paired-study/outputs/paired_study/paired_generalization_v2/summaries/20261009T175649556604Z`; this names an export directory and does not independently establish run completion time or remote export persistence.

Both branches reuse the source sequence L3, L0, L9, L5, L17, L1, L7, L14 and the same L15-only control. Differences below are final validation accuracy in the unlearn-then-learn branch minus the learning-only control, in percentage points. The retained-task mean excludes the forgotten task.

| Continuation | Task-15 difference | Mean retained-task difference | Largest retained-task drop |
| --- | ---: | ---: | --- |
| U3 then L15 | 0.0 | -1.086 | Task 9: 11.0 |
| U14 then L15 | -2.6 | -1.6 | Task 17: 14.2 |

| Retained task | U3 then L15 difference | U14 then L15 difference |
| --- | ---: | ---: |
| 0 | +10.6 | +18.6 |
| 3 | Forgotten, excluded | -2.2 |
| 5 | +4.8 | -2.4 |
| 7 | +2.2 | -3.0 |
| 9 | -11.0 | -7.0 |
| 14 | 0.0 | Forgotten, excluded |
| 17 | -8.0 | -14.2 |
| 1 | -6.2 | -1.0 |

The selected two-comparison pilot now has results, while the original multi-seed study remains incomplete. `Complete setting: False` means the planned seeds 1 and 2 are missing, not that the seed-0 comparison failed. Standard deviation is unavailable because each setting has one seed. Average retained-task differences are small relative to some individual drops; positive changes on other tasks offset the losses. The two comparisons share a control and have different retained-task sets, so they are not independent repetitions of one setting.

The supplied excerpt omits absolute task-15 accuracies, the forgotten task's accuracy after unlearning and after L15, and the forgetting gate. It therefore does not establish successful forgetting or persistence of forgetting through L15. No erasure, recovery, privacy, or across-seed robustness claim follows from these results. Next: inspect those missing fields in the saved CPU review reports and finalize the single-seed pilot report. Additional GPU training is not required to inspect saved evidence. Wiki publication remains on its scheduled days.
