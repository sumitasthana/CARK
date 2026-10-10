# Experiment trajectory: 10 October 2026

## Seed 1 comparisons complete

The user supplied the notebook-19 progress panel and session record `20261010T023852185516Z`. Both order-01, seed-1 branches, U3/L15 and U14/L15, completed. All four selected seed-1 jobs and all 13 of its requests are complete. The panel reports 8 of 12 jobs, 4 of 6 comparisons, and 26 of 39 requests complete across the narrowed study. Only seed 2 remains: its eight-lesson source, L15-only control, U3/L15 branch, and U14/L15 branch. This is four jobs, two comparisons, and 13 requests. The other orders and task-5 branches remain deferred in the original study plan.

A read-only R2 check at 12:21 a.m. America/New_York on 10 October independently verified the new session log. It records both seed-1 branches complete and a duration of 6,073.955492 seconds, approximately 1 h 41 m 14 s. The session began on 9 October at 22:38:52 Eastern daylight time and continued past midnight. The user-supplied panel gives the latest verified boundary as `2026-10-10T04:19:57.282775+00:00`. No seed-1 accuracy results were included in this update.

Seven verified R2 session logs now total 33,067.539814 seconds, rounded to 9 h 11 m 8 s. This includes the interrupted upload session and excludes earlier notebook setup, gaps, and sessions without an uploaded final log. It is runner wall time, not a complete measure of billed GPU time. The latest local audit is `outputs/r2-review/session-times-2026-10-09/sessions.json`; its internal check timestamp identifies this refreshed snapshot.

The requested Excel tracker was updated as `outputs/experiment-tracker/Paired_Study_Experiment_Tracker_2026-10-10.xlsx`. It adds the seventh session, marks both seed-1 comparisons complete, updates the grand total and remaining work, and leaves seed-1 accuracy fields blank because no paired effects were supplied. Workbook checks confirmed seven session rows, six comparison rows, correct completion labels, and the duration total. The dated 9 October workbook is preserved.

Next: change only SEED to 2 in notebook 19 and run all on A100. Keep SEED 2 on subsequent sessions until its four jobs complete. Use notebook 20 on CPU to inspect and save multi-seed effects when available. No new GPU training was started by this record or workbook update. Wiki publication remains on its scheduled days.
