# Experiment trajectory: 10 October 2026

## Seed 1 comparisons complete

The user supplied the notebook-19 progress panel and session record `20261010T023852185516Z`. Both order-01, seed-1 branches, U3/L15 and U14/L15, completed. All four selected seed-1 jobs and all 13 of its requests are complete. The panel reports 8 of 12 jobs, 4 of 6 comparisons, and 26 of 39 requests complete across the narrowed study. Only seed 2 remains: its eight-lesson source, L15-only control, U3/L15 branch, and U14/L15 branch. This is four jobs, two comparisons, and 13 requests. The other orders and task-5 branches remain deferred in the original study plan.

A read-only R2 check at 12:21 a.m. America/New_York on 10 October independently verified the new session log. It records both seed-1 branches complete and a duration of 6,073.955492 seconds, approximately 1 h 41 m 14 s. The session began on 9 October at 22:38:52 Eastern daylight time and continued past midnight. The user-supplied panel gives the latest verified boundary as `2026-10-10T04:19:57.282775+00:00`. No seed-1 accuracy results were included in this update.

Seven verified R2 session logs now total 33,067.539814 seconds, rounded to 9 h 11 m 8 s. This includes the interrupted upload session and excludes earlier notebook setup, gaps, and sessions without an uploaded final log. It is runner wall time, not a complete measure of billed GPU time. The latest local audit is `outputs/r2-review/session-times-2026-10-09/sessions.json`; its internal check timestamp identifies this refreshed snapshot.

The requested Excel tracker was updated as `outputs/experiment-tracker/Paired_Study_Experiment_Tracker_2026-10-10.xlsx`. It adds the seventh session, marks both seed-1 comparisons complete, updates the grand total and remaining work, and leaves seed-1 accuracy fields blank because no paired effects were supplied. Workbook checks confirmed seven session rows, six comparison rows, correct completion labels, and the duration total. The dated 9 October workbook is preserved.

Next: change only SEED to 2 in notebook 19 and run all on A100. Keep SEED 2 on subsequent sessions until its four jobs complete. Use notebook 20 on CPU to inspect and save multi-seed effects when available. No new GPU training was started by this record or workbook update. Wiki publication remains on its scheduled days.

## Seed 2 source saved and tracker refreshed

The user supplied session `20261010T042423429787Z` and a progress panel showing the seed-2 source resumable after four completed source lessons, with Learn 17 active. The panel reports 30 of 39 requests complete across the narrowed study; jobs remain at 8 of 12 and comparisons at 4 of 6. Nine requests are unfinished. The latest boundary reported by the panel is `2026-10-10T06:12:02.796895+00:00`. Its active epoch and step were not supplied. Keep SEED 2 and run all on A100 to resume.

A read-only R2 session audit at 1:09 p.m. Eastern on 10 October independently verified the eighth log, recording the seed-2 source as resumable with elapsed time 6,466.819694 seconds (1 h 47 m 47 s). All eight logs total 39,534.359508 seconds, rounded to 10 h 58 m 54 s. The detailed request count above remains user-reported; the session audit did not reread the active progress pointer. The October-10 Excel tracker now includes all eight sessions, the correct seed-2 label, the updated duration total, and the user's 30/39 progress snapshot. Workbook checks passed. No new accuracy result was supplied.

## Full planned-job tracker

The user requested that the Excel tracker include all planned work so remaining jobs are visible. A read-only R2 audit at 1:14 p.m. Eastern on 10 October checked all 45 progress pointers and validated their referenced report/checkpoint metadata. Eight jobs are complete, one is resumable, and 36 are pending. The seed-2 source has six completed learning requests and an active Learn 7 request saved at epoch 0, step 0. The narrowed study therefore has 32 of 39 requests complete, seven unfinished requests, and four unfinished jobs. Its two seed-2 comparisons remain unfinished.

The Excel tracker now includes a Planned jobs sheet with all 45 original jobs. Each row shows selected versus deferred scope, order, seed, source sequence, planned requests, purpose, saved status, planned/completed/remaining request counts, active saved request, timestamp, and job ID. Twelve jobs belong to the narrowed study and 33 are deferred. Overview separates these totals. Start dates and durations are not invented for pending work; session durations remain in the Sessions sheet. Checks confirmed 45 job rows, 12 selected jobs, 39 selected requests, and 135 requests across the original plan. No new accuracy effect was assessed.

## Seed 2 source and control complete

The user supplied session `20261010T163836495156Z`, with status saved and elapsed time 6,584.078991 seconds (1 h 49 m 44 s). The seed-2 source and L15 control completed. U3/L15 returned resumable with Unlearn 3 active; the exact saved step was not supplied. U14/L15 remains pending. The panel reports 10 of 12 jobs, 4 of 6 comparisons, and 35 of 39 requests complete. Its latest verified boundary is `2026-10-10T18:28:12.115066+00:00`. This update records user-supplied output; remote state was not independently reread here.

Next: reconnect to A100, keep SEED 2 and the same study ID/settings, and run all. The completed source and control are skipped, U3/L15 resumes, and U14/L15 follows if the session budget permits. Two branch jobs and four requests remain unfinished; U3 already has partial progress. Once the panel shows 12/12 jobs and 6/6 comparisons complete, stop GPU execution and run notebook 20 on CPU. Its verified R2 exports provide the three-seed means, variation, retained-task changes, and forgotten-task gate/relapse checks. Do not start deferred orders or targets for this narrowed study.

## Requested draft report and verified seed-1 effects

The user explicitly requested a report using available results and placeholders for unfinished comparisons. A fresh CPU review read verified R2 reports and checkpoint metadata without downloading models. Four selected comparisons pass all recorded pairing checks; seed-2 U3/L15 remains resumable and U14/L15 pending. In all four completed comparisons, the target accuracy falls from above chance to 10.0% after unlearning and remains 10.0% after L15. The fixed 12% forgetting gate is met, with no recorded relapse above that gate during the evaluated L15 epochs. These results establish observed classification suppression over one new lesson; they do not establish erasure or recovery resistance.

| Seed | Branch | L15 control | L15 branch | L15 difference | Mean retained difference | Largest final retained drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | U3/L15 | 50.8 | 50.8 | 0.0 | -1.086 | 11.0 |
| 0 | U14/L15 | 50.8 | 48.2 | -2.6 | -1.600 | 14.2 |
| 1 | U3/L15 | 44.0 | 47.2 | +3.2 | -1.571 | 11.6 |
| 1 | U14/L15 | 44.0 | 47.8 | +3.8 | -1.743 | 14.0 |

Accuracy columns are percentages; differences and drops are percentage points. Provisional task-15 means and sample SD across seeds 0 and 1 are +1.60 ± 2.26 points for U3/L15 and +0.60 ± 4.53 for U14/L15. These are partial summaries; final three-seed means remain pending. The CPU review independently verifies the previously user-supplied seed-0 values and fills the missing seed-1 effects and forgotten-task scores. Nine recorded sessions total 46,118.438499 seconds, rounded to 12 h 48 m 38 s.

The requested draft is [Paired study draft](../reports/Paired-study-2026-10-10-draft.md), with a committed evidence snapshot, PNG/SVG figure, and rebuild script. It separates actual diagnostic settings, validated results, provisional averages, limitations, and placeholders for both seed-2 branches and final runtime/conclusions. Figure and numeric checks passed. No GPU training was performed to create the draft.

The user requested an HTML copy for easier reading. The [HTML draft](../reports/Paired-study-2026-10-10-draft.html) contains the same report with readable tables, section links, and an embedded figure. It can be opened as a standalone local file without a server or an image download. Browser printing can save a PDF. The report builder now produces both Markdown and HTML so future result updates keep them aligned. HTML checks confirmed six tables, the embedded figure, portable evidence links, and the seed-2 placeholders. This is a local and main-repository artifact, not a website deployment.

## Report readability revision

The user found the numbers, language, and chart confusing. The Markdown and HTML drafts now separate three questions: forgetting the requested task, learning task 15, and changes to other tasks. Each result table defines its measure and gives a reading example. A directly labeled bar chart shows absolute task-15 accuracy separately for each completed seed. The main report reserves final averages for the completed three-seed study and marks the two unfinished comparisons as pending. Technical settings and individual-task differences appear after the findings. This revision changes presentation only; it uses the same verified four-comparison snapshot and introduces no new training results.
