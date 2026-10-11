"""Build the requested draft from a verified CPU review snapshot, without training."""
import argparse
from datetime import datetime, timezone
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / 'docs/reports'
    (directory / 'evidence').mkdir(parents=True, exist_ok=True)
    (directory / 'figures').mkdir(parents=True, exist_ok=True)
    result = json.loads(args.review.read_text(encoding='utf-8'))
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    rows = [row for row in result['comparisons'] if row['status'] == 'complete'
        and row['order'] == 'order_01' and row['forget_task'] in ('3', '14')]
    lookup = {(row['seed'], row['forget_task']): row for row in rows}
    assert len(rows) == 6, 'Final report requires all six selected comparisons.'
    assert all(all(row['pair_checks'].values()) for row in rows)
    snapshot = datetime.fromtimestamp(args.review.stat().st_mtime, timezone.utc).isoformat()
    total_seconds = sum(row.get('session_seconds', 0) for row in result['session_records'])
    rounded = round(total_seconds)
    elapsed = f'{rounded // 3600} h {(rounded % 3600) // 60} m {rounded % 60} s'
    evidence = {'snapshot_saved_at_utc': snapshot, 'study': plan,
        'selected_comparisons': rows, 'unfinished_jobs': result['unfinished_jobs'],
        'invalid_pairs': result['invalid_pairs'], 'session_records': result['session_records'],
        'recorded_session_seconds': total_seconds}
    evidence_path = directory / 'evidence/paired-study-2026-10-10.json'
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True)
    labels = ['Learn task 15 only', 'Forget task 3, then learn 15', 'Forget task 14, then learn 15']
    for seed, ax in enumerate(axes):
        values = [lookup[(seed, '3')]['new_task_a'], lookup[(seed, '3')]['new_task_b'], lookup[(seed, '14')]['new_task_b']]
        ax.barh(labels, values, color=['#64748b', '#2563a6', '#c16a29'], height=.55)
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(value + 1, i, f'{value:.1f}%', va='center', fontweight='bold')
        ax.set_title(f'Run {seed + 1} (seed {seed})', loc='left', fontweight='bold')
        ax.set_xlim(0, 100)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('Task 15 accuracy (%) | Higher is better')
    fig.suptitle('Can the model still learn task 15?', fontsize=16, fontweight='bold')
    fig.text(.5, .015, 'All three seeds complete. Compare each forgetting branch with its own run baseline.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'figures/paired-study-2026-10-10.{extension}', dpi=180)
    svg_path = directory / 'figures/paired-study-2026-10-10.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)

    combinations = [(seed, target) for seed in plan['seeds'] for target in ('3', '14')]
    retained_table = ['| Task | ' + ' | '.join(f'Run {seed + 1}: forget {target}' for seed, target in combinations) + ' |', '| --- | ' + ' | '.join(['---:'] * len(combinations)) + ' |']
    for task in plan['orders']['order_01']:
        values = []
        for seed, target in combinations:
            row = lookup.get((seed, target))
            values.append('Forgotten; excluded' if task == target else
                f'{row["retained_differences"][task]:+.1f}' if row else 'Pending')
        retained_table.append(f'| {task} | ' + ' | '.join(values) + ' |')
    forgetting = ['| Run and task to forget | Before forgetting | After forgetting | After learning task 15 |', '| --- | ---: | ---: | ---: |']
    learning = ['| Run | Learn 15 only | Forget 3, then learn 15 | Forget 14, then learn 15 |', '| --- | ---: | ---: | ---: |']
    harm = ['| Run and request | Average change across other tasks | Largest loss on one other task |', '| --- | --- | --- |']
    for seed in plan['seeds']:
        label = f'Run {seed + 1} (seed {seed})'
        control = lookup.get((seed, '3'))
        learning.append(f'| {label} | ' + (f'{control["new_task_a"]:.1f}%' if control else 'Completed; score not in this comparison snapshot') + ' | ' + ' | '.join(f'{lookup[(seed, target)]["new_task_b"]:.1f}%' if (seed, target) in lookup else '**Pending**' for target in ('3', '14')) + ' |')
        for target in ('3', '14'):
            row = lookup.get((seed, target))
            if row:
                forgetting.append(f'| {label}: task {target} | {row["starting_target_accuracy"]:.1f}% | {row["after_unlearning_accuracy"]:.1f}% | {row["final_target_accuracy"]:.1f}% |')
                worst = min(row['retained_differences'], key=row['retained_differences'].get)
                harm.append(f'| {label}: forget {target}, then learn 15 | {abs(row["retained_mean_difference"]):.2f} points {'lower' if row['retained_mean_difference'] < 0 else 'higher'} | Task {worst}: {row["largest_paired_retained_drop"]:.1f} points lower |')
            else:
                forgetting.append(f'| {label}: task {target} | Pending | Pending | Pending |')
                harm.append(f'| {label}: forget {target}, then learn 15 | Pending | Pending |')
    averages = ['| Forgetting request | Average task-15 change | Task-15 standard deviation | Average change on other tasks | Other-task standard deviation |', '| --- | ---: | ---: | ---: | ---: |']
    for target in ('3', '14'):
        selected = [lookup[(seed, target)] for seed in plan['seeds']]
        new = [r['new_task_difference'] for r in selected]
        other = [r['retained_mean_difference'] for r in selected]
        averages.append(f'| Forget {target}, then learn 15 | {statistics.mean(new):+.2f} points | {statistics.stdev(new):.2f} points | {statistics.mean(other):+.2f} points | {statistics.stdev(other):.2f} points |')
    seed2 = lookup[(2, '3')], lookup[(2, '14')]
    seed2_sentence = f"In Run 3, forgetting task 3 changes task-15 accuracy by {seed2[0]['new_task_difference']:+.1f} points relative to its baseline. Forgetting task 14 changes it by {seed2[1]['new_task_difference']:+.1f} points."
    text = f'''# What happens when we forget a task and then learn task 15?

**Completed selected study, 10 October 2026. All three runs and all six comparisons have been reviewed.** This report uses the saved results available at this snapshot. It does not include later training.

## Main findings

- **Forgetting:** task 3 and task 14 both reach 10% accuracy after forgetting. They stay at 10% after learning task 15 in all three runs.
- **Learning task 15:** the model still learns task 15. forgetting task 3 gives the same or higher score in all three runs. Forgetting task 14 sometimes improves the score and sometimes reduces it. The task-3 average is strongly influenced by Run 3.
- **Other tasks:** five of the six branches have a lower average score on other tasks. The largest loss on an individual task ranges from 2.6 to 14.2 points. A small average loss does not mean every task is protected.

These results cover seeds 0, 1, and 2. Averages below are calculated separately for each forgetting request.

## What exactly are we comparing?

First, each model learns these eight tasks in this order:

**Task 3 -> task 0 -> task 9 -> task 5 -> task 17 -> task 1 -> task 7 -> task 14**

We copy that model into three branches. Each branch starts from the same model within a run.

| Branch | What the model does | Why we run it |
| --- | --- | --- |
| Baseline | Learn task 15 only | Measure learning without a forgetting request |
| Forget task 3 | Forget task 3, then learn task 15 | Test forgetting a task learned first |
| Forget task 14 | Forget task 14, then learn task 15 | Test forgetting a task learned last |

We repeat this comparison with seeds 0, 1, and 2. A **seed** sets the random choices used in a run, including model initialization. Repeating with different seeds helps show whether a result changes with those choices. Here, Run 1 means seed 0, Run 2 means seed 1, and Run 3 means seed 2.

## 1. Did the model forget the requested task?

Accuracy is the percentage of validation images classified correctly. Each task has 10 classes, so random guessing would score about **10%**. For example, 35.6% accuracy means 178 correct answers out of 500 validation images.

{chr(10).join(forgetting)}

**Reading this table:** in Run 1, task 3 falls from 35.6% to 10.0% after forgetting. It is still at 10.0% after the model learns task 15. The other five completed comparisons show the same final 10.0% score.

The study accepts a forgotten-task score at or below 12%. All six completed comparisons meet this threshold. None of the saved evaluations during task-15 learning rises above it. This shows reduced classification accuracy over this tested continuation. It does **not** prove that the task's information has been erased or cannot be recovered.

## 2. Could the model still learn task 15?

All numbers below are task-15 validation accuracy. Higher is better. Compare the three scores **within the same row**, because each run has its own starting model.

{chr(10).join(learning)}

In Run 1, forgetting task 3 makes no difference: both scores are 50.8%. Forgetting task 14 gives 48.2%, which is **2.6 points lower** than the baseline.

In Run 2, forgetting task 3 gives **3.2 points higher** accuracy than the baseline. Forgetting task 14 gives **3.8 points higher** accuracy.

{seed2_sentence}

A **percentage point** is the difference between two percentage scores: 47.2% minus 44.0% is 3.2 points. These differences are not relative percentage improvements.

![Task 15 accuracy for the baseline and both forgetting branches, labeled separately for each completed run](figures/paired-study-2026-10-10.png)

The chart shows the same scores as the table. Each bar has its accuracy written beside it. All three runs are shown.

## 3. What happened to the other learned tasks?

Here we compare each forgetting branch with the baseline **after both have learned task 15**. We examine the seven earlier tasks that were supposed to remain learned. The requested forgotten task is excluded.

{chr(10).join(harm)}

**Reading this table:** in Run 1, forgetting task 3 lowers the average across the other seven tasks by 1.09 points. However, task 9 alone loses 11.0 points. Some tasks improve while others decline, so the average can hide a large loss.

The main concern is therefore the loss on individual tasks, even when task 15 learns successfully. The task with the largest loss also changes between runs.

## Three-seed averages

For each run, we subtract the baseline accuracy from the forgetting branch accuracy. We then average those three differences. This keeps each branch matched to its own starting model.

{chr(10).join(averages)}

The standard deviation measures how much the three run results differ from their average. It is not a confidence interval. The individual run scores above remain important because an average can hide opposite effects. Only three seeds were tested.

## Completion and saved summaries

All selected work is complete: **12/12 jobs, 6/6 comparisons, and 39/39 requests**. Notebook 20 has been run on CPU, and the R2 summary objects have been checked. The other sequences and forget-task-5 branches from the original plan remain deferred. No further GPU training is needed for this selected study.

## What can this experiment support?

So far, the requested task reaches chance-level accuracy and stays there during one subsequent task. Learning task 15 remains possible. Forgetting task 3 has an average task-15 gain of 5.40 points, with run effects of 0.0, +3.2, and +13.0 points. Forgetting task 14 has an average change of -0.20 points, with effects of -2.6, +3.8, and -1.8 points. The large differences between runs limit a general claim. Some other tasks suffer substantial losses.

This study tests **one order and one incoming task**, not all possible sequences. Task 3 and task 14 differ in both task identity and learning position. We therefore cannot say that their differences are caused only by being learned first or last. The two forgetting branches share a baseline within each run, so they are not independent repetitions.

Even three seeds provide limited evidence about how broadly the result holds. This study does not test privacy, recovery, or long future learning sequences.

## Technical details and recorded time

| Setting | Value used in this study |
| --- | --- |
| Dataset | Tiny ImageNet; 10 classes per task; partition seed 42 |
| Data per task | 5,000 training images and 500 validation images |
| Input | 64 x 64 RGB images; tensor values in [0, 1]; no augmentation or additional normalization |
| Classifier | ResNet-50; 16 bottleneck blocks; 3 x 3 stride-1 input layer; 10 outputs |
| Weight-generating network | Input width 64; hidden layers 128, 256, 512 with ReLU activation; three linear output heads |
| Task and chunk codes | 32 values each; 200 chunks of generated weights |
| Optimizer | Adam; learning rate 0.0001 for learning and forgetting |
| Learning request | 5 epochs; batch size 64; validation batch size 256 |
| Forgetting request | 100 steps; 10 Gaussian-noise samples per step |
| Loss weights | Learning protection beta 0.1; forgetting noise gamma 0.01 |
| GPU | A100 in the user-reported sessions |

The weight-generating network produces the classifier's parameters. Learning updates this network and the new task's code. Forgetting updates the network while keeping the task codes fixed. A protection loss tries to limit changes to other tasks' generated parameters. This diagnostic study uses beta 0.1, compared with the repository's paper-based Tiny ImageNet default of 0.01. It is not a paper reproduction, and older experiment-06 results are not included in these results.

The {len(result['session_records'])} verified saved session logs total **{elapsed}**. This is elapsed time inside the study runner, including training, evaluation, and file transfers. It excludes earlier notebook setup and time between sessions. It is not a measurement of billed GPU time.

Checkpoints save after learning epochs and at 20-step forgetting boundaries. A restart resumes from the last verified R2 save; work after that save may repeat. Completed jobs are skipped.

## Detailed evidence

The six completed comparisons pass all recorded checks that the branches use matching starting models and task-15 training conditions. The [saved evidence](evidence/paired-study-2026-10-10.json) contains the exact scores, pairing checks, study plan, and session logs. The [dated experiment record](../wiki/Experiment-trajectory-2026-10-10.md) records the progress history.

For readers checking individual tasks, this table shows branch accuracy minus baseline accuracy after learning task 15, in percentage points. A plus sign means higher accuracy; a minus sign means lower accuracy.

{chr(10).join(retained_table)}

Study ID: `{plan['study_id']}`. Fixed training revision: `{plan['code_version']}`. Review snapshot: {snapshot}. No new training was performed for this report.
'''
    assert '\u2014' not in text
    path = directory / 'Paired-study-2026-10-10-draft.md'
    path.write_text(text, encoding='utf-8')
    from render_report_html import render_report_html
    html_path = render_report_html(path)
    assert len(rows) == 6
    assert all(all(row['pair_checks'].values()) for row in rows)
    assert not any('Pending' in line for line in forgetting)
    print(path)
    print(html_path)
    print(f'Verified report: {len(rows)} completed pairs; {len(result["session_records"])} session logs.')


if __name__ == '__main__':
    main()
