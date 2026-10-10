"""Build the requested draft from a verified CPU review snapshot, without training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


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
    pending = {row['job']: row['status'] for row in result['unfinished_jobs']}
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(axes,
        ('new_task_difference', 'retained_mean_difference'),
        ('Task 15: branch minus control', 'Mean retained-task difference')):
        for target, color, marker in [('3', '#245d99', 'o'), ('14', '#b85b22', 's')]:
            points = [row for row in rows if row['forget_task'] == target]
            ax.scatter([row['seed'] for row in points], [row[metric] for row in points],
                color=color, marker=marker, s=65, label=f'U{target} then L15')
        for seed in plan['seeds']:
            if not any(row['seed'] == seed for row in rows):
                ax.axvspan(seed - .25, seed + .25, color='#eeeeee')
                ax.text(seed, .04, 'Pending', transform=ax.get_xaxis_transform(),
                    ha='center', color='#666666')
        ax.axhline(0, color='#666666', linewidth=.8)
        ax.set_xticks(plan['seeds'])
        ax.set_xlim(-.35, max(plan['seeds']) + .35)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Percentage points')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Draft: completed comparisons only; missing results are not zero')
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'figures/paired-study-2026-10-10.{extension}', dpi=180)
    plt.close(fig)

    table = ['| Seed | Branch | Status | Target before U | Target after U | Target after L15 | L15 control | L15 branch | L15 difference | Retained mean difference | Largest final retained drop |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in plan['seeds']:
        for target in ('3', '14'):
            row = lookup.get((seed, target))
            if row:
                fields = ['starting_target_accuracy', 'after_unlearning_accuracy', 'final_target_accuracy',
                    'new_task_a', 'new_task_b', 'new_task_difference', 'retained_mean_difference',
                    'largest_paired_retained_drop']
                table.append(f'| {seed} | U{target} → L15 | Complete | ' +
                    ' | '.join(f'{row[field]:.2f}' for field in fields) + ' |')
            else:
                status = pending.get(f'branches/order_01/seed_{seed}/forget_{target}/learn_15', 'Not available')
                table.append(f'| {seed} | U{target} → L15 | **PENDING: {status}** | ' +
                    ' | '.join(['Pending'] * 8) + ' |')
    averages = ['| Branch | Completed seeds | L15 difference: mean ± sample SD | Retained mean difference: mean ± sample SD | Final three-seed result |',
        '| --- | --- | ---: | ---: | --- |']
    for target in ('3', '14'):
        points = [row for row in rows if row['forget_task'] == target]
        measures = []
        for metric in ('new_task_difference', 'retained_mean_difference'):
            values = [row[metric] for row in points]
            measures.append(f'{statistics.mean(values):+.2f} ± {statistics.stdev(values):.2f}'
                if len(values) > 1 else 'Not enough completed seeds')
        averages.append(f'| U{target} → L15 | ' + ', '.join(str(row['seed']) for row in points) +
            f' | {measures[0]} | {measures[1]} | **Pending seed 2** |')
    retained_table = ['| Task | Seed 0: U3/L15 | Seed 0: U14/L15 | Seed 1: U3/L15 | Seed 1: U14/L15 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for task in plan['orders']['order_01']:
        values = []
        for seed, target in [(0, '3'), (0, '14'), (1, '3'), (1, '14')]:
            row = lookup.get((seed, target))
            values.append('Forgotten; excluded' if task == target else
                f'{row["retained_differences"][task]:+.1f}' if row else 'Pending')
        retained_table.append(f'| {task} | ' + ' | '.join(values) + ' |')
    text = f'''# Paired continual-unlearning study: draft report

Draft dated 10 October 2026. Study: `{plan['study_id']}`. This report covers the selected order-01 study, not the full original matrix. Review snapshot saved at {snapshot}. Results may be added after this cutoff.

## Current finding

Four of six selected comparisons are complete, covering seeds 0 and 1. In all four, the forgotten task falls from above chance to 10% validation accuracy after unlearning and remains at 10% after learning task 15. No evaluated L15 epoch exceeds the fixed 12% tolerance after the unlearning gate is met. This is evidence of suppressed classification performance during one subsequent lesson, not proof of information erasure or resistance to recovery.

Task-15 effects depend on the seed. U3/L15 changes task-15 accuracy by 0.0 points for seed 0 and +3.2 points for seed 1. U14/L15 changes it by -2.6 and +3.8 points. Both branches have negative average retained-task differences, with much larger losses on some individual tasks. Seed-2 outcomes and final three-seed summaries remain pending.

## Question and design

We test whether unlearning a previously learned task changes later learning and harms other tasks. The source sequence is:

**L3 → L0 → L9 → L5 → L17 → L1 → L7 → L14**

For each seed, the same completed source starts three independent continuations:

| Continuation | Requests | Role |
| --- | --- | --- |
| Control | L15 | Learn the new task without unlearning |
| Early-task branch | U3 → L15 | Unlearn task 3, which was learned first, then learn 15 |
| Recent-task branch | U14 → L15 | Unlearn task 14, which was learned last, then learn 15 |

Each branch is compared with its matching seed's L15-only control. The two branches share that control and are correlated. The selected study has 3 seeds, 12 jobs, 6 paired comparisons, and 39 requests. Other orders and U5 branches in the immutable 45-job plan are deferred. A job completion is distinct from a successful forgetting gate.

## Actual settings

| Setting | Selected diagnostic study |
| --- | --- |
| Dataset | Tiny ImageNet; fixed partition seed 42; 10 classes per task |
| Images per task | 5,000 training; 500 validation |
| Input and preprocessing | 64×64 RGB; tensor values in [0, 1]; no augmentation or additional normalization |
| Classifier | ResNet-50 with a 3×3, stride-1 stem; 10 outputs; no pretrained weights |
| Hypernetwork | 64-input shared MLP with hidden widths 128, 256, 512 and ReLU; three linear output heads |
| Codes and chunks | 32-value task code and 32-value chunk code; 200 generation chunks |
| Initialization | Hyperfan-in scaling of generated parameters |
| Optimizer | Adam; learning and unlearning learning rate 0.0001 |
| Learning | 5 epochs per request; batch size 64; validation batch size 256 |
| Unlearning | 100 steps; 10 Gaussian-noise samples per step |
| Loss weights | Learning protection beta 0.1; unlearning noise gamma 0.01 |
| Chance and forgetting gate | Chance 10%; fixed gate at or below 12% |
| GPU | A100 in the user-reported training sessions |
| Fixed training code | `{plan['code_version']}` |

The hypernetwork generates the classifier's weights. Learning updates its shared generator and the new task code; chunk codes are trained only for the first learned task. Unlearning updates the shared generator while keeping codes fixed. Protection compares generated parameters with the snapshot taken before the request. Per-task batch-normalization buffers are retained. The study uses the generated, scaled parameters in both protection and unlearning noise losses.

This is an extension of the earlier experiment-06 question, not a paper reproduction. The selected beta is 0.1; the repository's paper-based Tiny ImageNet default is 0.01. The source is freshly trained with uniform settings and differs from the older mixed-setting checkpoint chain. Historical experiment-06 results are not pooled into these averages. Implementation: [fixed hypernetwork](https://github.com/sumitasthana/CARK/blob/{plan['code_version']}/uncle/hypernet.py), [study training](https://github.com/sumitasthana/CARK/blob/{plan['code_version']}/uncle/study_training.py), and [pair review](https://github.com/sumitasthana/CARK/blob/{plan['code_version']}/uncle/paired_study.py).

## Paired results and placeholders

Absolute accuracy columns are percentages. Difference and drop columns are percentage points. Differences are branch minus control; positive values mean higher branch accuracy. Retained means exclude the forgotten target. Drop magnitudes are positive numbers describing a loss.

{chr(10).join(table)}

All recorded pairing checks pass for the four completed comparisons: matching source and starting model, configuration, task-15 initialization and random streams, frozen state, image values, batch order, and update count. The review labels these pairs complete. This validates the recorded pairing evidence; it does not establish cross-session determinism for every future runtime.

## Provisional averages

These means and sample standard deviations use only seeds 0 and 1. They are descriptive partial summaries, not the final planned three-seed results. Two seeds provide little evidence about the distribution of possible outcomes. No significance test or confidence interval is reported.

{chr(10).join(averages)}

![Completed-seed paired effects; seed 2 pending](figures/paired-study-2026-10-10.png)

Missing results are displayed as pending, not as zero. Do not combine the two forget targets into six independent observations; their controls are shared and their retained-task sets differ.

## Individual retained-task changes

{chr(10).join(retained_table)}

The average losses hide offsetting changes. For example, seed-0 U14/L15 improves task 0 by 18.6 points but reduces task 17 by 14.2 points. Seed-1 U14/L15 reduces task 0 by 14.0 points. The largest final retained loss ranges from 11.0 to 14.2 points across the completed branches. The largest recorded temporary drop during unlearning ranges from 18.2 to 21.6 points, relative to the starting model. Temporary and final drops use different baselines and should not be compared as the same metric.

## Runtime and interruptions

The CPU review contains {len(result['session_records'])} saved session logs totaling {total_seconds:.6f} seconds, rounded to **{elapsed}**. This includes the interrupted R2 upload session. It is runner wall time, covering in-run dataset preparation, training, evaluation, and transfers. Earlier notebook setup, gaps, active sessions without a final log, and disconnected sessions whose log was not uploaded are excluded. It is not a complete measure of billed GPU consumption.

Learning saves at epoch boundaries and unlearning at 20-step boundaries. Checkpoints and reports are content-verified before the progress pointer advances. Interrupted work resumes with optimizer and random state; work after the previous verified save may repeat. Completed branch models are discarded after their verification receipt and report are durable. Source models, unfinished state, and reports remain. CPU review reads reports and checkpoint metadata without downloading models.

## Limits on the conclusion

- Only one learning order is tested in this narrowed study. Task identity and original learning position change together, so these results do not isolate a causal position effect.
- Task 15 is the only incoming task. There is no evidence here for other new-task requests or long future sequences.
- A 10% validation score does not prove removal of information, membership privacy, or inability to relearn. No reserved-image recovery or privacy test is part of this study.
- The reported persistence check covers the saved evaluations during five epochs of one new lesson, not every future model state.
- Positive and negative task-15 effects vary across the first two seeds. A reliable general direction should not be claimed before seed 2 is available.

## Remaining placeholders and finalization

| Item | Current status | Required update |
| --- | --- | --- |
| Seed 2 U3/L15 | Resumable | Target gate, relapse checks, task-15 and retained-task results |
| Seed 2 U14/L15 | Pending | Same result fields |
| Three-seed means and SD | Pending | Average paired effects separately for U3/L15 and U14/L15 |
| Final runtime | Partial snapshot | Add the last session logs |
| Final conclusion | Draft | Reassess direction, variation, forgetting persistence, and retained-task losses |

Keep SEED 2 in notebook 19 until both branches complete, then run notebook 20 on CPU and save verified R2 exports. No extra training on deferred orders is needed to complete this report's selected scope. Update this draft from the final reports; preserve the partial snapshot for provenance.

## Evidence

The draft is based on verified R2 reports under `uncle/paired_generalization/{plan['study_id']}/` and the matching CPU review. A compact [evidence snapshot](evidence/paired-study-2026-10-10.json) contains the immutable plan, comparison checks/results, unfinished-job list, and session records. The [dated experiment record](../wiki/Experiment-trajectory-2026-10-10.md) documents user-supplied progress and subsequent verification. No GPU training was performed to create this report.
'''
    assert '\u2014' not in text
    path = directory / 'Paired-study-2026-10-10-draft.md'
    path.write_text(text, encoding='utf-8')
    from render_report_html import render_report_html
    html_path = render_report_html(path)
    assert len(rows) == 4, 'Reconcile new completions before regenerating this partial draft.'
    assert all(all(row['pair_checks'].values()) for row in rows)
    assert sum('Pending' in line for line in table) == 2
    print(path)
    print(html_path)
    print(f'Verified draft: {len(rows)} completed pairs, two pending rows; {len(result["session_records"])} session logs.')


if __name__ == '__main__':
    main()
