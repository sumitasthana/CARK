"""Check compact display counts and throttling without training or remote I/O."""
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
scope = {}
source = (ROOT / 'scripts/paired_progress.py').read_text(encoding='utf-8')
exec(source.split('\ndashboard = StudyDashboard(SEED)')[0], scope)
StudyDashboard = scope['StudyDashboard']


def study_jobs():
    rows = []
    for seed in range(3):
        source = f'sources/order_01/seed_{seed}'
        rows.append(dict(id=source, kind='source', order='order_01', seed=seed,
            source=None, operations=[['learn', str(i)] for i in range(8)]))
        rows.append(dict(id=f'controls/order_01/seed_{seed}/learn_15', kind='control',
            order='order_01', seed=seed, source=source, operations=[['learn', '15']]))
        for task in ('3', '5', '14'):
            rows.append(dict(id=f'branches/order_01/seed_{seed}/forget_{task}/learn_15',
                kind='unlearned', order='order_01', seed=seed, source=source,
                forget_task=task, operations=[['forget', task], ['learn', '15']]))
    return rows


class ProgressDisplay(unittest.TestCase):
    def test_counts_exclude_inherited_requests_and_deferred_jobs(self):
        dashboard = StudyDashboard(0, emit=lambda markup: None)
        pointers = {
            'sources/order_01/seed_0': dict(status='complete', completed_requests=8),
            'controls/order_01/seed_0/learn_15': dict(status='complete', completed_requests=9),
            'branches/order_01/seed_0/forget_3/learn_15':
                dict(status='resumable', completed_requests=8, active={'step': 60}),
        }
        dashboard.configure({}, SimpleNamespace(progress=lambda job: pointers.get(job['id'])), study_jobs())
        self.assertEqual(dashboard.counts(), (9, 39, 2, 12, 0))
        self.assertEqual(dashboard.counts(0), (9, 13, 2, 4, 0))
        branch = dashboard.jobs['branches/order_01/seed_0/forget_3/learn_15']
        dashboard.saved(branch, dict(status='complete', completed_requests=10,
            updated_at='2026-10-09T18:00:00Z'))
        self.assertEqual(dashboard.counts(), (11, 39, 3, 12, 1))
        self.assertIn('3/12 jobs complete', dashboard.markup())
        self.assertNotIn('U5 then L15', dashboard.markup())

    def test_many_transfer_updates_do_not_append_many_displays(self):
        output = []
        now = [0.0]
        dashboard = StudyDashboard(1, emit=output.append, clock=lambda: now[0])
        bar = dashboard.transfer_bar(1000, 'Download <model>')
        for _ in range(1000):
            bar.update(1)
        self.assertEqual(len(output), 1)
        now[0] = 1.1
        bar.update(0)
        self.assertEqual(len(output), 2)
        self.assertIn('Download &lt;model&gt;', output[-1])
        bar.reset()
        self.assertEqual(bar.n, 0)

    def test_nested_stage_restores_training_details_after_validation(self):
        dashboard = StudyDashboard(2, emit=lambda markup: None)
        training = dashboard.bar(395, 'Learn 15')
        training.set_postfix_str('epoch 3/5, loss 2.5')
        training.update(237)
        with dashboard.bar(2, 'Validate 3') as validation:
            validation.update()
            self.assertIn('Validate 3', dashboard.markup())
        self.assertIn('Learn 15', dashboard.markup())
        self.assertIn('237 / 395 steps', dashboard.markup())
        self.assertIn('epoch 3/5', dashboard.markup())


if __name__ == '__main__':
    unittest.main()
