"""Small CPU checks for matched initialization, isolation, and resume."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import torch
from torch import nn
from torch.utils.data import TensorDataset

from uncle import checkpoint
from uncle.config import Config
from uncle.recovery_setup import run_setup_stage, readiness, validate_manifest


def target(config):
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, bias=False),
                         nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, 2)).requires_grad_(False)


class ForbiddenData:
    def __len__(self):
        return 8
    def __getitem__(self, index):
        raise AssertionError('The never-learned branch must not read target images.')


class RecoverySetupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path('outputs/recovery_stage_checks') / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.config = Config(device='cpu', tasks=('0','3','9','2'), requests=(),
                             initialization='hyperfan_in', classes_per_task=2,
                             hidden=(8,), code_dim=4, chunks=8, epochs=1,
                             batch_size=4, eval_batch_size=4, noise_samples=2,
                             burn_in=2, burn_in_min=2, learning_rate=0.0001)
        data = TensorDataset(torch.rand(8,1,8,8), torch.arange(8) % 2)
        self.tasks = {t: {'train': data, 'test': data} for t in ('0','3','9')}

    def run_stage(self, stage, **kwargs):
        with patch('uncle.recovery_setup.build_target', side_effect=target):
            return run_setup_stage(self.config, self.tasks, self.root/stage,
                                   stage=stage, retained=('0','9'),
                                   manifest_sha256='test', code_version='test',
                                   progress=False, **kwargs)

    def test_matching_shared_state_target_exclusion_and_frozen_unlearning(self):
        self.run_stage('shared')
        source = self.root/'shared/checkpoint.pt'
        shared = checkpoint.load(source)
        real = self.tasks['3']['train']
        self.tasks['3']['train'] = ForbiddenData()
        reference = self.run_stage('never_learned', source=source)
        self.tasks['3']['train'] = real
        unlearned = self.run_stage('unlearned', source=source)
        self.assertEqual(reference['status'], 'complete')
        self.assertEqual(unlearned['status'], 'complete')
        never = checkpoint.load(self.root/'never_learned/checkpoint.pt')
        after = checkpoint.load(self.root/'unlearned/checkpoint.pt')
        before = checkpoint.load(self.root/'unlearned/before_unlearning.pt')
        self.assertNotIn('3', never['tasks_with_codes'])
        self.assertNotIn('2', after['tasks_with_codes'])
        self.assertEqual(after['forgotten'], ['3'])
        self.assertEqual(len(before['history']), 3)
        for final in (never, after):
            self.assertTrue(torch.equal(shared['hypernet']['chunk_codes'], final['hypernet']['chunk_codes']))
        for t in after['seen']:
            self.assertTrue(torch.equal(before['hypernet']['task_codes.'+t], after['hypernet']['task_codes.'+t]))
            for name, value in before['task_buffers'][t].items():
                self.assertTrue(torch.equal(value, after['task_buffers'][t][name]))
        self.assertEqual(after['history'][-1]['readiness'], unlearned['readiness'])

    def test_request_resume_and_complete_rerun_do_not_train_twice(self):
        events = []
        def interrupt(event, output):
            events.append(event)
            if event == 'request_complete':
                raise RuntimeError('Simulated disconnect after checkpoint save')
        with self.assertRaisesRegex(RuntimeError, 'Simulated disconnect'):
            self.run_stage('shared', on_saved=interrupt)
        saved = checkpoint.load(self.root/'shared/checkpoint.pt')
        self.run_stage('shared')
        again = checkpoint.load(self.root/'shared/checkpoint.pt')
        self.assertEqual(len(again['history']), 1)
        for name, value in saved['hypernet'].items():
            self.assertTrue(torch.equal(value, again['hypernet'][name]))
        self.config = replace(self.config, seed=1)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            self.run_stage('shared')

    def test_spill_uses_absolute_sum_and_strict_thresholds(self):
        gate = readiness({'3':27.4,'0':45.,'9':45.}, {'3':10.,'0':43.,'9':47.}, '3', ('0','9'))
        self.assertEqual(gate['spill_points'], 4.)
        self.assertFalse(gate['eligible'])
        gate = readiness({'3':27.4,'0':45.}, {'3':10.,'0':44.}, '3', ('0',))
        self.assertTrue(gate['eligible'])

    def test_manifest_rejects_wrong_partition(self):
        with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
            validate_manifest({'partition_sha256':'wrong'}, 'uncle/task_partition.json')


if __name__ == '__main__':
    unittest.main()
