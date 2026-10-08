"""Verify paired controls, generated-weight unlearning, and checkpoint history."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle import checkpoint
from uncle.config import Config, random_stream
from uncle.hypernet import HyperNetwork
from uncle.trainer import UnCLe
from uncle.paired_diagnostic import generated_noise_loss, run_paired_diagnostic
from uncle import paired_diagnostic
from uncle.learning_diagnostic import file_hash


def target(config=None):
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4),
        nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).requires_grad_(False)


class NoOldTraining:
    def __len__(self):
        return 8
    def __getitem__(self, index):
        raise AssertionError('Old training images must never be read.')


class PairedDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.config = Config(device='cpu', tasks=('3','0','15'),
            requests=(('learn','3'), ('learn','0')), initialization='hyperfan_in',
            classes_per_task=2, hidden=(16,), code_dim=4, chunks=8, epochs=2,
            batch_size=4, eval_batch_size=4, learning_rate=0.0001,
            beta=0.1, gamma=0.01, noise_samples=2, burn_in=3)
        data = TensorDataset(torch.rand(8,1,8,8), torch.arange(8)%2)
        self.tasks = {t: {'train': data, 'test': data} for t in self.config.tasks}
        self.h = HyperNetwork(target(), self.config)
        self.trainer = UnCLe(self.h, self.config, target(), self.tasks)
        self.history = []
        seen, before = [], {}
        for t in ('3','0'):
            losses = self.trainer.learn(t, seen)
            seen.append(t)
            after = {x:self.trainer.accuracy(x) for x in seen}
            self.history.append({'index':len(self.history), 'action':'learn', 'task':t,
                'before':before, 'after':after, 'seen':list(seen), 'forgotten':[],
                'final_loss':losses[-1], 'burn_in':None})
            before = after
        self.scores = before

    def save(self, root):
        path = root / 'source.pt'
        checkpoint.save(path, config=self.config, hypernet=self.h, uncle=self.trainer,
            history=self.history, seen=['3','0'], forgotten=[], previous=self.scores,
            costs=[], setup_seconds=0)
        return path

    def test_noise_objective_uses_scaled_weights_and_offsets(self):
        first = random_stream(0, 'forget', '3', 'cpu')
        second = random_stream(0, 'forget', '3', 'cpu')
        weights = torch.cat([v.reshape(-1) for v in self.h.weights_for('3').values()])
        expected = sum((weights-torch.randn(weights.shape, generator=first)).square().sum()
                       for _ in range(2))/2
        actual = generated_noise_loss(self.h, '3', second, 2)
        self.assertTrue(torch.equal(actual, expected))
        trainable = self.h.generator_parameters()
        expected_grad = torch.autograd.grad(expected, trainable, allow_unused=True)
        actual_grad = torch.autograd.grad(actual, trainable, allow_unused=True)
        for a, b in zip(actual_grad, expected_grad):
            self.assertTrue(a is None and b is None or torch.equal(a,b))

    def test_pair_controls_no_old_training_and_preserved_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.save(root)
            source_hash = file_hash(source)
            payload = checkpoint.load(source)
            old_buffers = copy.deepcopy(payload['task_buffers'])
            # Ordinary training is the same trajectory as the measured control.
            plain_config = replace(self.config, requests=self.config.requests+(('learn','15'),))
            plain_h = HyperNetwork(target(), plain_config)
            plain = UnCLe(plain_h, plain_config, target(), self.tasks)
            checkpoint.restore(payload, hypernet=plain_h, uncle=plain)
            plain.learn('15', ['3','0'])
            for t in ('3','0'):
                self.tasks[t]['train'] = NoOldTraining()
            saved = []
            with patch('uncle.paired_diagnostic.build_target', side_effect=target):
                result = run_paired_diagnostic(source, root/'pair', '3', '15', self.tasks,
                    device='cpu', progress=False, code_version='test',
                    on_saved=lambda *row:saved.append(row))
            self.assertTrue(all(result['pair_checks'].values()))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['selection']['learning_protected_tasks'], ['3','0'])
            self.assertEqual(result['selection']['unlearning_protected_tasks'], ['0'])
            self.assertEqual(file_hash(source), source_hash)
            self.assertEqual(result['retained_tasks'], ['0'])
            a = checkpoint.load(root/'pair/learning_only/checkpoint.pt')
            b = checkpoint.load(root/'pair/unlearn_then_learn/checkpoint.pt')
            u = checkpoint.load(root/'pair/unlearn_then_learn/after_unlearning.pt')
            self.assertEqual(a['forgotten'], [])
            self.assertEqual(b['forgotten'], ['3'])
            self.assertEqual(u['forgotten'], ['3'])
            self.assertEqual([r['action'] for r in b['history']], ['learn','learn','forget','learn'])
            self.assertEqual(b['history'][:2], self.history)
            self.assertEqual(a['history'][:2], self.history)
            for name, value in plain_h.state_dict().items():
                self.assertTrue(torch.equal(value, a['hypernet'][name]), name)
            for final in (a, b, u):
                for t in ('3','0'):
                    for name, value in old_buffers[t].items():
                        self.assertTrue(torch.equal(value, final['task_buffers'][t][name]))
                    self.assertTrue(torch.equal(payload['hypernet']['task_codes.'+t], final['hypernet']['task_codes.'+t]))
                self.assertTrue(torch.equal(payload['hypernet']['chunk_codes'], final['hypernet']['chunk_codes']))
            self.assertEqual(json.loads((root/'pair/paired_comparison.json').read_text()), result)
            self.assertEqual([r[0] for r in saved if r[1]=='complete'], ['learning_only','unlearn_then_learn'])
            with self.assertRaises(ValueError):
                run_paired_diagnostic(source, root/'pair', '3', '15', self.tasks)

    def test_callbacks_do_not_change_trajectory_and_bad_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.save(root)
            def noisy(*args):
                torch.rand(50)
            with patch('uncle.paired_diagnostic.build_target', side_effect=target):
                quiet = run_paired_diagnostic(source, root/'quiet', '3','15',self.tasks,progress=False)
                observed = run_paired_diagnostic(source, root/'observed', '3','15',self.tasks,
                                                progress=False,on_saved=noisy)
            # on_saved is outside trainer callbacks, so its RNG must be isolated by the runner.
            self.assertEqual(quiet['unlearn_then_learn_final_accuracies'], observed['unlearn_then_learn_final_accuracies'])
            for branch in ('learning_only','unlearn_then_learn'):
                a = checkpoint.load(root/'quiet'/branch/'checkpoint.pt')
                b = checkpoint.load(root/'observed'/branch/'checkpoint.pt')
                for name,value in a['hypernet'].items():
                    self.assertTrue(torch.equal(value,b['hypernet'][name]), name)
            real_restore = paired_diagnostic._restore_training_rng
            restores = []
            def different_rng(*args):
                real_restore(*args)
                restores.append(1)
                torch.rand(len(restores))
            with patch('uncle.paired_diagnostic.build_target', side_effect=target), patch('uncle.paired_diagnostic._restore_training_rng', side_effect=different_rng):
                with self.assertRaisesRegex(ValueError, 'Pair controls differ'):
                    run_paired_diagnostic(source,root/'bad','3','15',self.tasks,progress=False)
            self.assertEqual(json.loads((root/'bad/paired_comparison.json').read_text())['status'], 'invalid_pair')


if __name__ == '__main__':
    unittest.main()
