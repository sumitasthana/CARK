"""Hyperfan statistics, gradients, and historical checkpoint compatibility."""
import copy
from dataclasses import replace
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.config import Config
from uncle.hypernet import HyperNetwork
from uncle.trainer import UnCLe
from uncle import checkpoint


class HyperfanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_generated_affine_variances_across_initializations(self):
        # Average over independent generators, not only correlated chunks
        # from one generator. These are output statistics, not source checks.
        target = nn.Sequential(nn.Linear(64, 128, bias=False), nn.ReLU(),
                               nn.Linear(128, 32))
        config = Config(device='cpu', initialization='hyperfan_in', chunks=16)
        measured = {name: [] for name in ['0.weight', '2.weight', '2.bias']}
        for seed in range(24):
            torch.manual_seed(seed)
            h = HyperNetwork(target, replace(config, seed=seed))
            h.add_task('3')
            with torch.no_grad():
                weights = h.weights_for('3')
                for name in measured:
                    measured[name].append(weights[name].square().mean().item())
        expected = {'0.weight': 2 / 64, '2.weight': 1 / (2 * 128), '2.bias': 0.5}
        for name, variance in expected.items():
            ratio = sum(measured[name]) / len(measured[name]) / variance
            self.assertTrue(0.8 < ratio < 1.2, (name, ratio))

    def test_batchnorm_starts_at_identity_and_can_learn(self):
        target = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False),
                               nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(), nn.Linear(8, 10))
        config = Config(device='cpu', initialization='hyperfan_in', chunks=12,
                        hidden=(32, 64), code_dim=8)
        torch.manual_seed(0)
        h = HyperNetwork(target, config)
        h.add_task('3')
        weights = h.weights_for('3')
        self.assertTrue(torch.equal(weights['1.weight'], torch.ones(8)))
        self.assertTrue(torch.equal(weights['1.bias'], torch.zeros(8)))
        scores = functional_call(target, (weights, dict(target.named_buffers())),
                                 (torch.randn(4, 1, 8, 8),), strict=True)
        loss = nn.functional.cross_entropy(scores, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in h.parameters()))
        gradient = h.heads['batchnorm'].weight.grad
        self.assertGreater(gradient.abs().sum().item(), 0)
        optimizer = torch.optim.Adam(h.parameters(), lr=0.0001)
        optimizer.step()
        self.assertFalse(torch.equal(h.weights_for('3')['1.weight'], torch.ones(8)))

    def test_legacy_generation_matches_historical_code(self):
        source = subprocess.check_output(['git', 'show',
            '83e1c32773fc59886c8b43a0ed159ccf6424637a:uncle/hypernet.py'], text=True)
        old = types.ModuleType('uncle.old_hypernet')
        old.__package__ = 'uncle'
        exec(compile(source, 'historical_hypernet', 'exec'), old.__dict__)
        target = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 10))
        config = Config(device='cpu', hidden=(16,), code_dim=4, chunks=8)
        torch.manual_seed(7)
        historical = old.HyperNetwork(copy.deepcopy(target), config)
        historical.add_task('3')
        torch.manual_seed(7)
        current = HyperNetwork(copy.deepcopy(target), config)
        current.add_task('3')
        for name, value in historical.weights_for('3').items():
            self.assertTrue(torch.equal(value, current.weights_for('3')[name]), name)

    def test_checkpoint_cannot_switch_initialization(self):
        target = nn.Linear(4, 10)
        config = Config(device='cpu', initialization='hyperfan_in', chunks=4)
        h = HyperNetwork(target, config)
        trainer = UnCLe(h, config)
        with self.assertRaisesRegex(ValueError, 'initialization differs'):
            checkpoint.restore({'config': {}}, hypernet=h, uncle=trainer)
        self.assertEqual(h.known_tasks, [])

    def test_preservation_with_offsets_matches_direct_equation_and_gradient(self):
        target = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Linear(16, 10))
        config = Config(device='cpu', initialization='hyperfan_in', chunks=8,
                        hidden=(32,), code_dim=8)
        h = HyperNetwork(target, config)
        h.add_task('3'); h.add_task('0')
        snapshot = h.snapshot()
        trainer = UnCLe(h, config)
        self.assertEqual(trainer.preserve(['3', '0'], snapshot).item(), 0)
        with torch.no_grad():
            h.heads['batchnorm'].weight.add_(0.01)
            h.heads['weights'].weight.add_(0.01)
        actual = trainer.preserve(['3', '0'], snapshot)
        direct = 0
        for task in ['3', '0']:
            code = h.task_codes[task].detach()
            current, before = h.weights_from_code(code), snapshot.weights_from_code(code)
            direct += sum((current[n] - before[n]).square().sum() for n in current) / 2
        self.assertTrue(torch.equal(actual, direct))
        a = torch.autograd.grad(actual, h.generator_parameters())
        b = torch.autograd.grad(direct, h.generator_parameters())
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))


if __name__ == '__main__':
    unittest.main()
