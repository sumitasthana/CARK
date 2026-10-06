"""Run the pilot's training and finish cells with real CPU checkpoint files."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.config import Config
from uncle.experiments import run_requests as real_run_requests


class PilotNotebookTests(unittest.TestCase):
    def test_train_resume_and_finish_from_saved_files(self):
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / 'notebooks/13_hyperfan_learning_pilot.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        order = ('3', '0', '9')
        config = Config(dataset='tiny_imagenet', backbone='resnet50', device='cpu',
                        tasks=order, requests=tuple(('learn', t) for t in order),
                        initialization='hyperfan_in', hidden=(16,), code_dim=4,
                        chunks=8, epochs=1, batch_size=4, eval_batch_size=4,
                        learning_rate=0.0001, beta=0.01)
        torch.manual_seed(0)
        dataset = TensorDataset(torch.rand(8, 3, 8, 8), torch.arange(8) % 10)
        tasks = {task: {'train': dataset, 'test': dataset} for task in order}
        def target(_config):
            return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.Linear(4, 10)).requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory:
            namespace = {'OUTPUT': Path(directory), 'DATA': Path(directory), 'config': config,
                         'TASK_ORDER': order, 'CODE_VERSION': 'test-version', 'os': os,
                         'json': json, 'math': math}
            with patch('uncle.experiments.build_tasks', return_value=tasks), \
                 patch('uncle.experiment.build_target', side_effect=target), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                def interrupted_run(*args, **kwargs):
                    def stop_after_saved_task(record, checkpoint_path):
                        raise InterruptedError("Simulated disconnect after saving task 3")
                    kwargs['on_saved_checkpoint'] = stop_after_saved_task
                    return real_run_requests(*args, **kwargs)
                with patch('uncle.experiments.run_requests', side_effect=interrupted_run):
                    with self.assertRaises(InterruptedError):
                        exec(compile(cells[2], str(path), 'exec'), namespace)
                saved = json.loads((Path(directory) / 'accuracy_table.json').read_text())
                self.assertFalse(saved['complete'])
                self.assertEqual([r['task'] for r in saved['rows']], ['3'])
                exec(compile(cells[2], str(path), 'exec'), namespace)
                self.assertTrue(namespace['report']['complete'])
                first = namespace['report']
                # Re-running the cell reads saved history and skips training.
                exec(compile(cells[2], str(path), 'exec'), namespace)
                self.assertEqual(namespace['report'], first)
                events = []
                colab = ModuleType('google.colab')
                colab.runtime = type('Runtime', (), {'unassign': staticmethod(lambda: events.append('release'))})
                namespace['drive'] = type('Drive', (), {
                    'flush_and_unmount': staticmethod(lambda **kwargs: events.append('flush'))})
                with patch.dict(sys.modules, {'google.colab': colab}):
                    exec(compile(cells[3], str(path), 'exec'), namespace)
                self.assertEqual(events, ['flush', 'release'])


if __name__ == '__main__':
    unittest.main()
