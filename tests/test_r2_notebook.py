"""Run notebook modes with a tiny real model and a fake R2 service."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle import checkpoint
from uncle.config import Config
from uncle.hypernet import HyperNetwork
from uncle.trainer import UnCLe
from uncle.learning_diagnostic import run_learning_diagnostic as diagnostic
from uncle.storage import R2Store, sha256
from test_storage import Files


class R2NotebookTests(unittest.TestCase):
    def test_copy_then_continue_one_lesson_and_release_after_uploads(self):
        torch.set_num_threads(1)
        notebook = Path(__file__).resolve().parents[1] / 'notebooks/15_R2_learning_loss.ipynb'
        cells = [''.join(c['source']) for c in json.loads(notebook.read_text(encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
        for cell in cells:
            compile(cell, str(notebook), 'exec')
        torch.manual_seed(0)
        config = Config(dataset='tiny_imagenet', backbone='resnet50', device='cpu',
            tasks=('3','0','9','5','17'), requests=tuple(('learn', t) for t in ('3','0','9','5')),
            initialization='hyperfan_in', hidden=(16,), code_dim=4, chunks=8,
            epochs=1, batch_size=4, eval_batch_size=4, learning_rate=0.0001, beta=0.01)
        data = TensorDataset(torch.rand(8,3,8,8), torch.arange(8) % 10)
        tasks = {t: {'train': data, 'test': data} for t in config.tasks}
        def target(_config=None):
            return nn.Sequential(nn.Conv2d(3,4,3,padding=1,bias=False),nn.BatchNorm2d(4),
                nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,10)).requires_grad_(False)
        template = target()
        hypernet = HyperNetwork(template, config)
        trainer = UnCLe(hypernet, config, template, tasks)
        history, seen, before = [], [], {}
        for task in ('3','0','9','5'):
            losses = trainer.learn(task, seen)
            seen.append(task)
            after = {t: trainer.accuracy(t) for t in seen}
            history.append({'index': len(history), 'action': 'learn', 'task': task,
                'before': before, 'after': after, 'seen': list(seen), 'forgotten': [],
                'final_loss': losses[-1], 'burn_in': None})
            before = after
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drive_root = root / 'drive'
            drive_root.mkdir()
            source = drive_root / 'model.pt'
            checkpoint.save(source, config=config, hypernet=hypernet, uncle=trainer,
                history=history, seen=seen, forgotten=[], previous=before, costs=[], setup_seconds=0)
            dataset_folder = drive_root / 'data/tiny-imagenet-200/train'
            dataset_folder.mkdir(parents=True)
            (dataset_folder / 'image.JPEG').write_bytes(b'dataset image')
            (dataset_folder / 'labels.json').write_text('{}')
            (drive_root / 'tiny-imagenet-200.zip').write_bytes(b'dataset archive')
            original_hash = sha256(source)
            client = Files()
            store = R2Store(client, 'test')
            events = []
            colab = ModuleType('google.colab')
            colab.userdata = SimpleNamespace(get=lambda name: 'test-value')
            colab.drive = SimpleNamespace(mount=lambda _: events.append('mount'),
                flush_and_unmount=lambda **_: events.append('flush'))
            def release():
                report = json.loads(client.objects['uncle/learning_loss_diagnostic/20261006_learning_loss_r2_01/beta_0_01/report.json'][0])
                self.assertEqual(report['status'], 'complete')
                self.assertTrue(any(k.endswith('/checkpoint.pt') for k in client.objects))
                events.append('release')
            colab.runtime = SimpleNamespace(unassign=release)
            setup = cells[1].replace('Path("/content/drive/MyDrive/uncle")', 'TEST_ROOT / "drive"')
            setup = setup.replace('Path("/content/r2-cache")', 'TEST_ROOT / "cache"')
            setup = setup.replace('Path("/content/r2-results")', 'TEST_ROOT / "results"')
            setup = setup.replace('SOURCE_KEY = "uncle/learning_initialization/20261006_hyperfan_L3_L0_L9_seed0_02/checkpoint_seq1_resnet50_seed0.pt"', 'SOURCE_KEY = "uncle/model.pt"')
            # Production checks the user's reported scores; this fixture has its own saved scores.
            source_cell = cells[2].replace('{"3": 27.0, "0": 31.2, "9": 48.4, "5": 54.8}', repr(before))
            def cpu_diagnostic(*args, **kwargs):
                kwargs['device'] = 'cpu'
                with patch('torch.cuda.is_available', return_value=False):
                    return diagnostic(*args, **kwargs)
            def namespace():
                return {'TEST_ROOT': root, 'Path': Path, 'CODE_VERSION': 'test-version'}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch.dict(sys.modules, {'google.colab': colab}), patch.object(R2Store, 'connect', return_value=store):
                copy_session = namespace()
                exec(setup.replace('MODE = "COMPARE"', 'MODE = "COPY"'), copy_session)
                with patch('torch.cuda.is_available', return_value=False):
                    exec(source_cell, copy_session)
                exec(cells[3], copy_session)
                exec(cells[4], copy_session)
                self.assertEqual(events, ['mount', 'flush'])
                self.assertFalse(any('learning_loss_diagnostic' in k for k in client.objects))
                self.assertEqual(set(client.objects), {'uncle/model.pt'})
                self.assertEqual(copy_session['copied_files']['skipped_directories'], ['data'])
                run_session = namespace()
                exec(setup.replace('MODE = "COMPARE"', 'MODE = "RUN"'), run_session)
                with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.get_device_name', return_value='A100'):
                    exec(source_cell, run_session)
                run_session['run_learning_diagnostic'] = cpu_diagnostic
                with patch('uncle.streams.build_tasks', return_value=tasks), patch('uncle.learning_diagnostic.build_target', side_effect=target):
                    exec(cells[3], run_session)
                exec(cells[4], run_session)
                self.assertEqual(events, ['mount', 'flush', 'release'])
                report = run_session['reports'][0][2]
                self.assertEqual(report['new_task'], '17')
                self.assertEqual(report['source_sha256'], original_hash)
                self.assertEqual(report['source_history'], json.loads(json.dumps(history)))
                self.assertEqual(sha256(source), original_hash)
                # Reusing this experiment name in a fresh session stops before another lesson.
                with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.get_device_name', return_value='A100'), self.assertRaises(AssertionError):
                    exec(source_cell, run_session)

                earlier = json.loads(json.dumps(report))
                earlier['environment']['gpu'] = 'earlier GPU'
                earlier_path = root / 'earlier.json'
                earlier_path.write_text(json.dumps(earlier))
                store.upload(earlier_path, run_session['DRIVE_REPORT_KEY'])
                compare_session = namespace()
                exec(setup, compare_session)
                compare_cell = cells[5].replace('Path("/content/report-comparison.json")', 'TEST_ROOT / "comparison.json"')
                with patch('torch.cuda.is_available', return_value=False), patch('uncle.streams.build_tasks', side_effect=AssertionError('CPU comparison must not load data')):
                    exec(source_cell, compare_session)
                    exec(compare_cell, compare_session)
                self.assertTrue(compare_session['comparison']['ready_for_beta_batch'])
                self.assertIn('gpu', compare_session['comparison']['runtime_differences'])

                batch_session = namespace()
                exec(setup.replace('MODE = "COMPARE"', 'MODE = "BATCH"'), batch_session)
                trained_betas = []
                def batch_diagnostic(*args, **kwargs):
                    trained_betas.append(kwargs['beta'])
                    return cpu_diagnostic(*args, **kwargs)
                batch_session['run_learning_diagnostic'] = batch_diagnostic
                reference_uploads = client.uploads.count(run_session['R2_REPORT_KEY'])
                with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.get_device_name', return_value='A100'), patch('uncle.streams.build_tasks', return_value=tasks) as build, patch('uncle.learning_diagnostic.build_target', side_effect=target):
                    exec(source_cell, batch_session)
                    exec(cells[6], batch_session)
                    self.assertEqual(build.call_count, 1)
                    with self.assertRaises(AssertionError):
                        exec(cells[6], batch_session)
                self.assertEqual(trained_betas, [0.0, 0.001, 0.1, 1.0])
                self.assertEqual(client.uploads.count(run_session['R2_REPORT_KEY']), reference_uploads)
                self.assertEqual(events, ['mount', 'flush', 'release', 'release'])
                self.assertEqual(sha256(source), original_hash)
                self.assertEqual(len(batch_session['summaries']), 5)
                self.assertEqual([r['beta'] for r in batch_session['summaries'] if r['reused']], [0.01])
                self.assertIn(batch_session['batch_prefix'] + '/beta_batch_comparison.json', client.objects)

                repeat_session = namespace()
                exec(setup.replace('MODE = "COMPARE"', 'MODE = "REPEAT"'), repeat_session)
                repeat_calls = []
                def repeat_diagnostic(*args, **kwargs):
                    repeat_calls.append((sha256(args[0]), kwargs['beta']))
                    return cpu_diagnostic(*args, **kwargs)
                repeat_session['run_learning_diagnostic'] = repeat_diagnostic
                first_trial_uploads = client.uploads.count(repeat_session['REPEAT_REPORT_KEY'])
                with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.get_device_name', return_value='A100'), patch('uncle.streams.build_tasks', return_value=tasks), patch('uncle.learning_diagnostic.build_target', side_effect=target):
                    exec(source_cell, repeat_session)
                    # Run all earlier cells: none may train or release the session.
                    for cell in cells[3:7]:
                        exec(cell, repeat_session)
                    self.assertEqual(repeat_calls, [])
                    exec(cells[7], repeat_session)
                    with self.assertRaises(AssertionError):
                        exec(cells[7], repeat_session)
                self.assertEqual(repeat_calls, [(original_hash, 0.1)])
                self.assertEqual(client.uploads.count(repeat_session['REPEAT_REPORT_KEY']), first_trial_uploads)
                self.assertEqual(events, ['mount', 'flush', 'release', 'release', 'release'])
                self.assertEqual(sha256(source), original_hash)
                saved_comparison = json.loads(client.objects[repeat_session['repeat_prefix'] + '/repeat_comparison.json'][0])
                self.assertEqual(saved_comparison['first']['beta'], 0.1)
                self.assertEqual(saved_comparison['repeat']['beta'], 0.1)
                self.assertEqual(saved_comparison['repeat_minus_first'], {t: 0.0 for t in config.tasks})

                # A source mismatch is recorded on CPU and blocks all new training.
                earlier['source_sha256'] = 'different model'
                earlier_path.write_text(json.dumps(earlier))
                store.upload(earlier_path, run_session['DRIVE_REPORT_KEY'], overwrite=True)
                compare_session['COMPARISON_KEY'] = 'uncle/mismatch-comparison.json'
                with patch('torch.cuda.is_available', return_value=False):
                    exec(compare_cell, compare_session)
                self.assertFalse(compare_session['comparison']['ready_for_beta_batch'])
                batch_session['COMPARISON_KEY'] = compare_session['COMPARISON_KEY']
                with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.get_device_name', return_value='A100'), self.assertRaises(AssertionError):
                    exec(cells[6], batch_session)
                self.assertEqual(trained_betas, [0.0, 0.001, 0.1, 1.0])


if __name__ == '__main__':
    unittest.main()
