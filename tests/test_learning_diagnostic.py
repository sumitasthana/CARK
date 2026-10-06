"""Learning measurements and continuation use the same optimizer trajectory."""
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
from uncle.config import Config
from uncle.hypernet import HyperNetwork
from uncle.trainer import UnCLe
from uncle.learning_diagnostic import inspect_source, file_hash, run_learning_diagnostic


def target(_config=None):
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, bias=False),
                         nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, 2)).requires_grad_(False)


class LearningDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = Config(device='cpu', tasks=('3','0','9','5','17'),
            requests=(('learn','3'),), initialization='hyperfan_in',
            classes_per_task=2, hidden=(16,), code_dim=4, chunks=8,
            epochs=2, batch_size=4, eval_batch_size=4, learning_rate=0.0001, beta=0.01)
        torch.manual_seed(0)
        data = TensorDataset(torch.rand(8,1,8,8), torch.arange(8) % 2)
        self.tasks = {t: {'train': data, 'test': data} for t in self.config.tasks}
        self.target = target()
        self.h = HyperNetwork(self.target, self.config)
        self.trainer = UnCLe(self.h,self.config,self.target,self.tasks)
        self.trainer.learn('3', [])
        self.initial_scores = {'3': self.trainer.accuracy('3')}

    def save_source(self, directory):
        source = Path(directory) / 'source.pt'
        history = [{'index':0, 'action':'learn', 'task':'3','before':{},
            'after':self.initial_scores,'seen':['3'],'forgotten':[],
            'final_loss':0.0,'burn_in':None}]
        checkpoint.save(source, config=self.config, hypernet=self.h,
            uncle=self.trainer, history=history, seen=['3'], forgotten=[],
            previous=self.initial_scores, costs=[], setup_seconds=0.0)
        return source

    def test_observation_gradients_and_epoch_accuracy_do_not_change_training(self):
        measured_h = copy.deepcopy(self.h)
        measured = UnCLe(measured_h,self.config,target(),self.tasks)
        measured.task_buffers = copy.deepcopy(self.trainer.task_buffers)
        frozen_code = self.h.task_codes['3'].detach().clone()
        frozen_chunks = self.h.chunk_codes.detach().clone()
        old_buffers = copy.deepcopy(self.trainer.task_buffers['3'])
        plain_losses = self.trainer.learn('0',['3'])
        rows,epochs = [],[]
        def epoch(row):
            epochs.append(row)
            measured.accuracy('3'); measured.accuracy('0')
            torch.rand(20)
        measured_losses = measured.learn('0',['3'],on_step=rows.append,
                                        on_epoch=epoch,gradient_steps=(1,3))
        self.assertEqual(plain_losses,measured_losses)
        for name,value in self.h.state_dict().items():
            self.assertTrue(torch.equal(value,measured_h.state_dict()[name]),name)
        for task in self.trainer.task_buffers:
            for name,value in self.trainer.task_buffers[task].items():
                self.assertTrue(torch.equal(value,measured.task_buffers[task][name]))
        self.assertTrue(torch.equal(frozen_code,self.h.task_codes['3']))
        self.assertTrue(torch.equal(frozen_chunks,self.h.chunk_codes))
        self.assertFalse(self.h.task_codes['3'].requires_grad)
        self.assertFalse(self.h.chunk_codes.requires_grad)
        for name,value in old_buffers.items():
            self.assertTrue(torch.equal(value,self.trainer.task_buffers['3'][name]))
        self.assertTrue(measured.target.training and measured.hypernet.training)
        self.assertEqual(rows[1]['protection_loss'],0)
        self.assertGreater(rows[2]['protection_loss'],0)
        for row in rows[1:]:
            self.assertAlmostEqual(row['loss'], row['task_loss']+row['weighted_protection_loss'],places=5)
            self.assertAlmostEqual(row['weighted_protection_loss'],0.01*row['protection_loss'],places=5)
        self.assertIn('task_gradient_norms', rows[1])
        self.assertIn('task_gradient_norms', rows[3])
        self.assertNotIn('task_gradient_norms', rows[2])
        self.assertEqual(len(epochs),2)

    def test_continue_changes_only_next_lesson_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source=self.save_source(directory)
            digest=file_hash(source)
            self.assertEqual(inspect_source(source)['completed_tasks'],['3'])
            output=Path(directory)/'continue'
            with patch('uncle.learning_diagnostic.build_target',side_effect=target):
                report=run_learning_diagnostic(source,output,'0',self.tasks,
                    beta=0.1,code_version='test',progress=False)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['source_beta'],0.01)
            self.assertEqual(report['config']['beta'],0.1)
            self.assertEqual(file_hash(source),digest)
            self.assertTrue(report['starting_scores_match'])
            saved=checkpoint.load(output/'checkpoint.pt')
            self.assertEqual(saved['seen'],['3','0'])
            self.assertEqual(saved['history'][0],report['source_history'][0])
            self.assertEqual(len(report['trace']),4)
            self.assertEqual(len(report['epochs']),2)
            self.assertEqual(saved['previous'],report['final_accuracies'])
            with patch('uncle.learning_diagnostic.build_target',side_effect=AssertionError('retrained')):
                repeated=run_learning_diagnostic(source,output,'0',self.tasks,
                    beta=0.1,code_version='test',progress=False)
            self.assertEqual(repeated,report)
            with self.assertRaisesRegex(ValueError,'different source'):
                run_learning_diagnostic(source,output,'0',self.tasks,beta=1,code_version='test')

    def test_each_beta_starts_from_same_saved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source=self.save_source(directory)
            with patch('uncle.learning_diagnostic.build_target',side_effect=target):
                a=run_learning_diagnostic(source,Path(directory)/'a','0',self.tasks,
                    beta=0,progress=False)
                b=run_learning_diagnostic(source,Path(directory)/'b','0',self.tasks,
                    beta=0.1,progress=False)
            self.assertEqual(a['source_sha256'],b['source_sha256'])
            self.assertEqual(a['starting_accuracies'],b['starting_accuracies'])
            self.assertEqual(a['trace'][0]['task_loss'],b['trace'][0]['task_loss'])
            self.assertEqual(a['trace'][0]['protection_loss'],0)
            self.assertEqual(b['trace'][0]['protection_loss'],0)

    def test_interruption_keeps_partial_epoch_report_and_requires_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            source=self.save_source(directory)
            output=Path(directory)/'attempt'
            def interrupted(row):
                raise InterruptedError('disconnected')
            with patch('uncle.learning_diagnostic.build_target',side_effect=target):
                with self.assertRaises(InterruptedError):
                    run_learning_diagnostic(source,output,'0',self.tasks,
                        progress=False,on_epoch=interrupted)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'interrupted')
            self.assertEqual(len(report['epochs']),1)
            self.assertFalse((output/'checkpoint.pt').exists())
            with self.assertRaisesRegex(ValueError,'interrupted'):
                run_learning_diagnostic(source,output,'0',self.tasks,progress=False)

    def test_already_learned_task_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            source=self.save_source(directory)
            with self.assertRaisesRegex(ValueError,'already learned'):
                run_learning_diagnostic(source,Path(directory)/'bad','3',self.tasks)


if __name__ == '__main__':
    unittest.main()
