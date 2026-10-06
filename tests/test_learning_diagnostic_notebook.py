"""Exercise source selection and the continuation notebook with CPU models."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uncle import checkpoint
from uncle.config import Config
from uncle.hypernet import HyperNetwork
from uncle.trainer import UnCLe
from uncle.learning_diagnostic import run_learning_diagnostic as real_diagnostic


class ContinuationNotebookTests(unittest.TestCase):
    def test_cpu_selection_then_one_gpu_mode_lesson_with_mocked_services(self):
        torch.set_num_threads(1)
        path=Path(__file__).resolve().parents[1]/'notebooks/14_learning_loss_diagnostic.ipynb'
        cells=[''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
        torch.manual_seed(0)
        config=Config(dataset='tiny_imagenet',backbone='resnet50',device='cpu',
            tasks=('3','0','9','5','17'),requests=(('learn','3'),('learn','0'),('learn','9')),
            initialization='hyperfan_in',hidden=(16,),code_dim=4,chunks=8,
            epochs=1,batch_size=4,eval_batch_size=4,learning_rate=0.0001,beta=0.01)
        dataset=TensorDataset(torch.rand(8,3,8,8),torch.arange(8)%10)
        tasks={t:{'train':dataset,'test':dataset} for t in config.tasks}
        def target(_config=None):
            return nn.Sequential(nn.Conv2d(3,4,3,padding=1,bias=False),nn.BatchNorm2d(4),
                nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,10)).requires_grad_(False)
        template=target()
        h=HyperNetwork(template,config)
        trainer=UnCLe(h,config,template,tasks)
        history=[]; seen=[]; before={}
        for task in ('3','0','9'):
            losses=trainer.learn(task,seen)
            seen.append(task)
            after={t:trainer.accuracy(t) for t in seen}
            history.append({'index':len(history),'action':'learn','task':task,
                'before':before,'after':after,'seen':list(seen),'forgotten':[],
                'final_loss':losses[-1],'burn_in':None})
            before=after
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.pt'
            checkpoint.save(source,config=config,hypernet=h,uncle=trainer,history=history,
                seen=seen,forgotten=[],previous=before,costs=[],setup_seconds=0)
            namespace={'TEST_ROOT':root,'Path':Path,'CODE_VERSION':'test-version'}
            events=[]
            colab=ModuleType('google.colab')
            colab.runtime=type('Runtime',(),{'unassign':staticmethod(lambda:events.append('release'))})
            namespace['drive']=type('Drive',(),{'flush_and_unmount':staticmethod(lambda **kwargs:events.append('flush'))})
            def cpu_diagnostic(*args,**kwargs):
                kwargs['device']='cpu'
                return real_diagnostic(*args,**kwargs)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),patch.dict(sys.modules,{'google.colab':colab}):
                setup=cells[1].replace('ROOT = Path("/content/drive/MyDrive/uncle")','ROOT = TEST_ROOT')
                exec(compile(setup,str(path),'exec'),namespace)
                selection=cells[2].replace('SOURCE_NUMBER = None','SOURCE_NUMBER = 1')
                exec(compile(selection,str(path),'exec'),namespace)
                self.assertEqual(namespace['NEW_TASK'],'5')
                # SELECT mode does no training and releases no GPU.
                with patch('uncle.learning_diagnostic.build_target',side_effect=AssertionError('SELECT trained')):
                    exec(compile(cells[3],str(path),'exec'),namespace)
                exec(compile(cells[4],str(path),'exec'),namespace)
                self.assertEqual(events,['flush'])
                events.clear()
                namespace['MODE']='RUN'
                namespace['run_learning_diagnostic']=cpu_diagnostic
                with patch('uncle.streams.build_tasks',return_value=tasks),patch('uncle.learning_diagnostic.build_target',side_effect=target):
                    exec(compile(cells[3],str(path),'exec'),namespace)
                self.assertEqual(len(namespace['reports']),1)
                self.assertEqual(namespace['reports'][0][1]['new_task'],'5')
                self.assertEqual(namespace['reports'][0][1]['source_history'],json.loads(json.dumps(history)))
                exec(compile(cells[4],str(path),'exec'),namespace)
                self.assertEqual(events,['flush','release'])
                self.assertTrue((namespace['OUTPUT_ROOT']/'beta_comparison.json').is_file())


if __name__=='__main__':
    unittest.main()
