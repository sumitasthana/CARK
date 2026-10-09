"""Execute CPU notebook flows with fake R2 and no model or dataset downloads."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/'scripts'))
import build_paired_study_notebooks as builder
from test_paired_study import Objects
from uncle.storage import R2Store
from uncle.paired_study import StudyStore,make_plan,jobs,inspect_queue,run_queue,review


def cells(name):
    notebook=json.loads((ROOT/'notebooks'/name).read_text())
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code']


class NotebookFlows(unittest.TestCase):
    def test_multipart_retry_is_bounded_and_does_not_retry_access_denied(self):
        import types
        from boto3.exceptions import S3UploadFailedError
        source=''.join(builder.resilient_connection()['source'])
        source=source[source.index('# Transport-only fix:'):]
        calls=[]
        def upload(*args,**kwargs):
            calls.append(kwargs['Config'])
            if len(calls)<3:
                raise S3UploadFailedError('An error occurred (InvalidPart)')
            return 'uploaded'
        store=types.SimpleNamespace(client=types.SimpleNamespace(upload_file=upload))
        with contextlib.redirect_stdout(io.StringIO()), patch('time.sleep'):
            scope={'store':store}
            exec(source,scope)
            self.assertEqual(store.client.upload_file('model.pt','bucket','key'),'uploaded')
        self.assertEqual(len(calls),3)
        self.assertFalse(calls[0].use_threads)
        self.assertEqual(calls[0].multipart_chunksize,64*1024*1024)
        for message,expected in [('(InvalidPart)',3),('(AccessDenied)',1)]:
            attempts=[]
            def fail(*args,**kwargs):
                attempts.append(1)
                raise S3UploadFailedError(message)
            store=types.SimpleNamespace(client=types.SimpleNamespace(upload_file=fail))
            with contextlib.redirect_stdout(io.StringIO()), patch('time.sleep'):
                exec(source,{'store':store})
                with self.assertRaises(S3UploadFailedError):
                    store.client.upload_file('model.pt','bucket','key')
            self.assertEqual(len(attempts),expected)

    def test_setup_prints_error_before_runtime_release(self):
        import types
        import importlib.util
        spec=importlib.util.spec_from_file_location('notebook_builder',ROOT/'scripts/build_paired_study_notebooks.py')
        builder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        output=io.StringIO()
        runtime=types.SimpleNamespace(unassign=lambda: print('RUNTIME RELEASED'))
        colab=types.ModuleType('google.colab')
        colab.runtime=runtime
        scope={'sys':sys,'IN_COLAB':True,'RELEASE_GPU_WHEN_DONE':True}
        source=''.join(builder.guarded_setup(builder.code('raise FileNotFoundError("missing study.json")'))['source'])
        with patch.dict(sys.modules,{'google.colab':colab}), contextlib.redirect_stdout(output):
            with self.assertRaises(FileNotFoundError):
                exec(source,scope)
        captured=output.getvalue()
        self.assertIn('FileNotFoundError: missing study.json',captured)
        self.assertLess(captured.index('FileNotFoundError: missing study.json'),captured.index('RUNTIME RELEASED'))

    def test_cpu_prepare_and_empty_review_export_use_reports_only(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            store=R2Store(Objects(),'test')
            scope={'Path':Path,'REPO':ROOT,'CODE_VERSION':'test-revision','IN_COLAB':False,
                'torch':torch,'json':json,'store':store,'CACHE':Path(directory),
                'StudyStore':StudyStore,'make_plan':make_plan,'jobs':jobs,
                'inspect_queue':inspect_queue,'run_queue':run_queue,'review':review}
            prepare=cells('18_paired_study_prepare.ipynb')
            exec(prepare[1],scope)
            exec(prepare[2],scope)
            scope['MODE']='PREPARE'
            scope['BUCKET']='test'
            exec(prepare[-1],scope)
            self.assertEqual(len(store.client.objects),1)
            self.assertIn('uncle/paired_generalization/paired_generalization_v1/study.json',store.client.objects)
            report=cells('20_paired_study_review.ipynb')
            exec(report[1],scope)
            scope['BUCKET']='test'
            exec(report[3],scope)
            exec(report[4],scope)
            exec(report[5],scope)
            import matplotlib
            matplotlib.use('Agg')
            exec(report[6],scope)
            scope['SAVE_SUMMARY_TO_R2']=True
            exec(report[7],scope)
            self.assertEqual(scope['result']['status'],'partial')
            self.assertEqual(scope['result']['valid_pairs'],0)
            self.assertTrue((scope['EXPORT']/'comparisons.csv').exists())
            self.assertTrue((scope['EXPORT']/'retained_tasks.csv').exists())
            self.assertFalse(any(key.endswith('.pt') for key in store.client.objects))
            self.assertFalse(any(key.endswith('.pt') for key in store.client.download_keys))

    def test_gpu_notebook_requires_explicit_training_and_cpu_notebooks_reject_gpu(self):
        with contextlib.redirect_stdout(io.StringIO()):
            scope={'Path':Path,'REPO':ROOT,'IN_COLAB':False,'torch':torch,'json':json}
            gpu=cells('19_paired_study_run.ipynb')
            with self.assertRaisesRegex(RuntimeError,'Training is off'):
                exec(gpu[1],scope)
            for name in ('18_paired_study_prepare.ipynb','20_paired_study_review.ipynb'):
                with patch('torch.cuda.is_available',return_value=True):
                    with self.assertRaisesRegex(RuntimeError,'CPU runtime'):
                        exec(cells(name)[1],scope)


if __name__=='__main__':
    unittest.main()
