"""Verify resumed trajectories, shared controls, and durable R2 publication."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_storage import Files
from test_paired_diagnostic import target, NoOldTraining
from uncle.storage import R2Store
from uncle.config import Config
from uncle.hypernet import HyperNetwork
from uncle.trainer import UnCLe
from uncle.paired_study import (StudyStore, compare_reports, jobs, make_plan,
    review, run_job, run_queue, summarize)


class Objects(Files):
    def __init__(self):
        super().__init__()
        self.deleted = []
        self.fail_key = None
        self.fail_delete_key = None
        self.download_keys = []

    def upload_file(self, path, bucket, key, ExtraArgs, **kwargs):
        if key == self.fail_key:
            raise OSError('Injected R2 outage')
        super().upload_file(path, bucket, key, ExtraArgs)
        if 'Callback' in kwargs:
            kwargs['Callback'](Path(path).stat().st_size)

    def download_file(self, bucket, key, path, **kwargs):
        self.download_keys.append(key)
        super().download_file(bucket, key, path)
        if 'Callback' in kwargs:
            kwargs['Callback'](Path(path).stat().st_size)

    def delete_object(self, Bucket, Key, **kwargs):
        if Key==self.fail_delete_key:
            raise OSError('Injected cleanup outage')
        self.deleted.append(Key)
        self.objects.pop(Key, None)


class PairedStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        data = TensorDataset(torch.rand(8,1,8,8), torch.arange(8)%2)
        self.tasks = {task:{'train':data,'test':data} for task in ('3','0','15')}
        self.plan = make_plan('test-revision', Path(__file__).resolve().parents[1]/'uncle/task_partition.json')
        self.plan.update(orders={'order_01':['3','0']}, seeds=[0], forget_tasks=['3'],
                         forget_checkpoint_interval=2, forget_evaluation_interval=2,
                         keep_branch_checkpoints=True, keep_post_unlearning_checkpoint=True)
        self.plan['config'].update(tasks=['3','0','15'], device='cpu',
            classes_per_task=2, hidden=[16], code_dim=4, chunks=8, epochs=2,
            batch_size=4, eval_batch_size=4, noise_samples=2, burn_in=4)
        self.source, self.control, self.branch = jobs(self.plan)
        self.patch = patch('uncle.paired_study.build_target', side_effect=target)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def storage(self, root, client=None):
        client = client or Objects()
        result = StudyStore(R2Store(client,'test'),self.plan,root)
        result.prepare()
        return result

    def compare_states(self, first, second):
        for name,value in first['hypernet'].items():
            self.assertTrue(torch.equal(value,second['hypernet'][name]),name)
        for task,buffers in first['task_buffers'].items():
            for name,value in buffers.items():
                self.assertTrue(torch.equal(value,second['task_buffers'][task][name]),task+'.'+name)
        self.assertEqual(first['history'],second['history'])
        self.assertTrue(torch.equal(first['rng'],second['rng']))

    def test_matrix_reuses_controls_and_balances_target_positions(self):
        plan=make_plan('revision',Path(__file__).resolve().parents[1]/'uncle/task_partition.json')
        queue=jobs(plan)
        self.assertEqual(sum(j['kind']=='source' for j in queue),9)
        self.assertEqual(sum(j['kind']=='control' for j in queue),9)
        self.assertEqual(sum(j['kind']=='unlearned' for j in queue),27)
        for task in plan['forget_tasks']:
            self.assertEqual(sorted(order.index(task)+1 for order in plan['orders'].values()),[1,4,8])

    def test_source_training_matches_existing_learning_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            storage=self.storage(Path(directory))
            run_job(self.plan,self.source,self.tasks,storage,device='cpu',progress=False)
            saved=storage.load(self.source)
            config=Config(**self.plan['config'])
            torch.manual_seed(config.seed)
            h=HyperNetwork(target(),config)
            trainer=UnCLe(h,config,target(),self.tasks)
            seen=[]
            for task in ('3','0'):
                trainer.learn(task,seen)
                seen.append(task)
            for name,value in h.state_dict().items():
                self.assertTrue(torch.equal(value,saved['hypernet'][name]),name)

    def test_reports_only_branch_retention_keeps_sources_and_review(self):
        self.plan['keep_branch_checkpoints']=False
        self.plan['keep_post_unlearning_checkpoint']=False
        with tempfile.TemporaryDirectory() as directory:
            storage=self.storage(Path(directory))
            for job in jobs(self.plan):
                run_job(self.plan,job,self.tasks,storage,device='cpu',progress=False)
            models=[key for key in storage.store.client.objects if key.endswith('.pt')]
            self.assertEqual(models,[storage.prefix+'/'+self.source['id']+'/checkpoint.pt'])
            self.assertEqual(storage.load(self.source)['report']['status'],'complete')
            for job in (self.control,self.branch):
                pointer=storage.progress(job)
                self.assertFalse(pointer['checkpoint_retained'])
                self.assertTrue(pointer['checkpoint_verified_before_discard'])
                self.assertEqual(storage.completed_report(job)['status'],'complete')
                with self.assertRaisesRegex(ValueError,'discarded after verification'):
                    storage.load(job)
            self.assertEqual(review(self.plan,storage,progress=False)['valid_pairs'],1)
            result=run_queue(self.plan,storage,lambda job:self.fail('Completed work should not train'),
                max_jobs=1,max_minutes=2,reserve_minutes=1,device='cpu',progress=False)
            self.assertEqual(result['completed_jobs'],3)

    def test_cleanup_outage_keeps_completion_and_refuses_changed_model(self):
        self.plan['keep_branch_checkpoints']=False
        self.plan['keep_post_unlearning_checkpoint']=False
        with tempfile.TemporaryDirectory() as directory:
            storage=self.storage(Path(directory))
            run_job(self.plan,self.source,self.tasks,storage,device='cpu',progress=False)
            key=storage.prefix+'/'+self.control['id']+'/checkpoint.pt'
            storage.store.client.fail_delete_key=key
            with self.assertRaisesRegex(OSError,'cleanup outage'):
                run_job(self.plan,self.control,self.tasks,storage,device='cpu',progress=False)
            self.assertEqual(storage.completed_report(self.control)['status'],'complete')
            self.assertIn(key,storage.store.client.objects)
            data,metadata=storage.store.client.objects[key]
            storage.store.client.fail_delete_key=None
            storage.store.client.objects[key]=(b'changed',metadata)
            with self.assertRaisesRegex(ValueError,'changed R2 artifact'):
                storage.cleanup(self.control)
            self.assertIn(key,storage.store.client.objects)
            storage.store.client.objects[key]=(data,metadata)
            storage.cleanup(self.control)
            self.assertNotIn(key,storage.store.client.objects)
            self.assertEqual(storage.completed_report(self.control)['status'],'complete')

    def test_epoch_and_forget_resume_match_uninterrupted_with_no_old_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=self.storage(root/'first')
            run_job(self.plan,self.source,self.tasks,first,device='cpu',progress=False)
            other=self.storage(root/'other',copy.deepcopy(first.store.client))
            for task in ('3','0'):
                self.tasks[task]['train']=NoOldTraining()
            run_job(self.plan,self.control,self.tasks,first,device='cpu',progress=False)
            saw=[]
            def stop_epoch(job,pointer):
                saw.append(pointer['active'])
            result=run_job(self.plan,self.control,self.tasks,other,device='cpu',progress=False,
                on_boundary=stop_epoch,
                stop_requested=lambda:bool(saw[-1] and saw[-1]['epoch']==1))
            self.assertEqual(result['status'],'resumable')
            partial=other.load(self.control)
            self.assertEqual(partial['active_request']['epoch'],1)
            self.assertTrue(partial['active_request']['optimizer']['state'])
            run_job(self.plan,self.control,self.tasks,other,device='cpu',progress=False)
            self.compare_states(first.load(self.control),other.load(self.control))
            run_job(self.plan,self.branch,self.tasks,first,device='cpu',progress=False)
            saw=[]
            result=run_job(self.plan,self.branch,self.tasks,other,device='cpu',progress=False,
                on_boundary=stop_epoch,
                stop_requested=lambda:bool(saw[-1] and saw[-1]['action']=='forget' and saw[-1]['step']==2))
            self.assertEqual(result['status'],'resumable')
            partial=other.load(self.branch)
            self.assertEqual(partial['active_request']['step'],2)
            self.assertIsNotNone(partial['active_request']['generator'])
            saw=[]
            result=run_job(self.plan,self.branch,self.tasks,other,device='cpu',progress=False,
                on_boundary=stop_epoch,
                stop_requested=lambda:bool(saw[-1] and saw[-1]['action']=='learn' and saw[-1]['epoch']==1))
            self.assertEqual(result['status'],'resumable')
            run_job(self.plan,self.branch,self.tasks,other,device='cpu',progress=False)
            self.compare_states(first.load(self.branch),other.load(self.branch))
            compared=compare_reports(self.plan,other.completed_report(self.control),other.completed_report(self.branch))
            self.assertEqual(compared['status'],'complete')
            self.assertTrue(all(compared['pair_checks'].values()))
            self.assertFalse(any(key.rsplit('/',1)[-1] in
                ('resume_0.pt','resume_1.pt','resume_0.json','resume_1.json')
                for key in other.store.client.objects))
            self.assertTrue(all('/paired_generalization/' in key for key in other.store.client.deleted))
            before_downloads=len(other.store.client.download_keys)
            result=review(self.plan,other,progress=False)
            self.assertEqual(result['valid_pairs'],1)
            self.assertEqual(result['status'],'complete')
            self.assertFalse(any(key.endswith('.pt') for key in
                other.store.client.download_keys[before_downloads:]))

    def test_failed_upload_does_not_advance_resume_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            storage=self.storage(Path(directory))
            def inject(job,pointer):
                if pointer['generation']==0:
                    storage.store.client.fail_key=storage.prefix+'/'+job['id']+'/resume_1.json'
            with self.assertRaisesRegex(OSError,'Injected R2 outage'):
                run_job(self.plan,self.source,self.tasks,storage,device='cpu',progress=False,on_boundary=inject)
            pointer=storage.progress(self.source)
            self.assertEqual(pointer['generation'],0)
            self.assertEqual(storage.load(self.source)['active_request']['epoch'],0)
            storage.store.client.fail_key=None
            resumed=run_job(self.plan,self.source,self.tasks,storage,device='cpu',progress=False)
            self.assertEqual(resumed['status'],'complete')

    def test_job_limit_and_incompatible_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            storage=self.storage(Path(directory))
            result=run_queue(self.plan,storage,lambda job:self.tasks,max_jobs=1,
                max_minutes=2,reserve_minutes=1,device='cpu',progress=False)
            self.assertEqual(result['completed_jobs'],1)
            self.assertEqual(len(result['attempted']),1)
            self.assertIsNone(storage.progress(self.control))
            changed=copy.deepcopy(self.plan)
            changed['config']['beta']=1
            with self.assertRaisesRegex(ValueError,'Study settings'):
                StudyStore(storage.store,changed,Path(directory)/'changed').prepare()

    def test_pair_mismatch_is_retained_and_averages_use_equal_setting_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            storage=self.storage(Path(directory))
            for job in jobs(self.plan):
                run_job(self.plan,job,self.tasks,storage,device='cpu',progress=False)
            a=storage.completed_report(self.control)
            b=storage.completed_report(self.branch)
            b['learning_rng_sha256']='different'
            invalid=compare_reports(self.plan,a,b)
            self.assertEqual(invalid['status'],'invalid_pair')
            self.assertEqual(summarize(self.plan,[invalid])['valid_pairs'],0)
            valid=compare_reports(self.plan,a,storage.completed_report(self.branch))
            failed=copy.deepcopy(storage.completed_report(self.branch))
            failed['after_unlearning_accuracies']['3']=60
            failed['requests'][-1]['epochs'][-1]['accuracies']['3']=70
            not_relapse=compare_reports(self.plan,a,failed)
            self.assertFalse(not_relapse['forgetting_within_tolerance'])
            self.assertFalse(not_relapse['relapse_above_tolerance'])
            plan=copy.deepcopy(self.plan)
            plan.update(seeds=[0,1],orders={'order_01':['3','0'],'order_02':['0','3']})
            rows=[]
            for order,seed,effect in [('order_01',0,2),('order_01',1,4),('order_02',0,20)]:
                row={**valid,'order':order,'seed':seed,'new_task_difference':effect}
                rows.append(row)
            summary=summarize(plan,rows)
            self.assertEqual(summary['status'],'partial')
            self.assertEqual(summary['equal_setting_means']['new_task_difference'],3)
            self.assertEqual(summary['complete_settings'],1)


if __name__=='__main__':
    unittest.main()
