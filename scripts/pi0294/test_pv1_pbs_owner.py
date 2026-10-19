"""CPU intent and metadata tests, never submissions or site permission proof."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import pbs_owner as owner
from scripts.pi0294.pv1.neural_ledger import Ledger
from scripts.pi0294.test_pv1_neural_ledger import value
from scripts.pi0294.test_mp1_transport_fixture import put
from scripts.pi0294.native_dataset import sha


def submitted():
    return dict(status='SUBMITTED_PENDING_NATIVE',request_id='fixture-1',job_id='123.hopper-m-02',
        job_name='pi0294_pv1_01',project='CFP04-CF-007',reserved_H200_s=7200,accounting_pad_s=60,
        additional_output_upper_bytes=40*1024**3,stage='B0_DEV')


def root_ledger(tmp):
    parent=Path(tmp)/'campaign';root=parent/'children/fixture';root.mkdir(parents=True)
    put(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',dict(run_uuid='ORIGINAL_CAMPAIGN'))
    v=value();v.update(new_storage_peak_bytes=0,storage_reserved_bytes=0)
    put(parent/'HOPPER_RESOURCE_LEDGER.json',v)
    return root,Ledger(root)


class PbsOwnerContracts(unittest.TestCase):
    def test_replica_queues_have_separate_progress_and_no_duplicate_run_key(self):
        v=value();v.update(new_storage_peak_bytes=0,training_runs={'B_COVERAGE':dict(last_native_step=4096)})
        r=dict(stage='P4_REPLICA',walltime_s=7200,additional_output_upper_bytes=40*1024**3,
            dependency_afterok=None,training_block=dict(arm='B_COVERAGE',replica=True),expected_native_start_step=0)
        owner.budget_request(v,r)
        v['training_runs']['REPLICA_B_COVERAGE']=dict(last_native_step=1024)
        with self.assertRaisesRegex(PermissionError,'ACTUAL_CHARGED_PROGRESS'):owner.budget_request(v,r)
        r['expected_native_start_step']=1024;owner.budget_request(v,r)
        first=submitted();first['training_run_key']='REPLICA_B_COVERAGE';v['PBS_submissions']=[first]
        v['used']['PBS_submissions']=1;v['reserved']['H200_s']=7260;r['dependency_afterok']=first['job_id']
        with self.assertRaisesRegex(PermissionError,'DUPLICATE_QUEUED'):owner.budget_request(v,r)
        r['training_block']['replica']=False
        with self.assertRaisesRegex(PermissionError,'REPLICA_RUN_IDENTITY'):owner.budget_request(v,r)

    def test_existing_eligible_project_membership_dates_and_auth_fail_closed(self):
        p=dict(name='CFP05-CF-025',active=True,users=['fixture'],cluster=['hopper-m-02'],start_date='2000-01-01',end_date='2100-01-01')
        def query(args,**kwargs):
            return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps([p]) if args[1]=='ls' else "{'su': 100000}")
        with patch.dict(os.environ,{'USER':'fixture'}):
            self.assertEqual(owner.project(p['name'],query=query)['balance_su'],100000)
            p['users']=[]
            with self.assertRaises(PermissionError):owner.project(p['name'],query=query)
            expired=lambda *a,**kw:SimpleNamespace(returncode=1,stdout='Code 15006: token expired',stderr='')
            with self.assertRaisesRegex(PermissionError,'AUTH_EXPIRED_15006'):owner.project(p['name'],query=expired)
            with self.assertRaises(PermissionError):owner.project('invented-project',query=query)

    def test_ready_jobs_can_chain_across_projects_without_more_budget_or_duplicate_training(self):
        v=value();v.update(new_storage_peak_bytes=0);first=submitted();v['PBS_submissions']=[first]
        v['used']['PBS_submissions']=1;v['reserved']['H200_s']=7260
        r=dict(stage='P2_PRIMARY',walltime_s=7200,additional_output_upper_bytes=40*1024**3,
            dependency_afterok=first['job_id'],training_block=dict(arm='A_CONTROL'),expected_native_start_step=0)
        owner.budget_request(v,r)
        r['dependency_afterok']=None
        with self.assertRaisesRegex(PermissionError,'DEPENDENCY'):owner.budget_request(v,r)
        r['dependency_afterok']=first['job_id'];first['training_run_key']='A_CONTROL'
        with self.assertRaisesRegex(PermissionError,'DUPLICATE_QUEUED'):owner.budget_request(v,r)
        first.pop('training_run_key');v['used']['H200_s']=76*3600
        with self.assertRaises(PermissionError):owner.budget_request(v,r)

    def test_unknown_qsub_keeps_full_gpu_and_storage_reservations_and_is_not_resent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,ledger=root_ledger(tmp);directory=root/'pbs/alloc-01';directory.mkdir(parents=True)
            r=dict(stage='B0_DEV',evaluation_candidate='B0_CORR',walltime_s=7200,additional_output_upper_bytes=40*1024**3,
                dependency_afterok=None,created_epoch=owner.time.time(),submit_count=1,
                request_id='fixture-1',project='CFP04-CF-007',job_name='pi0294_pv1_01')
            put(directory/'REQUEST.json',r,True);(directory/'job.pbs').write_text('CPU fixture, not executable PBS')
            put(directory/'JOB_SCRIPT_BINDING.json',dict(request_sha256=sha(directory/'REQUEST.json'),job_script_sha256=sha(directory/'job.pbs')))
            with patch.object(owner,'validate'),patch.object(owner,'native_chain'),\
                    patch.object(owner.subprocess,'run',side_effect=subprocess.TimeoutExpired('qsub',45)) as qsub:
                with self.assertRaises(subprocess.TimeoutExpired):owner.submit(root,1)
                v=json.loads(ledger.path.read_text());self.assertEqual(v['reserved']['H200_s'],7260)
                self.assertEqual(v['storage_reserved_bytes'],40*1024**3);self.assertEqual(v['used']['PBS_submissions'],1)
                with self.assertRaises(PermissionError):owner.submit(root,1)
                self.assertEqual(qsub.call_count,1)

    def test_terminal_native_reads_are_outside_the_shared_training_counter_lock(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'USER':'fixture'}):
            root,ledger=root_ledger(tmp);v=json.loads(ledger.path.read_text());v['PBS_submissions']=[submitted()]
            v['used']['PBS_submissions']=1;v['reserved']['H200_s']=7260;put(ledger.path,v)
            in_lock=[False]
            class CheckedLedger(Ledger):
                @contextmanager
                def transaction(self):
                    with super().transaction() as data:
                        in_lock[0]=True
                        try:yield data
                        finally:in_lock[0]=False
            def native(job):
                self.assertFalse(in_lock[0])
                return dict(job_state='F',Job_Owner='fixture@host',project='CFP04-CF-007',Job_Name='pi0294_pv1_01',
                    Resource_List=dict(ngpus=1),resources_used=dict(walltime='00:10:00'),stime=1000,obittime=1600,Exit_status=0)
            with patch.object(owner,'validate'),patch.object(owner,'Ledger',CheckedLedger),patch.object(owner,'native',side_effect=native),\
                    patch.object(owner.subprocess,'check_output',return_value='1024 campaign'):
                self.assertEqual(owner.settle(root)['rows'][0]['conservative_GPU_s'],600)
            v=json.loads(ledger.path.read_text());self.assertEqual(v['used']['H200_s'],600);self.assertEqual(v['reserved']['H200_s'],0)
