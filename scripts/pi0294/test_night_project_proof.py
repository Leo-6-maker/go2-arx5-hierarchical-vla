"""CPU-only entitlement/PID fixtures, not actual PBS admission or authentication."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.pi0292.durable import read,write
from .allocation_admission import night_parent_proof,validate,T0,CUTOFF,DEADLINE,GIB,PROJECT
from .native_dataset import sha


class NightProjectProofTests(unittest.TestCase):
    def fixture(self,root):
        parent=root/'pbs/night-final/REQUEST.json';parent.parent.mkdir(parents=True)
        request=dict(root=str(root),source_sha='a'*40,run_manifest_sha256='b'*64,
            code_seal_sha256='c'*64,bindings='fixture',bindings_sha256='d'*64,
            walltime_s=7200,submit_count=4,prior_GPU_allocated_s=489,
            numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',microbatch=2,workers=0,
            project_admission_sha256='e'*64,request_nonce='c9e9f761-f1e8-4cf4-9851-68adb68d1f21',
            stage='night-final',mode='night-final')
        write(parent,request)
        child=root/'pbs/main-512/REQUEST.json';child.parent.mkdir()
        current=dict(request,stage='main',mode='main',night_parent_request_relative=str(parent.relative_to(root)),
            night_parent_request_sha256=sha(parent),allowed_peer_ids=[],predicted_remaining_peak_bytes=GIB)
        write(child,current)
        admission=dict(status='PASS_PI0294_OWN_NATIVE_PBS_ADMISSION_NOT_MODEL_QUALIFICATION',
            project_proof_mode='FRESH_LOGIN_PROJECT_AT_INITIAL_ADMISSION',job_id='1.server',project=PROJECT,
            request_sha256=sha(parent),project_attestation_sha256='e'*64,
            observed_epoch=T0+2,allocation_end_epoch=T0+7202)
        write(parent.parent/'ALLOCATION_ADMISSION.json',admission)
        state_path=parent.parent/'night-session/STATE.json';state_path.parent.mkdir()
        # A live harmless process tests PID reuse checks without running the module.
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
            'scripts.pi0294.night_session',str(parent)])
        start=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21]
        state=dict(job_id='1.server',run_uuid=root.name,parent_request_sha256=sha(parent),
            active_request_sha256=sha(child),status='RUNNING_MAIN',pid=process.pid,proc_start=start)
        write(state_path,state)
        now=T0+3602
        arguments=dict(now=now,user='owner',job_id='1.server',hostname='hopper-1',
            jobs={'1.server':dict(job_state='R',Job_Owner='owner@host',project=PROJECT,
                exec_host='hopper-1/0',Resource_List=dict(walltime='02:00:00',ngpus=1))},
            quota=dict(user='owner',report_epoch=now,hard_bytes=200*GIB,used_bytes=10*GIB),
            project_receipt=dict(project=dict(name=PROJECT,active=True,users=['owner'],
                cluster=['hopper-m-02'],start_date='2026-01-01',end_date='2026-12-31'),
                balance_su=1,observed_epoch=T0+1),
            adoption=dict(pi='PI0294',T0_epoch=T0,new_work_cutoff_epoch=CUTOFF,
                deadline_epoch=DEADLINE,reset_on_resume=False),request=current)
        return parent,child,current,state_path,state,process,arguments

    def test_only_same_live_allocation_can_continue_old_project_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent,child,current,state_path,state,process,arguments=self.fixture(root)
            try:
                with self.assertRaises(PermissionError):validate(**arguments)
                proof=night_parent_proof(root,child,current,job_id='1.server',now=arguments['now'])
                result=validate(**arguments,continued_allocation=proof)
                self.assertEqual(result['project_proof_mode'],'SAME_LIVE_NATIVE_ALLOCATION_INITIAL_ENTITLEMENT')
                self.assertEqual(arguments['project_receipt']['observed_epoch'],T0+1)
                for key,value in [('job_id','2.server'),('project','other'),
                    ('project_attestation_sha256','f'*64),('allocation_end_epoch',arguments['now']),
                    ('initial_admission_epoch',T0+1802)]:
                    with self.subTest(key=key),self.assertRaises(PermissionError):
                        validate(**arguments,continued_allocation=dict(proof,**{key:value}))
                for stage in ('night-final','smoke','benchmark'):
                    with self.subTest(stage=stage),self.assertRaises(PermissionError):
                        validate(**dict(arguments,request=dict(current,stage=stage)),continued_allocation=proof)
                for request in (dict(current,submit_count=5),dict(current,walltime_s=43200)):
                    with self.assertRaises(PermissionError):validate(**dict(arguments,request=request),continued_allocation=proof)
                job=copy.deepcopy(arguments['jobs']);job['1.server']['job_state']='F'
                with self.assertRaises(PermissionError):validate(**dict(arguments,jobs=job),continued_allocation=proof)
                with self.assertRaises(PermissionError):validate(**dict(arguments,now=CUTOFF),continued_allocation=proof)
                quota=dict(arguments['quota'],hard_bytes=arguments['quota']['used_bytes'])
                with self.assertRaises(PermissionError):validate(**dict(arguments,quota=quota),continued_allocation=proof)
            finally:process.terminate();process.wait()

    def test_parent_request_job_and_process_identity_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent,child,current,state_path,state,process,arguments=self.fixture(root)
            try:
                def check(value=current):return night_parent_proof(root,child,value,job_id='1.server',now=arguments['now'])
                for key,value in [('source_sha','f'*40),('bindings_sha256','f'*64),
                    ('project_admission_sha256','f'*64),('request_nonce','other'),
                    ('night_parent_request_relative','../outside')]:
                    with self.subTest(key=key),self.assertRaises(PermissionError):check(dict(current,**{key:value}))
                for key,value in [('job_id','other'),('parent_request_sha256','f'*64),
                    ('active_request_sha256','f'*64),('proc_start','wrong'),('status','WAIT_OWNER_NEXT_SEALED_STAGE')]:
                    write(state_path,dict(state,**{key:value}))
                    with self.subTest(key=key),self.assertRaises(PermissionError):check()
                write(state_path,state);process.terminate();process.wait()
                with self.assertRaises(PermissionError):check()
            finally:
                if process.poll() is None:process.terminate();process.wait()

    def test_incomplete_parent_reference_cannot_fall_back_to_a_new_login_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertIsNone(night_parent_proof(root,root/'unused',{},job_id='fixture'))
            with self.assertRaises(PermissionError):
                night_parent_proof(root,root/'unused',dict(night_parent_request_relative='x'),job_id='fixture')


if __name__=='__main__':unittest.main()
