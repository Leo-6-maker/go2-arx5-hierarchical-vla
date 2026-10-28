"""CPU fixtures only: final-allocation stage identity and original clock."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.pi0292.durable import read,write
from .allocation_admission import validate,T0,CUTOFF,DEADLINE,GIB,PROJECT
from .native_dataset import sha
from .neural import dev_panel,smoke_output_root
from .night_session import validate_stage,seal_single_measured_profile


class NightContracts(unittest.TestCase):
    def test_stage_cannot_change_source_data_recipe_or_reopen_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=dict(root=str(root),source_sha='a'*40,run_manifest_sha256='b'*64,
                code_seal_sha256='c'*64,bindings='private',bindings_sha256='d'*64,
                walltime_s=7200,submit_count=4,prior_GPU_allocated_s=489,
                numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',microbatch=2,workers=0)
            request=dict(original,mode='main',stage='main',allowed_modes=['main'],
                max_steps=4096,stop_at=512)
            path=root/'request.json';write(path,request)
            stage=dict(run_uuid=root.name,job_id='1.server',sequence=0,mode='main',
                request_relative='request.json',request_sha256=sha(path))
            def check():return validate_stage(root,original,stage,job_id='1.server',sequence=0)
            self.assertEqual(check(),path)
            for key,value in [('source_sha','changed'),('bindings_sha256','changed'),
                ('submit_count',5),('walltime_s',7201),('microbatch',4),('workers',2),
                ('numerical_mode',None),('allowed_modes',['main','smoke-all']),('stop_at',1024)]:
                with self.subTest(key=key):
                    write(path,dict(request,**{key:value}));stage['request_sha256']=sha(path)
                    with self.assertRaises(PermissionError):check()
            write(path,request);stage['request_sha256']=sha(path)
            with self.assertRaises(PermissionError):validate_stage(root,original,stage,job_id='2.server',sequence=0)
            with self.assertRaises(PermissionError):validate_stage(root,original,stage,job_id='1.server',sequence=1)
            stage['request_relative']='../request.json'
            with self.assertRaises(PermissionError):check()

    def test_exact_remaining_eight_numerical_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proof=root/'proof.json'
            value=dict(status='KNOWN_TERMINAL_NUMERICAL_SMOKE_REQUALIFICATION_8_REMAINING',
                run_uuid=root.name,actual_optimizer_updates=8,prior_job_terminal=True)
            write(proof,value)
            request=dict(smoke_output_relative='smoke_attempts/strict-new',mode='night-final',
                numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',prior_engineering_updates=8,
                engineering_retry_receipt=str(proof),engineering_retry_receipt_sha256=sha(proof))
            self.assertEqual(smoke_output_root(root,request),root/'smoke_attempts/strict-new')
            for key,value in [('mode','smoke-all'),('numerical_mode',None),('prior_engineering_updates',0)]:
                with self.subTest(key=key):
                    with self.assertRaises(PermissionError):smoke_output_root(root,dict(request,**{key:value}))

    def test_only_actual_microbatch_two_is_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence').mkdir()
            write(root/'evidence/MODEL_SMOKE.json',dict(status='TRAINER_MODEL_INTERFACE_QUALIFIED'))
            path=root/'training/smoke_continuous/session-start/TRAINING_RUN.json'
            path.parent.mkdir(parents=True)
            write(path,dict(global_step=4,recipe=dict(per_device_train_batch_size=2,
                numerical_configuration={'fixture':True}),actual_synchronized_optimizer_step_s=[1.]*4,
                actual_optimizer_steps_per_s=1.,peak_gpu_allocated_bytes=100,
                manifest_sha256='a'*64,statistics_sha256='b'*64))
            seal_single_measured_profile(root,dict(workers=0))
            seal=read(root/'evidence/TRAINING_PROFILE_SEAL.json')
            self.assertEqual(seal['untested_microbatches'],[4,8])
            self.assertEqual(seal['benchmark_updates'],0)
            self.assertEqual(seal['engineering_updates_total'],16)
            with self.assertRaises(FileExistsError):seal_single_measured_profile(root,dict(workers=0))

    def test_paired_dev_only_language_changes_denominator(self):
        rows=[dict(task=task,layout_id=str(group),attempted=True,
            proof_kind='REAL_CORELAB_MODEL_CLOSED_LOOP',result_sha256='a'*64,
            physics_sha256='b'*64,checkpoint_identity_sha256='c'*64,
            persistent_success=True,safety_valid=True,rejected_prefixes=0,total_prefixes=80)
            for task in ('T1','T2') for group in range(4)]
        self.assertEqual(dev_panel(rows)['successes'],{'T1':4,'T2':4})
        bad=copy.deepcopy(rows);bad[-1]['layout_id']='other'
        with self.assertRaises(ValueError):dev_panel(bad)
        with self.assertRaises(ValueError):dev_panel(rows[:-1]+[rows[0]])

    def test_closeout_only_resident_not_training_or_new_smoke(self):
        now=CUTOFF+1
        arguments=dict(now=now,user='owner',job_id='1.server',hostname='hopper-1',
            jobs={'1.server':dict(job_state='R',Job_Owner='owner@host',project=PROJECT,
                exec_host='hopper-1/0',Resource_List=dict(walltime='02:00:00',ngpus=1))},
            quota=dict(user='owner',report_epoch=now,hard_bytes=200*GIB,used_bytes=10*GIB),
            project_receipt=dict(project=dict(name=PROJECT,active=True,users=['owner'],
                cluster=['hopper-m-02'],start_date='2026-01-01',end_date='2026-12-31'),
                balance_su=1,observed_epoch=now),
            adoption=dict(pi='PI0294',T0_epoch=T0,new_work_cutoff_epoch=CUTOFF,
                deadline_epoch=DEADLINE,reset_on_resume=False),
            request=dict(stage='resident',submit_count=4,prior_GPU_allocated_s=489,
                walltime_s=7200,request_nonce='c9e9f761-f1e8-4cf4-9851-68adb68d1f21',
                allowed_peer_ids=[],predicted_remaining_peak_bytes=GIB))
        self.assertIn('PASS',validate(**arguments)['status'])
        for stage in ('main','smoke','night-final','benchmark'):
            with self.subTest(stage=stage):
                with self.assertRaises(PermissionError):validate(**dict(arguments,request=dict(arguments['request'],stage=stage)))
        with self.assertRaises(PermissionError):validate(**dict(arguments,now=DEADLINE))
        with self.assertRaises(PermissionError):validate(**dict(arguments,now=T0-1))


if __name__=='__main__':unittest.main()
