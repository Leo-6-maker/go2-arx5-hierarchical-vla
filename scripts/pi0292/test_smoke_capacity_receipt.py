import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .checkpoint_artifacts import REQUIRED
from .smoke_capacity_receipt import build
from . import test_checkpoint_capacity as checkpoint_fixtures
from .train_adapter import CyclicSampler, sha


class TestBoundSmokeCapacityReceipt(unittest.TestCase):
    def setUp(self):
        self.fixture=checkpoint_fixtures.TestCheckpointInventory();self.fixture.setUp()
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.recipe=dict(max_steps=4,effective_batch=32,workers=2,learning_rate=1e-4,seed=29200,eval_strategy='no')
        self.put('ADOPTION.json',dict(T0_epoch=1791860304,deadline_epoch=1791946704,
            new_work_cutoff_epoch=1791943104,reset_on_resume=False))
        self.put('evidence/HOPPER_DATA_QUALIFICATION.json',dict(status='PASS_BANK_V2_FULL_DATA_AND_CPU_INTERFACE',
            statistics_sha256=self.fixture.stats,manifest_sha256='m'*64))
        self.put('evidence/OPEN_LOOP_PROBE_SEAL.json',dict(status='SEALED_BEFORE_MAIN'))
        self.put('evidence/RESUME_EQUIVALENCE.json',dict(status='PASS_FIXED_RESUME_EQUIVALENCE'))
        self.put('evidence/MODEL_SMOKE.json',dict(status='TRAINER_MODEL_INTERFACE_QUALIFIED',
            equivalence_sha256=sha(self.root/'evidence/RESUME_EQUIVALENCE.json')))
        self.put('evidence/PBS_SMOKE_S3-smoke-02.json',dict(status='SMOKE_COMPLETE_NO_MAIN_STARTED',steps=[
            dict(mode=mode,wall_s=50,exit=0) for mode in ('smoke-continuous','smoke-first','smoke-resume')]))
        self.put('HOPPER_RESOURCE_LEDGER.json',dict(main_updates=0,model_smoke_updates=8,GPU_allocated_s=200,
            CPU_s=1000,CPU_upper_s=660,stages=[],PBS_submissions=[dict(status='SUBMIT_FAILED_PRESERVED'),dict(status='RECONCILED_TERMINAL')]))
        for relative,step in [('smoke_continuous/checkpoint-4',4),('smoke_split/checkpoint-2',2),('smoke_split/checkpoint-4',4)]:
            path=self.root/'training'/relative;shutil.copytree(self.fixture.root,path)
            self.put(str(path.relative_to(self.root)/'trainer_state.json'),dict(global_step=step))
            self.put(str(path.relative_to(self.root)/'SAMPLER_STATE.json'),dict(recipe=self.recipe,
                sampler=CyclicSampler(601,stop=128).state_dict(step*32),statistics_sha256=self.fixture.stats,manifest_sha256='m'*64,
                worker_trace=dict(status='PASS_ACTUAL_CONSUMED_WORKER_IDENTITIES',start=0,stop=step*32,count=step*32,prefetch_excluded=True)))
            self.put(str(path.relative_to(self.root)/'CHECKPOINT_COMPLETE.json'),dict(global_step=step,status='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING',
                required_files={name:sha(path/name) for name in REQUIRED}))
        for session,step in [('smoke_continuous/session-start',4),('smoke_split/session-start',2),('smoke_split/session-resume',4)]:
            self.put('training/'+session+'/TRAINING_RUN.json',dict(global_step=step,recipe=self.recipe,
                frozen_backbone_unchanged=True,changed_trainable_parameters=['SYNTHETIC_ONLY']))
        path=self.root/'training/smoke_continuous/session-start/OPTIMIZER_STEPS.jsonl'
        path.write_text('\n'.join(json.dumps(dict(step=i,cursor=i*32,epoch=100+i*2)) for i in range(1,5))+'\n')
        self.arguments=dict(owned_bytes=16*1024**3,filesystem_free_bytes=100*1024**3,quota_free_bytes=100*1024**3,
            extra_peak_bytes=8*1024**3,now=1791884000)

    def tearDown(self):
        self.fixture.tearDown();self.temp.cleanup()

    def put(self,relative,value):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_complete_chain_is_hash_bound_but_not_execution_authority(self):
        result=build(self.root,**self.arguments)
        self.assertFalse(result['authorizes_execution'])
        self.assertEqual(len(result['actual_checkpoint_inventories']),3)
        self.assertEqual(len(result['source_receipt_hashes']),9)
        self.assertTrue(result['main_and_eval_share_next_allocation'])

    def test_changed_equivalence_and_reset_are_refused(self):
        for name in ('equivalence','clock'):
            with self.subTest(name=name):
                relative='evidence/RESUME_EQUIVALENCE.json' if name=='equivalence' else 'ADOPTION.json'
                path=self.root/relative;before=path.read_text();value=json.loads(before)
                if name=='equivalence':value['changed']=True
                else:value['reset_on_resume']=True
                self.put(relative,value)
                with self.assertRaises(PermissionError):build(self.root,**self.arguments)
                path.write_text(before)

    def test_unreconciled_allocation_and_incomplete_checkpoint_are_refused(self):
        path=self.root/'HOPPER_RESOURCE_LEDGER.json';value=json.loads(path.read_text())
        value['PBS_submissions'][1]['status']='RUNNING';self.put('HOPPER_RESOURCE_LEDGER.json',value)
        with self.assertRaisesRegex(ValueError,'NOT_RECONCILED'):build(self.root,**self.arguments)
        value['PBS_submissions'][1]['status']='RECONCILED_TERMINAL';self.put('HOPPER_RESOURCE_LEDGER.json',value)
        (self.root/'training/smoke_split/checkpoint-2/optimizer.pt').unlink()
        with self.assertRaisesRegex(ValueError,'FILE_MISSING_OR_CHANGED'):build(self.root,**self.arguments)


if __name__=='__main__':unittest.main()
