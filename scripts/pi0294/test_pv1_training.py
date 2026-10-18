import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.fd1 import sha
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.pv1 import training


class TrainingContracts(unittest.TestCase):
    def test_no_pbs_or_wrong_native_identity_rejected_before_data_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'PBS_JOBID':'JOB1'}):
            for allocation,stop in [({},1024),
                    (dict(status='PASS_PV1_OWN_NATIVE_H200_ALLOCATION',job_id='OTHER'),1024),
                    (dict(status='PASS_PV1_OWN_NATIVE_H200_ALLOCATION',job_id='JOB1'),2048)]:
                with self.assertRaises(PermissionError):
                    training.run(tmp,None,allocation,lambda:None,None,arm='A_4MM',request_id='R',stop_after_step=stop)

    def test_exact_replica_seed_data_and_cursor_resume_without_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'training/A_4MM/checkpoint-1024';checkpoint.mkdir(parents=True)
            recipe=dict(seed=20261020)
            sampler=lambda dataset,stop:SimpleNamespace(state_dict=lambda n:dict(cursor=n,seed=20261020,stop=stop))
            dataset=SimpleNamespace(make_sampler=sampler,manifest_sha='FROZEN_BANK_AND_SUPPORT')
            def put(name,value):
                p=checkpoint/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            for name in ['config.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors','processor/config.json','statistics.json']:
                put(name,dict(fixture=True))
            put('trainer_state.json',dict(global_step=1024))
            put('SAMPLER_STATE.json',dict(recipe=recipe,sampler=sampler(dataset,4096*32).state_dict(1024*32),
                manifest_sha256=dataset.manifest_sha,statistics_sha256=STATS))
            files={str(p.relative_to(checkpoint)):STATS if p.name=='statistics.json' else sha(p)
                for p in checkpoint.rglob('*') if p.is_file()}
            put('CHECKPOINT_COMPLETE.json',dict(full_resume=True,global_step=1024,files=files))
            def digest(p):return STATS if Path(p).name=='statistics.json' else sha(p)
            with patch.object(training,'sha',side_effect=digest):
                self.assertEqual(training.checked_resume(root,'A_4MM',checkpoint,recipe,dataset),1024)
                with self.assertRaises(PermissionError):training.checked_resume(root,'B_12MM',checkpoint,recipe,dataset)
                with self.assertRaises(PermissionError):training.checked_resume(root,'A_4MM',checkpoint,dict(seed=20261019),dataset)
                (checkpoint/'optimizer.pt').write_text('CHANGED')
                with self.assertRaises(ValueError):training.checked_resume(root,'A_4MM',checkpoint,recipe,dataset)

    def test_control_interruption_saves_only_a_known_completed_step_once_never_replays_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'ledger.json';sync=[];saves=[]
            value=dict(pending_updates=[],unknown_effects=[],training_runs=dict(A_4MM=dict(last_native_step=3)))
            path.write_text(json.dumps(value));ledger=SimpleNamespace(path=path)
            checkpoint=root/'training/A_4MM/checkpoint-3'
            def save(model,trial):
                saves.append((model,trial));checkpoint.mkdir(parents=True)
                (checkpoint/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(dict(global_step=3,full_resume=True)))
                (checkpoint/'SAMPLER_STATE.json').write_text(json.dumps(dict(sampler=dict(cursor=96))))
            trainer=SimpleNamespace(state=SimpleNamespace(global_step=3),consumed=96,model='CPU_FIXTURE_NOT_MODEL',_save_checkpoint=save)
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:sync.append(True)))
            result=training.save_known_completed_step(root,'A_4MM',trainer,ledger,torch)
            self.assertEqual(result['extra_optimizer_updates'],0);self.assertEqual(len(saves),1);self.assertEqual(len(sync),1)
            again=training.save_known_completed_step(root,'A_4MM',trainer,ledger,torch)
            self.assertEqual(again['status'],'PRIOR_COMPLETE_STATE_PRESERVED_NO_REWRITE');self.assertEqual(len(saves),1)

    def test_pending_native_update_or_partial_save_is_never_overwritten_or_called_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'ledger.json';calls=[]
            value=dict(pending_updates=[dict(fixture=True)],unknown_effects=[],training_runs=dict(A_4MM=dict(last_native_step=3)))
            path.write_text(json.dumps(value));ledger=SimpleNamespace(path=path)
            trainer=SimpleNamespace(state=SimpleNamespace(global_step=3),consumed=96)
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:calls.append(True)))
            self.assertEqual(training.save_known_completed_step(root,'A_4MM',trainer,ledger,torch)['status'],
                'NOT_SAVED_UNKNOWN_OR_UNCOMMITTED_UPDATE_PRESERVED')
            value['pending_updates']=[];path.write_text(json.dumps(value))
            (root/'training/A_4MM/checkpoint-3').mkdir(parents=True)
            self.assertEqual(training.save_known_completed_step(root,'A_4MM',trainer,ledger,torch)['status'],'NOT_OVERWRITTEN_PRIOR_PARTIAL_SAVE')
            self.assertEqual(calls,[])
