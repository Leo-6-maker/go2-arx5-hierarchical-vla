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
