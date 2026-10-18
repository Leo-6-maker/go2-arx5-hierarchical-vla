"""Fixed NI16 transport never generates a new draw or broadens the file set."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pi0294.mp1 import endpoint_ni_delivery as d
from scripts.pi0294.native_dataset import sha


class DeliveryTests(unittest.TestCase):
    def fixture(self):
        native={i:dict(actual_native_noise_fingerprint=dict(shape=[1,40,132],dtype='torch.float32',sha256='a'*64),
            noise_seed=n,query_uuid='q'+str(n),observation_fingerprint='b'*64)
            for n,i in enumerate(sorted(d.ni.selected_ids()))}
        rows=[]
        for model in d.ni.CHECKPOINTS:
            for identifier,old in native.items():
                rows.append(dict(id=identifier,model=model,noise=old['actual_native_noise_fingerprint'],
                    noise_seed=old['noise_seed'],original_query_uuid=old['query_uuid'],
                    observation_fingerprint=old['observation_fingerprint'],checkpoint_manifest_sha256=d.ni.CHECKPOINTS[model],
                    inference_precision=dict(d.PRECISION),new_query=True,live_or_shadow_integrations=0,arrays_sha256='c'*64))
        value=dict(status='COMPLETE_SIXTEEN_SAME_INPUT_NOISE_ENDPOINT_DRAWS_NOT_POLICY_EVALUATION',records=rows,
            new_neural_queries=16,new_physics=0,new_optimizer_updates=0,closed_loop_or_task_success_claimed=False,
            original_eight_BC512_draws_reused=True,native_history_gate_result_sha256=d.ni.GATE_RESULT)
        return value,dict(native=native)

    def test_fixed_sixteen(self):
        v,b=self.fixture();self.assertEqual(len(d.checked_rows(v,b)),16)
        self.assertEqual(len({d.filename(r['model'],r['id']) for r in v['records']}),16)

    def test_changed_scope_or_identity_rejected(self):
        changes=[('new_neural_queries',17),('new_physics',1),('new_optimizer_updates',1),
            ('closed_loop_or_task_success_claimed',True),('original_eight_BC512_draws_reused',False)]
        for key,value in changes:
            with self.subTest(key=key):
                v,b=self.fixture();v[key]=value
                with self.assertRaises(PermissionError):d.checked_rows(v,b)
        for field,value in [('noise_seed',999),('observation_fingerprint','x'),('checkpoint_manifest_sha256','x'),
                ('new_query',False),('live_or_shadow_integrations',1),('noise',dict(sha256='x'))]:
            with self.subTest(field=field):
                v,b=self.fixture();v['records'][0][field]=value
                with self.assertRaises(PermissionError):d.checked_rows(v,b)

    def test_duplicate_and_private_metadata_rejected(self):
        v,b=self.fixture();v['records'][0]=deepcopy(v['records'][1])
        with self.assertRaises(PermissionError):d.checked_rows(v,b)
        v,b=self.fixture();v['private']='/scratch/user/file'
        with self.assertRaises(ValueError):d.checked_rows(v,b)
        for model,identifier in [('M0','x'),('S-BC-2048','../x')]:
            with self.assertRaises(PermissionError):d.filename(model,identifier)

    def test_actual_array_identity_and_no_pickle(self):
        v,b=self.fixture();row=v['records'][0]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'draw.npz'
            np.savez(p,physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132)),
                noise_seed=row['noise_seed'],noise_fingerprint_json=json.dumps(row['noise']),
                inference_precision_json=json.dumps(row['inference_precision']),
                observation_fingerprint=row['observation_fingerprint'],checkpoint_files_fingerprint='weights')
            row['arrays_sha256']=sha(p);d.checked_array(p,row,'weights')
            with self.assertRaises(ValueError):d.checked_array(p,row,'other-weights')
            row['noise_seed']+=1
            with self.assertRaises(ValueError):d.checked_array(p,row,'weights')


if __name__=='__main__':unittest.main()
