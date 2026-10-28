import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.pi0292.eval_contract import FRAME, TCP, gate_action
from scripts.pi0292.relay_protocol import PROTOCOL, digest, sealed
from scripts.pi0294.native_dataset import sha
from .mp1.probe_compare import PRECISION
from .mp1.student_probe_audit import audit


def fixture(folder):
    states, records, originals = [], [], {}
    stream = io.BytesIO()
    Image.new('RGB', (256,256), (120,30,90)).save(stream, format='PNG')
    raw = stream.getvalue()
    images = [dict(camera=c, png_base64=base64.b64encode(raw).decode(),
                   sha256=hashlib.sha256(raw).hexdigest()) for c in ['external_rgb','wrist_rgb']]
    state = [0.,0.,.5,1.,0.,0.,0.,1.,0.,1.]
    physical = np.tile(state, (40,1))
    precision = dict(PRECISION, training_flag_on_entry=True)
    noise = dict(sha256='a'*64, shape=[1,40,132], dtype='torch.float32')
    for parent in range(4):
        task = 'T1' if parent%2==0 else 'T2'
        episode = 'parent-'+str(parent)
        for sequence in range(3):
            identifier = 'STATE-TRAIN-'+str(parent)+'-S'+str(sequence)
            payload = dict(snapshot_uuid='00000000-0000-0000-0000-000000000001',
                state_eef10=state, language='move object', images=images, frame=FRAME,tcp=TCP)
            header = dict(protocol=PROTOCOL, run_uuid='old-E0', episode_uuid=episode,
                          sequence=sequence,tick=200*sequence)
            obs = sealed(dict(header,kind='observation',payload=payload))
            request = sealed(dict(header,kind='student_request',payload=payload,
                query_uuid=identifier,anchor_sha256=digest(state),observation_sha256=obs['sha256']))
            action = sealed(dict(header,kind='action',payload=dict(query_uuid=identifier,
                observation_sha256=obs['sha256'],trajectory_eef10=physical.tolist())))
            originals[identifier] = dict(request=request,action=action,
                receipt=dict(seed=sequence,noise=noise,action_sha256=action['sha256']))
            states.append(dict(id=identifier,task=task,original_episode_uuid=episode,
                sequence=sequence,tick=200*sequence,payload=payload,observation_sha256=obs['sha256'],
                original_action_sha256=action['sha256'],M0_physical_H40=physical.tolist(),noise_seed=sequence))
            path = folder/(identifier+'.npz')
            np.savez(path,physical_H40=physical,noise_seed=np.asarray(sequence,dtype=np.uint64),
                noise_fingerprint_json=np.asarray(json.dumps(noise)),
                inference_precision_json=np.asarray(json.dumps(precision)),
                checkpoint_files_fingerprint=np.asarray('b'*64),
                observation_fingerprint=np.asarray(digest({k:v for k,v in payload.items() if k!='snapshot_uuid'})))
            records.append(dict(id=identifier,new_query=True,noise=noise,noise_seed=sequence,
                physical_H40=physical.tolist(),inference_precision=precision,arrays_sha256=sha(path),
                gate=gate_action(physical,state,sequence*.4,identifier)))
    (folder/'ON_STUDENT_RESULTS.private.json').write_text(json.dumps(dict(
        status='FIXED_12_TRAIN_OBSERVATIONS_ONLY_NOT_LIVE_OR_TEACHER_LABELS',records=records)))
    return dict(states=states), originals


class StudentProbeTest(unittest.TestCase):
    def test_actual_identities_and_missing_label_or_S1_stay_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,originals=fixture(root)
            result=audit(root,plan,'b'*64,originals)
            self.assertEqual(result['probe_states'],12)
            self.assertEqual(result['independent_historical_student_parents'],4)
            self.assertEqual(result['task_probe_states'],dict(T1=6,T2=6))
            self.assertTrue(result['historical_E0_not_current_S_BC_induced_states'])
            self.assertEqual(sum(r['initial_anchor_old_label_exclusion_preserved'] for r in result['rows']),4)
            self.assertTrue(all(r['current_model_original_S1'].startswith('NOT_RUN') for r in result['rows']))

    def test_changed_actual_source_payload_noise_or_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,originals=fixture(root)
            changed=copy.deepcopy(plan);changed['states'][0]['payload']['language']='different'
            with self.assertRaises(ValueError):audit(root,changed,'b'*64,originals)
            changed=copy.deepcopy(originals);first=next(iter(changed));changed[first]['receipt']['noise']['sha256']='f'*64
            with self.assertRaises(ValueError):audit(root,plan,'b'*64,changed)
            with self.assertRaises(ValueError):audit(root,plan,'c'*64,originals)

    def test_missing_duplicate_dev_or_bad_clock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,originals=fixture(root)
            for kind in ['missing','duplicate','dev','clock']:
                changed=copy.deepcopy(plan)
                if kind=='missing':changed['states'].pop()
                elif kind=='duplicate':changed['states'][1]=changed['states'][0]
                elif kind=='dev':changed['states'][0]['id']='STATE-DEV-0'
                else:changed['states'][0]['tick']=25
                with self.subTest(kind=kind),self.assertRaises((ValueError,PermissionError)):
                    audit(root,changed,'b'*64,originals)

    def test_tampered_gate_and_array_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,originals=fixture(root);p=root/'ON_STUDENT_RESULTS.private.json'
            raw=p.read_text();value=json.loads(raw);value['records'][0]['gate']['decision']='REJECT';p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'GATE_RECOMPUTATION'):audit(root,plan,'b'*64,originals)
            p.write_text(raw);path=root/(plan['states'][0]['id']+'.npz');target=root/'saved.npz';path.rename(target);path.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'ARRAY_HASH_OR_ESCAPE'):audit(root,plan,'b'*64,originals)


if __name__=='__main__':unittest.main()
