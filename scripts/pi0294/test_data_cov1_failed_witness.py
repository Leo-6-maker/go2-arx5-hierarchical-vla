"""Saved-byte fixtures do not score, construct engines or run physics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pi0294.data_cov1.plan import candidate,capture_cards,slots
from scripts.pi0294.data_cov1.failed_witness import score_input_failure_witness
from scripts.pi0294.fd1 import sha


def fixture(root):
    card=capture_cards([candidate(**s,index=0) for s in slots()],'P24')[0]
    def put(name,value):
        (root/name).write_text(json.dumps(value))
    matrix=np.eye(4).tolist();samples=[]
    for camera in ('body_rgb','wrist_rgb'):
        (root/camera).mkdir()
        for i in range(641):(root/camera/f'{i:04d}.png').write_bytes(b'CPU fixture, not a physical image')
    for i in range(641):
        obs=dict(timestamp_s=i*.05,instruction=card['instruction'])
        for camera in ('body_rgb','wrist_rgb'):
            name=f'{camera}/{i:04d}.png';obs[camera]=dict(path=name,sha256=sha(root/name))
        label=None if i==640 else dict(ref_id='r',point_index=i%40,actually_committed=True,valid=True,
            desired=matrix,accepted=matrix,installed=matrix,committed=matrix,committed_gripper=.5)
        samples.append(dict(policy_obs=obs,supervision=label))
    put('samples.json',samples);put('references.json',[dict(ref_id='r',planned_H40=[matrix]*40,gripper_H40=[.5]*40)])
    put('CARD.json',card)
    put('RESULT.json',dict(id=card['id'],status='FAILED_PRESERVED',error=None,source_sha='CPU_FIXTURE',
        score_error='ValueError: source-goal separation must be at least 0.15 m',physics_steps=16000,rgb_pairs=641))
    put('WORKER_RECEIPT.json',dict(source_sha='CPU_FIXTURE',physics_steps=16000,samples_sha256=sha(root/'samples.json')))
    put('S1_RESULT.json',dict(live_native_failures=0,pre_execution_rejections=0,actions_modified=False,
        shadow_integrations=16000,maximum_live_shadow_difference=0.))
    np.savez(root/'highrate.npz',rows=np.arange(1,16001)[:,None]*.002,external_force_max=np.zeros((16000,2)))
    np.savez(root/'physics.npz',time_s=np.arange(641)*.05)
    put('FILE_MANIFEST.json',[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))
        for p in root.rglob('*') if p.is_file()])
    return card


class FailedScoreWitness(unittest.TestCase):
    def test_identity_clock_witness_does_not_invent_a_score_or_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);card=fixture(root);before=sha(root/'RESULT.json')
            report=score_input_failure_witness(root,card)
            self.assertTrue(report['auditable_identity_and_clock'])
            for k in ('task_success','task_success_known','admitted','valid_task_failure'):self.assertFalse(report[k])
            self.assertEqual(report['new_physics'],0);self.assertEqual(sha(root/'RESULT.json'),before)
            (root/'body_rgb/0001.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):score_input_failure_witness(root,card)

    def test_missing_safety_or_wrong_score_error_is_not_an_authorized_witness(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);card=fixture(root)
            value=json.loads((root/'RESULT.json').read_text());value['score_error']='other failure'
            (root/'RESULT.json').write_text(json.dumps(value))
            with self.assertRaises(PermissionError):score_input_failure_witness(root,card)


if __name__=='__main__':unittest.main()
