"""Transfer security plus optional actual closed A/B metadata regression."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from scripts.pi0294.pv1 import evaluation_transfer as t


class MemberTests(unittest.TestCase):
    def fixture(self):
        files={n:'1'*64 for n in (*t.SMALL,'model-00001-of-00001.safetensors',
            'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
            'training_args.bin','SAMPLER_STATE.json','extra-a.json','extra-b.json')}
        files['statistics.json']=t.STATS
        return dict(global_step=4096,full_resume=True,files=files),dict(weight_map={'weight':'model-00001-of-00001.safetensors'})

    def test_exact_subset_has_no_training_state(self):
        complete,index=self.fixture();result=t.model_members(complete,index)
        self.assertEqual(len(result),7)
        self.assertFalse({'optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin'}&set(result))

    def test_bad_indices_and_original_identity_rejected(self):
        complete,index=self.fixture()
        mutations=[('escape',lambda c,i:i['weight_map'].update(weight='../model-00001-of-00001.safetensors')),
            ('missing',lambda c,i:i.update(weight_map={})),
            ('unknown',lambda c,i:i['weight_map'].update(weight='model-00002-of-00002.safetensors')),
            ('traversal_member',lambda c,i:c['files'].update({'../x':'1'*64})),
            ('not_terminal',lambda c,i:c.update(global_step=2048)),
            ('boolean_step',lambda c,i:c.update(global_step=True)),
            ('not_complete',lambda c,i:c.update(full_resume=False)),
            ('different_R0',lambda c,i:c['files'].update({'statistics.json':'2'*64})),
            ('missing_optimizer_history',lambda c,i:c['files'].pop('optimizer.pt'))]
        for label,mutate in mutations:
            with self.subTest(label=label):
                c,i=deepcopy(complete),deepcopy(index);mutate(c,i)
                with self.assertRaises(PermissionError):t.model_members(c,i)

    def test_other_candidate_and_untrusted_capture_rejected(self):
        with self.assertRaises(PermissionError):t.prepare(b'{}',b'{}',candidate='REPLICA_A_CONTROL')
        with self.assertRaises(PermissionError):t.prepare(b'{}',b'{}',candidate='A_CONTROL')


class CapturedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=Path(__file__).resolve().parents[4]/'work/private/science-next/wdr1-target-identity'
        endpoint=cls.base/'ACTUAL_V4A_BOTH_SAVED4096_ENDPOINT_METADATA_1792831093566152235.private.json'
        small=cls.base/'v4a-core-eval-preparation-01/VANDA_SMALL_CHECKPOINTS.private.json'
        if not endpoint.is_file() or not small.is_file():raise unittest.SkipTest('private actual captured bytes absent')
        cls.endpoint=endpoint.read_bytes();cls.small=small.read_bytes()

    def test_actual_AB_public_plans_and_exact_two_metadata_restorations(self):
        for candidate in ('A_CONTROL','B_COVERAGE'):
            with self.subTest(candidate=candidate):
                result=t.prepare(self.endpoint,self.small,candidate=candidate)
                public=result['public_plan'];private=result['private_restoration']
                self.assertEqual(public['payload_bytes'],12576324408)
                self.assertEqual(len(public['files']),9)
                self.assertEqual(set(private['original_metadata_base64']),set(t.REWRITTEN))
                self.assertFalse(public['full_resume_payload']);self.assertFalse(result['HF_IO_performed'])
                complete=json.loads(base64.b64decode(private['original_complete_base64']))
                for n,encoded in private['original_metadata_base64'].items():
                    self.assertEqual(t.sha(base64.b64decode(encoded)),complete['files'][n])
                    self.assertNotEqual(public['files'][n]['sha256'],complete['files'][n])
                for n,v in public['files'].items():
                    if n not in t.REWRITTEN:self.assertEqual(v['sha256'],complete['files'][n])
                t.reject_private(public)

    def test_actual_tamper_candidate_swap_and_missing_file_fail(self):
        for candidate in ('A_CONTROL','B_COVERAGE'):
            actual=json.loads(self.small)
            for label,change in [
                ('cross_arm',lambda d:d['value']['endpoints'][candidate].update(checkpoint_sha256=t.ENDPOINTS['B_COVERAGE' if candidate=='A_CONTROL' else 'A_CONTROL'])),
                ('changed_metadata',lambda d:d['value']['endpoints'][candidate]['files']['config.json'].update(base64=base64.b64encode(b'{}').decode())),
                ('missing_metadata',lambda d:d['value']['endpoints'][candidate]['files'].pop('processor/statistics.json')),
                ('different_source',lambda d:d.update(endpoint_metadata_sha256='0'*64))]:
                with self.subTest(candidate=candidate,reason=label):
                    changed=deepcopy(actual);change(changed)
                    with self.assertRaises(PermissionError):t.prepare(self.endpoint,t.encode(changed),candidate=candidate)

    def test_restore_refuses_incomplete_download_and_wrong_arm(self):
        result=t.prepare(self.endpoint,self.small,candidate='A_CONTROL')
        private=result['private_restoration']
        with self.assertRaises(PermissionError):t.restore_original({},private,candidate='A_CONTROL')
        with self.assertRaises(PermissionError):t.restore_original({},private,candidate='B_COVERAGE')


if __name__=='__main__':unittest.main()
