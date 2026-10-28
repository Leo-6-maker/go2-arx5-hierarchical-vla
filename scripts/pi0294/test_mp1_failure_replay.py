import copy
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .mp1 import failure_config as config
from .mp1.science import Ledger
from .mp1.failure_replay import verify_images


def plan():
    return dict(selection='LAST_REJECTED_PREFIX_INPUT_AND_PREVIOUS_ACCEPTED_PREFIX_INPUT',
        new_queries=0,new_live_episodes=0,cases=[dict(attempt_id=i,sequences=config.selected_sequences(i),
            original_result_sha256=v['result_sha256'],main_steps=v['main_steps'],
            original_rejection_offset_ticks=v['offset']) for i,v in config.ATTEMPTS.items()])


class FailureReplayTests(unittest.TestCase):
    def test_only_four_fixed_existing_failures_and_eight_noninitial_states(self):
        p=plan();cases=config.validate_plan(p)
        self.assertEqual(len(cases),4);self.assertEqual(sum(len(c['sequences']) for c in cases),8)
        self.assertTrue(all(s>0 for c in cases for s in c['sequences']))
        self.assertEqual(sum(c['main_steps'] for c in cases),27600)
        self.assertEqual(sum(v['offset'] for v in config.ATTEMPTS.values()),401)
        self.assertEqual(config.CAPS['shadow_integrations'],27600+4*200)

    def test_duplicate_extra_or_missing_case_never_dispatches(self):
        p=plan()
        for cases in [p['cases'][:-1],p['cases']+[p['cases'][0]],[p['cases'][0]]*4]:
            with self.subTest(cases=len(cases)),self.assertRaises(PermissionError):
                config.validate_plan(dict(p,cases=cases))

    def test_no_new_query_live_repeat_or_arbitrary_anchor(self):
        p=plan()
        for key in ['new_queries','new_live_episodes']:
            with self.assertRaises(PermissionError):config.validate_plan(dict(p,**{key:1}))
        for key,value in [('sequences',[0,4]),('main_steps',16000),('original_rejection_offset_ticks',200),('original_result_sha256','a'*64)]:
            changed=copy.deepcopy(p);changed['cases'][0][key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):config.validate_plan(changed)

    def test_ledger_refuses_gpu_models_training_recovery_and_live_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'MP1_STAGE_BINDING.json').write_text(json.dumps(dict(schema=config.SCHEMA,stage_caps=config.CAPS,CPU_core_s=config.CPU_CAPS)))
            (root/'RUN_MANIFEST.json').write_text('{}')
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(main_starts=0,live_and_recovery_integrations=0,shadow_integrations=0,reservations={})))
            ledger=Ledger(root,'core')
            for key in ['optimizer_updates','generation_queries','recovery_forks']:
                with self.subTest(key=key),self.assertRaises(PermissionError):ledger.charge(key,1)
            with self.assertRaises(PermissionError):
                with ledger.steps('live_and_recovery_integrations',1,'forbidden'):pass
            ledger.charge('main_starts',4)
            with self.assertRaises(PermissionError):ledger.charge('main_starts',1)
            with ledger.steps('shadow_integrations',28400,'bounded') as meter:meter['actual']=28001
            value=json.loads((root/'RESOURCE_LEDGER.json').read_text())
            self.assertEqual(value['shadow_integrations'],28001);self.assertEqual(value['reservations'],{})
            with self.assertRaises(PermissionError):
                with ledger.steps('shadow_integrations',400,'excess'):pass

    def test_counter_caps_cannot_be_extended_by_stage_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);caps=dict(config.CAPS,generation_queries=24)
            (root/'MP1_STAGE_BINDING.json').write_text(json.dumps(dict(schema=config.SCHEMA,stage_caps=caps)))
            with self.assertRaisesRegex(PermissionError,'EXACT_SOURCE_STUDY_COUNTERS'):Ledger(root,'core')

    def test_original_model_camera_bytes_not_rerendered_or_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);raw=b'original-saved-camera-fixture';digest=hashlib.sha256(raw).hexdigest();images=[]
            for wire,camera in [('external_rgb','body_rgb'),('wrist_rgb','wrist_rgb')]:
                p=folder/camera/'0008.png';p.parent.mkdir();p.write_bytes(raw)
                images.append(dict(camera=wire,png_base64=base64.b64encode(raw).decode(),sha256=digest))
            body=dict(tick=200,payload=dict(images=images));verify_images(body,folder)
            changed=copy.deepcopy(body);changed['payload']['images'][1]['camera']='external_rgb'
            with self.assertRaisesRegex(ValueError,'DOUBLE_CAMERA'):verify_images(changed,folder)
            (folder/'body_rgb/0008.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'RENDERED_CAMERA_EQUALS'):verify_images(body,folder)


if __name__=='__main__':unittest.main()
