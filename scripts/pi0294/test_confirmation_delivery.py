import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.pi0292.durable import write
from scripts.pi0294.mp1 import confirmation_delivery as d
from scripts.pi0294.scene_cards import task_card,digest
from scripts.pi0294.native_dataset import sha
from scripts.pi0294 import test_confirmation_plan as plan_fixture


class ConfirmationDeliveryTests(unittest.TestCase):
    def test_new_static_json_can_exceed_model_verifier_limit_without_allowing_models(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'static.json';p.write_bytes(b' '*(5*1024**2+1))
            files={'SOURCE_CONFIRM_PLAN.json':dict(bytes=p.stat().st_size,sha256=sha(p))}
            api=SimpleNamespace(list_repo_tree=lambda *a,**k:[SimpleNamespace(path=d.PREFIX+'/SOURCE_CONFIRM_PLAN.json',size=p.stat().st_size)])
            download=lambda *a,**k:str(p)
            d.verify_static(api,'fixed-revision',files,download,'not-a-real-token')
            for name in ['weights.safetensors','../escape.json','/absolute.png']:
                with self.subTest(name=name),self.assertRaises(PermissionError):
                    d.verify_static(api,'fixed-revision',{name:next(iter(files.values()))},download,'not-a-real-token')
            with self.assertRaises(PermissionError):
                d.verify_static(api,'fixed-revision',{'too-large.json':dict(bytes=d.MAX_BYTES+1,sha256=sha(p))},download,'not-a-real-token')

    def setup_bundle(self,root):
        (root/'plans').mkdir();(root/'worker_processes').mkdir()
        c,g,r=plan_fixture.ConfirmationPlanTests().fixture()
        for index,row in enumerate(r):
            group=next(x for x in g if x['group_id']==row['scene_group']);folder=root/'evidence/CONFIRM_STATIC'/str(index)
            folder.mkdir(parents=True);p=folder/'CARD.json';write(p,task_card(group,row['task']))
            row['actual_card_file_sha256']=sha(p);row['actual_static_images']={}
            for camera in ['body_rgb','wrist_rgb']:
                p=folder/(camera+'.png');Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(p)
                row['actual_static_images'][camera]=dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
        plan=d.config.plan.evaluation_plan(g,c,r,[]);write(root/d.config.PLAN,plan)
        write(root/d.config.STATIC_INPUTS,dict(exclusion_layouts=[]))
        write(root/'worker_processes/confirmation-static-01.json',dict(pid=-1,proc_start='0'))
        write(root/'RESOURCE_LEDGER.json',dict(reservations={}))
        value=dict(status='PASS_TEN_NEW_SCENES_TWENTY_ORIGINAL_STATIC_TASK_QUALIFICATIONS_NO_POLICY',scene_groups=10,
            static_receipts=20,actual_physics_integrations=0,new_model_queries=0,new_positive_labels=0,
            Teacher_success_filtering=False,policy_opened=False,plan_sha256=sha(root/d.config.PLAN))
        write(root/'evidence/CONFIRM_STATIC/RESULT.json',value)
        return plan,value

    def test_exact_twenty_cards_and_forty_pixel_files_not_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.setup_bundle(root)
            with patch.object(d,'validate_runtime',return_value=dict(execution=dict(confirmation_phase='STATIC_PREPARATION'))):
                files,plan,_=d.inventory(root)
            self.assertEqual(len(files),62);self.assertEqual(sum(n.endswith('.png') for n in files),40)
            self.assertEqual(sum(n.endswith('CARD.json') for n in files),20)
            self.assertFalse(any(n.endswith(('.safetensors','.bin','.pt','.pkl')) for n in files))
            self.assertEqual(plan['conditions'],40)

    def test_changed_pixel_or_policy_phase_refused_before_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan,_=self.setup_bundle(root)
            p=root/plan['static_receipts'][0]['actual_static_images']['body_rgb']['path'];p.write_bytes(b'changed')
            with patch.object(d,'validate_runtime',return_value=dict(execution=dict(confirmation_phase='STATIC_PREPARATION'))):
                with self.assertRaises(PermissionError):d.inventory(root)
            with patch.object(d,'validate_runtime',return_value=dict(execution=dict(confirmation_phase='POLICY_FROZEN'))):
                with self.assertRaises(PermissionError):d.inventory(root)

    def test_missing_static_proof_is_not_a_task_success_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,value=self.setup_bundle(root);value['actual_physics_integrations']=1
            write(root/'evidence/CONFIRM_STATIC/RESULT.json',value)
            with patch.object(d,'validate_runtime',return_value=dict(execution=dict(confirmation_phase='STATIC_PREPARATION'))):
                with self.assertRaises(PermissionError):d.inventory(root)


if __name__=='__main__':unittest.main()
