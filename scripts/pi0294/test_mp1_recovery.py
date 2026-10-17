import json
from pathlib import Path
import tempfile
import unittest

from .mp1 import recovery_config as r


class RecoveryTests(unittest.TestCase):
    def test_fixed_weights_no_new_training_or_other_checkpoints(self):
        bindings={label:dict(checkpoint_manifest_sha256=digest,global_step=512) for label,digest in r.CHECKPOINTS.items()}
        r.validate_checkpoint_bindings(bindings)
        with self.assertRaises(PermissionError):r.validate_checkpoint_bindings(dict(bindings,M1={}))
        bindings['M2-BC']['global_step']=1024
        with self.assertRaises(PermissionError):r.validate_checkpoint_bindings(bindings)

    def test_only_original_M2_engineering_gaps_not_M1_or_valid_outcomes(self):
        cards=[dict(id=f'{f}-{t}',scene_group='PI0294/TRAIN/'+f+'/000',task=t,split='TRAIN') for f in ['F0','F1'] for t in ['T1','T2']]
        rows=[dict(model=m,scene_group=c['scene_group'],task=c['task'],status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,live_native_failures=0) for m in r.CHECKPOINTS for c in cards]
        p=r.plan(cards,rows);self.assertEqual(p['regular_starts'],8);self.assertEqual(p['optimizer_updates'],0)
        self.assertEqual({s['branch'] for s in p['panels'].values()},set(r.CHECKPOINTS))
        for changed in [dict(rows[0],status='VALID_TASK_FAIL',pipeline_trustworthy=True),dict(rows[0],live_native_failures=1)]:
            with self.assertRaises(PermissionError):r.plan(cards,[changed]+rows[1:])
        cards[0]['split']='DEV'
        with self.assertRaises(PermissionError):r.plan(cards,rows)

    def test_recovery_resource_ledger_refuses_one_training_update(self):
        from .mp1.science import Ledger
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'MP1_STAGE_BINDING.json').write_text(json.dumps(dict(schema=r.SCHEMA,stage_caps=r.CAPS)))
            (root/'HOPPER_RESOURCE_LEDGER.json').write_text(json.dumps(dict(optimizer_updates=0)))
            ledger=Ledger(root,'hopper')
            with self.assertRaises(PermissionError):ledger.charge('optimizer_updates',1)
            ledger.charge('generation_queries',1280)
            with self.assertRaises(PermissionError):ledger.charge('generation_queries',1)
        self.assertEqual(r.limits({'schema':r.SCHEMA})['PBS'],2)
        self.assertEqual(r.limits({'schema':r.SCHEMA})['GPU_s'],21600)
