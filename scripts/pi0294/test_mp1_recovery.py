import json
from pathlib import Path
import tempfile
import unittest

from .mp1 import recovery_config as r


class RecoveryTests(unittest.TestCase):
    def test_additional_authorization_matches_published_card_bytes(self):
        from .native_dataset import sha
        path=Path(__file__).resolve().parents[2]/'reports/PI0294/design/MP1_E1_RECOVERY_ONE_ADDITIONAL_PBS_V1.json'
        self.assertEqual(sha(path),r.ADDITIONAL_PBS_AUTHORIZATION['card_sha256'])
        card=json.loads(path.read_text());self.assertEqual(card['optimizer_updates'],0)
        self.assertEqual(card['maximum_additional_allocation_wall_s'],7200)

    def test_human_third_PBS_is_opt_in_two_hours_no_counter_reset(self):
        prior=r.prior_usage({});prior.update(GPU_s=453,PBS=2,main_starts=4,
            live_and_recovery_integrations=600,shadow_integrations=600,generation_queries=5)
        stage=dict(schema=r.SCHEMA,stage_caps=r.CAPS,CPU_core_s=r.CPU_CAPS,prior_recovery_usage=prior,
                   additional_PBS_authorization=r.ADDITIONAL_PBS_AUTHORIZATION)
        self.assertEqual(r.family_PBS_cap(stage),3)
        self.assertEqual(r.limits(stage)['PBS'],1)
        self.assertEqual(r.limits(stage)['GPU_s'],7200)
        from .mp1.science import Ledger
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'MP1_STAGE_BINDING.json').write_text(json.dumps(stage))
            (root/'RUN_MANIFEST.json').write_text('{}')
            ledger=Ledger(root,'hopper');self.assertEqual(ledger.caps['main_starts'],12)
            self.assertEqual(ledger.caps['generation_queries'],1275)
            self.assertEqual(ledger.caps['optimizer_updates'],0)
        for bad in [dict(stage,additional_PBS_authorization={}),dict(stage,prior_recovery_usage=None),
                    dict(stage,prior_recovery_usage=dict(prior,PBS=1))]:
            with self.assertRaises(PermissionError):r.limits(bad)
        with self.assertRaises(PermissionError):r.limits(dict(schema=r.SCHEMA,prior_recovery_usage=prior))

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
            root=Path(d);(root/'MP1_STAGE_BINDING.json').write_text(json.dumps(dict(schema=r.SCHEMA,stage_caps=r.CAPS,CPU_core_s=r.CPU_CAPS)))
            (root/'RUN_MANIFEST.json').write_text('{}')
            (root/'HOPPER_RESOURCE_LEDGER.json').write_text(json.dumps(dict(optimizer_updates=0)))
            ledger=Ledger(root,'hopper')
            with self.assertRaises(PermissionError):ledger.charge('optimizer_updates',1)
            ledger.charge('generation_queries',1280)
            with self.assertRaises(PermissionError):ledger.charge('generation_queries',1)
        self.assertEqual(r.limits({'schema':r.SCHEMA})['PBS'],2)
        self.assertEqual(r.limits({'schema':r.SCHEMA})['GPU_s'],21600)

    def test_new_child_deducts_prior_allocation_and_every_counter(self):
        from .mp1.science import Ledger
        prior=r.prior_usage({});prior.update(GPU_s=213,PBS=1,main_starts=2,
            live_and_recovery_integrations=400,shadow_integrations=400,generation_queries=2,CPU_core_s=200,CPU_hopper_s=215)
        stage=dict(schema=r.SCHEMA,stage_caps=r.CAPS,CPU_core_s=r.CPU_CAPS,prior_recovery_usage=prior)
        self.assertEqual(r.limits(stage)['PBS'],1);self.assertEqual(r.limits(stage)['GPU_s'],21387)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'MP1_STAGE_BINDING.json').write_text(json.dumps(stage))
            (root/'RUN_MANIFEST.json').write_text('{}')
            (root/'HOPPER_RESOURCE_LEDGER.json').write_text(json.dumps(dict(optimizer_updates=0)))
            ledger=Ledger(root,'hopper');self.assertEqual(ledger.caps['main_starts'],14)
            ledger.charge('generation_queries',1278)
            with self.assertRaises(PermissionError):ledger.charge('generation_queries',1)
        for bad in [dict(prior,PBS=0),dict(prior,GPU_s=-1),dict(prior,optimizer_updates=1),dict(prior,CPU_core_s=1.5)]:
            with self.assertRaises(PermissionError):r.limits(dict(stage,prior_recovery_usage=bad))
