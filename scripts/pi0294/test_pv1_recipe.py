import copy
import unittest

from scripts.pi0294.mp1.train import validate_recipe as legacy_validate
from scripts.pi0294.pv1.recipe import primary_recipe,validate_primary,begin_update,commit_update,PARENT


def old_recipe():
    return dict(seed=20261017,max_steps=2048,warmup_updates=32,effective_batch=32,
        per_device_train_batch_size=2,gradient_accumulation_steps=16,learning_rate=1e-5,end_learning_rate=1e-6,
        new_optimizer=True,exact_resume=False,parent='AUDITED_M1_AT_1024_INFERENCE_WEIGHTS_ONLY',
        VLM_frozen=True,statistics_fit=False,gradient_checkpointing=False,bf16=True,tf32=True,
        optimizer='adamw_torch_fused',weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,
        max_grad_norm=1.,numerical_configuration=dict(mode='FROZEN_FIXTURE_NUMERICS'),
        trainable_modules='EXACT_ORIGINAL_PROJECTOR_ACTION_DIT_MAP')


class RecipeTests(unittest.TestCase):
    def test_new_version_keeps_original_guard_and_frozen_hyperparameters(self):
        old=old_recipe();before=copy.deepcopy(old)
        a=primary_recipe(old,'CLEARANCE_PAIR','A_4MM')
        b=primary_recipe(old,'CLEARANCE_PAIR','B_12MM')
        self.assertEqual(old,before)
        for k in a:
            if k!='arm':self.assertEqual(a[k],b[k])
        self.assertEqual(a['initial_parent_checkpoint_manifest_sha256'],PARENT)
        self.assertEqual(a['max_steps'],4096)
        self.assertEqual(a['warmup_updates'],64)
        self.assertEqual(a['sample_visits'],4096*32)
        legacy_validate(old,'S-CORR')
        with self.assertRaises(PermissionError):legacy_validate(a,'S-CORR')
        validate_primary(a,old)

    def test_only_predeclared_replication_seed_and_no_development_at_1024(self):
        old=old_recipe();r=primary_recipe(old,'COVERAGE_FIRST','B_COVERAGE',replica=True)
        self.assertEqual(r['seed'],20261020)
        self.assertEqual(r['development_candidate_steps'],[4096])
        for k,v in [('learning_rate',1e-4),('max_steps',8192),('seed',123),('statistics_fit',True)]:
            bad=copy.deepcopy(r);bad[k]=v
            with self.assertRaises(PermissionError):validate_primary(bad,old)

    def test_pending_updates_not_erased_by_retry_or_wrong_job(self):
        r=primary_recipe(old_recipe(),'CLEARANCE_PAIR','A_4MM')
        l=dict(optimizer_updates=100,pending_updates=[])
        pending=begin_update(l,arm='A_4MM',local_step=99,request_id='J1',recipe=r)
        self.assertEqual(l['pending_updates'],[])
        with self.assertRaises(PermissionError):begin_update(pending,arm='A_4MM',local_step=99,request_id='J1',recipe=r)
        with self.assertRaises(PermissionError):commit_update(pending,arm='A_4MM',local_step_after=100,request_id='J2')
        final=commit_update(pending,arm='A_4MM',local_step_after=100,request_id='J1')
        self.assertEqual(final['optimizer_updates'],101)
        with self.assertRaises(PermissionError):commit_update(final,arm='A_4MM',local_step_after=100,request_id='J1')

    def test_total_updates_cap_includes_all_arms_and_replayed_work(self):
        r=primary_recipe(old_recipe(),'CLEARANCE_PAIR','A_4MM')
        with self.assertRaises(PermissionError):
            begin_update(dict(optimizer_updates=16384,pending_updates=[]),arm='A_4MM',local_step=1,request_id='J',recipe=r)
        with self.assertRaises(PermissionError):
            begin_update(dict(optimizer_updates=0,pending_updates=[]),arm='A_4MM',local_step=4096,request_id='J',recipe=r)
