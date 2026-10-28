"""CPU stage fixtures, not fresh forecast, native allocation or training proof."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import neural
from scripts.pi0294.pv1.dataset import POPULATION
from scripts.pi0294.pv1.evaluation_plan import POPULATION_FILE, development
from scripts.pi0294.test_mp1_transport_fixture import put
from scripts.pi0294.test_pv1_evaluation_plan import groups
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.pv1.recipe import primary_recipe
from .authority import AUTHORITY, BANK, HARD_CLOSE, PARENT, PARENT_AUTHORITY, R0
from .source_admission import BINDING, COPY, VERSION


class StageTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp)/'children'/'source-a';g=groups()
        block=dict(arm='A_CONTROL',replica=False,stop_after_step=1024,resume_checkpoint_relative=None)
        adopted=dict(schema='DEC1_NATIVE_ADOPTION_AND_BUDGET_V1',authority_release=AUTHORITY,
            original_authority_release=PARENT_AUTHORITY,status='DEC1_AUTHORITY_ADOPTED_CONTROL_ONLY',
            campaign_run_uuid='original',original_T0_epoch=1792348196,original_hard_close_epoch=HARD_CLOSE,
            clock_reset=False,additive_budget=False,source_training_requires_complete_B0=False,
            legacy_authority_and_baseline_relative_release_require_complete_B0=True)
        put(root.parent.parent/'authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json',adopted,True)
        raw=(root.parent.parent/'authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json').read_bytes()
        binding=dict(schema='DEC1_SOURCE_TRAIN_STAGE_ADMISSION_V1',authority_release=AUTHORITY,
            adoption_sha256=hashlib.sha256(raw).hexdigest(),parent_checkpoint_manifest_sha256=PARENT,
            statistics_sha256=R0,new_source_bank_manifest_sha256=BANK,source_only_original_recipe=True,
            use_target_task_data=False,baseline_relative_selection_authorized=False,LOCKBOX_authorized=False,
            original_T0_epoch=adopted['original_T0_epoch'])
        values={'PV1_STAGE_BINDING.json':dict(route='COVERAGE_FIRST',new_bank_manifest_sha256=BANK,campaign_run_uuid='original'),
            'plans/SOURCE_EVALUATION_PLAN.json':development(g,route='COVERAGE_FIRST',candidate='A_CONTROL',new_bank_manifest_sha256=BANK),
            'plans/PRIOR_EVALUATION_OUTCOMES.json':[],'plans/PRIMARY_TRAIN_BLOCK.json':block,
            POPULATION_FILE:dict(groups=g),'plans/INITIAL_CORR_PARAMETER_INVENTORY.json':{},
            'plans/PV1_FIXED_TRAIN_PROBES.json':{},BINDING:binding,COPY:adopted,
            'plans/FROZEN_SOURCE_PAIR_RECIPE.json':old_recipe(),
            'plans/PRIMARY_LEARNING_RECIPES.json':dict(recipes={arm:primary_recipe(old_recipe(),'COVERAGE_FIRST',arm)
                for arm in ('A_CONTROL','B_COVERAGE')}),
            'plans/PV1_B0_DEV_ADMISSION.json':dict(schema='PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1',status='PARTIAL14')}
        authority=dict(protocol_ref=PARENT_AUTHORITY,amendments=[AUTHORITY],training_admission_version=VERSION)
        original_sha=neural.sha
        def sha(path):
            return POPULATION if Path(path)==root/POPULATION_FILE else original_sha(path)
        def bind():
            for name,value in values.items():put(root/name,value,True)
            put(root/'RUN_MANIFEST.json',dict(authority=authority,
                execution=dict(sealed_inputs={name:sha(root/name) for name in values})),True)
        bind()
        return root,values,authority,dict(stage='P2_PRIMARY',training_block=block,scientific_admission_version=VERSION),bind,sha

    def test_new_authority_passes_stage_with_partial_b0_without_relabeling_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,values,authority,request,bind,sha=self.fixture(tmp)
            before=(root/'plans/PV1_B0_DEV_ADMISSION.json').read_bytes()
            with patch.object(neural,'sha',side_effect=sha):neural.stage_work(root,request)
            self.assertEqual((root/'plans/PV1_B0_DEV_ADMISSION.json').read_bytes(),before)
            self.assertEqual(json.loads(before)['status'],'PARTIAL14')

    def test_old_entry_still_rejects_partial_b0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,values,authority,request,bind,sha=self.fixture(tmp)
            authority.clear();request.pop('scientific_admission_version');bind()
            with patch.object(neural,'sha',side_effect=sha):
                with self.assertRaisesRegex(PermissionError,'COMPLETE_REAL_B0'):neural.stage_work(root,request)

    def test_scope_cannot_admit_wrong_model_stats_bank_target_training_or_release(self):
        mutations=[lambda v,a,r:v[BINDING].update(statistics_sha256='a'*64),
            lambda v,a,r:v['plans/PRIMARY_LEARNING_RECIPES.json']['recipes']['B_COVERAGE'].update(learning_rate=1e-4),
            lambda v,a,r:v[BINDING].update(parent_checkpoint_manifest_sha256='a'*64),
            lambda v,a,r:v[BINDING].update(use_target_task_data=True),
            lambda v,a,r:v[BINDING].update(LOCKBOX_authorized=True),
            lambda v,a,r:v[BINDING].update(baseline_relative_selection_authorized=True),
            lambda v,a,r:v['PV1_STAGE_BINDING.json'].update(new_bank_manifest_sha256='a'*64),
            lambda v,a,r:a.update(amendments=['a'*40]),
            lambda v,a,r:r.update(scientific_admission_version='OLD'),
            lambda v,a,r:r.update(stage='P4_DEV'),
            lambda v,a,r:r.update(stage='P5_FINAL'),
            lambda v,a,r:r['training_block'].update(replica=True)]
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory() as tmp:
                root,values,authority,request,bind,sha=self.fixture(tmp)
                mutate(values,authority,request);bind()
                with patch.object(neural,'sha',side_effect=sha):
                    with self.assertRaises(PermissionError):neural.stage_work(root,request)

    def test_child_copy_cannot_replace_native_adoption_and_1024_gate_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,values,authority,request,bind,sha=self.fixture(tmp)
            values[COPY]['clock_reset']=True;bind()
            with patch.object(neural,'sha',side_effect=sha):
                with self.assertRaises(PermissionError):neural.stage_work(root,request)
        with tempfile.TemporaryDirectory() as tmp:
            root,values,authority,request,bind,sha=self.fixture(tmp)
            request['training_block']['stop_after_step']=4096;bind()
            with patch.object(neural,'sha',side_effect=sha):
                with self.assertRaisesRegex(PermissionError,'1024_FIXED_PROBES'):neural.stage_work(root,request)


if __name__=='__main__':
    unittest.main()
