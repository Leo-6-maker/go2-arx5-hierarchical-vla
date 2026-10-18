"""CPU entry fixtures, not PBS allocation or model evidence."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import neural
from scripts.pi0294.pv1.dataset import POPULATION
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.evaluation_plan import development,POPULATION_FILE
from scripts.pi0294.pv1.recipe import PARENT
from scripts.pi0294.test_mp1_transport_fixture import put
from scripts.pi0294.test_pv1_evaluation_plan import groups


class NeuralEntryContracts(unittest.TestCase):
    def test_no_native_allocation_or_unimplemented_phase_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request=root/'REQUEST.json';put(request,dict(stage='B0_DEV'))
            with patch.object(neural,'admit',side_effect=PermissionError('NO_PBS')),patch.object(neural.model,'load') as load:
                with self.assertRaisesRegex(PermissionError,'NO_PBS'):neural.run(root,request)
                load.assert_not_called()
            put(request,dict(stage='OLD_N1'))
            with patch.object(neural,'admit') as admit, self.assertRaisesRegex(PermissionError,'ONLY_B0_PRIMARY'):
                neural.run(root,request)
            admit.assert_not_called()

    def test_primary_needs_complete_actual_b0_but_no_baseline_success_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=groups();bank='b'*64
            b0=development(g,route='COVERAGE_FIRST',candidate='B0_CORR',new_bank_manifest_sha256=bank)
            evaluation=development(g,route='COVERAGE_FIRST',candidate='A_CONTROL',new_bank_manifest_sha256=bank)
            conditions={c['condition_id'] for p in b0['panels'].values() for c in p['cards']}
            baseline=dict(schema='PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1',status='PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE',
                authority_release=AUTHORITY,population_plan_sha256=POPULATION,checkpoint_manifest_sha256=PARENT,
                new_bank_manifest_sha256=bank,condition_result_sha256={c:'c'*64 for c in conditions},
                independent_audit_sha256={c:'d'*64 for c in conditions},real_native_faults=0,Teacher_takeover=False,
                all_condition_evidence_audits_PASS=True,successes=dict(T1=0,T2=0))
            block=dict(arm='A_CONTROL',replica=False,stop_after_step=1024,resume_checkpoint_relative=None)
            values={'PV1_STAGE_BINDING.json':dict(route='COVERAGE_FIRST',new_bank_manifest_sha256=bank),
                'plans/SOURCE_EVALUATION_PLAN.json':evaluation,'plans/PRIOR_EVALUATION_OUTCOMES.json':[],
                'plans/PRIMARY_TRAIN_BLOCK.json':block,POPULATION_FILE:dict(groups=g),
                'plans/PV1_B0_DEV_ADMISSION.json':baseline,'plans/INITIAL_CORR_PARAMETER_INVENTORY.json':{},
                'plans/PV1_FIXED_TRAIN_PROBES.json':{}}
            for name,value in values.items():put(root/name,value,True)
            actual_sha=neural.sha
            def digest(path):return POPULATION if Path(path)==root/POPULATION_FILE else actual_sha(path)
            def bind():put(root/'RUN_MANIFEST.json',dict(execution=dict(sealed_inputs={n:digest(root/n) for n in values})),True)
            bind();request=dict(stage='P2_PRIMARY',training_block=block)
            with patch.object(neural,'sha',side_effect=digest):
                neural.stage_work(root,request)
                bad=deepcopy(baseline);bad['real_native_faults']=1;put(root/'plans/PV1_B0_DEV_ADMISSION.json',bad,True);bind()
                with self.assertRaisesRegex(PermissionError,'COMPLETE_REAL_B0'):neural.stage_work(root,request)
                put(root/'plans/PV1_B0_DEV_ADMISSION.json',baseline,True);bind()
                block['stop_after_step']=4096;put(root/'plans/PRIMARY_TRAIN_BLOCK.json',block,True);bind()
                with self.assertRaisesRegex(PermissionError,'1024_FIXED_PROBES'):neural.stage_work(root,request)

    def test_development_candidate_identity_is_unique_and_bounded_across_cold_children(self):
        class Ledger:
            value={}
            @contextmanager
            def transaction(self):yield self.value
        ledger=Ledger();identity=dict(checkpoint_manifest_sha256='a'*64,global_step=4096)
        neural.record_candidate(ledger,'A_CONTROL',identity);neural.record_candidate(ledger,'A_CONTROL',identity)
        self.assertEqual(len(ledger.value['development_candidates']),1)
        with self.assertRaises(PermissionError):neural.record_candidate(ledger,'A_CONTROL',dict(identity,checkpoint_manifest_sha256='b'*64))
        for i in range(5):neural.record_candidate(ledger,str(i),identity)
        with self.assertRaisesRegex(PermissionError,'AT_MOST_SIX'):neural.record_candidate(ledger,'seventh',identity)
