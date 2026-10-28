from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import core
from scripts.pi0294.pv1.evaluation_plan import development
from scripts.pi0294.test_pv1_evaluation_plan import groups


class AutonomousEntryContracts(unittest.TestCase):
    def test_missing_typed_runtime_cannot_start_physics_or_counter_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'not-created'
            with patch.object(core,'validate_runtime',side_effect=PermissionError('UNBOUND')):
                with self.assertRaisesRegex(PermissionError,'UNBOUND'):core.run(root,'card.json')
                with self.assertRaisesRegex(PermissionError,'UNBOUND'):core.qualify(root,'session')
            self.assertFalse(root.exists())

    def test_stale_or_unqualified_model_is_rejected_before_native_engine_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign/children/policy-01';root.mkdir(parents=True)
            plan=development(groups(),route='COVERAGE_FIRST',candidate='B0_CORR',new_bank_manifest_sha256='a'*64)
            label,spec=next(iter(plan['panels'].items()));card=deepcopy(spec['cards'][0])
            card.update(relay_request_relative='plans/request.json',relay_request_sha256='c'*64)
            manifest=dict(execution=dict(sealed_inputs={'plans/PRIOR_EVALUATION_OUTCOMES.json':'a'*64}))
            ready=dict(status='PV1_NATIVE_MODEL_READY',label=label,identity=dict(global_step=2048,relay_identity={}),
                native_pre_model_qualification=dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',receipt_sha256='b'*64),job_id='123')
            registry=dict(cases={'fixture':dict(attempt_id=card['id'],card=card)},job_id='123',native_pre_model_receipt_sha256='b'*64)
            base={str(root/'INPUT_BINDINGS.private.json'):{},str(root/'card.json'):card,
                str(root/'plans/PRIOR_EVALUATION_OUTCOMES.json'):[],str(root/'plans/request.json'):dict(identity={}),
                str(root/'evidence/COUNTER_NATIVE_QUALIFICATION.json'):dict(status='PASS_ORIGINAL_VS_COUNTED_NATIVE_MODEL_STATE_XML_AND_TWO_RGB',
                    integrations=0,model_constructor_attempts=2)}
            for change in ('old_status','wrong_step','wrong_session','no_native_proof','changed_proof','wrong_job'):
                r=deepcopy(ready)
                if change=='old_status':r['status']='MP1_NATIVE_MODEL_READY'
                if change=='wrong_step':r['identity']['global_step']=1024
                if change=='wrong_session':r['label']='other'
                if change=='no_native_proof':r['native_pre_model_qualification']['status']='CPU_TEST_ONLY'
                if change=='changed_proof':r['native_pre_model_qualification']['receipt_sha256']='d'*64
                if change=='wrong_job':r['job_id']='previous'
                values=dict(base);values[str(root/'plans'/label/'REMOTE_READY.json')]=r
                values[str(root/'plans'/label/'REGISTRY.json')]=registry
                ledger=SimpleNamespace(guard=lambda:None)
                with self.subTest(change=change),patch.object(core,'validate_runtime',return_value=manifest),\
                        patch.object(core,'Ledger',return_value=ledger),patch.object(core,'session_spec',return_value=spec),\
                        patch.object(core,'read',side_effect=lambda p:values[str(p)]),patch.object(core,'sha',return_value='a'*64),\
                        self.assertRaisesRegex(PermissionError,'CURRENT_NATIVE_MODEL_IDENTITY'):
                    core.run(root,'card.json')
            self.assertFalse((root/'episodes').exists())
