"""CPU evidence fixtures, not actual B0 evaluation or scientific admission."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1.baseline import admission,AUDIT_CHECKS
from scripts.pi0294.pv1.recipe import PARENT
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.pv1.evaluation_plan import development
from scripts.pi0294.test_pv1_evaluation_plan import groups


def raw(value):return (json.dumps(value,indent=2)+'\n').encode()


def fixtures():
    plan=development(groups(),route='COVERAGE_FIRST',candidate='B0_CORR',new_bank_manifest_sha256='b'*64)
    rows=[]
    for panel in plan['panels'].values():
        for card in panel['cards']:
            result=raw(dict(id=card['id'],source_sha='a'*40))
            row={k:card[k] for k in ('condition_id','task','split','scene_group','noise_bank','family')}
            row.update(attempt_id=card['id'],model='B0_CORR',status='VALID_TASK_FAIL',
                pipeline_trustworthy=True,failure_stage='OBJECT_TASK_RESULT',live_native_failures=0,main_integrations=16000,
                teacher_takeover=False,result_sha256=hashlib.sha256(result).hexdigest())
            audit=dict(attempt_id=row['attempt_id'],status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',
                adjudication=row,errors=[],checks={k:True for k in AUDIT_CHECKS},
                actual_native_safety_flag_rows=0,model_autonomous_evaluation=True,
                execution_source_sha='a'*40,video=dict(sha256='c'*64),
                new_integrations=0,new_native_constructors=0,new_model_queries=0)
            ready=dict(status='PV1_NATIVE_MODEL_READY',label=card['pv1_session_label'],
                job_id='CPU_FIXTURE_NOT_A_JOB',run_uuid='CPU_FIXTURE_NOT_A_RUN',
                identity=dict(label='B0_CORR',global_step=2048,checkpoint_manifest_sha256=PARENT,
                    statistics_sha256=STATS,relay_identity=dict(source_sha='a'*40)),
                native_pre_model_qualification=dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',
                    heartbeat_elapsed_s=600.01,transport_cycles=80,new_model_queries=0,new_physics=0,receipt_sha256='e'*64))
            chain=dict(status='PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN',attempt_id=row['attempt_id'],
                adjudication_sha256=hashlib.sha256(raw(row)).hexdigest(),identity=ready['identity'],
                new_integrations=0,new_model_queries=0,executed_ACKs=80,draws=80,
                core_proof_sha256='a'*64,hopper_proof_sha256='b'*64)
            rows.append(dict(adjudication=raw(row),result=result,audit=raw(audit),model_ready=raw(ready),causal_chain=raw(chain)))
    return rows


def build(rows):
    population=raw(dict(groups=groups()))
    with patch('scripts.pi0294.pv1.baseline.POPULATION',hashlib.sha256(population).hexdigest()):
        return admission(population,route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,evidence=rows)


class BaselineAdmission(unittest.TestCase):
    def test_sanitized_or_replaced_population_is_not_original_native_seal(self):
        with self.assertRaisesRegex(PermissionError,'FROZEN_POPULATION_BYTES'):
            admission(raw(dict(groups=groups())),route='COVERAGE_FIRST',new_bank_manifest_sha256='b'*64,evidence=[])

    def test_all_48_zero_success_is_admitted_with_exact_original_byte_hashes(self):
        rows=fixtures();value=build(rows)
        self.assertEqual(value['successes'],dict(T1=0,T2=0))
        self.assertEqual(len(value['condition_result_sha256']),48)
        self.assertEqual(len(value['all_actual_attempts']),48)
        row=json.loads(rows[0]['adjudication'])
        self.assertEqual(value['condition_result_sha256'][row['condition_id']],hashlib.sha256(rows[0]['adjudication']).hexdigest())

    def test_missing_duplicate_foreign_untrusted_or_safety_cannot_admit(self):
        rows=fixtures()
        for changed in (rows[:-1],rows+[rows[0]]):
            with self.subTest(size=len(changed)),self.assertRaises(PermissionError):build(changed)
        for key,value in [('condition_id','OTHER'),('model','A_CONTROL'),('noise_bank',999),
                ('split','TRAIN'),('teacher_takeover',True),('live_native_failures',1),
                ('pipeline_trustworthy',False),('status','ENGINEERING_INCOMPLETE'),('failure_stage','S1_INTEGRITY')]:
            changed=deepcopy(rows);row=json.loads(changed[0]['adjudication']);row[key]=value
            changed[0]['adjudication']=raw(row)
            with self.subTest(key=key),self.assertRaises(PermissionError):build(changed)

    def test_each_raw_audit_and_result_binding_is_required(self):
        rows=fixtures()
        variants=[dict(status='FAIL_PRESERVED_AUDIT'),dict(errors=['bad']),dict(video=None),
            dict(actual_native_safety_flag_rows=1),dict(model_autonomous_evaluation=False),
            dict(execution_source_sha='d'*40),dict(new_integrations=1),dict(new_model_queries=1),
            dict(adjudication={}),dict(checks={})]
        variants += [dict(checks={**{k:True for k in AUDIT_CHECKS},name:False}) for name in AUDIT_CHECKS]
        for change in variants:
            changed=deepcopy(rows);audit=json.loads(changed[0]['audit']);audit.update(change);changed[0]['audit']=raw(audit)
            with self.subTest(change=change),self.assertRaises(PermissionError):build(changed)
        changed=deepcopy(rows);changed[0]['result']=raw(dict(id='OTHER',source_sha='d'*40))
        with self.assertRaises(PermissionError):build(changed)
        changed=deepcopy(rows);changed[0]['audit']={}
        with self.assertRaisesRegex(PermissionError,'ACTUAL_FILE_BYTES'):build(changed)

    def test_missing_or_foreign_causal_evidence_never_admits_training(self):
        rows=fixtures()
        for change in [dict(status='NOT_VERIFIED'),dict(adjudication_sha256='f'*64),dict(identity={}),
                dict(executed_ACKs=79),dict(draws=81),dict(new_integrations=1),dict(core_proof_sha256='')]:
            changed=deepcopy(rows);chain=json.loads(changed[0]['causal_chain']);chain.update(change)
            changed[0]['causal_chain']=raw(chain)
            with self.subTest(change=change),self.assertRaises(PermissionError):build(changed)
        changed=deepcopy(rows);changed[0].pop('causal_chain')
        with self.assertRaises(PermissionError):build(changed)

    def test_wrong_checkpoint_stale_panel_or_incomplete_native_qualification_rejected(self):
        rows=fixtures()
        for section,key,value in [(None,'label','WRONG'),(None,'job_id',None),
                ('identity','global_step',4096),('identity','checkpoint_manifest_sha256','f'*64),
                ('identity','statistics_sha256','f'*64),
                ('native_pre_model_qualification','transport_cycles',79),
                ('native_pre_model_qualification','heartbeat_elapsed_s',599),
                ('native_pre_model_qualification','new_model_queries',1),
                ('native_pre_model_qualification','receipt_sha256','')]:
            changed=deepcopy(rows);ready=json.loads(changed[0]['model_ready'])
            target=ready if section is None else ready[section];target[key]=value
            changed[0]['model_ready']=raw(ready)
            with self.subTest(section=section,key=key),self.assertRaisesRegex(PermissionError,'CORR_MODEL'):build(changed)

    def test_diagnosed_replacement_preserves_original_without_repeating_policy_failure(self):
        rows=fixtures();original=json.loads(rows[0]['adjudication'])
        original.update(status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,
            failure_stage='TRANSPORT')
        audit=json.loads(rows[0]['audit']);audit.update(adjudication=original)
        original_bytes=raw(original)
        diagnosis=dict(status='CONFIRMED_ENGINEERING_FAILURE_NO_POLICY_REPLAY',
            original_adjudication_sha256=hashlib.sha256(original_bytes).hexdigest(),
            original_result_sha256=original['result_sha256'],native_process_terminal_verified=True,
            unknown_effects=[],policy_outcome_valid=False,evidence_sha256=dict(native_terminal='d'*64))
        rows.insert(0,dict(adjudication=original_bytes,result=rows[0]['result'],audit=raw(audit),
            engineering_diagnosis=raw(diagnosis),model_ready=rows[0]['model_ready']))
        replacement=json.loads(rows[1]['adjudication']);replacement['attempt_id']+='-REPLACEMENT-01'
        result=json.loads(rows[1]['result']);result['id']=replacement['attempt_id'];rows[1]['result']=raw(result)
        replacement['result_sha256']=hashlib.sha256(rows[1]['result']).hexdigest()
        audit=json.loads(rows[1]['audit']);audit.update(attempt_id=replacement['attempt_id'],adjudication=replacement)
        rows[1].update(adjudication=raw(replacement),audit=raw(audit))
        chain=json.loads(rows[1]['causal_chain']);chain.update(attempt_id=replacement['attempt_id'],
            adjudication_sha256=hashlib.sha256(raw(replacement)).hexdigest())
        rows[1]['causal_chain']=raw(chain)
        self.assertEqual(len(build(rows)['all_actual_attempts']),49)
        self.assertEqual(rows[0]['adjudication'],original_bytes)
        for change in [dict(original_adjudication_sha256='e'*64),dict(native_process_terminal_verified=False),
                dict(unknown_effects=['unknown']),dict(policy_outcome_valid=True),dict(evidence_sha256={})]:
            changed=deepcopy(rows);changed[0]['engineering_diagnosis']=raw(dict(diagnosis,**change))
            with self.subTest(change=change),self.assertRaises(PermissionError):build(changed)
