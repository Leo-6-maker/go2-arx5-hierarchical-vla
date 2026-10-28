"""Synthetic CPU contracts only: no LOCKBOX execution, model load or physics."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch
import unittest

from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.pv1 import p5,offline_policy
from scripts.pi0294.pv1.baseline import AUDIT_CHECKS
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.evaluation_plan import final_confirmation
from scripts.pi0294.test_pv1_evaluation_plan import groups


HASH='a'*64


def raw(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()


def admission(schema,status,successes,checkpoint):
    return dict(schema=schema,status=status,authority_release=AUTHORITY,population_plan_sha256='POPULATION',
        checkpoint_manifest_sha256=checkpoint,real_native_faults=0,Teacher_takeover=False,
        all_condition_evidence_audits_PASS=True,successes=successes,new_training_updates=0)


def frozen(replica='PASS_FIXED_SECOND_SEED'):
    candidate=admission('PI0294_PV1_CANDIDATE_ACTUAL_DEV_ADMISSION_V1',
        'PASS_COMPLETE_ACTUAL_CANDIDATE_DEV48_EVIDENCE',dict(T1=20,T2=19),'c'*64)
    baseline=admission('PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1',
        'PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE',dict(T1=16,T2=14),'b'*64)
    candidate_bytes=raw(candidate);baseline_bytes=raw(baseline)
    selection=dict(schema='PI0294_PV1_P4_SELECTION_V1',authority_release=AUTHORITY,route='COVERAGE_FIRST',
        LOCKBOX_opened=False,unique_candidate='B_COVERAGE',qualifying_candidates=['B_COVERAGE'],global_step=4096,
        checkpoint_manifest_sha256='c'*64,processor_manifest_sha256='d'*64,runtime_manifest_sha256='e'*64,
        precision_manifest_sha256='f'*64,selection_evidence_sha256='1'*64,
        candidate_DEV_admission_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        baseline_DEV_admission_sha256=hashlib.sha256(baseline_bytes).hexdigest())
    artifacts={k:selection[k] for k in ('checkpoint_manifest_sha256','processor_manifest_sha256',
        'runtime_manifest_sha256','precision_manifest_sha256')}
    with patch.object(p5,'POPULATION','POPULATION'):
        return p5.freeze_candidate(route='COVERAGE_FIRST',selection=selection,
            candidate_admission_bytes=candidate_bytes,baseline_admission_bytes=baseline_bytes,
            artifacts=artifacts,replica_status=replica)


def evidence(freeze,successes=(40,40)):
    plan=final_confirmation(groups(),freeze);rows=[];seen=dict(T1=0,T2=0)
    for panel in plan['panels'].values():
        for card in panel['cards']:
            task=card['task'];seen[task]+=1;success=seen[task]<=successes[int(task[1])-1]
            result=raw(dict(id=card['id'],source_sha='2'*40,physics_steps=16000,rgb_pairs=641))
            row={k:card[k] for k in ('condition_id','task','split','scene_group','noise_bank','family')}
            row.update(attempt_id=card['id'],model=freeze['candidate'],status='SUCCESS' if success else 'VALID_TASK_FAIL',
                pipeline_trustworthy=True,failure_stage='COMPLETE_SUCCESS' if success else 'OBJECT_TASK_RESULT',
                live_native_failures=0,main_integrations=16000,teacher_takeover=False,duration_s=32.,RGB_pairs=641,
                S1_rejections=0,result_sha256=hashlib.sha256(result).hexdigest())
            audit=dict(attempt_id=card['id'],status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',adjudication=row,errors=[],
                checks={k:True for k in AUDIT_CHECKS},actual_native_safety_flag_rows=0,model_autonomous_evaluation=True,
                new_integrations=0,new_model_queries=0,score=dict(complete_32s=True,clock_valid=True,
                    safety_valid=True,persistent_success=success))
            identity=dict(label=freeze['candidate'],global_step=4096,
                checkpoint_manifest_sha256=freeze['checkpoint_manifest_sha256'],statistics_sha256=STATS)
            ready=dict(status='PV1_NATIVE_MODEL_READY',label=card['pv1_session_label'],identity=identity,
                native_pre_model_qualification=dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',
                    heartbeat_elapsed_s=600,transport_cycles=80,new_model_queries=0,new_physics=0,receipt_sha256='3'*64))
            chain=dict(status='PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN',attempt_id=card['id'],
                adjudication_sha256=hashlib.sha256(raw(row)).hexdigest(),identity=identity,new_integrations=0,
                new_model_queries=0,executed_ACKs=80,draws=80)
            rows.append(dict(adjudication=raw(row),result=result,audit=raw(audit),model_ready=raw(ready),causal_chain=raw(chain)))
    return rows


def resources():return dict(schema='PI0294_PV1_RESOURCE_FINAL_V1',status='PASS_FROZEN_FINAL_RESOURCE_SETTLEMENT',unknown_effects=[])


class P5Contracts(unittest.TestCase):
    def test_freeze_opens_exact_shared_lockbox80_only_after_unique_dev_gate(self):
        freeze=frozen();plan=final_confirmation(groups(),freeze)
        cards=[c for panel in plan['panels'].values() for c in panel['cards']]
        self.assertEqual(len(cards),80);self.assertEqual(len({c['condition_id'] for c in cards}),80)
        self.assertEqual({c['split'] for c in cards},{'LOCKBOX'})
        self.assertEqual(sum(c['task']=='T1' for c in cards),40)
        self.assertEqual({c['policy_candidate'] for c in cards},{freeze['candidate']})
        bad=deepcopy(freeze);bad['qualifying_candidates'].append('A_CONTROL')
        with self.assertRaises(PermissionError):final_confirmation(groups(),bad)

    def test_dev_threshold_baseline_retention_and_artifact_hashes_fail_closed(self):
        freeze=frozen();self.assertFalse(freeze['LOCKBOX_opened'])
        bad=deepcopy(freeze);bad['checkpoint_manifest_sha256']='bad'
        with self.assertRaises(PermissionError):final_confirmation(groups(),bad)

    def test_complete_all_success_lockbox_is_validated_with_scene_cluster_intervals(self):
        freeze=frozen();population=raw(dict(groups=groups()))
        with patch.object(p5,'POPULATION',hashlib.sha256(population).hexdigest()):
            value=p5.acceptance(population,freeze=freeze,evidence=evidence(freeze),resource_final=resources())
        self.assertEqual(value['grade'],'PV1_VALIDATED');self.assertEqual(value['conditions'],80)
        self.assertEqual(value['tasks']['T1']['scene_groups_2_of_2'],20)
        self.assertFalse(value['tasks']['T2']['clustered_95_interval']['deployment_reliability_claim'])

    def test_replica_omission_caps_grade_and_low_result_stays_not_qualified(self):
        population=raw(dict(groups=groups()))
        freeze=frozen('OMITTED_RESOURCE_CRITICAL_MAX_BETA')
        with patch.object(p5,'POPULATION',hashlib.sha256(population).hexdigest()):
            beta=p5.acceptance(population,freeze=freeze,evidence=evidence(freeze),resource_final=resources())
            failed=p5.acceptance(population,freeze=freeze,evidence=evidence(freeze,(27,27)),resource_final=resources())
        self.assertEqual(beta['grade'],'PV1_BETA')
        self.assertEqual(failed['grade'],'BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED')
        self.assertEqual(failed['selected'],'NONE')

    def test_valid_failure_cannot_be_replaced_or_removed_from_denominator(self):
        freeze=frozen();rows=evidence(freeze);rows.append(deepcopy(rows[0]))
        row=json.loads(rows[-1]['adjudication']);row['attempt_id']+='-REPLACEMENT-01';rows[-1]['adjudication']=raw(row)
        population=raw(dict(groups=groups()))
        with patch.object(p5,'POPULATION',hashlib.sha256(population).hexdigest()),self.assertRaises(PermissionError):
            p5.acceptance(population,freeze=freeze,evidence=rows,resource_final=resources())

    def test_policy_package_and_offline_entry_load_generate_reset_close_and_failure_stop(self):
        freeze=frozen();acceptance=dict(candidate=freeze['candidate'],checkpoint_manifest_sha256='c'*64,grade='PV1_BETA')
        names=('shared_weights','processor','statistics','runtime','precision','action_schema','dependency_lock','offline_inference_entry')
        artifacts={name:dict(uri='hf://public/fixed/'+name,sha256='c'*64 if name=='shared_weights' else HASH) for name in names}
        retirement=dict(schema='PI0294_PV1_CHECKPOINT_RETIREMENT_LIST_V1',retired=[],protected=['c'*64])
        manifest=p5.delivery_manifest(freeze=freeze,acceptance=acceptance,artifacts=artifacts,
            resource_final=resources(),checkpoint_retirement=retirement)
        docs=p5.delivery_documents(manifest);self.assertIn('real-robot deployment is NONE',docs['POLICY_CARD.md'])
        self.assertEqual(offline_policy.dry_run(manifest)['models_loaded'],0)
        class Instance:
            reset_calls=close_calls=0
            def generate(self,obs,instruction):return [[0.]*10 for _ in range(40)]
            def reset(self):self.reset_calls+=1
            def close(self):self.close_calls+=1
        instance=Instance();obs=dict(body_rgb='a',wrist_rgb='b',eef10=[0.]*10,timestamp_ns=1)
        action=offline_policy.generate_once(manifest,obs,freeze['task_instructions']['T1'],loader=lambda m:instance)
        self.assertEqual(len(action),40);self.assertEqual((instance.reset_calls,instance.close_calls),(1,1))
        class Broken(Instance):
            def generate(self,obs,instruction):raise RuntimeError('fixture')
        broken=Broken()
        with self.assertRaises(RuntimeError):offline_policy.generate_once(manifest,obs,freeze['task_instructions']['T2'],loader=lambda m:broken)
        self.assertEqual((broken.reset_calls,broken.close_calls),(1,1))


if __name__=='__main__':unittest.main()
