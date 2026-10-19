"""CPU-only P4 freeze, P5 evidence acceptance and policy delivery contracts."""
import hashlib
import json
import math
import re

from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.scene_cards import INSTRUCTIONS
from .baseline import AUDIT_CHECKS
from .dataset import POPULATION
from .decision import AUTHORITY
from .evaluation_plan import final_confirmation


def _read(raw):
    if not isinstance(raw,bytes):
        raise PermissionError('PV1_P5_REQUIRES_ORIGINAL_FILE_BYTES')
    value=json.loads(raw)
    if not isinstance(value,dict):raise PermissionError('PV1_P5_JSON_OBJECT_REQUIRED')
    return value,hashlib.sha256(raw).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def freeze_candidate(*,route,selection,candidate_admission_bytes,baseline_admission_bytes,
        artifacts,replica_status):
    """Derive a unique immutable candidate binding from complete DEV evidence."""
    candidate,candidate_sha=_read(candidate_admission_bytes);baseline,baseline_sha=_read(baseline_admission_bytes)
    chosen=selection.get('unique_candidate')
    hashes=('checkpoint_manifest_sha256','processor_manifest_sha256','runtime_manifest_sha256',
        'precision_manifest_sha256','selection_evidence_sha256')
    if (route not in ('COVERAGE_FIRST','CLEARANCE_PAIR')
            or selection.get('schema')!='PI0294_PV1_P4_SELECTION_V1'
            or selection.get('authority_release')!=AUTHORITY
            or selection.get('route')!=route or selection.get('LOCKBOX_opened') is not False
            or selection.get('qualifying_candidates')!=[chosen]
            or candidate.get('schema')!='PI0294_PV1_CANDIDATE_ACTUAL_DEV_ADMISSION_V1'
            or candidate.get('status')!='PASS_COMPLETE_ACTUAL_CANDIDATE_DEV48_EVIDENCE'
            or baseline.get('status')!='PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE'
            or candidate.get('population_plan_sha256')!=POPULATION
            or baseline.get('population_plan_sha256')!=POPULATION
            or candidate.get('real_native_faults') or baseline.get('real_native_faults')
            or candidate.get('Teacher_takeover') or baseline.get('Teacher_takeover')
            or candidate.get('all_condition_evidence_audits_PASS') is not True
            or selection.get('candidate_DEV_admission_sha256')!=candidate_sha
            or selection.get('baseline_DEV_admission_sha256')!=baseline_sha
            or selection.get('checkpoint_manifest_sha256')!=candidate.get('checkpoint_manifest_sha256')
            or selection.get('global_step')!=4096 or candidate.get('new_training_updates')!=0
            or any(not re.fullmatch('[0-9a-f]{64}',selection.get(k,'')) for k in hashes)
            or artifacts!={k:selection[k] for k in hashes[:-1]}
            or replica_status not in ('PASS_FIXED_SECOND_SEED','OMITTED_RESOURCE_CRITICAL_MAX_BETA')):
        raise PermissionError('PV1_P4_EXACT_COMPLETE_UNIQUE_CANDIDATE_SELECTION_AND_ARTIFACT_FREEZE')
    score=candidate.get('successes',{});base=baseline.get('successes',{})
    if (set(score)!={'T1','T2'} or set(base)!={'T1','T2'}
            or min(score.values())<18 or any(score[t]<base[t]-1 for t in score)
            or min(score.values())<=min(base.values())):
        raise PermissionError('PV1_P4_CANDIDATE_MUST_PASS_PREDECLARED_DEV_THRESHOLD_AND_B0_RETENTION')
    return dict(schema='PI0294_PV1_P4_UNIQUE_CANDIDATE_FREEZE_V1',authority_release=AUTHORITY,
        status='PASS_UNIQUE_DEV_QUALIFIED_SHARED_CANDIDATE_FROZEN',route=route,candidate=chosen,
        qualifying_candidates=[chosen],global_step=4096,statistics_sha256=STATS,
        candidate_DEV_admission_sha256=candidate_sha,baseline_DEV_admission_sha256=baseline_sha,
        checkpoint_manifest_sha256=selection['checkpoint_manifest_sha256'],
        processor_manifest_sha256=selection['processor_manifest_sha256'],
        runtime_manifest_sha256=selection['runtime_manifest_sha256'],
        precision_manifest_sha256=selection['precision_manifest_sha256'],
        selection_evidence_sha256=selection['selection_evidence_sha256'],
        replica_status=replica_status,maximum_possible_grade='PV1_VALIDATED' if replica_status.startswith('PASS_') else 'PV1_BETA',
        shared_weights_for_tasks=['T1','T2'],task_instructions=dict(INSTRUCTIONS),
        action_schema='EEF10_ABSOLUTE_COMMITTED_H40_WITH_ORIGINAL_V3_0_RELATIVE_CODEC',
        Teacher_takeover=False,evaluation_training_updates=0,LOCKBOX_opened=False,
        WBC_modified=False,S1_modified=False,score_modified=False,real_robot_deployment='NONE')


def _cluster_interval(scene_successes):
    """Exact percentile interval for a 20-scene grouped empirical bootstrap."""
    if len(scene_successes)!=20 or any(x not in (0,1,2) for x in scene_successes):
        raise PermissionError('PV1_P5_EXACT_TWENTY_TWO_BANK_SCENE_CLUSTERS')
    counts=[scene_successes.count(x) for x in range(3)];dist=[1.0]
    probabilities=[c/20 for c in counts]
    for _ in range(20):
        nxt=[0.0]*(len(dist)+2)
        for total,p in enumerate(dist):
            for value,q in enumerate(probabilities):nxt[total+value]+=p*q
        dist=nxt
    def quantile(q):
        acc=0.0
        for total,p in enumerate(dist):
            acc+=p
            if acc+1e-15>=q:return total/40
        return 1.0
    return dict(method='EXACT_EMPIRICAL_BOOTSTRAP_20_SCENE_CLUSTERS',estimate=sum(scene_successes)/40,
        lower_95=quantile(.025),upper_95=quantile(.975),cluster_success_counts={'0':counts[0],'1':counts[1],'2':counts[2]},
        deployment_reliability_claim=False)


def acceptance(population_bytes,*,freeze,evidence,resource_final):
    """Validate all raw P5 attempt evidence and compute the frozen research grade."""
    population,population_sha=_read(population_bytes)
    if population_sha!=POPULATION:raise PermissionError('PV1_P5_ORIGINAL_FROZEN_POPULATION_BYTES_REQUIRED')
    plan=final_confirmation(population['groups'],freeze)
    cards={c['condition_id']:c for p in plan['panels'].values() for c in p['cards']}
    attempts={};terminal={};attempt_hashes={}
    for item in evidence:
        row,row_sha=_read(item['adjudication']);result,result_sha=_read(item['result'])
        audit,audit_sha=_read(item['audit']);ready,ready_sha=_read(item['model_ready'])
        identifier=row.get('attempt_id');condition=row.get('condition_id');card=cards.get(condition)
        if card is None or identifier in attempts or identifier not in (card['id'],card['id']+'-REPLACEMENT-01'):
            raise PermissionError('PV1_P5_ONLY_ONE_ORIGINAL_AND_ONE_ENGINEERING_REPLACEMENT_PER_CONDITION')
        identity=ready.get('identity',{});proof=ready.get('native_pre_model_qualification',{})
        if (row.get('model')!=freeze['candidate'] or result.get('id')!=identifier
                or row.get('result_sha256')!=result_sha or row.get('split')!='LOCKBOX'
                or any(row.get(k)!=card[k] for k in ('task','scene_group','noise_bank','family'))
                or row.get('teacher_takeover') is not False or row.get('live_native_failures')!=0
                or row.get('status')=='ACTUAL_SAFETY_FAIL' or row.get('failure_stage')=='S1_INTEGRITY'
                or ready.get('status')!='PV1_NATIVE_MODEL_READY' or ready.get('label')!=card['pv1_session_label']
                or identity.get('label')!=freeze['candidate'] or identity.get('global_step')!=4096
                or identity.get('checkpoint_manifest_sha256')!=freeze['checkpoint_manifest_sha256']
                or identity.get('statistics_sha256')!=STATS
                or proof.get('status')!='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
                or proof.get('heartbeat_elapsed_s',0)<600 or proof.get('transport_cycles')!=80
                or proof.get('new_model_queries') or proof.get('new_physics')
                or not re.fullmatch('[0-9a-f]{64}',proof.get('receipt_sha256',''))
                or audit.get('attempt_id')!=identifier or audit.get('adjudication')!=row
                or audit.get('errors') or audit.get('actual_native_safety_flag_rows')!=0
                or audit.get('model_autonomous_evaluation') is not True
                or audit.get('new_integrations')!=0 or audit.get('new_model_queries')!=0
                or audit.get('status') not in ('PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT','PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE')
                or not set(AUDIT_CHECKS)<=set(audit.get('checks',{}))
                or any(audit['checks'][k] is not True for k in AUDIT_CHECKS)):
            raise PermissionError('PV1_P5_EXACT_SHARED_MODEL_SAFETY_AND_INDEPENDENT_AUDIT')
        valid=row.get('status') in ('SUCCESS','VALID_TASK_FAIL') and row.get('pipeline_trustworthy') is True
        if valid:
            chain,chain_sha=_read(item.get('causal_chain'))
            if (chain.get('status')!='PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN'
                    or chain.get('attempt_id')!=identifier or chain.get('adjudication_sha256')!=row_sha
                    or chain.get('identity')!=identity or chain.get('new_integrations') or chain.get('new_model_queries')
                    or chain.get('executed_ACKs',-1)*200!=row.get('main_integrations')
                    or not chain.get('executed_ACKs',81)<=chain.get('draws',-1)<=80):
                raise PermissionError('PV1_P5_ACTUAL_FRESH_OBSERVATION_ACTION_ACK_CHAIN_REQUIRED')
            if row['status']=='SUCCESS' and (row.get('main_integrations')!=16000 or row.get('duration_s')!=32.
                    or row.get('RGB_pairs')!=641 or row.get('S1_rejections')!=0
                    or row.get('failure_stage')!='COMPLETE_SUCCESS'
                    or any(audit.get('score',{}).get(k) is not True
                        for k in ('complete_32s','clock_valid','safety_valid','persistent_success'))):
                raise PermissionError('PV1_P5_SUCCESS_REQUIRES_FULL_SAFE_PERSISTENT_32S_TASK')
            if condition in terminal:raise PermissionError('PV1_P5_VALID_TASK_OUTCOME_NEVER_REPEATED')
            terminal[condition]=row
        elif row.get('status')=='ENGINEERING_INCOMPLETE' and row.get('pipeline_trustworthy') is False:
            diagnosis,diagnosis_sha=_read(item.get('engineering_diagnosis'))
            if (diagnosis.get('status')!='CONFIRMED_ENGINEERING_FAILURE_NO_POLICY_REPLAY'
                    or diagnosis.get('original_adjudication_sha256')!=row_sha
                    or diagnosis.get('original_result_sha256')!=result_sha
                    or diagnosis.get('native_process_terminal_verified') is not True
                    or diagnosis.get('unknown_effects')!=[] or diagnosis.get('policy_outcome_valid') is not False):
                raise PermissionError('PV1_P5_SETTLED_ENGINEERING_ATTEMPT_REQUIRED')
            if identifier.endswith('-REPLACEMENT-01'):
                terminal[condition]=row
            attempt_hashes[identifier]=dict(engineering_diagnosis_sha256=diagnosis_sha)
        else:raise PermissionError('PV1_P5_UNKNOWN_OR_UNSAFE_OUTCOME')
        attempts[identifier]=row
        attempt_hashes.setdefault(identifier,{}).update(adjudication_sha256=row_sha,result_sha256=result_sha,
            independent_audit_sha256=audit_sha,model_ready_sha256=ready_sha)
    for condition,card in cards.items():
        original=attempts.get(card['id']);replacement=attempts.get(card['id']+'-REPLACEMENT-01')
        if original is None:raise PermissionError('PV1_P5_ALL_80_ORIGINAL_CONDITIONS_REQUIRED')
        if replacement is not None and not (original['status']=='ENGINEERING_INCOMPLETE' and not original['pipeline_trustworthy']):
            raise PermissionError('PV1_P5_REPLACEMENT_ONLY_AFTER_DIAGNOSED_ENGINEERING_ORIGINAL')
        if condition not in terminal:terminal[condition]=replacement or original
    if len(terminal)!=80:raise PermissionError('PV1_P5_EXACT_80_CONDITIONS_IN_DENOMINATOR')
    tasks={};family={}
    for task in ('T1','T2'):
        rows=[r for r in terminal.values() if r['task']==task];success=sum(r['status']=='SUCCESS' for r in rows)
        rejects=sum(r.get('S1_rejections',0)>0 or r.get('failure_stage')=='S1_PRE_EXECUTION_REJECT' for r in rows)
        scenes={g:[r for r in rows if r['scene_group']==g] for g in {r['scene_group'] for r in rows}}
        scene_success=[sum(r['status']=='SUCCESS' for r in pair) for pair in scenes.values()]
        tasks[task]=dict(successes=success,total=40,S1_rejected_conditions=rejects,
            scene_groups_2_of_2=sum(x==2 for x in scene_success),clustered_95_interval=_cluster_interval(scene_success))
        family[task]={f:dict(successes=sum(r['status']=='SUCCESS' for r in rows if r['family']==f),
            total=sum(r['family']==f for r in rows)) for f in ('F0','F1','F2')}
    validated=all(tasks[t]['successes']>=32 and tasks[t]['scene_groups_2_of_2']>=14
        and tasks[t]['S1_rejected_conditions']<=4 and all(x['successes']*2>=x['total'] for x in family[t].values()) for t in tasks)
    beta=all(tasks[t]['successes']>=28 and tasks[t]['scene_groups_2_of_2']>=12
        and tasks[t]['S1_rejected_conditions']<=8 and all(x['successes']*2>=x['total'] for x in family[t].values()) for t in tasks)
    grade='PV1_VALIDATED' if validated and freeze['maximum_possible_grade']=='PV1_VALIDATED' else 'PV1_BETA' if beta else 'BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED'
    if (resource_final.get('schema')!='PI0294_PV1_RESOURCE_FINAL_V1'
            or resource_final.get('status')!='PASS_FROZEN_FINAL_RESOURCE_SETTLEMENT'
            or resource_final.get('unknown_effects')!=[]):
        raise PermissionError('PV1_P5_FROZEN_RESOURCE_FINAL_WITH_NO_UNKNOWN_EFFECTS')
    return dict(schema='PI0294_PV1_ACCEPTANCE_V1',authority_release=AUTHORITY,status='PASS_COMPLETE_LOCKBOX80_ADJUDICATION',
        grade=grade,candidate=freeze['candidate'],checkpoint_manifest_sha256=freeze['checkpoint_manifest_sha256'],
        conditions=80,all_attempts=attempt_hashes,tasks=tasks,families=family,real_native_safety_faults=0,
        Teacher_takeover=False,evaluation_training_updates=0,original_S1_and_score=True,
        resource_final_sha256=_digest(resource_final),selected=freeze['candidate'] if grade!='BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED' else 'NONE',
        real_robot_deployment='NONE',deployment_reliability_claim=False)


def delivery_manifest(*,freeze,acceptance,artifacts,resource_final,checkpoint_retirement):
    required=('shared_weights','processor','statistics','runtime','precision','action_schema','dependency_lock','offline_inference_entry')
    if (set(artifacts)!=set(required) or acceptance.get('candidate')!=freeze.get('candidate')
            or acceptance.get('checkpoint_manifest_sha256')!=freeze.get('checkpoint_manifest_sha256')
            or resource_final.get('status')!='PASS_FROZEN_FINAL_RESOURCE_SETTLEMENT'
            or checkpoint_retirement.get('schema')!='PI0294_PV1_CHECKPOINT_RETIREMENT_LIST_V1'):
        raise PermissionError('PV1_P5_EXACT_POLICY_DELIVERY_COMPONENTS_AND_FINAL_RECEIPTS')
    for name,item in artifacts.items():
        if (set(item)!={'uri','sha256'} or not isinstance(item['uri'],str) or not item['uri']
                or not re.fullmatch('[0-9a-f]{64}',item['sha256'])):
            raise PermissionError('PV1_P5_HASHED_POLICY_ARTIFACT:'+name)
    if artifacts['shared_weights']['sha256']!=freeze['checkpoint_manifest_sha256']:
        raise PermissionError('PV1_P5_PACKAGE_MUST_USE_ACCEPTED_SHARED_WEIGHTS')
    ordered={name:artifacts[name] for name in required}
    return dict(schema='PI0294_PV1_POLICY_MANIFEST_V1',authority_release=AUTHORITY,
        candidate=freeze['candidate'],grade=acceptance['grade'],shared_weights_for_tasks=['T1','T2'],
        task_instructions=dict(INSTRUCTIONS),action_schema=freeze['action_schema'],artifacts=ordered,
        acceptance_sha256=_digest(acceptance),resource_final_sha256=_digest(resource_final),
        checkpoint_retirement_sha256=_digest(checkpoint_retirement),default_mode='OFFLINE_DRY_RUN',
        robot_or_WBC_connection_default=False,WBC_modified=False,real_robot_deployment='NONE')


def delivery_documents(manifest):
    """Render the two sanitized text members; raw weights, videos and paths stay outside Git."""
    grade=manifest['grade'];candidate=manifest['candidate']
    card=(f"# POLICY_CARD\n\nCandidate: `{candidate}`\n\nQualification: **{grade}**.\n\n"
        "This is one shared SOURCE_FIXED_BASE_ARX5 task-space policy for T1 pick-place and T2 push. "
        "It depends on the original S1 gate and frozen simulator scoring. WBC, plant and gripper PD are unchanged. "
        "Evaluation is paused simulation, not real-time control; real-robot deployment is NONE.\n\n"
        "Known failures and every LOCKBOX outcome are indexed by ACCEPTANCE.json.\n")
    reproduce=("# REPRODUCE\n\n1. Resolve every artifact at the URI and SHA-256 in POLICY_MANIFEST.json.\n"
        "2. Run `python -m scripts.pi0294.pv1.offline_policy --manifest POLICY_MANIFEST.json --dry-run`.\n"
        "3. Use the approved simulation scope only after its independent lease, resource and candidate checks pass.\n\n"
        "The offline entry never connects to a robot or WBC. Loading, one generation, reset, close and failure-stop "
        "are covered by CPU contract tests; this document does not authorize execution.\n")
    return {'POLICY_CARD.md':card,'REPRODUCE.md':reproduce}
