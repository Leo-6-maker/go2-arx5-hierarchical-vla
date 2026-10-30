"""Build training admission from complete, byte-bound B0 evidence, not scores."""
import hashlib
import json
import re

from scripts.pi0294.e1.scope import STATS
from .dataset import POPULATION
from .decision import AUTHORITY
from .evaluation_plan import development,remaining_panel
from .recipe import PARENT


AUDIT_CHECKS = (
    'raw_manifest_all_sizes_and_hashes',
    'actual_execution_source_not_later_root_manifest',
    'adjudication_result_hash',
    'original_submitted_installed_committed_prefix_labels',
    'original_20Hz_clock_tolerance',
    'independent_saved_state_score_exact',
    'actual_contact_tick_coverage',
    'actual_highrate_clock_and_count',
    'original_S1_no_action_modification',
    'original_S1_live_shadow_parity',
    'shadow_did_not_mutate_live',
    'S1_rejection_accounting',
    'paired_actual_camera_names',
    'actual_RGB_count',
)


def _read(raw):
    if not isinstance(raw,bytes):
        raise PermissionError('PV1_BASELINE_REQUIRES_ACTUAL_FILE_BYTES_NOT_HASH_ASSERTIONS')
    value=json.loads(raw)
    if not isinstance(value,dict):
        raise PermissionError('PV1_BASELINE_EVIDENCE_JSON_OBJECT')
    return value,hashlib.sha256(raw).hexdigest()


def _validated_attempts(population_bytes,*,route,new_bank_manifest_sha256,evidence,
        candidate='B0_CORR',checkpoint_manifest_sha256=PARENT):
    """Evidence contains original adjudication/result/audit bytes for ALL attempts."""
    population,population_sha=_read(population_bytes)
    if population_sha!=POPULATION:
        raise PermissionError('PV1_BASELINE_ORIGINAL_FROZEN_POPULATION_BYTES_REQUIRED')
    if (not re.fullmatch('[0-9a-f]{64}',checkpoint_manifest_sha256)
            or (candidate=='B0_CORR')!=(checkpoint_manifest_sha256==PARENT)):
        raise PermissionError('PV1_ACTUAL_DEV_CANDIDATE_CHECKPOINT_IDENTITY_REQUIRED')
    successor=candidate in ('C50','D25')
    if candidate=='E1':
        from scripts.pi0294.generalization import minimal_core_accounting as first512
        from scripts.pi0294.dec1.authority import BANK
        if route!='COVERAGE_FIRST' or new_bank_manifest_sha256!=BANK or checkpoint_manifest_sha256!=first512.CHECKPOINT:
            raise PermissionError('RAW512_ONLY_KNOWN_CLOSED_E1_ORIGINAL_PANEL')
        step=512;plan=first512.plan.candidate_plan(population['groups'],'E1',checkpoint_manifest_sha256=checkpoint_manifest_sha256)
    elif successor:
        from scripts.pi0294.dec1.authority import BANK
        from .mix_evaluation import candidate_plan
        if route!='COVERAGE_FIRST' or new_bank_manifest_sha256!=BANK:
            raise PermissionError('V5_ONLY_FROZEN_CD_SOURCE_AND_ORIGINAL_SINGLE_PANEL')
        step=1024;plan=candidate_plan(population['groups'],candidate)
    else:
        step=2048 if candidate=='B0_CORR' else 4096
        plan=development(population['groups'],route=route,candidate=candidate,
            new_bank_manifest_sha256=new_bank_manifest_sha256)
    cards={c['condition_id']:c for p in plan['panels'].values() for c in p['cards']}
    rows=[];result_hashes={};audit_hashes={};attempt_hashes={}
    for item in evidence:
        row,row_sha=_read(item['adjudication'])
        result,result_sha=_read(item['result'])
        audit,audit_sha=_read(item['audit'])
        ready,ready_sha=_read(item['model_ready'])
        identifier=row['attempt_id'];condition=row['condition_id']
        if identifier in attempt_hashes:
            raise PermissionError('PV1_BASELINE_DUPLICATE_ATTEMPT_NOT_EXTRA_EVIDENCE')
        if condition not in cards:
            raise PermissionError('PV1_BASELINE_ONLY_ORIGINAL_DEV48_CONDITIONS')
        card=cards[condition]
        identity=ready.get('identity',{});proof=ready.get('native_pre_model_qualification',{})
        if (ready.get('status')!='PV1_NATIVE_MODEL_READY' or ready.get('label')!=card['pv1_session_label']
                or not ready.get('job_id') or not ready.get('run_uuid')
                or identity.get('label')!=candidate or identity.get('global_step')!=step
                or identity.get('checkpoint_manifest_sha256')!=checkpoint_manifest_sha256
                or identity.get('statistics_sha256')!=STATS
                or identity.get('relay_identity',{}).get('source_sha')!=result['source_sha']
                or proof.get('status')!='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
                or proof.get('heartbeat_elapsed_s',0)<600 or proof.get('transport_cycles')!=80
                or proof.get('new_model_queries')!=0 or proof.get('new_physics')!=0
                or not re.fullmatch('[0-9a-f]{64}',proof.get('receipt_sha256',''))):
            raise PermissionError('PV1_BASELINE_ACTUAL_ORIGINAL_CORR_MODEL_AND_NATIVE_QUALIFICATION')
        if (row['model']!=candidate or row['result_sha256']!=result_sha or result['id']!=identifier
                or result['physics_steps']!=row['main_integrations'] or result['rgb_pairs']!=row['RGB_pairs']
                or row['duration_s']!=row['main_integrations']*.002
                or any(row[k]!=card[k] for k in ('task','split','scene_group','noise_bank','family'))
                or row['teacher_takeover'] is not False or row['live_native_failures']!=0
                or row['status']=='ACTUAL_SAFETY_FAIL' or row['failure_stage']=='S1_INTEGRITY'
                or audit['attempt_id']!=identifier or audit.get('errors') or audit.get('error')
                or audit.get('adjudication')!=row or not audit.get('checks')
                or any(v is not True for v in audit['checks'].values())
                or audit.get('actual_native_safety_flag_rows')!=0
                or audit.get('model_autonomous_evaluation') is not True
                or audit.get('execution_source_sha')!=result['source_sha']
                or not audit.get('video')
                or audit.get('new_integrations')!=0 or audit.get('new_native_constructors')!=0
                or audit.get('new_model_queries')!=0):
            raise PermissionError('PV1_BASELINE_EXACT_IDENTITY_SAFETY_AND_READ_SIDE_AUDIT')
        if row['status'] in ('SUCCESS','VALID_TASK_FAIL'):
            if 'engineering_diagnosis' in item:
                raise PermissionError('PV1_BASELINE_VALID_POLICY_OUTCOME_IS_NOT_ENGINEERING')
            chain,chain_sha=_read(item.get('causal_chain'))
            if (chain.get('status')!='PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN'
                    or chain.get('attempt_id')!=identifier or chain.get('adjudication_sha256')!=row_sha
                    or chain.get('identity')!=identity or chain.get('new_integrations')!=0
                    or chain.get('new_model_queries')!=0
                    or type(chain.get('executed_ACKs')) is not int or type(chain.get('draws')) is not int
                    or chain.get('executed_ACKs')*200!=row['main_integrations']
                    or not chain['executed_ACKs']<=chain.get('draws',-1)<=80
                    or any(not re.fullmatch('[0-9a-f]{64}',chain.get(k,''))
                        for k in ('core_proof_sha256','hopper_proof_sha256'))):
                raise PermissionError('PV1_BASELINE_INDEPENDENT_ACTUAL_CAUSAL_CHAIN_REQUIRED')
            if (row['pipeline_trustworthy'] is not True
                    or audit['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
                    or not set(AUDIT_CHECKS)<=set(audit['checks'])):
                raise PermissionError('PV1_BASELINE_COMPLETE_ACTUAL_RAW_AUDIT_FOR_EACH_VALID_OUTCOME')
            if row['status']=='SUCCESS' and (row['main_integrations']!=16000 or row['duration_s']!=32.
                    or row['RGB_pairs']!=641 or row['S1_rejections']!=0
                    or row['failure_stage']!='COMPLETE_SUCCESS'
                    or any(audit['score'].get(k) is not True
                        for k in ('complete_32s','clock_valid','safety_valid','persistent_success'))):
                raise PermissionError('PV1_BASELINE_SUCCESS_REQUIRES_FULL_ACTUAL_TASK_NOT_ONLY_TRUSTED_PIPELINE')
            if condition in result_hashes:
                raise PermissionError('PV1_BASELINE_VALID_CONDITION_MUST_NOT_BE_REPEATED')
            result_hashes[condition]=row_sha;audit_hashes[condition]=audit_sha
        elif row['status']=='ENGINEERING_INCOMPLETE' and row['pipeline_trustworthy'] is False:
            if audit['status'] not in ('PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT','PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'):
                raise PermissionError('PV1_BASELINE_ORIGINAL_ENGINEERING_ATTEMPT_AUDIT_REQUIRED')
            diagnosis,diagnosis_sha=_read(item.get('engineering_diagnosis'))
            if (diagnosis.get('status')!='CONFIRMED_ENGINEERING_FAILURE_NO_POLICY_REPLAY'
                    or diagnosis.get('original_adjudication_sha256')!=row_sha
                    or diagnosis.get('original_result_sha256')!=result_sha
                    or diagnosis.get('native_process_terminal_verified') is not True
                    or diagnosis.get('unknown_effects')!=[]
                    or diagnosis.get('policy_outcome_valid') is not False
                    or not diagnosis.get('evidence_sha256')
                    or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v)
                        for v in diagnosis['evidence_sha256'].values())):
                raise PermissionError('PV1_BASELINE_SEPARATE_SETTLED_ENGINEERING_DIAGNOSIS_REQUIRED')
            # Add replacement eligibility only to the derived prior-row view;
            # the original adjudication and its independent audit stay byte-exact.
            row=dict(row,confirmed_engineering_failure=True,engineering_diagnosis_sha256=diagnosis_sha)
        else:raise PermissionError('PV1_BASELINE_UNKNOWN_OUTCOME_NOT_ADMITTED')
        attempt_hashes[identifier]=dict(adjudication_sha256=row_sha,result_sha256=result_sha,
            independent_audit_sha256=audit_sha,model_ready_sha256=ready_sha)
        if row['status'] in ('SUCCESS','VALID_TASK_FAIL'):
            attempt_hashes[identifier]['causal_chain_sha256']=chain_sha
        if 'engineering_diagnosis' in item:
            attempt_hashes[identifier]['engineering_diagnosis_sha256']=hashlib.sha256(item['engineering_diagnosis']).hexdigest()
        rows.append(row)
    # Both complete admission and V4 partial history share every per-attempt
    # check above. Only the separate complete-admission entry requires all48.
    return plan,rows,result_hashes,audit_hashes,attempt_hashes


def admission(population_bytes,*,route,new_bank_manifest_sha256,evidence,
        candidate='B0_CORR',checkpoint_manifest_sha256=PARENT):
    """Original complete DEV48 gate, unchanged for training and selection."""
    if candidate in ('C50','D25','E1'):
        raise PermissionError('V5_CD_SINGLE_PANEL_HISTORY_IS_NOT_LEGACY_DEV48_TRAINING_ADMISSION')
    plan,rows,result_hashes,audit_hashes,attempt_hashes=_validated_attempts(population_bytes,
        route=route,new_bank_manifest_sha256=new_bank_manifest_sha256,evidence=evidence,
        candidate=candidate,checkpoint_manifest_sha256=checkpoint_manifest_sha256)
    cards={c['condition_id']:c for p in plan['panels'].values() for c in p['cards']}
    # The existing validator enforces no repeated valid failures, and the one
    # diagnosed engineering replacement with its original attempt preserved.
    for panel in plan['panels'].values():
        if remaining_panel(rows,panel):
            raise PermissionError('PV1_BASELINE_INCOMPLETE_DEV48_NO_TRAINING')
    if set(result_hashes)!=set(cards):
        raise PermissionError('PV1_BASELINE_EACH_DEV_CONDITION_NEEDS_VALID_AUDITED_RESULT')
    successes={task:sum(r['task']==task and r['status']=='SUCCESS' for r in rows) for task in ('T1','T2')}
    return dict(schema='PI0294_PV1_B0_ACTUAL_DEV_ADMISSION_V1' if candidate=='B0_CORR' else 'PI0294_PV1_CANDIDATE_ACTUAL_DEV_ADMISSION_V1',
        status='PASS_COMPLETE_ACTUAL_B0_DEV48_EVIDENCE' if candidate=='B0_CORR' else 'PASS_COMPLETE_ACTUAL_CANDIDATE_DEV48_EVIDENCE',authority_release=AUTHORITY,
        population_plan_sha256=POPULATION,checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        new_bank_manifest_sha256=new_bank_manifest_sha256,
        condition_result_sha256=result_hashes,independent_audit_sha256=audit_hashes,
        all_actual_attempts=attempt_hashes,real_native_faults=0,Teacher_takeover=False,
        all_condition_evidence_audits_PASS=True,successes=successes,
        task_success_rate_is_not_training_admission_gate=True,
        new_integrations=0,new_model_queries=0,new_training_updates=0)


def partial_history(population_bytes,*,candidate,evidence,checkpoint_manifest_sha256=None):
    """V4 validated supplied history; not training, selection or native GO.

    The owner must supply the complete original attempt inventory. This parser
    verifies its original bytes; it does not prove a remote directory is fully
    enumerated. Fresh native admission must bind that independent inventory.
    """
    from scripts.pi0294.dec1.authority import BANK
    from .evaluation_platform import ENDPOINTS
    if candidate in ('C50','D25','E1'):
        return _successor_history(population_bytes,candidate=candidate,evidence=evidence,
            checkpoint_manifest_sha256=checkpoint_manifest_sha256)
    if candidate not in ENDPOINTS:
        raise PermissionError('PV1_V4_ONLY_EXISTING_PRIMARY_DEV_ENDPOINTS')
    plan,rows,results,audits,attempts=_validated_attempts(population_bytes,route='COVERAGE_FIRST',
        new_bank_manifest_sha256=BANK,evidence=evidence,candidate=candidate,
        checkpoint_manifest_sha256=ENDPOINTS[candidate])
    remaining={label:remaining_panel(rows,panel) for label,panel in plan['panels'].items()}
    valid=len(results);eligible=sum(len(cards) for cards in remaining.values())
    return dict(schema='PI0294_V4_PRIMARY_DEV_VALIDATED_HISTORY_V1',authority_release=AUTHORITY,
        status='VALIDATED_SUPPLIED_ORIGINAL_ATTEMPTS_NOT_COMPLETE_ADMISSION',candidate=candidate,
        population_plan_sha256=POPULATION,checkpoint_manifest_sha256=ENDPOINTS[candidate],
        new_bank_manifest_sha256=BANK,prior_rows=rows,all_actual_attempts=attempts,
        condition_result_sha256=results,independent_audit_sha256=audits,
        valid_conditions=valid,remaining_eligible_conditions=eligible,
        exhausted_engineering_conditions=48-valid-eligible,remaining_panels=remaining,
        successes={task:sum(r['task']==task and r['status']=='SUCCESS' for r in rows) for task in ('T1','T2')},
        remote_attempt_inventory_complete_not_proven_here=True,complete_DEV48=valid==48,
        training_or_selection_admission=False,native_or_resource_admission=False,
        new_integrations=0,new_model_queries=0,new_training_updates=0)


def _successor_history(population_bytes,*,candidate,evidence,checkpoint_manifest_sha256):
    """Original per-attempt audit/causal checks on the predeclared CD24 panel."""
    from scripts.pi0294.dec1.authority import BANK
    from .successor_cd import TASK_SOURCE
    first512=candidate=='E1'
    if first512:
        from scripts.pi0294.generalization.minimal_core_accounting import CHECKPOINT
        if checkpoint_manifest_sha256!=CHECKPOINT:raise PermissionError('RAW512_EXACT_KNOWN_E1_HISTORY')
    if (type(checkpoint_manifest_sha256)is not str
            or not re.fullmatch('[0-9a-f]{64}',checkpoint_manifest_sha256)
            or checkpoint_manifest_sha256==PARENT):
        raise PermissionError('V5_EXPLICIT_CLOSED_1024_ENDPOINT_REQUIRED_FOR_HISTORY')
    plan,rows,results,audits,attempts=_validated_attempts(population_bytes,route='COVERAGE_FIRST',
        new_bank_manifest_sha256=BANK,evidence=evidence,candidate=candidate,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256)
    remaining={label:remaining_panel(rows,panel)for label,panel in plan['panels'].items()}
    valid=len(results);eligible=sum(len(cards)for cards in remaining.values())
    if len(attempts)-valid>4:raise PermissionError('V5_WHOLE_PAIR_ENGINEERING_MAX_FOUR')
    return dict(schema='ARX5_MINIMAL_FIXED512_ORIGINAL_DEV24_VALIDATED_HISTORY_V1'if first512 else'V5_CD_ORIGINAL_DEV24_VALIDATED_HISTORY_V1',authority_release=TASK_SOURCE,
        status='VALIDATED_SUPPLIED_ORIGINAL_ATTEMPTS_NOT_NATIVE_OR_SELECTION_ADMISSION',candidate=candidate,
        population_plan_sha256=POPULATION,checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        checkpoint_step=512 if first512 else 1024,prior_rows=rows,all_actual_attempts=attempts,
        condition_result_sha256=results,independent_audit_sha256=audits,valid_conditions=valid,
        remaining_eligible_conditions=eligible,exhausted_engineering_conditions=24-valid-eligible,
        remaining_panels=remaining,successes={task:sum(r['task']==task and r['status']=='SUCCESS'for r in rows)for task in ('T1','T2')},
        remote_attempt_inventory_complete_not_proven_here=True,complete_DEV24=valid==24,
        independent_generalization=False,training_or_selection_admission=False,
        native_or_resource_admission=False,new_integrations=0,new_model_queries=0,new_training_updates=0)
