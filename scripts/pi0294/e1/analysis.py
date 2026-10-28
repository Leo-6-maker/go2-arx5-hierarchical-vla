"""Pure evidence aggregation: episode-macro probe errors and separate outcome classes."""
from collections import Counter
import math
import statistics

from .remaining import BASE_LABELS

LAYOUTS={'PI0294/TRAIN/F0/000','PI0294/TRAIN/F1/000'}


def canonical_model(label):
    aliases={base+'-REMAINDER-03':base for base in BASE_LABELS}
    return aliases.get(label,label)


def audited_outcome(adjudication,audit,*,autonomous,excluded_scope_deviation=False):
    """Reporting only: missing audits must never turn into eligible physics retries."""
    if type(autonomous) is not bool or type(excluded_scope_deviation) is not bool:
        raise ValueError('EXPLICIT_AUTONOMY_AND_SCIENTIFIC_EXCLUSION_FLAGS')
    audit=audit or {};checks=audit.get('checks',{})
    if audit.get('adjudication',adjudication)!=adjudication or audit.get('attempt_id',adjudication['attempt_id'])!=adjudication['attempt_id']:
        raise ValueError('ACTUAL_AUDIT_ADJUDICATION_IDENTITY_MISMATCH')
    if 'model_autonomous_evaluation' in audit and audit['model_autonomous_evaluation'] is not autonomous:
        raise ValueError('ACTUAL_TEACHER_TAKEOVER_STATUS_MISMATCH')
    required={'raw_manifest_all_sizes_and_hashes','actual_execution_source_not_later_root_manifest','adjudication_result_hash',
        'original_submitted_installed_committed_prefix_labels','original_20Hz_clock_tolerance','independent_saved_state_score_exact',
        'actual_contact_tick_coverage','actual_highrate_clock_and_count','original_S1_no_action_modification',
        'original_S1_live_shadow_parity','shadow_did_not_mutate_live','S1_rejection_accounting','paired_actual_camera_names','actual_RGB_count'}
    verified=(audit.get('status')=='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT' and required.issubset(checks)
        and all(v is True for v in checks.values()) and audit.get('errors')==[])
    if verified and (type(audit.get('actual_native_safety_flag_rows')) is not int
            or audit['actual_native_safety_flag_rows']<0 or type(audit.get('score',{}).get('persistent_success')) is not bool):
        raise ValueError('ACTUAL_AUDIT_SAFETY_AND_TASK_RESULT_REQUIRED')
    if verified and adjudication['status']=='SUCCESS' and (audit['score']['persistent_success'] is not True
            or adjudication['duration_s']!=32. or audit['actual_native_safety_flag_rows']!=0):
        raise ValueError('SUCCESS_REQUIRES_FULL_ORIGINAL_TASK_AND_NATIVE_SAFETY_EVIDENCE')
    score=audit.get('score') or {}
    return dict(adjudication,adjudication_pipeline_trustworthy=adjudication['pipeline_trustworthy'],
        pipeline_trustworthy=adjudication['pipeline_trustworthy'] and verified,audit_verified=verified,
        evidence_audit_status=audit.get('status','MISSING'),persistent_success=score.get('persistent_success'),
        native_safety_flag_rows=audit.get('actual_native_safety_flag_rows'),native_duration_s=adjudication['duration_s'],
        model_autonomous_evaluation=autonomous,excluded_scope_deviation=excluded_scope_deviation)


def errors(rows):
    output={}
    for view,count in [('prefix8',8),('H40',40)]:
        output[view]={}
        for metric in ['XYZ_euclidean_m','SO3_rad','gripper_absolute']:
            episodes={};incomplete=set();bad_windows=0
            for row in rows:
                values=row['errors']['pointwise'][metric][:count]
                key=row['episode_id'];episodes.setdefault(key,[]).extend(values)
                if len(values)!=count or any(v is None or type(v) is bool or not math.isfinite(v) for v in values):
                    incomplete.add(key);bad_windows+=1
            valid={key:statistics.mean(values) for key,values in episodes.items() if values and key not in incomplete}
            output[view][metric]=dict(episodes=len(episodes),finite_complete_episodes=len(valid),
                episode_macro_mean=statistics.mean(valid.values()) if len(valid)==len(episodes) and valid else None,
                worst_episode_mean=max(valid.values()) if len(valid)==len(episodes) and valid else None,
                nonfinite_or_short_windows=bad_windows,
                expected_points=len(rows)*count,not_independent_samples=True)
    return output


def probe_summary(records):
    return {task:dict(windows=len(chosen),episodes=len({r['episode_id'] for r in chosen}),
        errors=errors(chosen),reference_gate_rejections=sum(r['gate']['decision']=='REJECT' for r in chosen),
        missing_close_events=sum(r['errors']['close_events']['missing'] for r in chosen),
        missing_open_events=sum(r['errors']['open_events']['missing'] for r in chosen))
        for task in ['T1','T2'] for chosen in [[r for r in records if r['task']==task]]}


def autonomous_summary(rows):
    result={}
    if any(r['task'] not in ['T1','T2'] for r in rows):raise ValueError('E1_EXACT_TWO_TASK_REPORTING_POPULATION')
    for label in sorted({canonical_model(r['model']) for r in rows}):
        result[label]={}
        for task in ['T1','T2']:
            chosen=[r for r in rows if canonical_model(r['model'])==label and r['task']==task and r['model_autonomous_evaluation'] and not r['excluded_scope_deviation']]
            if any(r['scene_group'] not in LAYOUTS for r in chosen):raise PermissionError('E1_REPORT_ONLY_FIXED_TRAIN_CONDITIONS_OR_EXPLICIT_EXCLUSION')
            selected=[]
            for layout in sorted(LAYOUTS):
                attempts=[r for r in chosen if r['scene_group']==layout]
                valid=[r for r in attempts if r['pipeline_trustworthy'] and r.get('audit_verified') is True and r['status'] in ['SUCCESS','VALID_TASK_FAIL']]
                if len(valid)>1:raise PermissionError('E1_REPEATED_VALID_RESULT_REQUIRES_EXPLICIT_SCOPE_ADJUDICATION')
                if len(attempts)>1:
                    originals=[r for r in attempts if r['model']==label];replacements=[r for r in attempts if r['model']==label+'-REMAINDER-03']
                    if (len(originals)!=1 or len(replacements)!=1 or originals[0]['status']!='ENGINEERING_INCOMPLETE'
                            or originals[0].get('adjudication_pipeline_trustworthy',originals[0]['pipeline_trustworthy'])):
                        raise PermissionError('E1_ONLY_ONE_SETTLED_ENGINEERING_REPLACEMENT_NOT_BEST_OF_REPEATS')
                selected.extend(valid)
            layouts={r['scene_group'] for r in chosen};complete=len(selected)==2
            successes=sum(r['status']=='SUCCESS' and r['persistent_success'] is True and r['native_safety_flag_rows']==0 and r['native_duration_s']==32. for r in selected)
            result[label][task]=dict(conditions_attempted=len(layouts),actual_attempts=len(chosen),planned_distinct_conditions=2,
                all_conditions_attempted=layouts==LAYOUTS,valid_policy_outcomes=len(selected),task_successes=successes,
                unverified_or_untrustworthy_attempts=sum(not r['pipeline_trustworthy'] or r.get('audit_verified') is not True for r in chosen),
                original_attempt_models=[r['model'] for r in chosen],
                failure_stages=dict(Counter(r['failure_stage'] for r in chosen)),
                complete_predeclared_panel=complete,
                source_task_competence_observed=complete and successes==2,
                DEV_generalization_proven=False,deployment_qualified=False)
    return result
