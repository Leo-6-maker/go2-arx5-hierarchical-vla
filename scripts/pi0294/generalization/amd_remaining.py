"""Original ten never-attempted GEN1 cells after saved T1 re-audit closes."""
from copy import deepcopy
import math
from . import amd_adoption as a

PROFILE='amd-gen1-teacher-capture-r2'
STATUS='RESERVED_TEN_UNATTEMPTED_ORIGINAL_CONDITIONS'
SOURCE='21aa09341f27848c840892bf1beb42c868123ebf'

def cards(packet_bytes,index):
    from .amd_capture import cards as original_cards
    values=original_cards(packet_bytes,index)
    a.need(values[0]['task']=='T1'and len(values)==6,'REMAINING_FIXED_FIRST_SAVED_T1')
    return values[1:]

def allocation(value,*,packet_bytes,workers,closed_reaudit_lease,now_epoch):
    from .amd_capture import COUNTER_POLICY
    mark=a.validate(value);old=mark['teacher_pilot']['engineering_recovery'];audit=old['saved_data_reaudit']
    a.need(old['status']=='NATIVE_AUDIT_ENGINEERING_TERMINAL'
        and audit['status']=='KNOWN_SAVED_RAW_REAUDIT_SETTLED'and 'remaining_conditions'not in old
        and set(audit['raw_admitted_cases'])=={'0','1'}and audit['actual']['unknown_native_effects']==[],
        'REMAINING_ONLY_AFTER_KNOWN_SAVED_DATA_AUDIT')
    a.need(closed_reaudit_lease['status']=='CLOSED'
        and closed_reaudit_lease['manifest_sha256']==audit['manifest_sha256']
        and closed_reaudit_lease['independent_terminal_audit_sha256']==audit['independent_terminal_audit_sha256'],
        'REMAINING_REAL_CLOSED_REAUDIT_ROLE')
    a.need(mark['T_CD']<=now_epoch<mark['stop_new_training_epoch']-18.25*3600,
        'REMAINING_EXPORT_TRAINING_TIME')
    a.need(set(workers)=={'0','1'},'REMAINING_TWO_DISJOINT_FINITE_WORKERS')
    ids=[]
    for index in (0,1):
        row=workers[str(index)];run='gen1-amd-teacher-r2-'+str(index).zfill(2)
        expected=[c['id']for c in cards(packet_bytes,index)];saved=audit['raw_admitted_cases'][str(index)]
        a.need(row['run_uuid']==run and row['root'].endswith('/children/'+run)
            and row['root']!=saved['raw_root']and row['attempt_ids']==expected
            and saved['attempt_id']not in expected and row['counter_policy']==COUNTER_POLICY
            and type(row['manifest_sha256'])is str and len(row['manifest_sha256'])==64,
            'REMAINING_NO_FULL_SAVED_CASE_OR_VALID_FAILURE_REPLAY')
        ids+=expected
    a.need(ids==old['unattempted_condition_ids']and len(set(ids))==10,'REMAINING_EXACT_ORIGINAL_TEN')
    out=deepcopy(value);row=dict(status=STATUS,workers=deepcopy(workers),logical_conditions=10,
        episode_starts=10,physics_ticks=320000,engine_starts=880,Core_CPU_s=36000,
        complete_saved_cases_not_recaptured=2,previous_partial_attempts_retained=2,
        new_GPU=0,new_PBS=0,valid_scientific_failure_replay=False,
        independent_saved_audit_sha256=audit['independent_terminal_audit_sha256'],
        closed_reaudit_lease_sha256=a.sha(a.raw(closed_reaudit_lease)))
    out[a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']=row;out['revision']+=1
    return dict(proposed_global=out,row=row,new_GPU=0,new_PBS=0)


def settlement(value,audit,*,now_epoch):
    """Known finite effects only; valid S1/quality failures stay negative."""
    mark=a.validate(value);old=mark['teacher_pilot']['engineering_recovery'];pool=old['remaining_conditions']
    a.need(pool['status']==STATUS and audit['status']=='PASS_INDEPENDENT_TEN_KNOWN_NATIVE_TERMINAL_EFFECTS'
        and audit['source']==SOURCE and audit['all_processes_terminal']is True
        and audit['worker_OS_returncodes']==[0,0]and audit['unknown_native_effects']==[]
        and audit['old_costs_and_raw_and_valid_failures_preserved']is True,
        'TEN_ACTUAL_FINITE_OS_TERMINALS_NOT_OBSERVATION_TIMEOUT')
    a.need(mark['T_CD']<=audit['epoch']<=now_epoch and math.isfinite(now_epoch),
        'TEN_FINANCIAL_ONLY_CURRENT_OR_LATE_CLOSE')
    a.need(set(audit['workers'])=={'0','1'}and all(audit[k]==0 for k in('models','queries','updates')),
        'TEN_NO_NEURAL_EFFECTS')
    ids=[];totals=dict(episode_starts=0,live_ticks=0,shadow_ticks=0,engine_starts=0,
        native_safety_failures=0,S1_rejections=0,independently_raw_admitted=0)
    outcomes=[]
    for key,worker in audit['workers'].items():
        bound=pool['workers'][key]
        a.need(worker['manifest_sha256']==bound['manifest_sha256']and worker['run_root']==bound['root']
            and worker['expected_ids']==bound['attempt_ids']and worker['attempted_ids']==bound['attempt_ids']
            and len(worker['cases'])==5 and worker['unknown_effects']==[]
            and worker['source_sha']==SOURCE and 0<worker['engine_starts']<=440,
            'TEN_EXACT_REGISTERED_WORKER_AND_FIVE_ORIGINAL_CONDITIONS')
        live=shadow=0
        for case,identifier in zip(worker['cases'],bound['attempt_ids']):
            a.need(case['attempt_id']==identifier and case['terminal']is True
                and case['unknown_effects']==[]and case['raw_and_custody_verified']is True
                and case['source_sha']==SOURCE and type(case['full_task_success'])is bool,
                'TEN_EACH_ORIGINAL_KNOWN_SAVED_CASE')
            for name in('live_ticks','shadow_ticks','native_safety_failures','S1_rejections'):
                a.need(type(case[name])is int and 0<=case[name]<=16000,'TEN_FINITE_NATIVE_EFFECT_COUNTERS')
            for name in('raw_manifest_sha256','audit_sha256','card_sha256','worker_receipt_sha256'):
                a.need(type(case[name])is str and len(case[name])==64,'TEN_BOUND_ORIGINAL_RAW_IDENTITIES')
            if case['full_task_success']:
                a.need(case['independent_raw_audit']=='PASS_INDEPENDENT_RAW_AUDIT'
                    and case['live_ticks']==16000 and case['paired_frames']==641
                    and case['native_safety_failures']==case['S1_rejections']==0,
                    'TEN_SUCCESS_IS_FULL_RAW_AUDITED_NOT_SCORE_ONLY')
            else:a.need(case['independent_raw_audit']in('PASS_INDEPENDENT_RAW_AUDIT',
                'PASS_INDEPENDENT_PARTIAL_NEGATIVE_WITNESS','AUDIT_FAIL_PRESERVED'),
                'TEN_VALID_NEGATIVE_PRESERVED_NOT_ADMITTED')
            ids.append(identifier);outcomes.append(deepcopy(case));live+=case['live_ticks'];shadow+=case['shadow_ticks']
            totals['episode_starts']+=1;totals['native_safety_failures']+=case['native_safety_failures'];totals['S1_rejections']+=case['S1_rejections'];totals['independently_raw_admitted']+=int(case['full_task_success'])
        a.need(live==worker['live_ticks']and shadow==worker['shadow_ticks'],'TEN_COUNTERS_JOIN_REAL_S1_AND_RAW')
        totals['live_ticks']+=live;totals['shadow_ticks']+=shadow;totals['engine_starts']+=worker['engine_starts']
    a.need(ids==old['unattempted_condition_ids']and len(set(ids))==10,'TEN_NO_SAVED_CASE_REPLAY_OR_DROPPED_NEGATIVE')
    cost=audit['Core_CPU_s'];a.need(type(cost)in(int,float)and math.isfinite(cost)and 0<cost<=36600,
        'TEN_OWNER_CHILD_CPU_PAID_ONCE_NOT_DOUBLE_SUMMED')
    out=deepcopy(value);row=out[a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']
    row.update(status='NATIVE_TEN_KNOWN_TERMINAL',actual=dict(**totals,Core_CPU_s=cost,
        unknown_native_effects=[],models=0,queries=0,updates=0),outcomes=outcomes,
        independent_terminal_audit_sha256=a.sha(a.raw(audit)),
        saved_two_T1_not_recaptured=True,valid_failures_preserved=True,
        original_R0_export_required=True,native_training_admitted=False)
    out['revision']+=1
    a.need(all(out[k]==v for k,v in value.items()if k not in(a.FIELD,'revision')),
        'TEN_ONLY_GEN_CLASSIFICATION_NO_OLD_FINANCIAL_REWRITE')
    return dict(proposed_global=out,actual=deepcopy(row['actual']),new_GPU=0,new_PBS=0)
