"""Pure once accounting for the exact saved GEN1 counter-guard failure.

This is neither a native audit nor a physics dispatcher. All completed native
effects and old files stay charged. It never admits a scientific failure as a
success and never replays an unknown write.
"""
from copy import deepcopy
import math
from . import amd_adoption as a

FAILED_SOURCE='363d5eb753d3f56d50883b5df220ede00ec5b647'
CLASSIFICATION='ENGINEERING_COUNTER_GUARD_INTERRUPTION_WITH_SAVED_NATIVE_EFFECTS'


def failure_proposal(value, audit, *, now_epoch):
    mark=a.validate(value);old=mark['teacher_pilot']
    a.need(old['status']=='RESERVED_FIRST_ATTEMPTS_ONLY', 'FAILURE_ONCE_TERMINAL_NOT_REPLAY')
    a.need(audit['status']=='PASS_SAVED_EFFECTS_ENGINEERING_FAILURE_RECONCILED_NO_REPLAY'
        and audit['source']==FAILED_SOURCE and audit['unknown_effects']==[]
        and audit['all_original_files_retained'] is True and audit['raw_member_SHA_verified'] is True
        and audit['admitted_demonstrations']==0 and audit['unattempted_conditions']==10
        and all(audit[k]==0 for k in ('new_constructors','new_integrations','new_GPU','new_PBS')),
        'EXACT_INDEPENDENT_SAVED_FAILURE_PROOF')
    a.need(mark['T_CD']<=audit['epoch']<=now_epoch<mark['deadline_epoch']
        and set(audit['workers'])==set(old['workers'])=={'0','1'},'FAILURE_REAL_TIME_AND_TWO_WORKERS')
    totals=dict(episode_starts=0,live_ticks=0,shadow_ticks=0,engine_starts=0)
    cap_errors=0
    for key,row in audit['workers'].items():
        bound=old['workers'][key]
        a.need(row['manifest_sha256']==bound['manifest_sha256']
            and row['full_first_attempt_ids']==bound['attempt_ids']
            and row['attempt_id']==bound['attempt_ids'][0]
            and row['source']==FAILED_SOURCE and row['actual_OS_returncode']==1
            and row['classification']==CLASSIFICATION and row['scientific_negative'] is False
            and row['data_admitted'] is False and row['all_raw_member_hashes_verified'] is True
            and row['episode_starts']==1 and row['engine_starts']==81
            and all(row[k]==0 for k in ('model_loads','queries','updates','native_safety_failures','S1_rejections')),
            'BOUND_ENGINEERING_FAILURE_NOT_VALID_REJECTION_OR_TASK_FAILURE')
        a.need(type(row['live_ticks'])is int and 0<row['live_ticks']<=15200
            and type(row['shadow_ticks'])is int and row['shadow_ticks']==15200
            and row['persisted_live_lower_bound']<=row['live_ticks']
            and row['last_persisted_counter_is_lower_bound'] is True,
            'ACTUAL_RAW_CLOCK_INSTEAD_OF_STALE_PROGRESS')
        if row['error']=='PermissionError: GEN1_AMD_ADOPTION_TASK_OR_READONLY_AUDIT_CAP_BEFORE_CALL':cap_errors+=1
        else:a.need(row['error']=='TimeoutError: PV1_CAPTURE_ORIGINAL_BOUND_OR_STOP','ONLY_EXACT_SHARED_STOP')
        for name in ('raw_manifest_sha256','worker_receipt_sha256'):
            a.need(type(row[name])is str and len(row[name])==64,'BOUND_RAW_METADATA')
        for k in totals:totals[k]+=row[k]
    a.need(cap_errors>=1 and totals['episode_starts']==audit['actual_episode_starts']==2
        and totals['live_ticks']==audit['live_ticks']and totals['shadow_ticks']==audit['shadow_ticks']
        and totals['engine_starts']==audit['engine_starts']==162,'INDEPENDENT_DUAL_TOTALS')
    cost=audit['Core_CPU_s']
    a.need(type(cost)in(int,float)and math.isfinite(cost)and 0<cost<=43200,
        'ACTUAL_OWNER_CHILD_CPU_NO_DOUBLE_SUM')
    out=deepcopy(value);row=out[a.FIELD]['teacher_pilot']
    row.update(status='NATIVE_ENGINEERING_TERMINAL',actual=dict(**totals,Core_CPU_s=cost,
        raw_payload_bytes=audit['raw_payload_bytes'],unknown_effects=[],models=0,queries=0,updates=0,
        admitted_demonstrations=0,independent_audit_sha256=a.sha(a.raw(audit))),
        failed_condition_ids=[audit['workers'][k]['attempt_id']for k in ('0','1')],
        unattempted_condition_ids=[x for k in ('0','1')for x in old['workers'][k]['attempt_ids'][1:]],
        all_old_costs_and_files_preserved=True,unused_reservation_not_training_admission=True)
    out['revision']+=1
    a.need(all(out[k]==v for k,v in value.items()if k not in(a.FIELD,'revision')),
        'OLD_NEURAL_FINANCIAL_HISTORY_AND_OTHER_WORK_UNCHANGED')
    return dict(proposed_global=out,actual=deepcopy(row['actual']),new_PBS=0,new_GPU=0,
        native_training_or_physics_admission=False)


def recovery_allocation(value, *, packet_bytes, workers, closed_leases, now_epoch):
    """One cold engineering continuation, with every failed physical cost kept.

    The twelve logical layouts/tasks stay fixed. Two interrupted physical
    attempts remain in the predecessor; they are not renamed as zero effects
    or scientific failures. This does not authorize a replacement after S1,
    task-quality or native-safety failure.
    """
    from .amd_capture import cards,COUNTER_POLICY
    mark=a.validate(value);old=mark['teacher_pilot']
    a.need(old['status']=='NATIVE_ENGINEERING_TERMINAL'and 'engineering_recovery'not in old
        and old['all_old_costs_and_files_preserved'] is True
        and old['actual']['unknown_effects']==[]and old['actual']['admitted_demonstrations']==0,
        'ONCE_KNOWN_ENGINEERING_BOUNDARY_ONLY')
    a.need(mark['T_CD']<=now_epoch<mark['stop_new_training_epoch']-18.25*3600,
        'FULL_EXPORT_TRAINING_TIME_REMAINS')
    a.need(set(closed_leases)=={'owner','0','1'}and set(workers)=={'0','1'},
        'THREE_ACTUAL_CLOSURES_AND_TWO_DISJOINT_WORKERS')
    for key,lease in closed_leases.items():
        a.need(lease['status']=='CLOSED'
            and lease['terminal_kind']=='KNOWN_ENGINEERING_FAILURE_WITH_REAL_PHYSICS'
            and lease['independent_failure_audit_sha256']==old['actual']['independent_audit_sha256']
            and lease['old_costs_and_raw_preserved']is True and lease['admitted_demonstrations']==0,
            'EXACT_CLOSED_FAILURE_CUSTODY')
        if key!='owner':a.need(lease['manifest_sha256']==old['workers'][key]['manifest_sha256'],
            'CLOSED_WORKER_IDENTITY')
    for index in (0,1):
        row=workers[str(index)];run='gen1-amd-teacher-r1-'+str(index).zfill(2)
        a.need(row['run_uuid']==run and row['root'].endswith('/children/'+run)
            and row['root']!=old['workers'][str(index)]['root']
            and row['attempt_ids']==[x['id']for x in cards(packet_bytes,index)]
            and row['counter_policy']==COUNTER_POLICY
            and type(row['manifest_sha256'])is str and len(row['manifest_sha256'])==64,
            'NEW_COLD_ROOT_SAME_TWELVE_LOGICAL_CONDITIONS')
    a.need(len({x['root']for x in workers.values()})==2,'DISJOINT_REPAIR_OUTPUTS')
    out=deepcopy(value);row=dict(status='RESERVED_ENGINEERING_CONTINUATION',workers=deepcopy(workers),
        logical_conditions=12,prior_failed_physical_attempts=2,possible_total_physical_attempts=14,
        episode_starts=12,physics_ticks=384000,engine_starts=1056,Core_CPU_s=43200,
        prior_effects_remain_charged=True,valid_scientific_failure_replay=False,
        counter_policy=COUNTER_POLICY,closed_predecessor_leases_sha256=a.sha(a.raw(closed_leases)))
    out[a.FIELD]['teacher_pilot']['engineering_recovery']=row;out['revision']+=1
    return dict(proposed_global=out,row=row,new_GPU=0,new_PBS=0)
