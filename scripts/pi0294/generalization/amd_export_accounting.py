"""One official GEN1 CPU export inside the original GLOBAL/onceIO system.

This is a pure proposal layer, not qsub, registration, a fresh quota proof or
permission to start a processor. Original C/D escrow and final work stay held.
"""
from copy import deepcopy
import math,re,uuid
from . import amd_adoption as a,amd_pilot_scope as scope
from scripts.pi0294.pv1 import mix_accounting as original

VERSION='GEN1_VANDA_ORIGINAL_OFFICIAL_EXPORT_CPU_V1'
PROFILE='vanda-gen1-official-export-cpu'
JOB_NAME='pi0294_gen1_export'
STATES=('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE','NATIVE_TERMINAL')
LIMITS=dict(ngpus=0,active_cpus=1,walltime_s=36600,CPU_s=36000,
    control_site_CPU_s=600,LOCAL_control_CPU_s=600,output_bytes=10*1024**3,
    constructors=0,physics_ticks=0,episode_starts=0,model_loads=0,
    neural_queries=0,optimizer_updates=0)


def rows(g):
    mark=a.validate(g);items=[x for x in g['PBS_submissions']if x.get('platform_data_version')==VERSION]
    a.need(len(items)<=1,'EXPORT_ONE_ORIGINAL_CPU_REQUEST_NOT_AUTO_RETRY')
    binding=mark.get('official_export')
    a.need(bool(items)==bool(binding),'EXPORT_GLOBAL_ROW_AND_ORIGINAL_MARKER_JOIN')
    for row in items:
        expected=dict(platform_data_version=VERSION,native_platform='VANDA_A40',project='CFP04-CF-007',
            job_name=JOB_NAME,physical_GPU_count=0,scope_grant_sha256=a.GRANT_SHA,
            T_CD=mark['T_CD'],deadline_epoch=mark['deadline_epoch'],limits=LIMITS)
        a.need(all(row.get(k)==v and type(row[k])is type(v)for k,v in expected.items()),
            'EXPORT_EXACT_ZERO_GPU_CPU_BOUNDS_AND_CLOCK')
        a.need(a.sha(a.raw(row['limits']))==a.sha(a.raw(LIMITS)),
            'EXPORT_EXACT_TYPED_CPU_AND_ZERO_EFFECT_LIMIT_FIELDS')
        for name,size in [('source_sha',40),('request_sha256',64),('run_manifest_sha256',64),
            ('closed_capture_proof_sha256',64),('raw_plan_sha256',64)]:
            a.need(type(row.get(name))is str and re.fullmatch('[0-9a-f]{'+str(size)+'}',row[name])is not None,
                'EXPORT_SEALED_IDENTITY:'+name)
        for name in('request_id','submission_nonce'):
            try:ok=str(uuid.UUID(row[name]))==row[name]
            except(ValueError,TypeError,AttributeError,KeyError):ok=False
            a.need(ok,'EXPORT_NEW_CANONICAL_REQUEST_ID')
        a.need(row['status']in STATES and (row['job_id']is None and row['status']==STATES[0]
            or type(row['job_id'])is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',row['job_id'])is not None),
            'EXPORT_REAL_JOB_OR_UNRESOLVED_INTENT')
        keys=('request_id','request_sha256','run_manifest_sha256','closed_capture_proof_sha256','raw_plan_sha256')
        a.need({k:binding[k]for k in keys}=={k:row[k]for k in keys},
            'EXPORT_SAME_REGISTERED_ROLE_NOT_OLD_TRAIN_REQUEST')
        before=binding['reservation_before'];a.need(re.fullmatch('[0-9a-f]{64}',binding['GLOBAL_before_sha256'])is not None
            and g['used']['PBS_submissions']>=before['PBS_submissions']+1
            and g['CPU_s']>=before['CPU_s']+600
            and g['new_storage_current_bytes']>=before['new_storage_current_bytes'],
            'EXPORT_REAL_PREPAID_CONTROL_AND_OLD_BALANCES_PRESERVED')
        if row['status']!=STATES[-1]:
            package=g[original.PACKAGE]['remaining_resources']
            a.need(g['CPU_reserved_s']>=package['Hopper_CPU_s']+36000
                and g['storage_reserved_bytes']>=package['Hopper_storage_bytes']+LIMITS['output_bytes'],
                'EXPORT_SEPARATE_ACTUAL_CPU_OUTPUT_HOLD_NO_C_D_ESCROW_BORROWING')
        if row['status']==STATES[-1]:
            a.need(type(row['native_exit_status'])is int and type(row['export_finished'])is bool
                and type(row['actual_CPU_s'])in(int,float)and math.isfinite(row['actual_CPU_s'])
                and 0<=row['actual_CPU_s']<=36000 and row['charged_output_upper_bytes']==LIMITS['output_bytes']
                and re.fullmatch('[0-9a-f]{64}',row['terminal_native_sha256'])is not None
                and re.fullmatch('[0-9a-f]{64}',row['terminal_export_audit_sha256'])is not None,
                'EXPORT_ACTUAL_TERMINAL_COST_AND_PRESERVED_OUTPUT_BOUND')
            a.need(g['CPU_s']>=before['CPU_s']+600+row['actual_CPU_s']
                and g['new_storage_current_bytes']>=before['new_storage_current_bytes']+LIMITS['output_bytes'],
                'EXPORT_TERMINAL_COSTS_REALLY_DEBITED_IN_ORIGINAL_LEDGER')
    return items


def raw_support(packet_bytes,outcomes):
    plan=scope.pilot_plan(packet_bytes);expected={x['attempt_id']:x for x in plan['attempts']}
    a.need(type(outcomes)is list and len(outcomes)==12
        and len({x['attempt_id']for x in outcomes})==12
        and {x['attempt_id']for x in outcomes}==set(expected),'EXPORT_ALL_TWELVE_RAW_OUTCOMES_REQUIRED')
    support=set()
    for row in outcomes:
        a.need(row['terminal']is True and row['unknown_effects']==[]
            and type(row['native_safety_failures'])is int and row['native_safety_failures']==0
            and type(row['full_task_success'])is bool,'EXPORT_KNOWN_SAFE_ALL_OUTCOMES_NOT_ONLY_SUCCESSES')
        if row['full_task_success']:
            a.need(row['independent_raw_audit']=='PASS_INDEPENDENT_RAW_AUDIT'
                and row['raw_and_custody_verified']is True,'EXPORT_RAW_SUCCESS_NOT_SCORE_ONLY')
            x=expected[row['attempt_id']];support.add((x['task'],x['layout_stratum']))
    required={(t,d+'/'+b)for t in('T1','T2')for d in('+Y','-Y')for b in('NEAR','FAR')}
    return dict(ready=support==required,missing=sorted(required-support),
        formal_R0_export_complete=False,native_training_admission=False)


def register(g,row,*,closed_capture_proof,packet_bytes,historical_LOCAL_bytes,now_epoch):
    mark=a.validate(g);a.need(not rows(g),'EXPORT_NO_SECOND_REQUEST_OR_UNKNOWN_WRITE_REPLAY')
    a.need(mark['T_CD']<=now_epoch<mark['stop_new_training_epoch']-18.25*3600,
        'EXPORT_TIME_FOR_OWN_BOUNDED_CPU_AND_TRAINING')
    old=mark['teacher_pilot']['engineering_recovery'];ten=old['remaining_conditions']
    a.need(ten['status']=='NATIVE_TEN_KNOWN_TERMINAL'and ten['actual']['unknown_native_effects']==[]
        and ten['actual']['native_safety_failures']==0
        and old['saved_data_reaudit']['status']=='KNOWN_SAVED_RAW_REAUDIT_SETTLED',
        'EXPORT_REAL_NATIVE_KNOWN_CAPTURE_PREDECESSOR')
    a.need(closed_capture_proof['status']=='INDEPENDENT_GEN1_ALL_SOURCE_ROLES_CLOSED'
        and closed_capture_proof['remaining_terminal_audit_sha256']==ten['independent_terminal_audit_sha256']
        and closed_capture_proof['saved_reaudit_sha256']==old['saved_data_reaudit']['independent_terminal_audit_sha256']
        and closed_capture_proof['all_processes_terminal']is True
        and closed_capture_proof['original_files_and_valid_failures_preserved']is True,
        'EXPORT_EXACT_CLOSED_SOURCE_PROOF_NOT_ACTIVE_CAPTURE')
    leases=closed_capture_proof['leases']
    a.need(set(leases)=={'remaining_owner','remaining_0','remaining_1','R1_owner','R1_0','R1_1','saved_reaudit'},
        'EXPORT_ALL_SEVEN_REAL_SOURCE_ROLE_CLOSURES')
    for key,lease in leases.items():
        a.need(lease['status']=='CLOSED'and type(lease['manifest_sha256'])is str
            and re.fullmatch('[0-9a-f]{64}',lease['manifest_sha256'])is not None,
            'EXPORT_EACH_OWN_SOURCE_CLOSED_NOT_RESERVED')
        if key.startswith('remaining_'):
            a.need(lease['independent_terminal_audit_sha256']==ten['independent_terminal_audit_sha256'],
                'EXPORT_EXACT_TEN_TERMINAL_CLOSURES')
            if key!='remaining_owner':a.need(lease['manifest_sha256']==ten['workers'][key[-1]]['manifest_sha256'],
                'EXPORT_REMAINING_WORKER_MANIFEST')
        elif key=='saved_reaudit':
            a.need(lease['manifest_sha256']==old['saved_data_reaudit']['manifest_sha256']
                and lease['independent_terminal_audit_sha256']==old['saved_data_reaudit']['independent_terminal_audit_sha256'],
                'EXPORT_EXACT_SAVED_REAUDIT_CLOSURE')
        else:
            a.need(lease['independent_saved_effects_audit_sha256']==old['independent_saved_effects_audit_sha256'],
                'EXPORT_ORIGINAL_R1_FAILURE_REMAINS_CLOSED')
            if key!='R1_owner':a.need(lease['manifest_sha256']==old['workers'][key[-1]]['manifest_sha256'],
                'EXPORT_R1_WORKER_MANIFEST')
    outcomes=closed_capture_proof['outcomes'];a.need(raw_support(packet_bytes,outcomes)['ready'],
        'EXPORT_MISSING_RAW_SUPPORT_NO_PREMATURE_CPU_OR_GPU')
    a.need(row['closed_capture_proof_sha256']==a.sha(a.raw(closed_capture_proof))
        and row['job_id']is None and row['status']==STATES[0]
        and all(x['status']in original.TERMINAL for x in g['PBS_submissions'])
        and g['pending_updates']==g['unknown_effects']==[],
        'EXPORT_FRESH_UNIQUE_INACTIVE_BOUNDARY')
    out=deepcopy(g);out['PBS_submissions'].append(deepcopy(row));out['used']['PBS_submissions']+=1
    out['CPU_reserved_s']+=LIMITS['CPU_s'];out['CPU_s']+=LIMITS['control_site_CPU_s']
    out['storage_reserved_bytes']+=LIMITS['output_bytes']
    out[a.FIELD]['official_export']={k:row[k]for k in ('request_id','request_sha256','run_manifest_sha256','closed_capture_proof_sha256','raw_plan_sha256')}
    out[a.FIELD]['official_export'].update(GLOBAL_before_sha256=a.sha(a.raw(g)),reservation_before=dict(
        PBS_submissions=g['used']['PBS_submissions'],CPU_s=g['CPU_s'],
        new_storage_current_bytes=g['new_storage_current_bytes']))
    rows(out);original.validate_global(out);out['revision']+=1
    a.need(type(historical_LOCAL_bytes)is bytes
        and a.sha(historical_LOCAL_bytes)==mark['historical_LOCAL_sha256'],
        'EXPORT_EXACT_HISTORICAL_LOCAL_NOT_FAKE_AMD_OR_FRESH_COPY')
    proof=scope.full_forecast(a.raw(out),historical_LOCAL_bytes)
    a.need(proof['cap_changes_required']=={}and proof['matrix3_and_final20GPUh40GiB_retained'],
        'EXPORT_ALL_REMAINING_WORK_PAYABLE_AFTER_REAL_RESERVATION')
    return dict(proposed_global=out,new_GPU=0,new_PBS=1,
        original_C_D_package_unchanged=out[original.PACKAGE]==g[original.PACKAGE],
        full_remaining_proof=proof,quota_and_native_process_admission_still_required=True,native_admission=False)


def bind(g,*,request_id,job_id):
    out=deepcopy(g);items=rows(out);a.need(len(items)==1 and items[0]['request_id']==request_id
        and items[0]['job_id']is None and items[0]['status']==STATES[0], 'EXPORT_BIND_ONLY_EXACT_ACTUAL_QSUB_REPLY')
    items[0].update(job_id=job_id,status=STATES[1]);rows(out);out['revision']+=1;return out


def settle(g,*,request_id,native,audit):
    out=deepcopy(g);items=rows(out);a.need(len(items)==1,'EXPORT_ONE_ACTUAL_REQUEST')
    row=items[0];a.need(row['request_id']==request_id and row['status']==STATES[1]
        and native['job_state']=='F'and native['job_id']==row['job_id']
        and native['request_id']==request_id and native['submission_nonce']==row['submission_nonce']
        and type(native['Exit_status'])is int,'EXPORT_NATIVE_F_NOT_OBSERVATION_TIMEOUT')
    a.need(audit['status']=='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS'
        and audit['request_sha256']==row['request_sha256']and audit['all_processes_terminal']is True
        and audit['unknown_effects']==[]and type(audit['export_finished'])is bool
        and all(type(audit[k])is int and audit[k]==0 for k in('model_loads','queries','updates','constructors','physics_ticks','episode_starts')),
        'EXPORT_ACTUAL_CPU_ONLY_EFFECT_AUDIT')
    cpu=native['conservative_CPU_s'];a.need(type(cpu)in(int,float)and math.isfinite(cpu)and 0<=cpu<=36000,
        'EXPORT_BOUNDED_REAL_CPU_NO_RESERVATION_AS_ACTUAL')
    row.update(status=STATES[-1],native_exit_status=native['Exit_status'],actual_CPU_s=cpu,
        export_finished=native['Exit_status']==0 and audit['export_finished'],
        charged_output_upper_bytes=LIMITS['output_bytes'],terminal_native_sha256=a.sha(a.raw(native)),
        terminal_export_audit_sha256=a.sha(a.raw(audit)),training_bank_qualified=audit.get('training_bank_qualified',False))
    out['CPU_reserved_s']-=36000;out['CPU_s']+=cpu
    out['storage_reserved_bytes']-=LIMITS['output_bytes'];out['new_storage_current_bytes']+=LIMITS['output_bytes']
    out['new_storage_peak_bytes']=max(out['new_storage_peak_bytes'],out['new_storage_current_bytes'])
    rows(out);original.validate_global(out);out['revision']+=1;return out


def forecast_credit(g):
    """Avoid charging the original planned export twice after real reservation."""
    items=rows(g)
    if not items:return {}
    row=items[0]
    if row['status']==STATES[-1]and not row['export_finished']:
        # The failed export consumes the existing engineering PBS allowance;
        # full future export/output is retained until exact cold repair facts.
        return dict(PBS_submissions=1,Hopper_CPU_s=3600)
    return dict(PBS_submissions=1,Hopper_CPU_s=36600,Hopper_storage_bytes=LIMITS['output_bytes'])
