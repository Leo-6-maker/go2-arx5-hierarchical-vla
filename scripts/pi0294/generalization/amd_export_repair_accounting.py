"""One authorized CPU engineering replacement after exact missing-git closure.

Keep the failed ten-GiB charge. Its unchanged raw namespace and one output
envelope are shared by exact references and an enforced aggregate bound;
only the declared engineering metadata is additional storage. No refund,
new budget system or second control prepayment is introduced.
"""
from copy import deepcopy
import math,re,uuid
from . import amd_export_accounting as original,amd_adoption as a,amd_pilot_scope as scope
from . import amd_export_git_failure as failure

VERSION='GEN1_VANDA_OFFICIAL_CPU_EXPORT_KNOWN_GIT_REPAIR_V1'
PROFILE='vanda-gen1-official-export-git-repair-cpu'
JOB_NAME=original.JOB_NAME
STATES=original.STATES
METADATA=64*1024**2
LIMITS=dict(original.LIMITS,control_site_CPU_s=0,LOCAL_control_CPU_s=0,
    shared_previous_paid_output_upper_bytes=original.LIMITS['output_bytes'],
    additional_metadata_upper_bytes=METADATA)
FIELD='official_export_git_repair'

def rows(g):
    mark=a.validate(g);old=original.rows(g);items=[x for x in g['PBS_submissions']if x.get('platform_data_version')==VERSION]
    binding=mark.get(FIELD);a.need(len(items)<=1 and bool(items)==bool(binding),'GIT_REPAIR_ONE_EXTRA_CPU_REQUEST')
    if not items:return items
    a.need(len(old)==1 and old[0]['request_id']==failure.REQUEST and old[0]['job_id']==failure.JOB
        and old[0]['status']==STATES[-1]and old[0]['native_exit_status']==1
        and old[0]['actual_CPU_s']==12 and not old[0]['export_finished'],
        'GIT_REPAIR_ACTUAL_OLD_FAILURE_COST_NOT_ROLLED_BACK')
    item=items[0]
    expected=dict(platform_data_version=VERSION,native_platform='VANDA_A40',project='CFP04-CF-007',
        job_name=JOB_NAME,physical_GPU_count=0,scope_grant_sha256=a.GRANT_SHA,
        T_CD=mark['T_CD'],deadline_epoch=mark['deadline_epoch'],limits=LIMITS,
        raw_plan_sha256=old[0]['raw_plan_sha256'],closed_capture_proof_sha256=old[0]['closed_capture_proof_sha256'],
        predecessor_request_id=failure.REQUEST,predecessor_job_id=failure.JOB,
        predecessor_manifest_sha256=old[0]['run_manifest_sha256'])
    a.need(all(item.get(k)==v and type(item[k])is type(v)for k,v in expected.items()),
        'GIT_REPAIR_EXACT_OLD_CASES_NEW_CPU_TYPE_AND_EXISTING_CLOCK')
    for name,size in [('source_sha',40),('request_sha256',64),('run_manifest_sha256',64),
        ('predecessor_closure_readback_sha256',64)]:
        a.need(type(item.get(name))is str and re.fullmatch('[0-9a-f]{'+str(size)+'}',item[name])is not None,
            'GIT_REPAIR_SEALED:'+name)
    for key in ['request_id','submission_nonce']:
        try:ok=str(uuid.UUID(item[key]))==item[key]
        except(ValueError,TypeError,AttributeError,KeyError):ok=False
        a.need(ok and item[key]!=old[0][key],'GIT_REPAIR_NEW_EXACT_REQUEST_NOT_REPLAY')
    a.need(item['status']in STATES and (item['job_id']is None and item['status']==STATES[0]
        or type(item['job_id'])is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',item['job_id'])),
        'GIT_REPAIR_ONLY_REAL_JOB_OR_EXPLICIT_UNBOUND')
    a.need(all(item[key]==binding[key]for key in ['request_id','request_sha256','run_manifest_sha256',
        'predecessor_closure_readback_sha256'])and binding['predecessor_request_id']==failure.REQUEST
        and binding['shared_paid_output_upper_bytes']==original.LIMITS['output_bytes']
        and binding['aggregate_source_and_repair_output_upper_bytes']==original.LIMITS['output_bytes']+METADATA
        and binding['old_closed_CPU_lease']['status']=='CLOSED'
        and binding['old_closed_CPU_lease']['terminal_audit_sha256']==old[0]['terminal_export_audit_sha256']
        and binding['old_closed_CPU_lease']['terminal_native_sha256']==old[0]['terminal_native_sha256']
        and binding['old_failed_counter_and_UNKNOWN_preserved']is True
        and binding['additional_control_prepay']==0,
        'GIT_REPAIR_REAL_CLOSED_PREDECESSOR_ONE_SHARED_OUTPUT_ENVELOPE')
    a.need(g['new_storage_current_bytes']>=binding['before_new_storage_current_bytes']
        and g['used']['PBS_submissions']>=binding['before_PBS']+1,'GIT_REPAIR_NO_OLD_COST_REFUND')
    if item['status']!=STATES[-1]:
        package=g[original.original.PACKAGE]['remaining_resources']
        a.need(g['CPU_reserved_s']>=package['Hopper_CPU_s']+36000
            and g['storage_reserved_bytes']>=package['Hopper_storage_bytes']+METADATA,
            'GIT_REPAIR_SEPARATE_CPU_METADATA_HOLD_KEEP_C_D_ESCROW')
    else:
        a.need(type(item['actual_CPU_s'])in(int,float)and math.isfinite(item['actual_CPU_s'])
            and 0<=item['actual_CPU_s']<=36000 and type(item['native_exit_status'])is int
            and type(item['export_finished'])is bool and item['charged_metadata_bytes']==METADATA
            and item['shared_old_output_charge_preserved']is True
            and all(re.fullmatch('[0-9a-f]{64}',item[k])for k in ['terminal_native_sha256','terminal_export_audit_sha256']),
            'GIT_REPAIR_ACTUAL_TERMINAL_NO_SECOND_TEN_GIB_CHARGE')
    return items

def register(g,item,*,closed_failure,failed_audit,historical_LOCAL_bytes,now_epoch):
    mark=a.validate(g);a.need(not rows(g),'GIT_REPAIR_NO_SECOND_ENGINEERING_REPLACEMENT')
    old=original.rows(g);a.need(len(old)==1 and old[0]['status']==STATES[-1],
        'GIT_REPAIR_PRIOR_CPU_REALLY_SETTLED')
    a.need(closed_failure['status']=='INDEPENDENT_ACTUAL_CPU1497552_FAILED_SETTLED_OWN_LEASE_CLOSED'
        and closed_failure['GLOBAL_sha256']==a.sha(a.raw(g))
        and closed_failure['original_UNKNOWN_unchanged']is True and closed_failure['symlinks']==[]
        and 0<closed_failure['old_output_bytes']<=original.LIMITS['output_bytes']
        and failed_audit['status']=='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS'
        and failed_audit['causal_boundary']=='EXACT_GIT_EXEC_FAILED_BEFORE_PROCESSOR_CONSTRUCTOR_AND_FIRST_EXPORT'
        and failed_audit['unknown_effects']==[]and failed_audit['export_finished']is False
        and old[0]['terminal_export_audit_sha256']==a.sha(a.raw(failed_audit)),
        'GIT_REPAIR_REAL_KNOWN_ZERO_EXPORT_CLOSURE_AND_FULL_OLD_OUTPUT')
    import base64
    lease_bytes=base64.b64decode(closed_failure['lease_base64'],validate=True)
    a.need(a.sha(lease_bytes)==closed_failure['lease_sha256'],'GIT_REPAIR_ACTUAL_CLOSED_LEASE_BYTES')
    lease=__import__('json').loads(lease_bytes)
    a.need(lease['status']=='CLOSED'and lease['known_preprocessor_git_failure']is True
        and lease['manifest_sha256']==old[0]['run_manifest_sha256']
        and lease['terminal_audit_sha256']==old[0]['terminal_export_audit_sha256']
        and lease['terminal_native_sha256']==old[0]['terminal_native_sha256']
        and closed_failure['original_counter_sha256']==failed_audit['resource_ledger_sha256'],
        'GIT_REPAIR_EXACT_CLOSED_ROLE_AND_UNCHANGED_OLD_FAILED_COUNTER')
    a.need(mark['T_CD']<=now_epoch<mark['stop_new_training_epoch']-18.25*3600
        and all(x['status']in original.original.TERMINAL for x in g['PBS_submissions'])
        and g['pending_updates']==g['unknown_effects']==[]and item['job_id']is None
        and item['status']==STATES[0],'GIT_REPAIR_INACTIVE_FRESH_WINDOW_NEW_INTENT')
    a.need(item['predecessor_closure_readback_sha256']==a.sha(a.raw(closed_failure)),
        'GIT_REPAIR_SEALED_ACTUAL_CLOSURE_NOT_A_PREPARED_HINT')
    out=deepcopy(g);out['PBS_submissions'].append(deepcopy(item));out['used']['PBS_submissions']+=1
    out['CPU_reserved_s']+=36000;out['storage_reserved_bytes']+=METADATA
    binding={k:item[k]for k in ['request_id','request_sha256','run_manifest_sha256','predecessor_closure_readback_sha256']}
    binding.update(predecessor_request_id=failure.REQUEST,old_source_root=closed_failure['old_root'],
        old_closed_CPU_lease=lease,old_closed_CPU_lease_sha256=closed_failure['lease_sha256'],
        old_failed_counter_and_UNKNOWN_preserved=True,shared_paid_output_upper_bytes=original.LIMITS['output_bytes'],
        aggregate_source_and_repair_output_upper_bytes=original.LIMITS['output_bytes']+METADATA,
        old_output_bytes_at_closed_boundary=closed_failure['old_output_bytes'],additional_control_prepay=0,
        before_new_storage_current_bytes=g['new_storage_current_bytes'],before_PBS=g['used']['PBS_submissions'])
    out[a.FIELD][FIELD]=binding;out['revision']+=1;rows(out);original.original.validate_global(out)
    a.need(a.sha(historical_LOCAL_bytes)==mark['historical_LOCAL_sha256'],'GIT_REPAIR_HISTORICAL_LOCAL_NOT_FAKE_FRESH')
    proof=scope.full_forecast(a.raw(out),historical_LOCAL_bytes)
    a.need(not proof['cap_changes_required'],'GIT_REPAIR_ALL_REMAINING_MATRIX_FINAL_INSIDE_ORIGINAL_CAPS')
    return dict(proposed_global=out,full_remaining_proof=proof,native_admission=False,
        new_PBS=1,new_GPU=0,additional_control_prepay=0,old_output_charge_refunded=False)

def bind(g,*,request_id,job_id):
    out=deepcopy(g);item=rows(out)[0];a.need(item['request_id']==request_id and item['job_id']is None
        and item['status']==STATES[0],'GIT_REPAIR_ONE_ACTUAL_QSUB_REPLY')
    item.update(job_id=job_id,status=STATES[1]);out['revision']+=1;rows(out);return out

def forecast_credit(g):
    items=rows(g)
    if not items:return {}
    row=items[0]
    if row['status']==STATES[-1]and not row['export_finished']:return {}
    # The primary failed request already consumes the one engineering PBS.
    # This request consumes the planned CPU export, sharing the original paid
    # control and ten-GiB output. Its extra metadata is actually held here.
    return dict(PBS_submissions=1,Hopper_CPU_s=36600,
        Hopper_storage_bytes=original.LIMITS['output_bytes']+METADATA)

def settle(g,*,request_id,native,audit):
    out=deepcopy(g);item=rows(out)[0];a.need(item['request_id']==request_id
        and item['status']==STATES[1]and native['job_state']=='F'
        and native['job_id']==item['job_id']and native['request_id']==request_id
        and native['submission_nonce']==item['submission_nonce']and type(native['Exit_status'])is int,
        'GIT_REPAIR_EXACT_NATIVE_TERMINAL_NOT_TIMEOUT')
    a.need(audit['status']=='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS'
        and audit['request_sha256']==item['request_sha256']and audit['all_processes_terminal']is True
        and audit['unknown_effects']==[]and type(audit['export_finished'])is bool
        and all(type(audit[k])is int and audit[k]==0 for k in ['model_loads','queries','updates','constructors','physics_ticks','episode_starts'])
        and audit['shared_source_and_repair_output_bytes']<=original.LIMITS['output_bytes']+METADATA,
        'GIT_REPAIR_INDEPENDENT_CPU_EFFECTS_AND_ACTUAL_AGGREGATE_OUTPUT')
    cpu=native['conservative_CPU_s'];a.need(type(cpu)in(int,float)and math.isfinite(cpu)
        and 0<=cpu<=36000,'GIT_REPAIR_BOUNDED_REAL_CPU')
    item.update(status=STATES[-1],native_exit_status=native['Exit_status'],actual_CPU_s=cpu,
        export_finished=native['Exit_status']==0 and audit['export_finished'],charged_metadata_bytes=METADATA,
        shared_old_output_charge_preserved=True,terminal_native_sha256=a.sha(a.raw(native)),
        terminal_export_audit_sha256=a.sha(a.raw(audit)),training_bank_qualified=audit.get('training_bank_qualified',False))
    out['CPU_reserved_s']-=36000;out['CPU_s']+=cpu;out['storage_reserved_bytes']-=METADATA
    out['new_storage_current_bytes']+=METADATA;out['new_storage_peak_bytes']=max(out['new_storage_peak_bytes'],out['new_storage_current_bytes'])
    out['revision']+=1;rows(out);original.original.validate_global(out);return out
