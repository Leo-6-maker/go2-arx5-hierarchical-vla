"""Exact prerelease C50 failure audit and repair proposal, no native IO.

Missing files alone never prove zero effects. The immutable executed source,
positive CPU_BOOTSTRAP terminal, PBS identity and release ordering must agree.
Historical paid output bounds and all failed allocation costs stay unchanged.
"""
import ast,base64,json
from copy import deepcopy
from . import mix_accounting as a, successor_cd as s

SOURCE='3983367bff4cc88361053f97c2904f3968ba345b'
PROGRAM_SHA='c403a1fdb4dce5e6ce63872275f881dc6126f9b810b008e486dedb94bff9424f'
REQUEST='396e9816-d6a4-490b-896c-c965de774bac'
JOB='1492540.stdct-mgmt-02'
REQUEST_SHA='4b81bf4e61d020130d38affa8009bdc03da17a7d87b0bb9bea6152fb548ca715'
VERSION='V5_EXACT_C50_PRERELEASE_CPU_BOOTSTRAP_FAILURE_V1'
REPAIR_METADATA_UPPER=48*1024**2
REPAIR_CONTROL_CPU_UPPER=1800

def need(ok,msg):
    if not ok:raise PermissionError(VERSION+':'+msg)

def audit(c):
    def data(key,parse=True):
        row=c['files'][key];b=base64.b64decode(row['base64'],validate=True)
        need(a.sha(b)==row['sha256'],'CAPTURE_BYTES:'+key)
        return json.loads(b) if parse else b
    req=data('REQUEST.json');final=data('PBS_PROCESS_FINAL.json');ready=data('PBS_READY.json');wrapper=data('PBS_WRAPPER_FINAL.json')
    manifest=data('RUN_MANIFEST.json');seal=data('SOURCE_SEAL.json');owner=data('control/RESIDENT_OWNER_OS_EXIT.json')
    need(req['request_id']==REQUEST and req['source_sha']==SOURCE and a.sha(data('REQUEST.json',False))==REQUEST_SHA,'EXACT_OLD_REQUEST_NOT_A_GENERAL_ZERO_BYPASS')
    need(req['arm']=='C50' and req['expected_native_start_step']==0 and req['optimizer_update_limit']==1024,'SAME_ORIGINAL_TRAINING')
    program=data('mix_native.py',False)
    need(a.sha(program)==PROGRAM_SHA==seal['files']['scripts/pi0294/pv1/mix_native.py'] and seal['source_sha']==SOURCE==manifest['code']['vla_commit'],'EXECUTED_IMMUTABLE_CAUSAL_SOURCE')
    tree=ast.parse(program);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    text=ast.unparse(run)
    need(text.index('wait_release(root, directory, {}, first_update=False)')<text.index('ledger.initialize()')<text.index('import torch'),'RELEASE_BEFORE_EFFECT_LEDGER_AND_MODEL')
    q=c['qstat'];need(q['returncode']==0 and q['stderr']=='' and q['argv'][-1]==JOB,'REAL_CLEAN_EXACT_PBS_HISTORY')
    jobs=json.loads(q['stdout']);need(jobs['pbs_server']=='stdct-mgmt-02','REAL_SERVER_NOT_EMPTY_DIAGNOSTIC')
    native=jobs['Jobs'][JOB];need(native['job_state']=='F' and native['Exit_status']==1 and native['run_count']==1,'ACTUAL_FINAL_SINGLE_FAILED_EXECUTION')
    env=native['Variable_List'];need(env['PV1_REQUEST_ID']==REQUEST and env['PV1_SUBMISSION_NONCE']==req['submission_nonce'],'PBS_REQUEST_AND_NONCE')
    need(final['source_sha']==SOURCE and final['job_id']==wrapper['job_id']==ready['job_id']==JOB and final['request_sha256']==ready['request_sha256']==REQUEST_SHA,'MATCH_POSITIVE_NATIVE_FINALS')
    need(final['phase']=='CPU_BOOTSTRAP' and final['error']==dict(type='TimeoutError',message='V5_NATIVE_OWNER_RELEASE_MISSING_NO_IDLE_MODEL') and final['result']is None and 'effects'not in final,'EXACT_PRERELEASE_EXCEPTION_NO_SYNTHETIC_EFFECT_LEDGER')
    need(ready['models_loaded']==0 and wrapper['returncode']==1 and final['new_physics']==final['new_episodes']==0,'POSITIVE_BOOTSTRAP_ZERO_MODEL_AND_FAILED_WRAPPER')
    need(owner['returncode']==1 and owner['source_sha']==SOURCE and owner['foreground_wait_completed']is True,'OLD_OWNER_ACTUAL_FAILED_OS_EXIT')
    for key in ['control/resident-owner/setup-rpc/RPC_EFFECT_STATE.json','control/resident-owner/running-rpc/RPC_EFFECT_STATE.json']:
        need(data(key)['status']=='CONFIRMED','NO_UNKNOWN_WRITES_SETTLED_AS_ZERO')
    need(all(c['absent_neural_release'].values()) and c['absent_neural_release'].get('root/A40_NATIVE_EFFECT_LEDGER.json')is True and len(c['absent_neural_release'])>=9,'NO_RELEASE_OR_EFFECT_RECORD')
    need(len(c['namespace_inventory'])==5 and all(not r['neural_outputs']for r in c['namespace_inventory']),'ALL_RETAINED_NAMESPACES_NO_TRAIN_OUTPUT')
    logical=sum(r['logical_bytes']for r in c['namespace_inventory']);allocated=sum(r['allocated_bytes']for r in c['namespace_inventory'])
    need(logical<=64*1024**2,'EXISTING_PREPAID_CONTROL_OUTPUT_BOUND')
    from scripts.pi0294.e0_scope import pbs_epoch
    hms=lambda x:sum(int(v)*m for v,m in zip(x.split(':'),(3600,60,1)))
    resources=native['resources_used'];gpu=max(hms(resources['walltime']),pbs_epoch(native['obittime'])-pbs_epoch(native['stime']))+60
    cpu=max(hms(resources['cput']),final['CPU_actual_s'])+120
    proof=dict(job_id=JOB,job_state='F',request_id=REQUEST,submission_nonce=req['submission_nonce'],Exit_status=1,conservative_GPU_s=gpu,conservative_CPU_s=cpu,observed_epoch=c['epoch'],raw_native_sha256=a.sha(a.raw(native)),process_final_sha256=c['files']['PBS_PROCESS_FINAL.json']['sha256'],wrapper_final_sha256=c['files']['PBS_WRAPPER_FINAL.json']['sha256'])
    effects=dict(schema='V5_CD_ACTUAL_NATIVE_EFFECT_SUMMARY_V1',request_id=REQUEST,request_sha256=REQUEST_SHA,job_id=JOB,arm='C50',optimizer_updates_committed=0,query_attempts=0,model_load_attempts=0,pending_updates=[],unknown_query_intents=[],derivation='EXACT_FROZEN_PRERELEASE_SOURCE_AND_POSITIVE_NATIVE_TERMINAL_NOT_A_NATIVE_EFFECT_LEDGER')
    return dict(status=VERSION,source_sha=SOURCE,request_id=REQUEST,job_id=JOB,native=proof,effects=effects,capture_sha256=a.sha(a.raw(c)),retained_logical_control_bytes=logical,retained_filesystem_allocated_bytes=allocated,scientific_output_created_bytes=0,prepaid_control_output_bytes=64*1024**2,scientific_endpoint_complete=False)

def settle_and_restore_original_stage(global_bytes,capture,*,now_epoch):
    g=json.loads(global_bytes);a.validate_global(g);audit_result=audit(capture)
    need(a.sha(global_bytes)==capture['files']['GLOBAL']['sha256'],'EXACT_AUDITED_BEFORE_LEDGER')
    old=next(r for r in g['PBS_submissions']if r.get('request_id')==REQUEST)
    need(old['status']not in a.TERMINAL and g['unknown_effects']==g['pending_updates']==[],'ONLY_UNSETTLED_KNOWN_FAILED_REQUEST')
    need(REQUEST not in g.get('V5_CD_bootstrap_failure_repairs',{}),'ONCE_REPAIR')
    v=a.settle_training(g,REQUEST,native=audit_result['native'],effects=audit_result['effects'],audit=audit_result,now_epoch=now_epoch)
    # The generic conservative fallback is retained for every other failure.
    # This exact prerelease path created no training output; its real files
    # already fit the control payment. No previously charged bound is refunded.
    v['new_storage_current_bytes']-=70000000000
    v['new_storage_peak_bytes']=max(g['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    row=next(r for r in v['PBS_submissions']if r.get('request_id')==REQUEST)
    row.update(actual_scientific_output_charge_bytes=0,retained_control_bytes=audit_result['retained_logical_control_bytes'],output_accounting_basis=VERSION,original_requested_output_upper_bytes=70000000000)
    p=a.package_view(v);remaining=p['remaining_resources'];restore=dict(H200_s=22560,optimizer_updates=1024,neural_queries=29,Hopper_CPU_s=54000,Hopper_storage_bytes=70000000000,PBS_submissions=1)
    for k,x in restore.items():remaining[k]+=x
    for k in ('H200_s','optimizer_updates','neural_queries'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=54000;v['storage_reserved_bytes']+=70000000000
    # Charge this failed allocation's CPU against the existing 3000s repair
    # allowance while retaining the FULL original 54000s next training bound.
    cpu=audit_result['native']['conservative_CPU_s'];remaining['Hopper_CPU_s']-=cpu+REPAIR_CONTROL_CPU_UPPER;v['CPU_reserved_s']-=cpu+REPAIR_CONTROL_CPU_UPPER;v['CPU_s']+=REPAIR_CONTROL_CPU_UPPER
    # New cold repair files have their own explicit retained-output upper,
    # paid from existing cap headroom, never another control prepayment.
    v['new_storage_current_bytes']+=REPAIR_METADATA_UPPER
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    marker=dict(audit=audit_result,restored_original_stage=restore,repair_CPU_charged_inside_existing_allowance_s=cpu,repair_control_CPU_prepaid_from_existing_3000_s=REPAIR_CONTROL_CPU_UPPER,new_retained_metadata_output_upper_bytes=REPAIR_METADATA_UPPER,old_paid_history_unchanged=True,old_request_never_reissued=True)
    v.setdefault('V5_CD_bootstrap_failure_repairs',{})[REQUEST]=marker
    a.validate_global(v)
    return dict(proposed_GLOBAL=v,audit=audit_result,marker=marker,native_IO=False)

def project_closed_failure(local_bytes,global_bytes,*,settled_bytes,lease_bytes,closed_bytes):
    g,l=json.loads(global_bytes),json.loads(local_bytes);a.validate_global(g);marker=g['V5_CD_bootstrap_failure_repairs'][REQUEST]
    settled,lease,closed=map(json.loads,(settled_bytes,lease_bytes,closed_bytes));row=next(r for r in g['PBS_submissions']if r.get('request_id')==REQUEST)
    need(row['status']=='NATIVE_TERMINAL' and not row['scientific_endpoint_complete'] and settled['marker']==marker,'REAL_FAILURE_SETTLEMENT')
    need(lease['status']=='CLOSED' and lease['terminal_receipt_sha256']==a.sha(settled_bytes) and closed['lease_sha256']==a.sha(lease_bytes) and closed['job_id']==JOB,'REAL_CLOSED_GPU_LEASE')
    need(l[s.FIELD]==g[s.FIELD] and l['unknown_effects']==l['pending_updates']==l['pending_queries']==[],'SAME_CURRENT_SCOPE_NO_UNKNOWN')
    need(REQUEST not in l.get('V5_CD_bootstrap_failure_repairs',{}) and not any(r.get('request_id')==REQUEST for r in l['PBS_submissions']),'ONCE_ALREADY_PAID_CLASSIFICATION')
    v=deepcopy(l);v['GPU_allocated_s']+=row['conservative_H200_s'];v['CPU_s']['Hopper']+=row['conservative_CPU_s']+REPAIR_CONTROL_CPU_UPPER;v['CPU_reserved_s']['Hopper']-=row['conservative_CPU_s']+REPAIR_CONTROL_CPU_UPPER
    v['PBS_submissions'].append(deepcopy(row));v[a.PACKAGE]=deepcopy(g[a.PACKAGE]);v.setdefault('V5_CD_bootstrap_failure_repairs',{})[REQUEST]=deepcopy(marker)
    need(v['CPU_reserved_s']['Hopper']>=0,'NO_HOLD_UNDERFLOW')
    return dict(proposed_LOCAL=v,global_bytes_sha256=a.sha(global_bytes),no_new_GLOBAL_fees=True)
