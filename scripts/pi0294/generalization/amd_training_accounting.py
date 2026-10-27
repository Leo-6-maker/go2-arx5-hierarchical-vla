"""One explicit GEN1 training row in the original campaign, never a C/D arm.

Pure proposals only. Actual owner/PBS, project, source, data and release checks
remain required. Existing C/D escrow, paid failures and clocks are immutable.
"""
from copy import deepcopy
import json,math,re,uuid
from . import amd_adoption as authority,amd_bank as bank,amd_export_repair_accounting as export
from . import amd_pilot_scope as scope
from scripts.pi0294.pv1 import mix_accounting as original

VERSION='GEN1_VANDA_INDEPENDENT_CORR2048_LAYOUT_V0_TRAIN1024_V1'
FIELD='independent_CORR1024_training'
PROFILE='vanda-gen1-independent-corr1024-training'
AUDIT='PASS_INDEPENDENT_GEN1_1024_COMPLETE_STATE'
EFFECTS='GEN1_ACTUAL_NATIVE_TRAINING_EFFECT_SUMMARY_V1'
STATES=('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE','NATIVE_TERMINAL')
raw=authority.raw
sha=authority.sha
need=authority.need

def closed_export(g,files,*,lease_bytes,owner_exit_bytes,packet_bytes):
    """Join original terminal bytes; a READY file or CPU PASS alone is insufficient."""
    rows=export.rows(g);need(len(rows)==1,'TRAIN_ONE_ORIGINAL_CLOSED_CPU_EXPORT')
    row=rows[0]
    need(row['status']==STATES[-1]and row['native_exit_status']==0
        and row['export_finished']is True and row['training_bank_qualified']is True,
        'TRAIN_EXPORT_REAL_SUCCESS_SETTLED_BEFORE_GPU')
    names=['RUN_MANIFEST.json','REQUEST.json','RESOURCE_LEDGER.json',
        'NATIVE_CPU_TERMINAL.json','INDEPENDENT_CPU_TERMINAL_AUDIT.json',
        'evidence/GEN1_EXPORT_RESULT.private.json','datasets/FROZEN_GEN1_BANK/READY.json',
        'datasets/FROZEN_GEN1_BANK/DATASET_MANIFEST.json',
        'datasets/FROZEN_GEN1_BANK/PAYLOAD_MANIFEST.json',
        'datasets/FROZEN_GEN1_BANK/CROSS_PARENT_QUALIFICATION.json']
    need(set(files)==set(names)and all(type(v)is bytes for v in files.values()),
        'TRAIN_EXACT_REAL_EXPORT_TERMINAL_BYTES')
    d={k:json.loads(v)for k,v in files.items()};manifest=d[names[0]];request=d[names[1]]
    ledger=d[names[2]];native=d[names[3]];audit=d[names[4]];result=d[names[5]]
    ready=d[names[6]];dataset=d[names[7]];lease=json.loads(lease_bytes);exit=json.loads(owner_exit_bytes)
    need(sha(files[names[0]])==row['run_manifest_sha256']
        and sha(files[names[1]])==row['request_sha256']
        and request['request_id']==row['request_id']and request['submission_nonce']==row['submission_nonce']
        and native['job_id']==row['job_id']and native['request_id']==row['request_id']
        and native['submission_nonce']==row['submission_nonce']and native['job_state']=='F'
        and type(native['Exit_status'])is int and native['Exit_status']==0
        and sha(files[names[3]])==row['terminal_native_sha256']
        and sha(files[names[4]])==row['terminal_export_audit_sha256'],
        'TRAIN_SAME_REGISTERED_NATIVE_F_AND_SETTLED_AUDIT')
    need(audit['status']=='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS'
        and audit['all_processes_terminal']is True and audit['unknown_effects']==[]
        and audit['export_finished']is True and audit['training_bank_qualified']is True
        and audit['resource_ledger_sha256']==sha(files[names[2]])
        and audit['result_sha256']==sha(files[names[5]])
        and ledger['status']=='KNOWN_GEN1_EXPORT_TERMINAL'and ledger['unknown_effects']==[]
        and ledger['error']is None and result['status']=='KNOWN_GEN1_OFFICIAL_EXPORT_TERMINAL'
        and result['export_finished']is True and result['bank_ready']==ready,
        'TRAIN_KNOWN_COMPLETE_EXPORT_NOT_SAVED_PREPARATION')
    for k in ('model_loads','queries','updates','constructors','physics_ticks','episode_starts'):
        need(type(audit[k])is int and audit[k]==ledger[k]==result[k]==0,'TRAIN_EXPORT_ZERO_NON_CPU_EFFECTS')
    need(lease['status']=='CLOSED'and lease['manifest_sha256']==row['run_manifest_sha256']
        and lease['terminal_native_sha256']==row['terminal_native_sha256']
        and lease['terminal_audit_sha256']==row['terminal_export_audit_sha256']
        and type(exit['returncode'])is int and exit['returncode']==0,
        'TRAIN_CPU_OWN_LEASE_CLOSED_AND_OWNER_NATURAL_OS0')
    need(ready['dataset_manifest_sha256']==sha(files[names[7]])
        and ready['payload_manifest_sha256']==sha(files[names[8]])
        and ready['cross_parent_qualification_sha256']==sha(files[names[9]]),
        'TRAIN_ACTUAL_BANK_READY_ALL_SEALED_MEMBERS')
    support=bank.validate(dataset,packet_bytes)
    return dict(status='PASS_CLOSED_OFFICIAL_GEN1_EXPORT_FOR_FRESH_TRAINING',
        root=manifest['output_root'],operation_deadline_epoch=manifest['execution']['operation_deadline_epoch'],request_id=row['request_id'],job_id=row['job_id'],
        manifest_sha256=row['run_manifest_sha256'],native_sha256=row['terminal_native_sha256'],
        audit_sha256=row['terminal_export_audit_sha256'],lease_sha256=sha(lease_bytes),
        owner_exit_sha256=sha(owner_exit_bytes),bank_manifest_sha256=sha(files[names[7]]),
        bank_payload_sha256=sha(files[names[8]]),bank_ready_sha256=sha(files[names[6]]),
        source_packet_sha256=scope.PACKET_SHA,support=support,
        actual_files_sha256={k:sha(v)for k,v in files.items()},native_GPU_admission=False)

def closed_saved_exports(g,capture,*,lease_bytes,packet_bytes):
    """Closed failed pack, not failed science: retain all saved official receipts.

    This permits only allocation/CPU packaging. Real bank READY and the mature
    cross-parent processor gate still precede model release and optimization.
    """
    from . import amd_export_seal_failure as failure
    result=failure.audit(capture);audit=result['audit'];row=export.rows(g)[0]
    lease=json.loads(lease_bytes)
    need(row['status']==STATES[-1]and row['native_exit_status']==1
        and row['terminal_export_audit_sha256']==sha(raw(audit))
        and row['terminal_native_sha256']==sha(raw(result['native']))
        and row['training_bank_qualified']is False and row['export_finished']is False
        and lease['status']=='CLOSED'and lease['manifest_sha256']==row['run_manifest_sha256']
        and lease['terminal_native_sha256']==row['terminal_native_sha256']
        and lease['terminal_audit_sha256']==row['terminal_export_audit_sha256']
        and lease['known_saved_export_seal_failure']is True,
        'TRAIN_EXACT_REAL_SAVED_FAILURE_SETTLED_OWN_LEASE_CLOSED')
    bank.validate(result['prospective_bank'],packet_bytes)
    return dict(status='PASS_CLOSED_GEN1_SAVED_EXPORTS_FOR_PREMODEL_PACK_ONLY',
        root=json.loads(base64_decode(capture['files']['RUN_MANIFEST.json']['base64']))['output_root'],
        operation_deadline_epoch=json.loads(base64_decode(capture['files']['RUN_MANIFEST.json']['base64']))['execution']['operation_deadline_epoch'],
        request_id=row['request_id'],job_id=row['job_id'],manifest_sha256=row['run_manifest_sha256'],
        native_sha256=row['terminal_native_sha256'],audit_sha256=row['terminal_export_audit_sha256'],
        lease_sha256=sha(lease_bytes),bank_manifest_sha256=audit['prospective_bank_manifest_sha256'],
        saved_numeric_members=deepcopy(audit['saved_numeric_members']),
        saved_dataset_manifest_sha256={identifier:next(x['sha256']for x in capture['inventory']if x['path']=='datasets/'+identifier+'/DATASET_MANIFEST.json')for identifier in audit['saved_numeric_members']},
        all_captured_files_sha256={k:v['sha256']for k,v in capture['files'].items()},
        support=audit['support'],source_packet_sha256=scope.PACKET_SHA,
        complete_bank_ready=False,CPU_pack_required_before_model_release=True,
        native_GPU_admission=False,native_training_permission=False)

def base64_decode(x):
    import base64
    return base64.b64decode(x,validate=True)

def validate_request(r,*,central=False):
    need(r.get('platform_training_version')==VERSION and r.get('arm')==bank.IDENTITY
        and r.get('stage')=='GEN1_INDEPENDENT_TRAIN1024'
        and r.get('native_platform')=='VANDA_A40'and r.get('scope_grant_sha256')==authority.GRANT_SHA,
        'TRAIN_DISTINCT_GEN_IDENTITY_AND_REAL_GRANT')
    fixed=dict(schema='GEN1_ACTUAL_NATIVE_TRAINING_REQUEST_V1',
        parent_checkpoint_manifest_sha256=original.PARENT,statistics_sha256=original.STATS,
        optimizer_update_limit=1024,query_limit=29,CPU_upper_s=54000,
        reserved_H200_s=22500,accounting_pad_s=60,physical_GPU_count=1,
        physical_GPU_kind='NVIDIA_A40',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',
        additional_output_upper_bytes=70000000000,expected_native_start_step=0,
        training_block=dict(arm=bank.IDENTITY,replica=False,stop_after_step=1024,resume_checkpoint_relative=None))
    need(all(r.get(k)==v and type(r[k])is type(v)for k,v in fixed.items()),'TRAIN_EXACT_FRESH_COMMON_CORR_RECIPE_AND_BOUNDS')
    for key in ('request_id','submission_nonce'):
        try:ok=str(uuid.UUID(r[key]))==r[key]
        except(KeyError,ValueError,TypeError,AttributeError):ok=False
        need(ok,'TRAIN_CANONICAL_UNIQUE_UUID')
    for key,n in [('source_sha',40),('run_manifest_sha256',64),('cold_input_audit_sha256',64),
        ('closed_export_sha256',64),('bank_manifest_sha256',64),('recipe_sha256',64)]:
        need(type(r.get(key))is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',r[key])is not None,
            'TRAIN_SEALED_EXPLICIT_INPUT:'+key)
    need(r['status']in STATES,'TRAIN_KNOWN_TYPED_STATE')
    if r.get('job_id')is None:need(r['status']==STATES[0],'TRAIN_UNBOUND_INTENT_NOT_PASS')
    else:need(re.fullmatch(r'[0-9]+\.stdct-mgmt-02',r['job_id'])is not None,'TRAIN_ACTUAL_VANDA_JOB')
    if central:need(re.fullmatch('[0-9a-f]{64}',r.get('request_sha256',''))is not None,'TRAIN_CENTRAL_REQUEST_HASH')
    else:need('request_sha256'not in r,'TRAIN_NO_SELF_REFERENTIAL_REQUEST_HASH')
    return r

def rows(g):
    adopted=authority.validate(g);items=[r for r in g['PBS_submissions']if r.get('platform_training_version')==VERSION]
    marker=adopted.get(FIELD);need(len(items)<=1 and bool(items)==bool(marker),'TRAIN_ONE_GEN_REQUEST_NO_AUTORETRY')
    for r in items:
        validate_request(r,central=True)
        need(r['T_CD']==adopted['T_CD']and r['deadline_epoch']==adopted['deadline_epoch']
            and all(marker[k]==r[k]for k in ('request_id','request_sha256','run_manifest_sha256',
                'closed_export_sha256','bank_manifest_sha256','recipe_sha256')),'TRAIN_EXACT_ACTUAL_BINDING_AND_CLOCK')
        need(g['used']['PBS_submissions']>=marker['before']['PBS_submissions']+1
            and g['CPU_s']>=marker['before']['CPU_s']+1800
            and g['new_storage_current_bytes']>=marker['before']['new_storage_current_bytes']+64*1024**2,
            'TRAIN_OLD_COST_AND_SINGLE_CONTROL_DEBIT_PRESERVED')
        if r['status']!=STATES[-1]:
            p=g[original.PACKAGE]['remaining_resources']
            need(all(g['reserved'][k]>=p[k]+v for k,v in dict(H200_s=22560,optimizer_updates=1024,neural_queries=29).items())
                and g['CPU_reserved_s']>=p['Hopper_CPU_s']+54000
                and g['storage_reserved_bytes']>=p['Hopper_storage_bytes']+70000000000,
                'TRAIN_SEPARATE_FULL_HOLD_NO_BORROWED_C_D_ESCROW')
        else:
            need(type(r['native_exit_status'])is int and type(r['scientific_endpoint_complete'])is bool
                and r['reserved_optimizer_updates']==r['reserved_neural_queries']==0
                and all(re.fullmatch('[0-9a-f]{64}',r[k])for k in
                    ('terminal_native_sha256','terminal_effect_summary_sha256','full_checkpoint_audit_sha256')),
                'TRAIN_REAL_KNOWN_TERMINAL_EFFECTS_AND_STATE_AUDIT')
    return items

def effective_original_eval_PBS(g,remaining_PBS):
    # The GEN1 adopted full forecast already moves deferred original D Eval
    # back to Core (no PBS). Keep its original escrow bytes and both original
    # GPU/Core-CPU panels; expose that same physical projection to validation.
    m=authority.validate(g);rows(g);marker=m[FIELD]
    need(marker['original_C_D_Eval_deferred_to_Core_no_new_Vanda_PBS']is True
        and g[original.PACKAGE]['completed_stages']==['TRAIN:C50:1024','TRAIN:D25:1024']
        and remaining_PBS==1,'TRAIN_EXACT_UNDISPATCHED_ORIGINAL_D_CORE_EVAL_PROJECTION')
    return 0

def register(g,r,*,closed,original_recipe,historical_LOCAL_bytes,now_epoch):
    from . import amd_learning
    adopted=authority.validate(g);validate_request(r,central=True);need(not rows(g),'TRAIN_REGISTER_ONLY_ONCE')
    source=export.rows(g)[0]
    need(source['status']==STATES[-1]and (
            source['export_finished']is True and source['training_bank_qualified']is True
            and closed['status']=='PASS_CLOSED_OFFICIAL_GEN1_EXPORT_FOR_FRESH_TRAINING'
            or source['export_finished']is False and source['training_bank_qualified']is False
            and closed['status']=='PASS_CLOSED_GEN1_SAVED_EXPORTS_FOR_PREMODEL_PACK_ONLY'
            and closed['CPU_pack_required_before_model_release']is True
            and closed['complete_bank_ready']is False and closed['native_training_permission']is False)
        and closed['request_id']==source['request_id']and closed['job_id']==source['job_id']
        and closed['manifest_sha256']==source['run_manifest_sha256']
        and closed['native_sha256']==source['terminal_native_sha256']
        and closed['audit_sha256']==source['terminal_export_audit_sha256']
        and r['closed_export_sha256']==sha(raw(closed))
        and r['bank_manifest_sha256']==closed['bank_manifest_sha256']
        and r['CPU_bank_deadline_epoch']==closed['operation_deadline_epoch']
        and r['recipe_sha256']==sha(raw(amd_learning.recipe(original_recipe))),
        'TRAIN_ACTUAL_CLOSED_QUALIFIED_BANK_AND_EXACT_FROZEN_RECIPE')
    need(adopted['T_CD']<=now_epoch and now_epoch+600<closed['operation_deadline_epoch']
        and now_epoch+22560<adopted['stop_new_training_epoch']
        and all(x['status']in original.TERMINAL for x in g['PBS_submissions'])
        and all(x['status']=='NATIVE_TERMINAL'for x in g.get(original.CORE_ROWS,{}).values())
        and g['pending_updates']==g['unknown_effects']==[]and r['job_id']is None and r['status']==STATES[0],
        'TRAIN_FRESH_INACTIVE_WINDOW_FULL_BOUND_NO_UNKNOWN')
    need(sha(historical_LOCAL_bytes)==adopted['historical_LOCAL_sha256'],'TRAIN_HISTORICAL_LOCAL_IS_NOT_FRESH')
    out=deepcopy(g);out['PBS_submissions'].append(deepcopy(r));out['used']['PBS_submissions']+=1
    for k,n in dict(H200_s=22560,optimizer_updates=1024,neural_queries=29).items():out['reserved'][k]+=n
    out['CPU_reserved_s']+=54000;out['CPU_s']+=1800;out['storage_reserved_bytes']+=70000000000
    out['new_storage_current_bytes']+=64*1024**2;out['new_storage_peak_bytes']=max(out['new_storage_peak_bytes'],out['new_storage_current_bytes'])
    marker={k:r[k]for k in ('request_id','request_sha256','run_manifest_sha256','closed_export_sha256','bank_manifest_sha256','recipe_sha256')}
    marker.update(before=dict(PBS_submissions=g['used']['PBS_submissions'],CPU_s=g['CPU_s'],
        new_storage_current_bytes=g['new_storage_current_bytes']),control_paid_once=dict(site_CPU_s=1800,
        LOCAL_CPU_s=15,site_output_bytes=64*1024**2,transfer_bytes=256*1024**2),
        actual_LOCAL_written=False,closed_export=deepcopy(closed),registered_epoch=now_epoch,
        original_C_D_Eval_deferred_to_Core_no_new_Vanda_PBS=True)
    out[authority.FIELD][FIELD]=marker;out['revision']+=1;rows(out);original.validate_global(out)
    forecast=scope.full_forecast(raw(out),historical_LOCAL_bytes)
    need(not forecast['cap_changes_required'],'TRAIN_ALL_FUTURE_MATRIX_FINAL_NO_SHORTFALL')
    return dict(proposed_global=out,full_remaining_proof=forecast,native_admission=False,
        original_C_D_package_unchanged=out[original.PACKAGE]==g[original.PACKAGE])

def bind(g,request_id,job_id,*,now_epoch):
    out=deepcopy(g);r=rows(out)[0];need(r['request_id']==request_id and r['job_id']is None
        and r['status']==STATES[0],'TRAIN_ONE_KNOWN_QSUB_REPLY_NOT_RESEND')
    r.update(job_id=job_id,status=STATES[1],native_reply_epoch=now_epoch);out['revision']+=1;rows(out);return out

def forecast_credit(g):
    items=rows(g)
    if not items or items[0]['status']==STATES[-1]and not items[0]['scientific_endpoint_complete']:return {}
    credit=dict(H200_s=22560,PBS_submissions=1,optimizer_updates=1024,neural_queries=29,
        Hopper_CPU_s=55800,Hopper_storage_bytes=70000000000+64*1024**2)
    closed=authority.validate(g)[FIELD]['closed_export']
    if closed['status']=='PASS_CLOSED_GEN1_SAVED_EXPORTS_FOR_PREMODEL_PACK_ONLY':
        # All eleven official exports are already paid/saved. The small CPU
        # bank finalization shares this same training allocation and its CPU
        # envelope, so there is no second future export/PBS/output reservation.
        credit['PBS_submissions']+=1;credit['Hopper_CPU_s']+=36600
        credit['Hopper_storage_bytes']+=export.original.LIMITS['output_bytes']+export.METADATA
    return credit

def settle_training(g,request_id,*,native,effects,audit,now_epoch):
    out=deepcopy(g);r=rows(out)[0];need(r['request_id']==request_id and r['status']==STATES[1],
        'TRAIN_KNOWN_BOUND_NOT_ALREADY_SETTLED')
    need(native['job_state']=='F'and native['job_id']==r['job_id']and native['request_id']==request_id
        and native['submission_nonce']==r['submission_nonce']and type(native['Exit_status'])is int,
        'TRAIN_REAL_NATIVE_F_NOT_TIMEOUT')
    need(effects['schema']==EFFECTS and effects['request_id']==request_id
        and effects['request_sha256']==r['request_sha256']and effects['job_id']==r['job_id']
        and effects['arm']==bank.IDENTITY and effects['pending_updates']==effects['unknown_query_intents']==[],
        'TRAIN_ACTUAL_KNOWN_SAME_GEN_EFFECTS')
    keys=['optimizer_updates_committed','query_attempts','model_load_attempts'];vals=[effects[k]for k in keys]
    need(all(type(x)is int and 0<=x<=n for x,n in zip(vals,[1024,29,4])),'TRAIN_REAL_BOUNDED_INTEGER_EFFECTS')
    gpu,cpu=native['conservative_GPU_s'],native['conservative_CPU_s']
    need(all(type(x)in(int,float)and math.isfinite(x)and 0<=x<=n for x,n in [(gpu,22560),(cpu,54000)]),
        'TRAIN_REAL_BOUNDED_ALLOCATION_COST')
    steps,queries,loads=vals;success=native['Exit_status']==0 and steps==1024 and audit.get('status')==AUDIT
    if success:need(audit['arm']==bank.IDENTITY and audit['actual_consumed_cursor']==32768
        and audit['parent_checkpoint_manifest_sha256']==original.PARENT
        and audit['statistics_sha256']==original.STATS,'TRAIN_FULL_INDEPENDENT_STATE_NOT_COUNTS_ONLY')
    for k,n in dict(H200_s=22560,optimizer_updates=1024,neural_queries=29).items():out['reserved'][k]-=n
    out['CPU_reserved_s']-=54000;out['storage_reserved_bytes']-=70000000000
    out['used']['H200_s']+=gpu;out['optimizer_updates']+=steps;out['used']['optimizer_updates']=out['optimizer_updates']
    out['generation_queries']+=queries;out['used']['neural_queries']=out['generation_queries'];out['model_loads']+=loads;out['CPU_s']+=cpu
    out['new_storage_current_bytes']+=70000000000;out['new_storage_peak_bytes']=max(out['new_storage_peak_bytes'],out['new_storage_current_bytes'])
    r.update(status=STATES[-1],native_exit_status=native['Exit_status'],scientific_endpoint_complete=success,
        committed_optimizer_updates=steps,query_attempts=queries,model_load_attempts=loads,
        conservative_H200_s=gpu,conservative_CPU_s=cpu,reserved_optimizer_updates=0,reserved_neural_queries=0,
        terminal_native_sha256=sha(raw(native)),terminal_effect_summary_sha256=sha(raw(effects)),
        full_checkpoint_audit_sha256=sha(raw(audit)),settled_epoch=now_epoch)
    if success:out[authority.FIELD][FIELD]['endpoint']=deepcopy(audit)
    out['revision']+=1;rows(out);original.validate_global(out);return out
