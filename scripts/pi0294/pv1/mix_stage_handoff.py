"""Byte-bound training handoff through the TWO EXISTING campaign ledgers.

Pure validation/proposal only. The original once IO persists any proposal;
neither a prepared endpoint nor this CPU check grants model/physics admission.
"""
from copy import deepcopy
import json,re
from . import mix_accounting as a,successor_cd as s


def need(ok,msg):
    if not ok:raise PermissionError('V5_CLOSED_TRAIN_HANDOFF_'+msg)


def load(raw):
    need(type(raw)is bytes,'ACTUAL_CAPTURED_BYTES_REQUIRED')
    return json.loads(raw)


def closed_endpoint(*,global_bytes,settled_bytes,lease_bytes,closed_bytes,owner_exit_bytes,OS_exit_bytes,arm):
    g=load(global_bytes);a.validate_global(g);p=a.package_view(g)
    need(arm in ('C50','D25'),'ONLY_PREDECLARED_TWO_ARMS')
    expected=list(s.ORDER[:1 if arm=='C50' else 3])
    need(p['completed_stages']==expected,'EXACT_TRAIN_THEN_EVAL_SEQUENCE')
    settled=load(settled_bytes);lease=load(lease_bytes);closed=load(closed_bytes)
    owner=load(owner_exit_bytes);os_exit=load(OS_exit_bytes)
    native,effects,audit=settled['native'],settled['effects'],settled['audit']
    endpoint=p['endpoints'][arm];matches=[r for r in g['PBS_submissions']if r.get('request_id')==endpoint['request_id']]
    need(len(matches)==1,'ONE_ACTUAL_TERMINAL_REQUEST');row=matches[0];a.validate_item(row)
    need(row['status']=='NATIVE_TERMINAL' and row['scientific_endpoint_complete']is True
         and row['arm']==arm and row['native_exit_status']==0 and row['committed_optimizer_updates']==1024,
         'ACTUAL_SUCCESSFUL_F_1024_NOT_FAILED_SAVED_EXCEPTION')
    need(native['job_state']=='F' and native['Exit_status']==0 and native['job_id']==row['job_id']
         and native['request_id']==row['request_id'] and native['submission_nonce']==row['submission_nonce']
         and a.sha(a.raw(native))==row['terminal_native_sha256']
         and a.sha(a.raw(effects))==row['terminal_effect_summary_sha256']
         and a.sha(a.raw(audit))==row['full_checkpoint_audit_sha256']==endpoint['audit_sha256'],
         'EXACT_TERMINAL_EFFECT_AND_INDEPENDENT_AUDIT_BYTES')
    need(audit.get('status')=='PASS_INDEPENDENT_V5_CD_1024_COMPLETE_STATE'
         and audit.get('arm')==arm and audit.get('member_count')==15
         and audit.get('actual_consumed_cursor')==32768 and audit.get('optimizer_steps')==[1024]
         and audit.get('scheduler_last_epoch')==1024 and {'python','numpy','cpu'}<=set(audit.get('rng_keys',[]))
         and audit.get('parent_checkpoint_manifest_sha256')==a.PARENT
         and audit.get('statistics_sha256')==a.STATS
         and audit.get('checkpoint_manifest_sha256')==endpoint['checkpoint_manifest_sha256']
         and audit.get('checkpoint_root')==endpoint['checkpoint_root']
         and re.fullmatch('[0-9a-f]{64}',audit.get('probe_result_sha256','')),
         'ALL_FIFTEEN_REAL_MEMBERS_OPTIMIZER_RNG_SAMPLER_AND_ORIGINAL_PROBES')
    need(lease['status']=='CLOSED' and lease['run_uuid']==row['run_uuid']
         and lease['terminal_receipt_sha256']==a.sha(settled_bytes)
         and closed['lease_sha256']==a.sha(lease_bytes) and closed['job_id']==row['job_id']
         and closed['native_exit_status']==0 and closed['complete_endpoint']is True,
         'ACTUAL_GPU_LEASE_CLOSE_AND_READBACK_NOT_PREPARED_CLOSE')
    need(type(owner.get('returncode'))is int and type(os_exit.get('returncode'))is int
         and owner['returncode']==os_exit['returncode']==0
         and owner.get('endpoint_status')==audit['status'] and os_exit.get('source_sha')==row['source_sha']
         and os_exit.get('foreground_wait_completed')is True
         and lease['closed_epoch']<=owner['observed_epoch']<=os_exit['finished_epoch'],
         'OWNER_NATURAL_SUCCESSFUL_OS_EXIT_NOT_LIVENESS_TIMEOUT')
    return dict(schema='V5_CD_REAL_CLOSED_1024_TRAINING_HANDOFF_V1',arm=arm,
        request_id=row['request_id'],job_id=row['job_id'],source_sha=row['source_sha'],
        checkpoint_manifest_sha256=endpoint['checkpoint_manifest_sha256'],global_step=1024,
        global_sha256=a.sha(global_bytes),settled_sha256=a.sha(settled_bytes),
        closed_lease_sha256=a.sha(lease_bytes),closed_receipt_sha256=a.sha(closed_bytes),
        owner_exit_sha256=a.sha(owner_exit_bytes),OS_exit_sha256=a.sha(OS_exit_bytes),
        independent_audit_sha256=row['full_checkpoint_audit_sha256'],
        no_new_native_effects=True,model_or_physics_admission=False)


def project_training_to_LOCAL(*,local_bytes,global_bytes,terminal_capture,arm):
    """Exact already-paid delta; no second reserve, refund or global mutation."""
    handoff=closed_endpoint(global_bytes=global_bytes,arm=arm,**terminal_capture)
    g,l=load(global_bytes),load(local_bytes);a.validate_global(g)
    need(handoff.get('schema')=='V5_CD_REAL_CLOSED_1024_TRAINING_HANDOFF_V1'
         and handoff.get('global_sha256')==a.sha(global_bytes),'SAME_FULL_CLOSED_GLOBAL_CAPTURE')
    need(l.get(s.FIELD)==s.validated(g) and l['unknown_effects']==l['pending_updates']==l['pending_queries']==[],
         'SAME_ACTUAL_LOCAL_SCOPE_WITHOUT_UNKNOWN_EFFECTS')
    key=handoff['request_id'];arm=handoff['arm'];row=next(r for r in g['PBS_submissions']if r.get('request_id')==key)
    mirrors=l.get('V5_CD_terminal_training_mirrors',{})
    need(key not in mirrors and not any(r.get('request_id')==key for r in l['PBS_submissions']),
         'NO_SECOND_ACCOUNTING_MIRROR_OR_SUBMISSION')
    local_package=a.package_view(l);global_package=a.package_view(g)
    need(local_package['completed_stages']==list(s.ORDER[:0 if arm=='C50' else 2])
         and global_package['completed_stages']==list(s.ORDER[:1 if arm=='C50' else 3]),'ONE_EXACT_LOCAL_STAGE_ADVANCE')
    # Controls were prepaid in BOTH ledgers before this request. Only this
    # actual allocation's original reserved vector is now removed once.
    allocation=dict(H200_s=22560,PBS_submissions=1,optimizer_updates=1024,neural_queries=29,
                    Hopper_CPU_s=54000,Hopper_storage_bytes=70000000000)
    expected=deepcopy(local_package['remaining_resources'])
    for k,value in allocation.items():expected[k]-=value
    need(expected==global_package['remaining_resources'],'NO_DROPPED_OTHER_STAGE_OR_GEN_ESCROW')
    need(row['terminal_native_sha256'] and row['status']=='NATIVE_TERMINAL' and row['arm']==arm
         and row['native_exit_status']==0 and row['scientific_endpoint_complete']is True,'SAME_COMPLETE_ACCOUNTED_STAGE')
    v=deepcopy(l);v['GPU_reserved_s']-=22560;v['GPU_allocated_s']+=row['conservative_H200_s']
    v['CPU_reserved_s']['Hopper']-=54000;v['CPU_s']['Hopper']+=row['conservative_CPU_s']
    v['optimizer_updates']+=row['committed_optimizer_updates'];v['neural_queries']+=row['query_attempts']
    v['PBS_submissions'].append(deepcopy(row));v[a.PACKAGE]=deepcopy(global_package)
    v.setdefault('V5_CD_terminal_training_mirrors',{})[key]=dict(handoff=deepcopy(handoff),
        actual_GPU_s=row['conservative_H200_s'],actual_CPU_s=row['conservative_CPU_s'],
        actual_updates=row['committed_optimizer_updates'],actual_queries=row['query_attempts'],
        old_COSTS_unchanged=True,new_GLOBAL_or_PBS_writes=0)
    need(v['GPU_reserved_s']>=0 and v['CPU_reserved_s']['Hopper']>=0,'NO_RESERVATION_UNDERFLOW')
    return dict(proposed_LOCAL=v,before_LOCAL_sha256=a.sha(local_bytes),
        exact_global_sha256=a.sha(global_bytes),native_IO_performed=False,
        original_onceIO_required=True,admission=False)
