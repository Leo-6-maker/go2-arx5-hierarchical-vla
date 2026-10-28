"""Finite five-session terminal in the original GLOBAL, never a dispatcher.

PBS F, the actual worker/wrapper finals and known local effect links precede
this pure proposal. The original owner independently audits saved state and
outputs; runtime COMPLETE alone cannot qualify either512 endpoint. Existing
onceIO, a native readback and own lease closure still follow this proposal.
"""
from copy import deepcopy
import math,re

from . import minimal_execution as e
from .minimal_effects import SUMMARY

NATIVE='ARX5_MINIMAL_ACTUAL_PBS_AND_PROCESS_TERMINAL_V1'
AUDIT='PASS_INDEPENDENT_ARX5_MINIMAL_FIVE_SESSION_COMPLETE_STATE_AND_OUTPUTS'
PARTIAL='KNOWN_FAILED_OR_PARTIAL_MINIMAL_SESSION_INVENTORY_PRESERVED'
SCHEMA='ARX5_MINIMAL_ORIGINAL_GLOBAL_ONCE_NATIVE_TERMINAL_V1'
need=e.need


def bounded(value,upper,*,integer=False):
    need(type(value)is int if integer else type(value)in(int,float),'TERMINAL_EXPLICIT_NUMERIC_TYPE')
    need(math.isfinite(value)and 0<=value<=upper,'TERMINAL_EFFECT_WITHIN_OWN_ORIGINAL_HOLD')
    return value


def digest(value):
    need(type(value)is str and re.fullmatch('[0-9a-f]{64}',value),'TERMINAL_REAL_FULL_SHA256')
    return value


def terminal_values(row,native,effects,audit,*,retry_global=None):
    need(native.get('schema')==NATIVE and native.get('job_state')=='F'
        and native.get('job_id')==effects.get('job_id')==row['job_id']
        and native.get('request_id')==effects.get('request_id')==row['request_id']
        and native.get('submission_nonce')==row['submission_nonce']
        and native.get('source_sha')==effects.get('source_sha')==row['source_sha']
        and native.get('request_sha256')==effects.get('request_sha256')==row['request_sha256']
        and type(native.get('Exit_status'))is int
        and native.get('all_native_workers_terminal')is True
        and native.get('terminal_inventory_stable')is True,
        'TERMINAL_SAME_NATIVE_F_WORKER_WRAPPER_NOT_TIMEOUT_OR_OLD_ARM')
    for key in ('raw_native_sha256','process_final_sha256','wrapper_final_sha256','inventory_sha256'):
        digest(native.get(key))
    need(type(native.get('finished_epoch'))in(int,float)and math.isfinite(native['finished_epoch'])
        and row['submitted_epoch']<=native['finished_epoch']<=row['deadline_epoch']
        and type(native.get('observed_epoch'))in(int,float)and math.isfinite(native['observed_epoch'])
        and native['finished_epoch']<=native['observed_epoch'],
        'TERMINAL_ACTUAL_NATIVE_FINISH_INSIDE_UNRESET_WINDOW')
    gpu=bounded(native['conservative_GPU_s'],row['reserved_H200_s'])
    cpu=bounded(native['conservative_CPU_s'],row['CPU_upper_s'])
    # The original 900s remains in its original settled row. Only the explicit
    # retry contract can prove that history and exclude a second charge here.
    prior=e.PREPARATION_CPU_UPPER
    if retry_global is not None:
        from .minimal_retry_execution import terminal_CPU_prior
        prior=terminal_CPU_prior(retry_global,row)
    bounded(native.get('new_native_conservative_CPU_s'),row['CPU_upper_s']-prior)
    need(native.get('prior_readonly_preparation_CPU_upper_s')==prior
        and cpu==native.get('new_native_conservative_CPU_s',-1)+prior,
        'TERMINAL_PRIOR900_INCLUDED_ONCE_NOT_REFUNDED_OR_DOUBLE_ADDED')
    need(effects.get('schema')==SUMMARY and effects.get('pending_updates')==[]
        and effects.get('unknown_query_intents')==[]
        and effects.get('global_budget_is_only_external_reservation')is True
        and effects.get('new_physics')==effects.get('new_episodes')==0,
        'TERMINAL_ORIGINAL_LINKED_KNOWN_EFFECTS_NO_UNKNOWN_CLEAR')
    counts={key:bounded(effects[key],row[limit],integer=True)for key,limit in
        [('optimizer_updates_committed','optimizer_update_limit'),('query_attempts','query_limit'),
         ('model_load_attempts','model_load_limit')]}
    sessions=effects.get('sessions');ids=[s['id']for s in e.native_sessions()]
    need(type(sessions)is dict and set(sessions)==set(ids),'TERMINAL_ALL_FIVE_SESSION_SLICES')
    complete=[];totals=[0,0,0];active=[];not_started=False
    for spec,loads in zip(e.native_sessions(),(1,3,3,2,2)):
        s=sessions[spec['id']];state=s.get('status')
        need(state in('NOT_STARTED','ACTIVE','COMPLETE'),'TERMINAL_FINITE_SESSION_STATE')
        values=[bounded(s[k],upper,integer=True)for k,upper in
                [('optimizer_updates',spec['optimizer_limit']),('generation_queries',spec['query_limit']),('model_loads',loads)]]
        totals=[a+b for a,b in zip(totals,values)]
        if state=='NOT_STARTED':not_started=True;need(values==[0,0,0],'TERMINAL_FUTURE_SESSION_HAS_NO_EFFECT')
        elif state=='ACTIVE':
            need(not not_started and not active,'TERMINAL_ONLY_NEXT_SESSION_ACTIVE');active.append(spec['id'])
        else:
            need(not not_started and not active and values==[spec['optimizer_limit'],spec['query_limit'],loads],
                'TERMINAL_COMPLETE_SESSION_ORIGINAL_EXACT_ORDER_AND_COUNTS')
            digest(s.get('terminal_receipt_sha256'));complete.append(spec['id'])
    need(totals==list(counts.values())and effects.get('active_session')==(active[0]if active else None)
        and effects.get('all_five_sessions_runtime_complete')==(len(complete)==5),
        'TERMINAL_AGGREGATE_COUNTS_NOT_FORMAL_ONLY_OR_SELF_REPORTED_COMPLETE')
    need(audit.get('status')in(AUDIT,PARTIAL)and audit.get('request_sha256')==row['request_sha256']
        and audit.get('source_sha')==row['source_sha']and audit.get('job_id')==row['job_id']
        and audit.get('effect_summary_sha256')==e.sha(e.raw(effects))
        and audit.get('inventory_sha256')==native['inventory_sha256']
        and audit.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
        and audit.get('statistics_sha256')==e.R0
        and audit.get('new_model_loads')==audit.get('new_queries')==audit.get('new_updates')==audit.get('new_physics')==0,
        'TERMINAL_INDEPENDENT_SAME_JOB_FULL_STATE_OR_PRESERVED_PARTIAL_AUDIT')
    success=native['Exit_status']==0 and len(complete)==5 and audit['status']==AUDIT
    if audit['status']==AUDIT:
        need(success and audit.get('audited_session_ids')==ids
            and audit.get('same15_zero_step_raw_wrapped_arms_exact')is True
            and audit.get('two_full_optimizer_scheduler_RNG_sampler_restore_pairs_exact')is True
            and audit.get('official_full_model_reconstruction_output_pairs_exact')is True
            and audit.get('frozen_parameters_and_buffers_unchanged')is True,
            'TERMINAL_ACTUAL_MODEL_CONTRACT_NOT_CPU_FLAGS_OR_RUNTIME_COUNTS')
        endpoints=audit.get('endpoints',{})
        need(set(endpoints)==set(e.training.ARMS),'TERMINAL_BOTH_MATCHED512_ENDPOINTS')
        for arm,cp in endpoints.items():
            need(cp.get('global_step')==512 and cp.get('cursor')==16384 and cp.get('scheduler_horizon')==1024
                and cp.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
                and cp.get('statistics_sha256')==e.R0 and cp.get('full_resume')is True
                and cp.get('qualification_only')is False and cp.get('TRAIN_probes')==12
                and cp.get('language_probes')==24,'TERMINAL_FULL_OWN512_STATE_NOT_QUALIFICATION_OR_OTHER_ARM')
            for k in('checkpoint_manifest_sha256','official_inference_manifest_sha256','full_streamed_members_sha256'):
                digest(cp.get(k))
    return dict(GPU_s=gpu,CPU_s=cpu,updates=counts['optimizer_updates_committed'],queries=counts['query_attempts'],
        loads=counts['model_load_attempts'],complete_sessions=complete,success=success)


def validate_terminal(g,row,marker):
    """Typed terminal is rechecked by all original global readers."""
    saved=marker.get('terminal',{})
    need(saved.get('schema')==SCHEMA,'TERMINAL_REAL_ONCE_INDEX_NOT_STATUS_ONLY')
    amounts=terminal_values(row,saved['native'],saved['effects'],saved['audit'])
    expected=dict(native_exit_status=saved['native']['Exit_status'],conservative_H200_s=amounts['GPU_s'],
        conservative_CPU_s=amounts['CPU_s'],committed_optimizer_updates=amounts['updates'],
        query_attempts=amounts['queries'],model_load_attempts=amounts['loads'],
        terminal_native_sha256=e.sha(e.raw(saved['native'])),terminal_effect_summary_sha256=e.sha(e.raw(saved['effects'])),
        independent_audit_sha256=e.sha(e.raw(saved['audit'])),scientific_endpoint_complete=amounts['success'])
    need(all(row.get(k)==v and type(row.get(k))is type(v)for k,v in expected.items())
        and type(row.get('settled_epoch'))in(int,float)
        and math.isfinite(row['settled_epoch'])and saved['native']['observed_epoch']<=row['settled_epoch'],
        'TERMINAL_FULL_PROOF_DERIVES_EXACT_ROW_AND_ORIGINAL_WINDOW')
    p=g[e.package.FIELD]
    need(p['status']==('MAIN_NATIVE_TERMINAL_COMPLETE'if amounts['success']else'MAIN_NATIVE_TERMINAL_INCOMPLETE')
        and p['completed_sessions']==amounts['complete_sessions'], 'TERMINAL_PACKAGE_NOT_ADVANCED_BY_PREPARED_FLAGS')
    need(g['used']['H200_s']>=amounts['GPU_s']and g['CPU_s']>=amounts['CPU_s']+e.CONTROL['Hopper_CPU_s']
        and g['optimizer_updates']>=amounts['updates']and g['generation_queries']>=amounts['queries']
        and g['model_loads']>=amounts['loads'],'TERMINAL_COSTS_RETAINED_IN_ORIGINAL_GLOBAL')
    return amounts


def settle(g,request_id,*,native,effects,audit,now_epoch):
    """Known failure may close financially, never claim a scientific endpoint.

    The original full output upper stays charged, including failures. Releasing
    unused compute is not a refund of paid history or the engineering hold.
    No original C/D/GEN row, Core stage, GLOBAL pending/UNKNOWN or clock changes.
    """
    marker=e.validate(g);need(marker is not None,'TERMINAL_ADOPTED_MAIN_REQUIRED')
    row=e.rows(g)[0];need(row['request_id']==request_id and row['status']==e.STATES[1],
                         'TERMINAL_EXACT_BOUND_MAIN_ONCE_NO_SETTLE_REPLAY')
    need(g['unknown_effects']==g['pending_updates']==[]and type(now_epoch)in(int,float)
        and math.isfinite(now_epoch)and native['observed_epoch']<=now_epoch,
         'TERMINAL_KNOWN_FULL_GLOBAL_AND_ORIGINAL_WINDOW')
    amounts=terminal_values(row,native,effects,audit);v=deepcopy(g);p=v[e.package.FIELD];r=v['PBS_submissions'][-1]
    need(r['request_id']==request_id,'TERMINAL_ORIGINAL_MAIN_INVENTORY_NOT_FOREIGN_ROW')
    held=e.native_resources()
    for key in('H200_s','optimizer_updates','neural_queries'):v['reserved'][key]-=held[key]
    v['CPU_reserved_s']-=held['Hopper_CPU_s'];v['storage_reserved_bytes']-=held['Hopper_storage_bytes']
    v['used']['H200_s']+=amounts['GPU_s'];v['CPU_s']+=amounts['CPU_s'];v['optimizer_updates']+=amounts['updates']
    v['used']['optimizer_updates']=v['optimizer_updates'];v['generation_queries']+=amounts['queries']
    v['used']['neural_queries']=v['generation_queries'];v['model_loads']+=amounts['loads']
    v['new_storage_current_bytes']+=held['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    v[e.FIELD]['terminal']=dict(schema=SCHEMA,native=deepcopy(native),effects=deepcopy(effects),audit=deepcopy(audit))
    r.update(status=e.STATES[2],native_exit_status=native['Exit_status'],conservative_H200_s=amounts['GPU_s'],
        conservative_CPU_s=amounts['CPU_s'],committed_optimizer_updates=amounts['updates'],query_attempts=amounts['queries'],
        model_load_attempts=amounts['loads'],terminal_native_sha256=e.sha(e.raw(native)),
        terminal_effect_summary_sha256=e.sha(e.raw(effects)),independent_audit_sha256=e.sha(e.raw(audit)),
        scientific_endpoint_complete=amounts['success'],settled_epoch=now_epoch)
    p.update(completed_sessions=amounts['complete_sessions'],
        status='MAIN_NATIVE_TERMINAL_COMPLETE'if amounts['success']else'MAIN_NATIVE_TERMINAL_INCOMPLETE')
    from scripts.pi0294.pv1.neural_ledger import check
    v['revision']+=1;e.validate(v);check(v,allow_unknown=True);return v
