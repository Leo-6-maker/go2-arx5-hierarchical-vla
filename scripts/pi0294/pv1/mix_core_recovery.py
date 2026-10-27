"""Recover the original C/D measurement chain after actual Core access returns.

Pure byte-bound proposals; original onceIO, fresh process/device/queue and
model/physics admission remain mandatory. Historical pending/failed records
are immutable. The unconsumed SAME C50 panel IO has one new explicit custodian;
no paid cost, peak, or scientific failure is refunded or erased.
"""
import base64
from copy import deepcopy
import json
from . import mix_accounting as a, successor_cd as s

FIELD='V5_CD_actual_Core_recovery_20261027'
SCHEMA='V5_CD_RECOVERED_CORE_CANONICAL_LOCAL_AND_UNCONSUMED_C50_PANEL_V1'
FAILED_VANDA='72acd2ad-f3aa-42be-b93d-01993f9ee023'
RESTORE=dict(H200_s=28860,neural_queries=2097,episode_starts=26,
             physics_ticks=832000,engine_starts=2108)


def need(ok,why):
    if not ok:raise PermissionError('V5_CORE_RECOVERY_'+why)


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    from . import mix_core_premodel_repair as repair, mix_core_accounting as core
    from .mix_vanda_first import LOCAL_SHA
    old=repair.validated(g);clock=s.validated(g)
    need(m['schema']==SCHEMA and m['scope_grant_sha256']==s.GRANT_SHA
         and m['before_LOCAL_sha256']==LOCAL_SHA
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch']
         and m['compute_reescrow']==RESTORE and m['PBS_added']==0
         and m['old_costs_peaks_rows_and_unknowns_preserved']is True
         and m['Core_dispatch_requires_actual_paired_bytes']is True,
         'ONE_EXISTING_SCOPE_WITHOUT_CAP_CLOCK_OR_EFFECT_RESET')
    rows=[x for x in g['PBS_submissions']if x.get('request_id')==FAILED_VANDA]
    need(len(rows)==1,'ORIGINAL_FAILED_VANDA_ROW_RETAINED')
    r=rows[0];p=r['actual_physical_effects']
    need(r['status']=='NATIVE_TERMINAL'and r['native_exit_status']==1
         and r['query_attempts']==r['model_load_attempts']==r['committed_optimizer_updates']==0
         and r['unknown_query_intents']==[]and r['conservative_H200_s']==64
         and r['settlement_financial_deltas']['output_bytes']==0
         and all(p[n]==0 for n in ('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and p['unknown_effects']==[]and p['reservations']=={},
         'KNOWN_ZERO_EFFECT_CLOSED_PREENTRY_FAILURE_NOT_VALID_BEHAVIOR')
    need(m['old_Vanda_request_id']==FAILED_VANDA
         and m['old_Vanda_terminal_native_sha256']==r['terminal_native_sha256']
         and m['original_paid_IO_marker_sha256']==a.sha(a.raw(old))
         and m['unconsumed_same_panel_IO']==old['paid_IO_credit']
         and m['maximum_new_Core_panels']==1,
         'ONLY_UNCONSUMED_ORIGINAL_C50_IO_NO_NEW_CREDIT_OR_REFUND')
    need(core.canonical_uuid(m['replacement_request_id'])
         and core.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-04'
         and m['replacement_request_id']not in(old['failed_request_id'],FAILED_VANDA)
         and m['replacement_submission_nonce']!=old['replacement_submission_nonce'],
         'NEW_DISTINCT_CORE_CUSTODY_NOT_REISSUED_OLD_REQUEST')
    return m


def paired(g,l):
    m=validate(g)
    if m is None:return
    need(l.get(FIELD)==m,'ACTUAL_CANONICAL_LOCAL_CATCHUP_NOT_GLOBAL_ONLY')
    need(l['PBS_submissions']==g['PBS_submissions']or all(
        any(x.get('request_id')==r.get('request_id')for x in l['PBS_submissions'])
        for r in g['PBS_submissions']), 'COMPLETE_PRIOR_PBS_HISTORY_MIRRORED')


def budget_LOCAL(g,l):
    """A historical-byte projection stays a projection, never a native receipt."""
    from . import mix_vanda_first as first
    m=validate(g);need(m is not None,'RECOVERY_MARKER_REQUIRED')
    if l.get(FIELD)==m:return l
    need(a.sha(a.raw(l))==first.LOCAL_SHA,'HISTORICAL_OR_ACTUAL_CANONICAL_LOCAL_ONLY')
    old=deepcopy(g);old.pop(FIELD)
    w=first.budget_LOCAL(old,l)
    # Mirror actual Vanda aggregate spending exactly once; retain all original
    # Core spending and old physical holds. New minimal-adaptation holds remain
    # exclusively in GLOBAL, not duplicated into this old Core budget view.
    w['GPU_allocated_s']=max(w['GPU_allocated_s'],g['used']['H200_s'])
    w['CPU_s']['Hopper']=max(w['CPU_s']['Hopper'],g['CPU_s'])
    from . import mix_core_prebind_terminal as prebind
    known=prebind.validate(g)
    if known is not None:w['CPU_s']['CoreLab']+=known['physical_CPU_s']+known['neural_CPU_s']
    from . import mix_core_caps_continuation as caps_cont
    cap_retry=caps_cont.validate(g)
    if cap_retry is not None:w['CPU_s']['CoreLab']+=cap_retry['physical']['Core_CPU_s']+cap_retry['native_proof']['conservative_CPU_s']
    rpc_retry=caps_cont.validate_RPC(g)
    if rpc_retry is not None:w['CPU_s']['CoreLab']+=rpc_retry['physical']['Core_CPU_s']+rpc_retry['native_proof']['conservative_CPU_s']
    peer_retry=caps_cont.validate_peer(g)
    if peer_retry is not None:w['CPU_s']['CoreLab']+=peer_retry['physical']['Core_CPU_s']+peer_retry['native_proof']['conservative_CPU_s']
    limit_retry=caps_cont.validate_rpc_limit(g)
    if limit_retry is not None:w['CPU_s']['CoreLab']+=limit_retry['physical']['Core_CPU_s']+limit_retry['native_proof']['conservative_CPU_s']
    bound_retry=caps_cont.validate_bound_age(g)
    if bound_retry is not None:w['CPU_s']['CoreLab']+=bound_retry['physical']['Core_CPU_s']+bound_retry['native_proof']['conservative_CPU_s']
    desktop_retry=caps_cont.validate_recorded_desktop(g)
    if desktop_retry is not None:w['CPU_s']['CoreLab']+=desktop_retry['physical']['Core_CPU_s']+desktop_retry['native_proof']['conservative_CPU_s']
    w['optimizer_updates']=max(w['optimizer_updates'],g['used']['optimizer_updates'])
    w['neural_queries']=max(w['neural_queries'],g['used']['neural_queries'])
    w['PBS_submissions']=deepcopy(g['PBS_submissions'])
    w['CPU_reserved_s']['Hopper']=max(w['CPU_reserved_s']['Hopper'],g[a.PACKAGE]['remaining_resources']['Hopper_CPU_s'])
    w['GPU_reserved_s']=max(w['GPU_reserved_s'],2*28860)
    for name in (a.PACKAGE,'V5_CD_D25_Vanda_first','V5_CD_Vanda_colocated_original_panels',
                 'V5_CD_C50_exact_zero_model_Pump_cold_repair',
                 'GEN1_AMD_layout_CORR1024_authority','GEN1_exact1497713_retry_PBS37_authority',
                 'ARX5_minimal_first512_seven_cap_authority','ARX5_minimal_first512_execution_package',FIELD):
        if name in g:w[name]=deepcopy(g[name])
    proof=g['V5_CD_Vanda_colocated_original_panels']['closed_endpoints']['D25']
    row=next(x for x in g['PBS_submissions']if x.get('request_id')==proof['request_id'])
    w.setdefault('V5_CD_terminal_training_mirrors',{})[row['request_id']]=dict(
        handoff=deepcopy(proof),actual_GPU_s=row['conservative_H200_s'],
        actual_CPU_s=row['conservative_CPU_s'],actual_updates=row['committed_optimizer_updates'],
        actual_queries=row['query_attempts'],old_COSTS_unchanged=True,new_GLOBAL_or_PBS_writes=0)
    return w


def propose(global_bytes,local_bytes,*,core_readback,closed_Vanda_readback,
            new_request_id,new_nonce,now_epoch):
    from . import mix_core_premodel_repair as repair, mix_core_accounting as core
    from .mix_vanda_first import LOCAL_SHA
    g,l=json.loads(global_bytes),json.loads(local_bytes);core.checked(g)
    need(FIELD not in g and a.sha(local_bytes)==LOCAL_SHA
         and core_readback['files']['LOCAL']['sha256']==LOCAL_SHA
         and base64.b64decode(core_readback['files']['LOCAL']['base64'])==local_bytes
         and core_readback['host']=='corelab-4080s'and core_readback['uid']==2009
         and 0<=now_epoch-core_readback['observed_epoch']<=120,
         'FRESH_ACTUAL_HOST_CANONICAL_LOCAL_BYTES_NOT_CACHED_PERMISSION')
    need(all('MainPID=0'in x['stdout']for x in core_readback['units'].values())
         and all(x['lease']['status']=='CLOSED'and x['worker_process_files']==[]
                 for x in core_readback['peers'].values()), 'OLD_REAL_OWNER_AND_WORKERS_TERMINAL')
    decode=lambda name:json.loads(base64.b64decode(core_readback['files'][name]['base64']))
    lease=decode('owner_lease');result=decode('owner_result')
    need(lease['status']=='CLOSED'and lease['all_actual_workers_terminal']is True
         and result['history']['valid_conditions']==0
         and result['effects']['query_attempts']==result['effects']['model_load_attempts']==0,
         'OLD_CONTROL_ACTUALLY_FINISHED_NO_UNKNOWN_REPLAY')
    old=repair.validated(g);r=next(x for x in g['PBS_submissions']if x.get('request_id')==FAILED_VANDA)
    cap=closed_Vanda_readback
    need(cap['read_only']is True and cap['no_models']is True and cap['no_physics']is True
         and cap['no_scientific_stage_advance']is True
         and cap['native']['job_state']=='F'and cap['native']['Exit_status']==1
         and cap['native']['Job_Name']==r['job_name']
         and set(cap['leases'])=={'GPU','physical','control'},'INDEPENDENT_VANDA_KNOWN_ZERO_FULL_TERMINAL')
    for x in cap['leases'].values():
        body=base64.b64decode(x['base64'],validate=True);value=json.loads(body)
        need(a.sha(body)==x['sha256']and value.get('status')=='CLOSED'and value['all_actual_workers_terminal']is True
             and value['terminal_native_sha256']==r['terminal_native_sha256'],'ALL_PRIOR_LEASES_CLOSED')
    need(all(x['status']=='NATIVE_TERMINAL'for x in g['PBS_submissions'])
         and all(x['status']=='NATIVE_TERMINAL'for x in g.get(core.FIELD,{}).values())
         and g['pending_updates']==g['unknown_effects']==[], 'COLD_KNOWN_BOUNDARY_WITHOUT_PENDING_EFFECTS')
    need(not any(x.get('request_id')==new_request_id or x.get('submission_nonce')==new_nonce
                 for x in [*g['PBS_submissions'],*g.get(core.FIELD,{}).values()]),'UNUSED_NEW_CUSTODY')
    v=deepcopy(g);m=dict(schema=SCHEMA,scope_grant_sha256=s.GRANT_SHA,
        before_GLOBAL_sha256=a.sha(global_bytes),before_LOCAL_sha256=LOCAL_SHA,
        T_CD=s.validated(g)['T_CD'],deadline_epoch=s.validated(g)['deadline_epoch'],
        adopted_epoch=now_epoch,Core_readback_sha256=a.sha(a.raw(core_readback)),
        old_Vanda_readback_sha256=a.sha(a.raw(cap)),old_Vanda_request_id=FAILED_VANDA,
        old_Vanda_terminal_native_sha256=r['terminal_native_sha256'],
        original_paid_IO_marker_sha256=a.sha(a.raw(old)),
        unconsumed_same_panel_IO=deepcopy(old['paid_IO_credit']),maximum_new_Core_panels=1,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-04',compute_reescrow=RESTORE.copy(),PBS_added=0,
        old_costs_peaks_rows_and_unknowns_preserved=True,Core_dispatch_requires_actual_paired_bytes=True)
    need(m['T_CD']<=now_epoch<m['deadline_epoch']-28860,'ORIGINAL_COMPLETE_PANEL_WINDOW')
    v[FIELD]=m
    for k,x in RESTORE.items():v[a.PACKAGE]['remaining_resources'][k]+=x
    for k in ('H200_s','neural_queries'):v['reserved'][k]+=RESTORE[k]
    v['revision']+=1;validate(v);w=budget_LOCAL(v,l)
    core.checked(v);core.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),local_bytes)
    need(not full['cap_changes_required'],'ALL_OLD_NEW_MATRIX_FINAL_DIMENSIONS_PAYABLE:'+str(full['cap_changes_required']))
    need(all(v[k]==x for k,x in g.items()if k not in(FIELD,'revision',a.PACKAGE,'reserved')),
         'ALL_PAID_SPENDING_AND_ROWS_UNCHANGED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,
                native_dispatch_admitted=False,original_onceIO_required=True)
