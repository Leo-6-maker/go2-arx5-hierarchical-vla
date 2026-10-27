"""Exact known-zero C50 owner-read omission; no invented binding or physics.

The real process-start receipt exists, but GLOBAL never bound it. Keep its
identity as terminal evidence, never label a dead process RUNNING. Ordinary
cold continuation has a new request after all three actual leases close.
"""
import base64,json,math
from copy import deepcopy
from . import mix_accounting as a,successor_cd as s

FIELD='V5_CD_exact033195ac_prebind_terminal'
FAILED='033195ac-d1d9-46ec-b7c4-59c17691559a'
SOURCE='8120b9e47da0d33b3728ad1673d3de4c49e3b574'
KIND='V5_CD_KNOWN_ZERO_PREBIND_DESKTOP_CHECK_TERMINAL_V1'
HELPER='925bb0c0ae4709df25636906c353a39d32a22ddd0758634909e5a5ff60aa08e1'


def need(ok,why):
    if not ok:raise PermissionError('V5_CD_ZERO_PREBIND_'+why)


def row(r):
    need(r['request_id']==FAILED and r['source_sha']==SOURCE
         and r['status']=='NATIVE_TERMINAL'and r['native_handle']is None
         and r['native_exit_status']==1 and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0
         and r.get('terminal_identity_kind')==KIND
         and not any(k in r for k in('pid','boot_id','proc_start','process_started_epoch')),
         'EXACT_ACTUAL_TERMINAL_NOT_FABRICATED_RUNNING_BIND')
    return r


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    from . import mix_core_accounting as c
    r=row(g[c.FIELD][FAILED]);need(m['request_id']==FAILED and m['source_sha']==SOURCE
        and m['terminal_native_sha256']==r['terminal_native_sha256']
        and m['GPU_s']==r['conservative_GPU_s']and m['neural_CPU_s']==r['conservative_CPU_s']
        and m['query_attempts']==m['model_load_attempts']==m['new_physics']==m['new_episodes']==0
        and m['native_handle']is None and m['stage_completed']is False
        and m['actual_process_started_witness_only']is True,'ONE_ZERO_EFFECT_TERMINAL_FINANCIAL_PROOF')
    return m


def propose(global_bytes,local_bytes,*,proof,now_epoch):
    from . import mix_core_accounting as c,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);r=g[c.FIELD][FAILED]
    need(FIELD not in g and r['status']==c.STATES[0]and r['native_handle']is None
        and c.progress(r)==(0,0,0)and r['source_sha']==SOURCE,'EXACT_ONCE_UNBOUND_ZERO_ESCROW')
    p=proof;decode=lambda k:json.loads(base64.b64decode(p['files'][k]['base64'],validate=True))
    need(p['status']=='ACTUAL_033195AC_PREBIND_DESKTOP_OWNER_OMISSION_KNOWN_ZERO_READONLY'
        and p['request_id']==FAILED and p['request_sha256']==r['request_sha256']and p['source']==SOURCE
        and p['host']=='corelab-4080s'and p['uid']==2009
        and 0<=now_epoch-p['observed_epoch']<=120
        and p['qualification_or_model_or_phase_release']is False
        and p['episodes']==p['workers']==[]and p['derived_zero_effects_from_saved_ordered_premodel_control_flow']is True,
        'FRESH_COMPLETE_NATIVE_ZERO_EFFECT_CONTROL_FLOW_EVIDENCE')
    for f in p['files'].values():need(a.sha(base64.b64decode(f['base64'],validate=True))==f['sha256'],'EXACT_CAPTURED_FILE_BYTES')
    need(p['files']['HELPERS']['sha256']==HELPER,'EXACT_FAILED_FROZEN_OWNER_KERNEL_READER')
    final=decode('NATIVE_FINAL');started=decode('NATIVE_STARTED');exc=decode('OWNER_EXCEPTION');phys=decode('PHYSICAL_LEDGER')
    need(final['status']=='FAILED_PRESERVED'and final['phase']=='WAIT_ACTUAL_GLOBAL_BIND_AND_PREMODEL_RELEASE'
        and 'PV1_CORE_EVALUATOR_ORIGINAL_STOP_FENCES'in final['error']
        and exc['error']=='PV1_CORE_NATIVE_NO_FOREIGN_COMPUTE_ON_SELECTED_DEVICE'
        and started['request_id']==FAILED and started['source_sha']==SOURCE
        and started['request_sha256']==r['request_sha256']and started['native_model_permission']is False
        and started['physics_authorized']is False and p['actual_process_start']!=started['proc_start'],
        'ACTUAL_SAVED_PREMODEL_FAILURE_AND_TERMINATED_EXACT_PROCESS')
    native=next(v for k,v in p['units'].items()if k.endswith('-neural.service'))
    need(all(v['MainPID']=='0'and v['ExecMainStatus']=='1'and v['ActiveState']=='failed'for v in p['units'].values())
        and int(native['ExecMainPID'])==started['pid']and native['InvocationID'], 'REAL_SYSTEMD_EXIT_NOT_TIMEOUT_OR_PID_ABSENCE')
    need(phys['unknown_effects']==[]and phys['reservations']=={}and phys['Core_CPU_s']==300
        and all(phys[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations')),
        'COMPLETE_ORIGINAL_PHYSICAL_COUNTER_ZERO')
    end=p['boot_epoch']+int(native['ExecMainExitTimestampMonotonic'])/1e6
    gpu=math.ceil(end-started['started_epoch'])+60;cpu=math.ceil(int(native['CPUUsageNSec'])/1e9)
    need(r['created_epoch']<=started['started_epoch']<=end<=now_epoch and 60<=gpu<=120 and 0<=cpu<=3,
        'FULL_ACTUAL_PADDED_GPU_AND_PAYLOAD_CPU_COST')
    v,w=deepcopy(g),deepcopy(l);rr=v[c.FIELD][FAILED]
    effects=dict(schema=KIND,request_id=FAILED,query_attempts=0,model_load_attempts=0,new_physics=0,new_episodes=0,
        actual_process_started_witness_only=True,derived_from_ordered_frozen_premodel_source_and_actual_terminals=True)
    digest=c.digest(p);rr.update(status=c.STATES[3],terminal_identity_kind=KIND,native_exit_status=1,
        conservative_GPU_s=gpu,conservative_CPU_s=cpu,query_attempts=0,model_load_attempts=0,unknown_query_intents=[],
        reserved_neural_queries=0,terminal_native_sha256=digest,terminal_effect_summary_sha256=c.digest(effects),settled_epoch=now_epoch)
    v['used']['H200_s']+=gpu;v['reserved']['H200_s']-=28860;v['reserved']['neural_queries']-=2097
    v['CPU_s']+=cpu;v['CPU_reserved_s']-=80000
    allocation=repair.allocation(g,'C50');need(repair.output_credit(g,r)==r['additional_output_upper_bytes'],'SAME_ALREADY_PAID_UNCONSUMED_IO')
    w['GPU_reserved_s']-=28860;w['GPU_allocated_s']+=gpu
    w['CPU_reserved_s']['Hopper']-=80000;w['CPU_s']['Hopper']+=cpu
    w['CPU_reserved_s']['CoreLab']-=173900;w['CPU_s']['CoreLab']+=300+cpu
    held=w['Core_native_GPU_reserved']
    for key,x in [('episode_starts',26),('constructors',2108),('physics_ticks',832000),('CoreLab_output_bytes',allocation['Core_storage_bytes']),('transfer_bytes',allocation['transfer_bytes'])]:
        need(held[key]>=x,'NO_ESCROW_UNDERFLOW:'+key);held[key]-=x
    w[c.FIELD][FAILED]=deepcopy(rr);w['V5_CD_Core_CPU_substages'][c.key('C50')]['status']='SETTLED_KNOWN_NATIVE_TERMINAL'
    receipt=dict(schema=KIND,request_id=FAILED,source_sha=SOURCE,terminal_native_sha256=digest,
        native_handle=None,actual_process_started_witness_only=True,GPU_s=gpu,neural_CPU_s=cpu,physical_CPU_s=300,
        query_attempts=0,model_load_attempts=0,new_physics=0,new_episodes=0,stage_completed=False,observed_epoch=now_epoch,
        old_metadata_fees_and_paid_IO_upper_preserved=True,leases_closed=False)
    for value in(v,w):value[FIELD]=deepcopy(receipt)
    w.setdefault('V5_CD_Core_terminal_settlements',{})[FAILED]=deepcopy(receipt)
    v['revision']+=1;c.checked(v);c.paired(v,w)
    return dict(proposed_global=v,proposed_LOCAL=w,receipt=receipt,effects=effects,original_onceIO_required=True,new_native_admission=False)

CONTINUATION='V5_CD_exact033195ac_known_zero_cold_continuation'


def continuation(g):
    m=g.get(CONTINUATION)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_recovery as recovered,mix_core_desktop as desktop
    prior=validate(g);base=recovered.validate(g)
    need(prior is not None and base is not None and m['schema']=='V5_CD_EXACT033195AC_KNOWN_ZERO_COLD_CONTINUATION_V1'
        and m['failed_request_id']==FAILED and m['terminal_native_sha256']==prior['terminal_native_sha256']
        and m['scope_grant_sha256']==s.GRANT_SHA and m['desktop_grant_sha256']==desktop.GRANT_SHA
        and m['maximum_continuations']==1 and m['new_PBS']==0
        and m['all_paid_costs_peaks_rows_unknowns_retained']is True
        and m['replacement_run_uuid']=='v5-cd-c50-n294101-05'
        and m['T_CD']==s.validated(g)['T_CD']and m['deadline_epoch']==s.validated(g)['deadline_epoch']
        and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
        and m['replacement_request_id']not in(FAILED,base['old_Vanda_request_id'])
        and m['replacement_submission_nonce']!=base['replacement_submission_nonce'],
        'ONE_NEW_CUSTODY_AFTER_EXACT_ZERO_FAILURE_NOT_A_SCIENTIFIC_RETRY')
    for role,lease in m['closed_leases'].items():
        need(role in('GPU','physical','control')and lease['status']=='CLOSED'
            and lease['terminal_native_sha256']==prior['terminal_native_sha256']
            and lease['all_actual_workers_terminal']is True
            and lease['authority_release']==s.TASK_SOURCE
            and lease['run_uuid']==('v5-cd-c50-eval-owner-03'if role=='control'else g[c.FIELD][FAILED]['child_run_uuid']),'ACTUAL_CLOSED_PREDECESSOR_LEASES')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_TERMINAL_LEASES')
    return m


def continue_proposal(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,fixed_owner_helper_sha256,now_epoch):
    from . import mix_core_accounting as c,mix_core_premodel_repair as repair,mix_core_desktop as desktop
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=validate(g)
    need(p is not None and CONTINUATION not in g and all(x['status']=='NATIVE_TERMINAL'for x in g[c.FIELD].values())
        and g['pending_updates']==g['unknown_effects']==[], 'COLD_TERMINAL_FINANCIAL_AND_NO_CURRENT_EFFECTS')
    need(closed_readback['status']=='INDEPENDENT_033195AC_TERMINAL_THREE_CLOSED_LEASES_READONLY'
        and closed_readback['request_id']==FAILED and closed_readback['host']=='corelab-4080s'and closed_readback['uid']==2009
        and closed_readback['GLOBAL_sha256']==a.sha(global_bytes)
        and closed_readback['LOCAL_sha256']==a.sha(local_bytes)
        and 0<=now_epoch-closed_readback['observed_epoch']<=120,
        'FRESH_ACTUAL_BOTH_LEDGERS_AND_THREE_NATIVE_LEASES')
    need(not any(x.get('request_id')==new_request_id or x.get('submission_nonce')==new_nonce
        for x in [*g['PBS_submissions'],*g[c.FIELD].values(),*g.get('Core_native_GPU_allocations',{}).values()]),'DISTINCT_NEW_REQUEST_AND_NONCE')
    # This is the published cold helper, not an arbitrary changed control file.
    old=base64.b64decode(closed_readback['original_helper_base64'],validate=True)
    need(a.sha(old)==HELPER and a.sha(desktop.owner_kernel_helpers(old))==fixed_owner_helper_sha256,
        'EXACT_ORIGINAL_READER_WITH_ONLY_NEW_SCOPE_OWN_PID_WIRING')
    v,w=deepcopy(g),deepcopy(l);clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'ORIGINAL_FULL_PANEL_WINDOW')
    marker=dict(schema='V5_CD_EXACT033195AC_KNOWN_ZERO_COLD_CONTINUATION_V1',failed_request_id=FAILED,
        terminal_native_sha256=p['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        desktop_grant_sha256=desktop.GRANT_SHA,maximum_continuations=1,new_PBS=0,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-05',fixed_owner_helper_sha256=fixed_owner_helper_sha256,
        T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch,
        closed_leases=deepcopy(closed_readback['leases']),closed_readback_sha256=a.sha(a.raw(closed_readback)),
        all_paid_costs_peaks_rows_unknowns_retained=True)
    restore=repair.allocation(g,'C50')
    for value in(v,w):
        value[CONTINUATION]=deepcopy(marker)
        for key,x in restore.items():value[a.PACKAGE]['remaining_resources'][key]+=x
    for key in ('H200_s','neural_queries','optimizer_updates'):v['reserved'][key]+=restore[key]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for key,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[key]
    v['revision']+=1;continuation(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_REMAINING_PACKAGE_MATRIX_FINAL_FITS:'+str(full['cap_changes_required']))
    need(all(v[key]==x for key,x in g.items()if key not in(CONTINUATION,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),
        'ALL_PRIOR_COSTS_AND_EFFECTS_UNCHANGED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)
