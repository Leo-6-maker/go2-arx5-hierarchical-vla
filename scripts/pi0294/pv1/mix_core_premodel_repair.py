"""One exact zero-model Pump failure: retain fees and reuse its paid panel IO.

The original complete panel output/transfer uppers were already charged at
terminal, although no model, query, case or physics ran. They fund ONE same-arm
replacement panel, not D25 or another scientific intervention. Gross execution
limits stay unchanged; only a second invoice for that same paid IO is excluded.
No IO, native retry or scientific permission is performed by this module.
"""
from copy import deepcopy
import json
from . import mix_accounting as a,successor_cd as s

FIELD='V5_CD_C50_exact_zero_model_Pump_cold_repair'
FAILED='37e1b20b-35c6-405a-b5de-73ca7bdaa5ae'
SOURCE='92dac22ef2181c1912b80982ac2bb7d50a4686c6'
SCHEMA='V5_CD_C50_EXACT_ZERO_MODEL_PUMP_PAID_IO_ONCE_V1'


def need(ok,why):
    if not ok:raise PermissionError('V5_CD_PREMODEL_REPAIR_'+why)


def validated(g):
    """No recursive ledger validation: called by the original row validator."""
    from . import mix_core_accounting as c
    marker=g.get(FIELD)
    if marker is None:return None
    r=g[c.FIELD][FAILED];n=marker['failed_native'];e=marker['failed_effects'];p=marker['failed_physical'];f=marker['failed_financial']
    need(marker['schema']==SCHEMA and marker['scope_grant_sha256']==s.GRANT_SHA
         and marker['arm']=='C50'and marker['same_original_panel']=='C50-DEV-N294101'
         and marker['failed_request_id']==FAILED and marker['maximum_replacements']==1
         and marker['original_execution_uppers_unchanged']is True,'ONE_EXACT_ALREADY_APPROVED_PANEL')
    need(r['status']==c.STATES[3]and r['source_sha']==SOURCE and r['evaluation_candidate']=='C50'
         and r['native_exit_status']==1 and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and c.progress(r)[:2]==(0,0)and 0<=c.progress(r)[2]<=2
         and r['conservative_GPU_s']==79 and r['conservative_CPU_s']==2,
         'ORIGINAL_FAILED_KNOWN_ZERO_ROW_AND_ALL_COSTS_RETAINED')
    need(c.digest(n)==r['terminal_native_sha256']and c.digest(e)==r['terminal_effect_summary_sha256']
         and n['native_handle']==e['native_handle']==r['native_handle']
         and n['native_exit_status']==1 and e['effect_journal_complete']is True
         and all(type(e[k])is int and e[k]==0 for k in('optimizer_updates_committed','new_physics','new_episodes','query_attempts','model_load_attempts'))
         and e['unknown_query_intents']==[],'COMPLETE_ACTUAL_ZERO_MODEL_EFFECTS_NOT_ABSENCE_ONLY')
    need(p['schema']=='V5_CD_CORE_POLICY_EFFECT_LEDGER_V1'and p['run_uuid']==r['child_run_uuid']
         and p['authority_release']==s.TASK_SOURCE and p['Core_CPU_s']==300
         and all(type(p[k])is int and p[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and p['reservations']=={}and p['unknown_effects']==[],'NO_CASE_PREFIX_CONSTRUCTOR_OR_PHYSICS_OUTCOME_TO_REPLAY')
    original=c.remaining_stage();credit={k:original[k]for k in('Hopper_storage_bytes','Core_storage_bytes','transfer_bytes')}
    need(marker['paid_IO_credit']==credit
         and n['new_resident_allocated_upper_bytes']==credit['Hopper_storage_bytes']
         and f['Core_output_charged_upper_bytes']==credit['Core_storage_bytes']
         and f['transfer_charged_upper_bytes']==credit['transfer_bytes']
         and f['terminal_native_sha256']==r['terminal_native_sha256']
         and f['request_id']==FAILED,'EXACT_ORIGINAL_GROSS_IO_ALREADY_CHARGED_ONCE_NOT_A_CAP_CREDIT')
    need(marker['old_raw_model_query_case_outputs_absent']is True
         and marker['all_preparation_metadata_inside_original_control_IO']is True
         and marker['actual_failed_readback_sha256']and marker['all_actual_workers_terminal']is True,
         'ALL_OLD_METADATA_RETAINED_IN_SEPARATE_ALREADY_PAID_CONTROL')
    for lease in marker['failed_closed_leases'].values():
        need(lease['status']=='CLOSED'and lease['run_uuid']==r['child_run_uuid']
             and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True,'BOTH_ACTUAL_CLOSED_LEASES')
    need(set(marker['failed_closed_leases'])=={'GPU','physical'}
         and c.canonical_uuid(marker['replacement_request_id'])
         and c.canonical_uuid(marker['replacement_submission_nonce'])
         and marker['replacement_request_id']!=FAILED
         and marker['replacement_run_uuid']=='v5-cd-c50-n294101-03'
         and marker['T_CD']==s.validated(g)['T_CD']
         and marker['deadline_epoch']==s.validated(g)['deadline_epoch']
         and marker['T_CD']<=a.number(marker['adopted_epoch'])<marker['deadline_epoch']-28860, 'FRESH_EXACT_IDENTITIES_SAME_WINDOW')
    others=[v for v in [*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]if v.get('request_id')!=marker['replacement_request_id']]
    need(not any(v.get('request_id')==marker['replacement_request_id']or v.get('submission_nonce')==marker['replacement_submission_nonce']for v in others), 'NO_REPLACEMENT_IDENTITY_REUSE')
    return marker


def allocation(g,arm):
    from . import mix_core_accounting as c
    result=c.remaining_stage();m=validated(g)
    if m and arm=='C50':
        for k,v in m['paid_IO_credit'].items():result[k]-=v
    from . import mix_core_caps_continuation as cont
    if arm=='C50'and cont.validate_rpc_limit(g)is not None:
        result['episode_starts']-=1;result['engine_starts']-=1
    return result


def output_credit(g,row):
    m=validated(g)
    from . import mix_core_recovery as recovered
    continuation=recovered.validate(g)
    from . import mix_core_prebind_terminal as prebind
    next_continuation=prebind.continuation(g)
    from . import mix_core_caps_continuation as caps_cont
    cap_retry=caps_cont.validate(g)
    rpc_retry=caps_cont.validate_RPC(g)
    peer_retry=caps_cont.validate_peer(g)
    desktop_retry=caps_cont.validate_recorded_desktop(g)
    bound_retry=caps_cont.validate_bound_age(g)
    limit_retry=caps_cont.validate_rpc_limit(g)
    identity_marker=(limit_retry if limit_retry and row['request_id']==limit_retry['replacement_request_id']else bound_retry if bound_retry and row['request_id']==bound_retry['replacement_request_id']else desktop_retry if desktop_retry and row['request_id']==desktop_retry['replacement_request_id']else peer_retry if peer_retry and row['request_id']==peer_retry['replacement_request_id']else rpc_retry if rpc_retry and row['request_id']==rpc_retry['replacement_request_id']else cap_retry if cap_retry and row['request_id']==cap_retry['replacement_request_id']else next_continuation if next_continuation and row['request_id']==next_continuation['replacement_request_id']else continuation if continuation and row['request_id']==continuation['replacement_request_id']else m)
    if m and row['request_id']==identity_marker['replacement_request_id']:
        need(row['evaluation_candidate']=='C50'and row['submission_nonce']==identity_marker['replacement_submission_nonce']
             and row['child_run_uuid']==identity_marker['replacement_run_uuid']
             and row.get('prepaid_panel_IO_credit_sha256')==a.sha(a.raw(m)), 'NO_CREDIT_ON_ANOTHER_REQUEST_OR_ARM')
        return m['paid_IO_credit']['Hopper_storage_bytes']
    need('prepaid_panel_IO_credit_sha256'not in row,'NO_ORPHAN_PREPAID_PANEL_IO')
    return 0


def propose(global_bytes,local_bytes,*,closure_capture_bytes,metadata_capture_bytes,replacement_request_id,
            replacement_submission_nonce,now_epoch):
    from . import mix_core_accounting as c
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    need(FIELD not in g and FIELD not in l and set(g[c.FIELD])=={FAILED}
         and g[a.PACKAGE]['completed_stages']==['TRAIN:C50:1024']
         and not g['unknown_effects']and not g['pending_updates'], 'ONE_INACTIVE_ORIGINAL_C50_PANEL_NO_PRIOR_REPAIR')
    cap=json.loads(closure_capture_bytes);need(cap['status']=='INDEPENDENT_C50_PREMODEL_PUMP_WRONG_LEGACY_STATUS_KNOWN_ZERO_AND_DUAL_CLOSURE_READBACK'
         and cap['request_id']==FAILED and cap['exact_cause']['all_other_registry_bindings_match']is True
         and cap['exact_cause']['saved_admission_status']==cap['exact_cause']['actual_CD_expected']
         and cap['exact_cause']['saved_admission_status']!=cap['exact_cause']['old_Pump_expected'], 'REAL_EXACT_PUMP_TYPED_STATUS_CAUSE')
    decode=lambda key:json.loads(__import__('base64').b64decode(cap['files'][key]['base64'],validate=True))
    need(decode('LOCAL')==l and all(cap['units'][k]['MainPID']=='0'for k in('owner','GPU'))
         and cap['units']['owner']['ExecMainStatus']=='0'and cap['units']['GPU']['ExecMainStatus']=='1',
         'ACTUAL_FULL_LOCAL_AND_NATIVE_TERMINALS')
    metadata=json.loads(metadata_capture_bytes)
    need(metadata['status']=='ACTUAL_C50_ALL_NEW_CONTROL_COLD_PEER_AND_SOURCE_UNIQUE_METADATA_READONLY'
         and metadata['root']==cap['root']and metadata['unmodified_model_input_inodes_excluded']==3
         and metadata['unique_logical_metadata_bytes']+48*1024**2<=metadata['paid_Core_control_output_bytes']==128*1024**2
         and metadata['unique_allocated_metadata_bytes']+48*1024**2<=128*1024**2, 'ACTUAL_ALL_RETAINED_METADATA_AND_FINITE_NEW_COLD_PREPARATION_INSIDE_PAID_CONTROL')
    result=decode('OWNER_RESULT');need(result['history']['valid_conditions']==0
         and decode('NATIVE_FINAL')['phase']=='ACTUAL_NATIVE_600S_80_CYCLE_QUALIFICATION'
         and 'PV1_NATIVE_PUMP_EXACT_CURRENT_ADMISSION_AND_REGISTRY'in decode('NATIVE_FINAL')['error'], 'KNOWN_PREMODEL_PATH_NOT_A_VALID_FAILURE')
    marker=dict(schema=SCHEMA,scope_grant_sha256=s.GRANT_SHA,arm='C50',same_original_panel='C50-DEV-N294101',
        failed_request_id=FAILED,failed_native=result['native'],failed_effects=result['effects'],
        failed_physical=decode('PHYSICAL_LEDGER'),failed_financial=decode('DUAL_FINANCE'),
        failed_closed_leases={k:decode(k+'_LEASE')for k in('GPU','physical')},
        actual_failed_readback_sha256=a.sha(closure_capture_bytes),actual_metadata_readback_sha256=a.sha(metadata_capture_bytes),
        actual_old_Core_metadata_logical_bytes=metadata['unique_logical_metadata_bytes'],actual_old_Core_metadata_allocated_bytes=metadata['unique_allocated_metadata_bytes'],
        new_cold_preparation_metadata_upper_bytes=48*1024**2,old_raw_model_query_case_outputs_absent=True,
        all_preparation_metadata_inside_original_control_IO=True,all_actual_workers_terminal=True,
        paid_IO_credit={k:c.remaining_stage()[k]for k in('Hopper_storage_bytes','Core_storage_bytes','transfer_bytes')},
        replacement_request_id=replacement_request_id,replacement_submission_nonce=replacement_submission_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-03',maximum_replacements=1,original_execution_uppers_unchanged=True,
        T_CD=s.validated(g)['T_CD'],deadline_epoch=s.validated(g)['deadline_epoch'],adopted_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l)
    for value in(v,w):value[FIELD]=deepcopy(marker)
    validated(v);added=allocation(v,'C50')
    # The failed terminal already released unused computation. Reserve the same
    # remaining computation again, while retaining its real79GPU/2CPU/300Core.
    for value in(v,w):
        p=value[a.PACKAGE];p['engineering_replacements_used']+=1
        for k,x in added.items():p['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=added[k]
    v['CPU_reserved_s']+=added['Hopper_CPU_s'];v['storage_reserved_bytes']+=added['Hopper_storage_bytes']
    w['GPU_reserved_s']+=added['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=added[k]
    c.checked(v);c.paired(v,w)
    full=s.recompute(v,w,Core_storage_upper_bytes=335556053016+marker['paid_IO_credit']['Core_storage_bytes'],local_storage_upper_bytes=9126805504)
    need(full['shortfalls']=={},'FULL_ORIGINAL_FUTURE_MATRIX_FINAL_STILL_PAYABLE:'+str(full['shortfalls']))
    return dict(proposed_global=v,proposed_LOCAL=w,marker=marker,remaining_compute_reescrow=added,
        full_remaining_proof=full,historical_paid_fees_unchanged=True,original_gross_execution_IO_uppers_unchanged=True,
        original_caps_and_clocks_unchanged=True,original_onceIO_required=True,native_IO=False)
