"""Complete arithmetic for one current target zero-integration IO component.

No IO writer/admission. Keep the original matrix and controller envelopes.
Closed SDK CPU receives only its already-qualified same-work credit; the new
IO component receives NO credit. Its eight holds remain visible, conservatively
promoted to committed uppers solely in the legacy forecaster's arithmetic view.
"""
from copy import deepcopy
import base64

from . import target_zero_counter_contract as common
from . import target_sdk_budget as sdk
from . import target_sdk_closed_costs as closed_sdk
from .target_zero_quota import ROLE, SCHEMA as REQUEST_SCHEMA
from . import target_native_io_predecessor as predecessor

CAMPAIGN,T0,STOP_NEW,HARD_CLOSE = sdk.CAMPAIGN,sdk.T0,sdk.STOP_NEW,sdk.HARD_CLOSE
ROWS='target_native_IO_reservations'
SETTLEMENTS='target_native_IO_settlements'
AGGREGATE='target_native_IO_reserved'
ROW_SCHEMA='DEC1_TARGET_NATIVE_ZERO_IO_RESOURCE_RESERVATION_V1'
WORK=sdk.WORK
COMPONENT='ORIGINAL_TARGET_ZERO_INTEGRATION_IO_BINDING_V1'
REQUESTED=dict(zip(common.DIMS,(3600,60,0,0,8,512*1024**2,64*1024**2,128*1024**2)))
LOCAL_NAMESPACE='science-next'
PROJECTION='plans/TARGET_NATIVE_IO_COMPLETE_RESOURCE_PROJECTION.private.json'
FORECAST='plans/TARGET_NATIVE_IO_FULL_REMAINING_RESOURCE_INPUTS.private.json'
SOURCE_COUNTER_OUTPUT=7147094016
need=common.need


def identity(p,c):
    need(p['run_uuid']==c['campaign_run_uuid']==CAMPAIGN
         and p['T0_epoch']==c['original_T0_epoch']==T0
         and p['hard_closeout_epoch']==c['hard_close_epoch']==HARD_CLOSE
         and c['authority_release']==common.AUTHORITY and c['VAP1_authority_release']==common.VAP1_AUTHORITY
         and p['unknown_effects']==p.get('Hopper_unknown_effects',[])==p['pending_updates']==c['unknown_effects']==[],
         'IO_CURRENT_ORIGINAL_AUTHORITY_CLOCK_AND_KNOWN_LEDGER_EFFECTS')


def rows(p,c):
    identity(p,c);a=p.get(ROWS,{})
    need(common.same(a,c.get(ROWS,{})) and len(a)<=2,'IO_EXACT_PAIRED_COMPONENT_AND_ONE_KNOWN_ZERO_COLD_REPAIR')
    need(sum(row['status']=='RESERVED' for row in a.values())<=1,'IO_ONE_CURRENT_NATIVE_OWNER')
    if len(a)==2:
        need(predecessor.REQUEST in a,'IO_COLD_REPAIR_ONLY_EXACT_ORIGINAL_IMPORT_FAILURE')
        original=a[predecessor.REQUEST]
        current=next(row for rid,row in a.items() if rid!=predecessor.REQUEST)
        packet=base64.b64decode(current['predecessor_packet_base64'],validate=True)
        cert=predecessor.verify(packet,prior_row=original,prior_settlement=p[SETTLEMENTS][predecessor.REQUEST])
        need(current['predecessor_certificate']==cert,'IO_ORIGINAL_CLOSED_PREDECESSOR_CERTIFICATE_UNCHANGED')
    spent=dict.fromkeys(common.DIMS,0);held=dict.fromkeys(common.DIMS,0)
    for rid,row in a.items():
        wanted=common.vector(row['requested']);left=common.vector(row['remaining'])
        need(row['schema']==ROW_SCHEMA and row['scope']==ROLE and row['request_id']==rid
             and row['funded_work_id']==WORK and row['component_id']==COMPONENT
             and row['status'] in ('RESERVED','SETTLED') and row['unknown_effects']==[]
             and wanted==REQUESTED,'IO_FIXED_ZERO_INTEGRATION_COMPONENT_AND_ALL_EIGHT_DIMENSIONS')
        if row['status']=='RESERVED':
            need(left==wanted,'IO_FULL_UNSETTLED_HOLD');amount=left;dest=held
        else:
            need(not any(left.values()),'IO_SETTLED_HOLDS_ZERO')
            record=p[SETTLEMENTS][rid]
            need(common.same(record,c[SETTLEMENTS][rid])
                 and common.sha(common.encode(record))==row['settlement_sha256']
                 and record['schema']=='DEC1_TARGET_NATIVE_IO_CPU_SETTLEMENT_V1'
                 and record['scope']==ROLE and record['request_id']==rid
                 and record['manifest_sha256']==row['manifest_sha256']
                 and record['target_task_native_qualified'] is False
                 and record['native_prefix_safety_qualified'] is False and record['core64_cells']==0,
                 'IO_EXACT_PAIRED_FINANCIAL_SETTLEMENT_NOT_TASK_PASS')
            amount=common.vector(record['financial_charged']);dest=spent
            need(all(amount[k]<=wanted[k] for k in common.DIMS),'IO_NO_UNRESERVED_COST')
        for k in common.DIMS:dest[k]+=amount[k]
    for ledger in (p,c):
        need(common.same(ledger.get(AGGREGATE,dict.fromkeys(common.DIMS,0)),held),'IO_FULL_EIGHT_HOLD_AGGREGATE')
    need(c['CPU_s']>=spent['CoreLab_CPU_s'] and p['CPU_s']['local_PC']>=spent['local_CPU_s']
         and c['constructors']>=spent['constructors']
         and p['network_transfer_accounted_bytes']>=spent['transfer_bytes'],'IO_PAID_COSTS_ALREADY_IN_ACTUAL_TOTALS')
    return a,spent,held


def forecast_pack(inputs,*,DEC1_bytes,SDK_closures):
    from .target_sdk_resources import forecast_pack as original_pack
    common.obj(DEC1_bytes)
    closures={rid:{k:base64.b64encode(v).decode() for k,v in value.items()}
              for rid,value in SDK_closures.items()}
    need(all(set(v)=={'dual_bytes','lease_bytes'} for v in closures.values()),'IO_EXACT_DUAL_CLOSURE_BYTE_NAMES')
    return common.encode(dict(schema='DEC1_TARGET_NATIVE_IO_COMPLETE_FORECAST_INPUTS_V1',
        original_inputs_base64=base64.b64encode(original_pack(inputs)).decode(),
        DEC1_base64=base64.b64encode(DEC1_bytes).decode(),SDK_closures_base64=closures))


def unpack_forecast(raw):
    from .target_sdk_resources import unpack_forecast as original_unpack
    v=common.obj(raw)
    need(v['schema']=='DEC1_TARGET_NATIVE_IO_COMPLETE_FORECAST_INPUTS_V1','IO_COMPLETE_FORECAST_PACK_SCHEMA')
    inputs=original_unpack(base64.b64decode(v['original_inputs_base64'],validate=True))
    db=base64.b64decode(v['DEC1_base64'],validate=True)
    closures={rid:{k:base64.b64decode(value,validate=True) for k,value in row.items()}
              for rid,row in v['SDK_closures_base64'].items()}
    need(common.same(common.obj(forecast_pack(inputs,DEC1_bytes=db,SDK_closures=closures)),v),
         'IO_CANONICAL_ALL_ORIGINAL_GRANTS_AND_SDK_HISTORY_PACK')
    return inputs,db,closures


def forecast(*,snapshot,plan,parent_bytes,Core_bytes,DEC1_bytes,SDK_closures,
             namespace_proof,now_epoch,**original_inputs):
    p,c,d=map(common.obj,(parent_bytes,Core_bytes,DEC1_bytes));a,spent,held=rows(p,c)
    sdk_rows,sdk_spent,sdk_held=sdk.rows(p,c)
    need(sdk_rows and not any(sdk_held.values()),'IO_ONLY_AFTER_ACTUAL_SDK_COMPONENTS_FINANCIALLY_CLOSED')
    projection=closed_sdk.propose(parent_bytes,Core_bytes,DEC1_bytes,closures=SDK_closures)
    need(set(d.get(closed_sdk.MARKER,{})) in (set(),set(sdk_rows)),
         'IO_NO_PARTIAL_OR_UNRECOGNIZED_SDK_NATIVE_PROJECTION')
    need(not d.get('closed_Core_target_native_IO_projections')
         and not snapshot.get('target_native_IO_component_arithmetic_view')
         and 'closed_SDK_storage_projection' not in snapshot,
         'IO_NEW_FULL_CONTEXT_NOT_A_SECOND_CREDIT_OR_STORAGE_APPLICATION')
    need(snapshot['actual_DEC1_ledger_sha256']==common.sha(DEC1_bytes)
         and snapshot['dec1_used']['engine_starts']==d['constructors']
         and snapshot['dec1_used']['transfer_bytes']==d['network_transfer_accounted_bytes'],
         'IO_EXACT_CURRENT_NATIVE_DEC1_COUNTER_AND_TRANSFER_BASE')
    need(snapshot['used']['Core_storage_bytes']==100*1024**3+SOURCE_COUNTER_OUTPUT
         and snapshot['dec1_used']['Core_storage_bytes']==80*1024**3+SOURCE_COUNTER_OUTPUT,
         'IO_ORIGINAL_SOURCE_AND_COUNTER_STORAGE_BASE_ONCE')
    need(p['CPU_reserved_s']['CoreLab']==held['CoreLab_CPU_s']
         and p['CPU_reserved_s']['local_PC']==held['local_CPU_s'],
         'IO_NO_HIDDEN_OTHER_CORE_OR_LOCAL_CPU_HOLD')
    # LOCAL output stays inside the original inclusive pool. This additional
    # check includes the IO hold even though the legacy SDK check cannot see it.
    sdk.namespace(namespace_proof,requested_local_bytes=held['local_output_bytes'],now_epoch=now_epoch)
    pv,cv,s,q=map(deepcopy,(p,c,snapshot,plan))
    pv['CPU_reserved_s']['CoreLab']-=held['CoreLab_CPU_s']
    pv['CPU_reserved_s']['local_PC']-=held['local_CPU_s']
    pv['CPU_s']['CoreLab']+=held['CoreLab_CPU_s'];pv['CPU_s']['local_PC']+=held['local_CPU_s']
    cv['CPU_s']+=held['CoreLab_CPU_s'];cv['constructors']+=held['constructors']
    pv['engine_starts']+=held['constructors'];pv['network_transfer_accounted_bytes']+=held['transfer_bytes']
    # The legacy SDK wrapper re-adds every SDK row. Remove ONLY the exact
    # already-projected SDK portion from its arithmetic base to avoid counting
    # it twice. Preserve the original actual snapshot in the returned witness.
    already_ctor=sdk_spent['constructors']-projection['delta_fields']['constructors']
    already_net=sdk_spent['transfer_bytes']-projection['delta_fields']['network_transfer_accounted_bytes']
    need(already_ctor>=0 and already_net>=0,'IO_NO_NEGATIVE_SDK_PROJECTION_DEDUCTION')
    s['dec1_used']['engine_starts']+=spent['constructors']+held['constructors']-already_ctor
    s['dec1_used']['transfer_bytes']+=spent['transfer_bytes']+held['transfer_bytes']-already_net
    output=spent['CoreLab_output_bytes']+held['CoreLab_output_bytes']
    for label,peaks in (('used','storage_peaks'),('dec1_used','dec1_storage_peaks')):
        s[label]['Core_storage_bytes']+=output
        s[peaks]['Core_storage_bytes']=max(s[peaks]['Core_storage_bytes'],s[label]['Core_storage_bytes'])
    q['storage_lifecycle']['new_output_peak_upper_bytes']['Core_storage_bytes']+=output
    s['target_native_IO_component_arithmetic_view']=dict(
        schema='DEC1_NATIVE_IO_LEGACY_ARITHMETIC_INPUT_NOT_ACTUAL_OBSERVATION_V1',
        original_actual_snapshot_sha256=common.sha(common.encode(snapshot)),IO_held=deepcopy(held),
        SDK_already_projected_deduction=dict(constructors=already_ctor,transfer_bytes=already_net))
    result=sdk.forecast(snapshot=s,plan=q,parent_bytes=common.encode(pv),Core_bytes=common.encode(cv),
        namespace_proof=namespace_proof,now_epoch=now_epoch,**original_inputs)
    # Keep the SDK's complete conservative wall bound and add this component's
    # finite interval. No CPU, constructor, output, GPU or physics credit for IO.
    result['projected_finish_epoch']+=1800
    if result['projected_finish_epoch']>=result['clock']['hard_closeout_epoch']:
        result['clock_failures'].append('NATIVE_IO_HARD_CLOSE')
    if result['projected_finish_epoch']-q['package_closeout_wall_s']>=result['clock']['stop_physics_epoch']:
        result['clock_failures'].append('NATIVE_IO_PHYSICS_CUTOFF')
    if now_epoch+1800>=STOP_NEW:result['clock_failures'].append('NATIVE_IO_NEW_COMPONENT_CUTOFF')
    result['admissible_forecast']=not result['global_shortfalls'] and not result['DEC1_shortfalls'] and not result['clock_failures']
    result['status']='PASS_NATIVE_IO_FULL_RESOURCES_NOT_NATIVE_ADMISSION' if result['admissible_forecast'] else 'STOP_RESOURCE_OR_CLOCK'
    result['native_IO_accounting']=dict(schema='DEC1_NATIVE_IO_FULL_COMMITMENT_ARITHMETIC_VIEW_V1',
        parent_sha256=common.sha(parent_bytes),Core_sha256=common.sha(Core_bytes),DEC1_sha256=common.sha(DEC1_bytes),
        original_actual_snapshot=deepcopy(snapshot),requests=list(a),actual_spent=spent,actual_reserved=held,
        IO_credit=dict.fromkeys(common.DIMS,0),SDK_existing_native_projection_deducted_once=dict(
            constructors=already_ctor,transfer_bytes=already_net),SDK_projection_missing=deepcopy(projection['delta_fields']),
        missing_SDK_marker_not_claimed_as_native_adoption=True,unknown_journal_not_cleared_or_reissued=True,
        holds_shown_as_committed_uppers_not_measured_use=True,LOCAL_namespace=LOCAL_NAMESPACE,
        original_matrix_and_all_controller_envelopes_preserved=True,native_dispatch_admitted=False)
    return result
