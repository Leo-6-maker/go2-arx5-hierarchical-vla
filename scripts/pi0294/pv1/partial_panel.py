"""Preserve actual C50 outcomes and forecast only its original remainder.

Byte-only cold preparation. No ledger adoption, registration, device, source
worker or retry is supplied here. A native continuation must separately bind
this history, its complete saved inventory and aggregate paid-IO custody.
"""
from copy import deepcopy
import json
from . import baseline,mix_accounting as a,mix_core_accounting as c
from . import mix_core_premodel_repair as repair
from .future_budget import _vector

FAILED='bb72707c-5e58-448d-9285-a34affd8993a'
SOURCE='8fba44f38aa5c6858614d8bd4e34ad7f969dd40c'
HISTORY_SHA='dd2806c99b34044db90e9b94d28106bdfb15d56530807d381c9e0d5b508e3cbc'


def need(ok,why):
    if not ok:raise PermissionError('PV1_ORIGINAL_PARTIAL_PANEL_'+why)


def history(population_bytes,evidence,*,checkpoint_manifest_sha256):
    result=baseline.partial_history(population_bytes,candidate='C50',evidence=evidence,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256)
    need(a.sha(a.raw(result))==HISTORY_SHA and result['valid_conditions']==11
         and result['remaining_eligible_conditions']==13 and result['exhausted_engineering_conditions']==0,
         'EXACT_TWELVE_ACTUAL_ATTEMPTS_AND_ELEVEN_VALID_OUTCOMES')
    cards=[row for panel in result['remaining_panels'].values()for row in panel]
    need(sum(row['id'].endswith('-REPLACEMENT-01')for row in cards)==1
         and not(set(result['condition_result_sha256'])&{row['condition_id']for row in cards}),
         'NO_VALID_SUCCESS_OR_FAILURE_REPLAY')
    return result


def resources():
    #13*900s source case walls +600s qualification +2700s close reserve fit5h.
    # Per-source CPU3600 remains; each original prefix/query/watchdog stays.
    return _vector(dict(H200_s=18060,episode_starts=13,neural_queries=13*80+17,
        physics_ticks=13*32000,engine_starts=13*81+2,
        Hopper_CPU_s=40000+7200,Core_CPU_s=40000+13*3600+300,
        local_CPU_s=7200,Hopper_storage_bytes=64*1024**2,
        Core_storage_bytes=128*1024**2,transfer_bytes=128*1024**2))


def forecast(global_bytes,local_bytes,*,validated_history,native_proof,effects,closed_leases):
    """Existing full fourteen-dimension forecast with an explicit13-case delta.

    The intermediate full-stage view is a calculation input only, not a
    proposed ledger. Additional metadata/control is charged in full. The raw
    continuation must prove aggregate retained IO fits the already paid panel;
    until then this conditional forecast is not an adoptable native proposal.
    """
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    row=g[c.FIELD][FAILED]
    need(row['status']=='NATIVE_TERMINAL'and row['source_sha']==SOURCE
         and row['native_exit_status']==1 and row['query_attempts']==574
         and row['model_load_attempts']==1 and row['unknown_query_intents']==[]
         and row['terminal_native_sha256']==c.digest(native_proof)
         and row['terminal_effect_summary_sha256']==c.digest(effects)
         and a.sha(a.raw(validated_history))==HISTORY_SHA,'KNOWN_NONZERO_EFFECTS_AND_VALIDATED_HISTORY')
    need(all(r['status']=='NATIVE_TERMINAL'for r in g[c.FIELD].values())
         and not g['unknown_effects']and not g['pending_updates'],'COLD_REAL_TERMINALS')
    need(set(closed_leases)=={'GPU','physical','control'},'ALL_THREE_REAL_CLOSED_LEASES')
    for role,lease in closed_leases.items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==row['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True
             and lease['original_valid_conditions']==11 and lease['original_remaining_conditions']==13,
             'EXACT_PARTIAL_CLOSURE:'+role)
    old=repair.allocation(g,'C50');new=resources();v,w=deepcopy(g),deepcopy(l)
    for value in(v,w):
        for k,cost in old.items():value[a.PACKAGE]['remaining_resources'][k]+=cost
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=old[k]
    v['CPU_reserved_s']+=old['Hopper_CPU_s'];v['storage_reserved_bytes']+=old['Hopper_storage_bytes']
    w['GPU_reserved_s']+=old['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=old[k]
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));dimensions={};gaps={}
    for k,prior in full['dimensions'].items():
        # Change only the exact restored C compute vector. Every actual old
        # cost and unrelated stage in the original forecast remains paid/held.
        total=prior['projected']-old[k]+new[k];gap=max(0,total-prior['cap'])
        dimensions[k]=dict(prior,projected=total,shortfall=gap,
            hypothetical_full_C_restore=old[k],exact_remaining13=new[k],headroom=prior['cap']-total)
        if gap:gaps[k]=gap
    return dict(status='CONDITIONAL_EXACT_REMAINING13_FULL_FUTURE_FORECAST_NOT_NATIVE_ADMISSION',
        GLOBAL_sha256=a.sha(global_bytes),LOCAL_sha256=a.sha(local_bytes),history_sha256=HISTORY_SHA,
        dimensions=dimensions,shortfalls=gaps,resources=new,
        original_valid_conditions=11,remaining_conditions=13,old_queries_paid=574,
        old_complete_and_partial_cases_retained=12,no_valid_outcome_replayed=True,
        aggregate_paid_panel_IO=dict(original_full_physical_case_capacity=26,
            old_physical_case_outputs=12,new_physical_case_outputs_max=13,
            aggregate_physical_case_outputs_max=25,
            old_neural_queries=574,new_neural_queries_max=1057,aggregate_neural_queries_max=1631,
            original_panel_query_capacity=2097,
            unchanged_model_input_no_tensor_copy_required=True,
            native_all_retained_outputs_and_model_inode_join='NOT_RUN',
            old_gross_current_peak_fees_refunded=False),
        original_900s_case_and_action_watchdogs_unchanged=True,
        original_600s80cycle_qualification_required_again=True,
        baseline_full_forecast_sha256=a.sha(a.raw(full)),
        native_hold_restored=False,native_registration=False,new_queries_updates_physics=0)


def retained_IO(readback):
    """Use actual ALL retained role/model inodes, with finite future uppers.

    This is a cold capacity calculation, not permission to copy a tensor or
    discard old logs. New native guards must enforce these reduced uppers and
    bind the same delivered model inodes before its paid IO may be consumed.
    """
    import base64,gzip,stat
    need(readback.get('status')=='ACTUAL_C12_ALL_THREE_ROOTS_AND_ORIGINAL_MODEL_RETAINED_INODE_CENSUS_READONLY'
         and readback.get('checkpoint_manifest_sha256')=='179af34b0d1b0922aa3679516b44fe4755d122738eb53fab87b4e710bc975475'
         and readback.get('new_tensor_RGB_reads')==readback.get('new_model_query_physics_writes')==0,
         'ACTUAL_ALL_RETAINED_INPUT_AND_OUTPUT_CENSUS')
    ib=gzip.decompress(base64.b64decode(readback['inventory_gzip_base64'],validate=True))
    need(a.sha(ib)==readback['inventory_sha256'],'COMPLETE_NATIVE_INVENTORY_BYTES')
    inventory=json.loads(ib);need(set(inventory['files'])=={'control','neural','physical','delivered_model'},'ALL_THREE_ROLES_AND_MODEL')
    unique={}
    for entries in inventory['files'].values():
        for row in entries.values():
            need(stat.S_ISREG(row['mode']),'REGULAR_RETAINED_BYTES')
            key=(row['dev'],row['inode'])
            need(key not in unique or unique[key]==row,'IDENTICAL_ACTUAL_SHARED_INODE')
            unique[key]=row
    logical=sum(r['size']for r in unique.values());allocated=sum(r['allocated_bytes']for r in unique.values())
    need(readback['actual']==dict(unique_inodes=len(unique),unique_logical_bytes=logical,unique_allocated_bytes=allocated),
         'COMPLETE_ACTUAL_STORAGE_TOTALS')
    # Count the delivered model once in the old retained census. New views
    # must reuse those exact inodes; no file-size or fake-hardlink discount.
    future=dict(neural_output_bytes=7*1024**3,physical_raw_bytes=13*768*1024**2,
        physical_metadata_bytes=512*1024**2,new_control_bytes=128*1024**2,new_source_bytes=64*1024**2)
    total=max(logical,allocated)+sum(future.values());paid=c.remaining_stage()['Core_storage_bytes']
    need(total<=paid,'ALL_OLD_AND_NEW_OUTPUTS_WITHIN_ORIGINAL_PAID_CORE_GROSS')
    return dict(status='ACTUAL_OLD_CENSUS_AND_CONDITIONAL_REMAINING13_FINITE_OUTPUT_BOUNDS',
        inventory_sha256=readback['inventory_sha256'],old_unique_logical_bytes=logical,
        old_unique_allocated_bytes=allocated,future_output_uppers=future,
        combined_retained_upper_bytes=total,original_Core_gross_already_paid=paid,
        Core_paid_gross_headroom_bytes=paid-total,
        original_GPU_gross_already_paid=c.remaining_stage()['Hopper_storage_bytes'],
        old_control_actual_over_128MiB_recorded=True,
        old_control_and_every_old_raw_file_retained_in_combined_total=True,
        future_native_output_guards_and_exact_input_inode_bind='NOT_RUN',
        new_native_IO_credit_adopted=False,old_cost_refund=False)
