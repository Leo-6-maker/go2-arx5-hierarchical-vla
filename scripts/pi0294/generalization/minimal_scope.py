"""Pure first512 package forecast on the existing complete campaign forecast.

No admission, ledger mutation, reservation, scheduler or source deployment.
Historical LOCAL remains historical. Proposed caps require explicit adoption
through the original owner; this module never supplies that authority.
"""
import hashlib
import math

from .amd_pilot_scope import full_forecast
from scripts.pi0294.pv1.future_budget import _vector
from scripts.pi0294.pv1.mix_accounting import stage_vector
from scripts.pi0294.pv1.evaluation_core_accounting import MODEL_BYTES

SCOPE = 'ARX5_MINIMAL_MATCHED_RAW_PARENT_FIRST512_V1'
GIB = 1024**3
INFERENCE_BYTES = 16 * GIB


def components():
    """Conservative old envelopes; do not infer speedup from fewer parameters.

    Two full checkpoint envelopes retain qualification and own512 state.
    Each two-step qualification is real additional accounting, followed by a
    fresh raw start for formal training; its effects/costs are never erased.
    Each arm's17 qualification calls comprise15 matched zero-step draws and
    the trained/restored step2 same-noise pair. Each original12 end-TRAIN probe remains.
    The additional language24/model is separate from those qualifications.
    """
    rows = {}
    for arm in ('E1', 'E2'):
        train = stage_vector('TRAIN')
        train['optimizer_updates'] = 512 + 2
        # Both arms share one serial PBS, not each other's optimizer or model.
        train['PBS_submissions'] = 0
        rows[arm + '_QUALIFICATION_AND_FORMAL512'] = train
        ev = stage_vector('EVAL', Core_GPU_projection=True)
        delta = INFERENCE_BYTES - MODEL_BYTES
        # Existing Eval output already includes one inference package at each
        # site and two package-sized transfer legs. Do not add a third copy.
        ev['Hopper_storage_bytes'] += delta
        ev['Core_storage_bytes'] += delta
        ev['transfer_bytes'] += 2 * delta
        rows[arm + '_CORE_ORIGINAL24'] = ev
    rows['FINITE_OFFLINE_DIAGNOSTIC_AND_SHARED_PBS'] = _vector(dict(
        H200_s=4*3600, PBS_submissions=1, Hopper_CPU_s=4*4*3600,
        local_CPU_s=600, neural_queries=17 + 3*24,
        Hopper_storage_bytes=4*GIB, transfer_bytes=256*1024**2))
    rows['ONE_KNOWN_ZERO_EFFECT_COLD_ENGINEERING_REPLACEMENT'] = _vector(dict(
        H200_s=3600, PBS_submissions=1, Hopper_CPU_s=3600, local_CPU_s=600,
        Hopper_storage_bytes=64*1024**2, Core_storage_bytes=64*1024**2,
        transfer_bytes=64*1024**2))
    return rows


def first512_forecast(global_bytes, historical_LOCAL_bytes):
    """Add the entire first phase while preserving every baseline dimension."""
    baseline = full_forecast(global_bytes, historical_LOCAL_bytes)
    # A new direct grant overlays only its seven dimensions. This is still a
    # pure forecast, not a package reservation or native model admission.
    import json
    from .minimal_adoption import effective_caps, FIELD
    global_value = json.loads(global_bytes)
    caps = effective_caps(global_value,
                          {k:r['cap'] for k,r in baseline['dimensions'].items()})
    parts = components()
    from .minimal_package import validate as package, reservation_credit
    adopted_package=package(global_value)
    from . import minimal_execution as execution, minimal_retry_execution as retry,minimal_remaining_execution as remaining
    adapter=remaining if remaining.FIELD in global_value else retry if retry.FIELD in global_value else execution
    additional = (adapter.forecast_remaining(global_value)if adopted_package is not None else
                  _vector({key: sum(row[key] for row in parts.values())
                           for key in baseline['dimensions']}))
    own_credit=reservation_credit(global_value,historical_LOCAL_bytes,baseline)
    from . import minimal_followup_scope as followup
    planned=followup.planned_unregistered(global_value)
    from . import minimal_entry_continuation_scope as entry
    entry_plan=entry.planned_unregistered(global_value)
    from . import minimal_partial_completion_scope as completion
    completion_plan=completion.planned_unregistered(global_value)
    planned={k:planned[k]+entry_plan[k]+completion_plan[k]for k in planned}
    dimensions, changes = {}, {}
    for key, old in baseline['dimensions'].items():
        projected = old['projected'] - own_credit[key] + additional[key] + planned[key]
        cap = caps[key]
        shortfall = max(0, projected - cap)
        proposed = cap
        if shortfall:
            quantum = GIB if key.endswith('bytes') else 3600 if key.endswith('_s') else 1
            proposed = math.ceil(projected / quantum) * quantum
            changes[key] = dict(before=cap, after=proposed,
                                before_display=cap/quantum, after_display=proposed/quantum,
                                unit='GiB' if key.endswith('bytes') else 'hours' if key.endswith('_s') else 'count')
        dimensions[key] = dict(old, cap=cap, baseline_projected=old['projected'],
            first512_additional=additional[key], projected=projected,
            exact_first512_reservation_projection_credit=own_credit[key],
            shortfall=shortfall, proposed_cap=proposed,
            headroom_after_proposed_cap=proposed-projected)
        if any(planned.values()):dimensions[key]['planned_unregistered_original_first512']=planned[key]
        if any(completion_plan.values()):dimensions[key]['planned_saved_prefix_completion']=completion_plan[key]
    return dict(schema='ARX5_MINIMAL_FIRST512_COMPLETE_FUTURE_FORECAST_V1', scope=SCOPE,
        status=('GLOBAL_PACKAGE_RESERVED_CORE_LOCAL_PENDING_NO_NATIVE_ADMISSION'if adopted_package is not None
                else 'FIRST512_PACKAGE_NOT_RESERVED_NO_NATIVE_ADMISSION'),
        cap_authority_adopted=FIELD in global_value,
        GLOBAL_sha256=hashlib.sha256(global_bytes).hexdigest(),
        historical_LOCAL_sha256=hashlib.sha256(historical_LOCAL_bytes).hexdigest(),
        LOCAL_is_historical=True, native_admission=False, scope_adopted=adopted_package is not None,
        baseline=baseline, components=parts, additional=additional,
        dimensions=dimensions, cap_changes_required=changes,
        formal_optimizer_updates=2*512, qualification_optimizer_updates=4,
        qualification_warmup_correction='First original LambdaLR optimizer call has LR0; two calls per arm required for actual update',
        finite_offline_generation_limit=147,
        first512_main_PBS=dict(count=1, serial_independent_raw_arms=True,
            requested_walltime_upper_s=59400, accounting_GPU_upper_s=59520,
            GPUs=1, CPU_slots=8),
        engineering_replacement=dict(maximum_count=1, unknown_write_replay=False,
            requires_actual_closed_zero_model_query_update_constructor_physics=True,
            cannot_replace_valid_behavior_failure=True),
        inference_package_upper_per_arm=INFERENCE_BYTES,
        checkpoint_retention_upper_per_arm=70000000000,
        qualification_and_first512_states_both_retained=True,
        memory_speed_and_save_size_measured=False,
        preparation_readonly_CPU_uncertainty=dict(
            observed_raw_hash_process_gone_without_final_digest=True,
            provisional_site_CPU_upper_s=900,
            included_in_finite_diagnostic_CPU_not_an_extra_credit=True,
            actual_CPU_unknown_not_zero=True, settled=False),
        existing_C_D_GEN_Evals_and_old_costs_retained=True,
        matrix3_and_final20GPUh40GiB_retained=True,
        conditional1024_not_reserved_or_authorized=True,
        Core_recovery_and_original_native_qualification_still_required=True,
        T_CD=baseline['T_CD'], deadline=baseline['deadline'],
        stop_new_training=baseline['stop_new_training'])
