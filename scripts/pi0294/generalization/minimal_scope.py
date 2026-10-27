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
    Each one-step qualification is a real additional update, followed by a
    fresh raw start for formal training; its effects/costs are never erased.
    The original17 zero-step qualification draws per arm are also used for
    matched E0/E1/E2 equivalence. Each original12 end-TRAIN probes remains.
    The additional language24/model is separate from those qualifications.
    """
    rows = {}
    for arm in ('E1', 'E2'):
        train = stage_vector('TRAIN')
        train['optimizer_updates'] = 512 + 1
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
    parts = components()
    additional = _vector({key: sum(row[key] for row in parts.values())
                          for key in baseline['dimensions']})
    dimensions, changes = {}, {}
    for key, old in baseline['dimensions'].items():
        projected = old['projected'] + additional[key]
        cap = old['cap']
        shortfall = max(0, projected - cap)
        proposed = cap
        if shortfall:
            quantum = GIB if key.endswith('bytes') else 3600 if key.endswith('_s') else 1
            proposed = math.ceil(projected / quantum) * quantum
            changes[key] = dict(before=cap, after=proposed,
                                before_display=cap/quantum, after_display=proposed/quantum,
                                unit='GiB' if key.endswith('bytes') else 'hours' if key.endswith('_s') else 'count')
        dimensions[key] = dict(old, baseline_projected=old['projected'],
            first512_additional=additional[key], projected=projected,
            shortfall=shortfall, proposed_cap=proposed,
            headroom_after_proposed_cap=proposed-projected)
    return dict(schema='ARX5_MINIMAL_FIRST512_COMPLETE_FUTURE_FORECAST_V1', scope=SCOPE,
        status='PROPOSED_NOT_ADOPTED_NO_NATIVE_ADMISSION',
        GLOBAL_sha256=hashlib.sha256(global_bytes).hexdigest(),
        historical_LOCAL_sha256=hashlib.sha256(historical_LOCAL_bytes).hexdigest(),
        LOCAL_is_historical=True, native_admission=False, scope_adopted=False,
        baseline=baseline, components=parts, additional=additional,
        dimensions=dimensions, cap_changes_required=changes,
        formal_optimizer_updates=2*512, qualification_optimizer_updates=2,
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
