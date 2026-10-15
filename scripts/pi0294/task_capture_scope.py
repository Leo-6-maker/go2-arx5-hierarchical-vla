"""Task-data-first policy checks, with no dispatch or resource authorization.

The future native task worker must FIRST run the deployed entry/manifest/lease
validator and reserve its actual ledger ticket. This policy cannot replace it.
The legacy native hold launcher and all historical receipts remain unchanged.
"""
import math

SCOPE = 'GR00T_TASK_CAPTURE_FROZEN_WBC_V1'
RELEASE = '4a7187e3ac4a1f57a4e696815a1e0d3df74d15c8'
LEGACY_SCOPE = 'R4A_NATIVE_HOLD_V1'
HOLD_FAIL = 'VALID_NATIVE_HOLD_QUALITY_FAIL'
HARD_GATES = (
    'standard_entry_manifest_owner_output', 'frozen_executor_identity',
    'fullbody_free_root_active_legs', 'native_safety', 'reference_identity',
    'source_clock', 'finite_gripper', 'actual_body_wrist_rgb',
    'actual_observation_submitted_label_sync', 'stop_expiry_reset',
)
DEMO_GATES = ('integrity', 'native_safety', 'label_schema', 'split')
EXECUTOR_KEYS = ('controller', 'leg_policy', 'plant_overlay', 'reference_interface')


def historical_hold_classification(scope, verdict):
    if scope not in (SCOPE, LEGACY_SCOPE):
        raise ValueError('UNKNOWN_CAPTURE_SCOPE')
    if verdict != HOLD_FAIL:
        raise ValueError('EXACT_HISTORICAL_HOLD_VERDICT_REQUIRED')
    return ('DIAGNOSTIC_ONLY_HISTORICAL_WBC_HOLD' if scope == SCOPE else verdict)


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def assess_capture(*, scope, stage, checks, resource, teacher_executor,
                   model_executor, historical_hold_verdict, now):
    """Assess supplied receipts; does not inspect files or grant launch rights.

    Missing/UNKNOWN/FAIL checks block. A pilot does not require a prior successful
    pilot; production does. Runtime must revalidate and reserve atomically before
    starting any child, including when this preparation assessment says ready.
    """
    history = historical_hold_classification(scope, historical_hold_verdict)
    blocked = []
    if scope != SCOPE:
        blocked.append('LEGACY_HOLD_FAIL_PRESERVED')
    if stage not in ('pilot', 'production'):
        blocked.append('UNKNOWN_CAPTURE_STAGE')
    for name in HARD_GATES:
        if checks.get(name) != 'PASS':
            blocked.append(name)
    if stage == 'production' and checks.get('task_pilot') != 'PASS':
        blocked.append('task_pilot')
    if (not isinstance(teacher_executor, dict) or not isinstance(model_executor, dict)
            or set(teacher_executor) != set(EXECUTOR_KEYS)
            or set(model_executor) != set(EXECUTOR_KEYS)
            or not all(_digest(teacher_executor[k]) for k in EXECUTOR_KEYS)
            or teacher_executor != model_executor):
        blocked.append('TEACHER_MODEL_FROZEN_EXECUTOR_MISMATCH_OR_UNBOUND')
    start, cutoff, end = (resource.get(k) for k in ('start', 'new_work_cutoff', 'deadline'))
    valid_clock = (all(_finite_number(v) for v in (start, cutoff, end, now))
                   and start < cutoff <= end and start <= now < cutoff)
    if (resource.get('authorizes_execution') is not True
            or resource.get('lease_status') != 'ACTIVE'
            or resource.get('budget_reserved') is not True
            or not valid_clock):
        blocked.append('VALID_RESOURCE_AUTHORITY_LEASE_CLOCK_AND_RESERVATION_REQUIRED')
    return dict(scope=scope, history=history, blocked=blocked,
                readiness='BLOCKED' if blocked else 'POLICY_READY_REQUIRES_RUNTIME_VALIDATION',
                dispatch_performed=False, resource_authorization_granted=False)


def admit_success_demo(*, checks, task_success):
    """Tracking RMS is diagnostic; unsuccessful/unsafe attempts remain failures."""
    blocked = [name for name in DEMO_GATES if checks.get(name) != 'PASS']
    if task_success is not True:
        blocked.append('ACTUAL_TASK_NOT_SUCCESSFUL')
    return dict(admit_success_demo=not blocked, blocked=blocked,
                tracking_metrics_role='DIAGNOSTIC_ONLY', failure_record_required=True)
