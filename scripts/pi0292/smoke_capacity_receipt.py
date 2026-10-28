"""Derive a bound capacity plan from actual terminal S3 and full checkpoints."""
from pathlib import Path

from .checkpoint_artifacts import inventory
from .durable import read
from .main_capacity import calculate
from .train_adapter import sha


def build(root, *, owned_bytes, filesystem_free_bytes, quota_free_bytes,
          extra_peak_bytes, now, prior_ledger=None):
    root = Path(root)
    adoption = read(root/'ADOPTION.json')
    if (adoption['T0_epoch'] != 1791860304 or adoption['deadline_epoch'] != 1791946704
            or adoption['new_work_cutoff_epoch'] != 1791943104 or adoption['reset_on_resume']):
        raise PermissionError('ORIGINAL_CLOCK_REQUIRED')
    gate_path = root/'evidence/MODEL_SMOKE.json'
    wrapper_path = root/'evidence/PBS_SMOKE_S3-smoke-02.json'
    data_path = root/'evidence/HOPPER_DATA_QUALIFICATION.json'
    ledger_path = root/'HOPPER_RESOURCE_LEDGER.json'
    gate, wrapper, data, ledger = map(read, (gate_path, wrapper_path, data_path, ledger_path))
    if prior_ledger is not None:
        import copy
        expected = copy.deepcopy(ledger)
        if (len(expected['PBS_submissions']) != 3
                or expected['PBS_submissions'][-1]['stage'] != 'main-eval'
                or expected['PBS_submissions'][-1]['status'] != 'SUBMITTED'):
            raise PermissionError('EXACT_JUST_STARTED_MAIN_ALLOCATION_VIEW_REQUIRED')
        expected['PBS_submissions'] = expected['PBS_submissions'][:2]
        if prior_ledger != expected:
            raise PermissionError('PRIOR_CAPACITY_VIEW_MUST_NOT_HIDE_ANY_USED_RESOURCES')
    if (data['status'] != 'PASS_BANK_V2_FULL_DATA_AND_CPU_INTERFACE'
            or read(root/'evidence/OPEN_LOOP_PROBE_SEAL.json')['status'] != 'SEALED_BEFORE_MAIN'):
        raise PermissionError('QUALIFIED_FIXED_DATA_AND_PREMAIN_PROBE_REQUIRED')
    paths = [root/'training/smoke_continuous/session-start/TRAINING_RUN.json',
             root/'training/smoke_split/session-start/TRAINING_RUN.json',
             root/'training/smoke_split/session-resume/TRAINING_RUN.json']
    runs = [read(path) for path in paths]
    equivalent_path = root/'evidence/RESUME_EQUIVALENCE.json'
    equivalent = read(equivalent_path)
    if (equivalent['status'] != 'PASS_FIXED_RESUME_EQUIVALENCE'
            or sha(equivalent_path) != gate['equivalence_sha256']):
        raise PermissionError('ACTUAL_HASH_BOUND_RESUME_EQUIVALENCE_REQUIRED')
    checkpoints = []
    for relative, step in [('smoke_continuous/checkpoint-4', 4),
                           ('smoke_split/checkpoint-2', 2), ('smoke_split/checkpoint-4', 4)]:
        checkpoints.append(inventory(root/'training'/relative, expected_step=step,
            recipe=runs[0]['recipe'], statistics_sha256=data['statistics_sha256'],
            manifest_sha256=data['manifest_sha256']))
    step_path = root/'training/smoke_continuous/session-start/OPTIMIZER_STEPS.jsonl'
    import json
    steps = [json.loads(line) for line in step_path.read_text().splitlines()]
    full_bytes = max(row['total_bytes'] for row in checkpoints)
    plan = calculate(smoke_gate=gate, smoke_runs=runs, step_records=steps, wrapper=wrapper,
        ledger=ledger if prior_ledger is None else prior_ledger, checkpoint_bytes=full_bytes, owned_bytes=owned_bytes,
        filesystem_free_bytes=filesystem_free_bytes, quota_free_lower_bound_bytes=quota_free_bytes,
        extra_peak_bytes=extra_peak_bytes, now=now, new_work_cutoff=adoption['new_work_cutoff_epoch'])
    plan.update(actual_checkpoint_inventories=checkpoints,
        original_T0=adoption['T0_epoch'], original_deadline=adoption['deadline_epoch'], budget_reset=False,
        source_receipt_hashes={str(path.relative_to(root)):sha(path) for path in
            [gate_path, wrapper_path, data_path, ledger_path, equivalent_path, step_path, *paths]},
        pre_main_probe_sha256=sha(root/'evidence/OPEN_LOOP_PROBE_SEAL.json'),
        receipt_binding='Actual successful smoke, completed full checkpoints and reconciled scheduler ledger only',
        selected='NONE', deployment='NONE')
    return plan
