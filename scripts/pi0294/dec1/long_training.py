"""Opt-in same-allocation 1024 probes and full-state training continuation."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from .source_admission import VERSION

POLICY = 'plans/VAP1_LONG_ALLOCATION_SCHEDULING.json'
POLICY_SHA = '25fa1dcf70db3974bc7a0462293a1a0360777f6f47a1ca1c5ab2c304e736c3f6'


def admitted(root, manifest, request):
    block = request.get('training_block', {})
    if 'same_allocation_1024_probes' not in block:
        return False
    root = Path(root); p = root/POLICY
    if (block['same_allocation_1024_probes'] is not True
            or request.get('scientific_admission_version') != VERSION
            or request.get('stage') != 'P2_PRIMARY'
            or block.get('arm') not in ('A_CONTROL', 'B_COVERAGE')
            or block.get('replica') is not False or block.get('probe_only', False)
            or block.get('stop_after_step') != 4096
            or p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
            or p.stat().st_mode & 0o222
            or manifest['execution']['sealed_inputs'].get(POLICY) != POLICY_SHA
            or request.get('sealed_inputs', {}).get(POLICY) != POLICY_SHA
            or sha(p) != POLICY_SHA):
        raise PermissionError('VAP1_LONG_TRAIN_EXACT_SEALED_COLD_POLICY_AND_ORIGINAL_TWO_ARMS')
    return True


def probe_receipt(root, arm):
    root = Path(root); checkpoint = root/'training'/arm/'checkpoint-1024'
    complete = read(checkpoint/'CHECKPOINT_COMPLETE.json')
    plan = read(root/'plans/PV1_FIXED_TRAIN_PROBES.json')
    result = read(root/'evidence'/(arm+'-1024')/'TRAIN_PROBES/RESULT.json')
    identifiers = [r['id'] for r in plan['windows']]
    if (complete.get('global_step') != 1024 or complete.get('full_resume') is not True
            or len(identifiers) != 12 or len(set(identifiers)) != 12
            or [r.get('id') for r in result.get('records', [])] != identifiers
            or result.get('status') != 'ACTUAL_FIXED_TRAIN_PROBES_NOT_TASK_SUCCESS'
            or result.get('complete') is not True or len(result.get('records', [])) != 12
            or result.get('new_physics') != 0 or result.get('physical_success_gate') is not False
            or result.get('checkpoint_manifest_sha256') != sha(checkpoint/'CHECKPOINT_COMPLETE.json')
            or result.get('probe_plan_sha256') != sha(root/'plans/PV1_FIXED_TRAIN_PROBES.json')):
        raise PermissionError('VAP1_LONG_TRAIN_ACTUAL_FIXED_PROBES_BEFORE_UPDATE_1025')
    return result


def phase(root, manifest, request, *, name, stop, resume_relative):
    if not admitted(root, manifest, request):
        raise PermissionError('VAP1_LONG_TRAIN_NOT_A_LEGACY_PHASE_OVERRIDE')
    block = request['training_block']
    if name == 'BEFORE_PROBES':
        valid = stop == 1024 and resume_relative == block.get('resume_checkpoint_relative')
    elif name == 'AFTER_PROBES':
        valid = stop == 4096 and resume_relative == f"training/{block['arm']}/checkpoint-1024"
        if valid: probe_receipt(root, block['arm'])
    else:
        valid = False
    if not valid:
        raise PermissionError('VAP1_LONG_TRAIN_ONLY_PREDECLARED_1024_AND_4096_PHASES')


def execute(*, start_step, train, probe, unload, load, guard, checkpoint):
    """Keep native model calls injectable so ordering is testable without CUDA."""
    if type(start_step) is not int or not 0 <= start_step < 1024:
        raise PermissionError('VAP1_LONG_TRAIN_FIRST_PHASE_START_BEFORE_FIXED_PROBES')
    first = train(stop=1024, phase='BEFORE_PROBES', checkpoint=None)
    records = [first]
    if first['global_step'] != 1024:
        return first, records
    unload(); guard(); load(checkpoint, trainable=False)
    records.append(probe())
    unload(); guard(); load(checkpoint, trainable=True)
    final = train(stop=4096, phase='AFTER_PROBES', checkpoint=checkpoint)
    records.append(final)
    return final, records
