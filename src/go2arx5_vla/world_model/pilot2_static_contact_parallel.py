"""Reuse the two-slot owner for already executed static contact programs."""
from pathlib import Path
import json
import numpy as np
from .storage import digest

SCHEMA = 'WM02_PILOT2_TWO_STATIC_CONTACT_BATCH_V1'
COMMISSION = 'SUPPORTED_STATIC_DUAL_COMMISSION'
PRODUCTION = 'SUPPORTED_STATIC_PRODUCTION'


def validate_single(spec, episode):
    from .pilot2_supported import contract, stage_for
    from .pilot2_contact_model import planned_openings
    contract(episode)
    if episode['purpose'] != 'SUPPORTED_CONTACT_COLLECTION':
        raise PermissionError('STATIC_CONTACT_COLLECTION_ONLY')
    proof = spec['single_static_qualification']; path = Path(proof['path'])
    root = path.parent.parent
    paths = dict(native=path, episode=root/'EPISODE_MANIFEST.private.json',
        score=path.parent/'SUPPORTED_CLOSE_RESULT.private.json', spec=root/'EXECUTION_SPEC.json')
    if any(digest(p) != proof[k+'_sha256'] for k, p in paths.items()):
        raise PermissionError('FROZEN_SINGLE_STATIC_CONTACT_EVIDENCE_CHANGED')
    native, old, score, prior = [json.loads(paths[k].read_text()) for k in ('native', 'episode', 'score', 'spec')]
    lease = json.loads((root/'control/LEASE.json').read_text())
    contract(old)
    same = ('schema', 'purpose', 'parent', 'split', 'scene_id', 'task_goal', 'command_program', 'close_start_atom')
    if (lease['status'] != 'CLOSED' or lease.get('worker_exit_code') != 0
            or lease.get('spec_sha256') != proof['spec_sha256']
            or any(old.get(k) != episode.get(k) for k in same)
            or prior.get('parallel_batch_spec')
            or prior.get('contact_forecast_observer') != spec.get('contact_forecast_observer')
            or native['status'] != 'NATIVE_STAGE_PASSED' or not native['shutdown_clean']
            or native['physics_stops'] or native['actual_target_integrations'] != 6703
            or native.get('native_outer_guard_rejection_count', 0)
            or native['source_sha'] != prior['source_sha']
            or native['native_binary_sha256'] != digest(spec['native_binary'])
            or native['scene_sha256'] != digest(spec['model_path'])
            or native['observer_sha256'] != digest(spec['observer_library'])
            or score.get('first_blocker') or not score['same_native_process']
            or score['physics_resets'] or score['state_teleports'] or score['actual_task_atoms'] != 168):
        raise PermissionError('ACTUAL_MATCHING_SINGLE_STATIC_CONTACT_REQUIRED')
    rows = score['samples']
    if (len(rows) != 168
            or any((v['native_start_step'], v['native_end_step']) != (2503+25*i, 2528+25*i)
                or v['stage'] != stage_for(i, episode) for i, v in enumerate(rows))
            or not np.allclose([v['executed_reference_EEF10'][-1] for v in rows],
                planned_openings(episode), rtol=0, atol=1e-7)):
        raise PermissionError('WHOLE_STATIC_PROGRAM_MUST_HAVE_ACTUALLY_EXECUTED')
    # Native safety and actual command execution qualify the condition. A task
    # failure is retained rather than excluded to select successful samples.


def validate_batch(batch, children, proof, value):
    from .pilot2_supported import GOALS
    prior_root = Path(proof).parent.parent
    prior = json.loads((prior_root/'EXECUTION_SPEC.json').read_text())
    lease = json.loads((prior_root/'control/LEASE.json').read_text())
    if (lease['status'] != 'CLOSED' or lease.get('worker_exit_code') != 0
            or lease.get('spec_sha256') != digest(prior_root/'EXECUTION_SPEC.json')
            or any(prior.get(k) != batch.get(k) for k in
                ('native_binary', 'observer_library', 'pilot1_binding', 'pilot1_binding_sha256', 'pilot2_plan_sha'))):
        raise PermissionError('ACTUAL_CLOSED_SAME_EXECUTOR_STATIC_PARALLEL_PREDECESSOR_REQUIRED')
    purpose = batch.get('purpose')
    if purpose == COMMISSION:
        episodes = [json.loads(Path(c['episode_manifest']).read_text()) for c in children]
        if (prior.get('schema') != 'WM02_PILOT2_TWO_COVERAGE_BATCH_V1'
                or prior.get('purpose') != 'COVERAGE_DUAL_DURATION_COMMISSION'
                or any(e['split'] != 'TRAIN' or e['close_start_atom'] != 56 for e in episodes)
                or len({e['parent'] for e in episodes}) != 1
                or {e['command_program'] for e in episodes} != {'CLOSE_KEEP', 'CLOSE_RELEASE'}
                or {e['task_goal'] for e in episodes} != set(GOALS)):
            raise PermissionError('STATIC_COMMISSION_REQUIRES_DURATION_PROOF_AND_BOTH_SINGLE_TRAIN56_PROGRAMS')
        ticks = 4103
    elif purpose == PRODUCTION and prior.get('schema') == SCHEMA and prior.get('purpose') in (COMMISSION, PRODUCTION):
        ticks = 6703
    else:
        raise PermissionError('FULL_STATIC_DUAL_QUALIFICATION_REQUIRED_BEFORE_PRODUCTION')
    for row in value['children']:
        if Path(row['run']).name != row['run']:
            raise PermissionError('PRIOR_STATIC_CHILD_IDENTITY_REQUIRED')
        native = json.loads((Path(prior['native_runs_root'])/row['run']/'native-01/RESULT.json').read_text())
        if (row['ticks'] != ticks or native['actual_target_integrations'] != ticks
                or native['status'] != 'NATIVE_STAGE_PASSED' or not native['shutdown_clean']
                or native['physics_stops'] or native['native_binary_sha256'] != digest(batch['native_binary'])
                or native['observer_sha256'] != digest(batch['observer_library'])):
            raise PermissionError('ACTUAL_SAME_EXECUTOR_DUAL_NATIVE_RESULTS_REQUIRED')
