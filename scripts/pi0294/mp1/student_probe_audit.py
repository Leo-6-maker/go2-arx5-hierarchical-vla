"""Verify existing historical student-state draws without creating labels."""
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.eval_contract import gate_action, matrix10
from scripts.pi0292.relay_protocol import digest, unseal, validate_observation
from scripts.pi0294.native_dataset import sha
from .probe_compare import PRECISION


def audit(folder, plan, checkpoint_fingerprint, originals):
    folder = Path(folder)
    path = folder / 'ON_STUDENT_RESULTS.private.json'
    if folder.is_symlink() or path.is_symlink():
        raise PermissionError('NO_STUDENT_PROBE_SYMLINK')
    raw = path.read_bytes()
    results = json.loads(raw)
    states = plan['states']
    by_id = {r['id']: r for r in results['records']}
    ids = [r['id'] for r in states]
    if (len(states) != 12 or len(set(ids)) != 12 or len(by_id) != 12
            or len(results['records']) != 12 or set(ids) != set(by_id)
            or set(originals) != set(ids)
            or results['status'] != 'FIXED_12_TRAIN_OBSERVATIONS_ONLY_NOT_LIVE_OR_TEACHER_LABELS'):
        raise ValueError('EXACT_TWELVE_EXISTING_STUDENT_PROBES')
    parents, task_count, rows = set(), dict(T1=0, T2=0), []
    for state in states:
        identifier = state['id']
        if (Path(identifier).name != identifier or identifier in ['.', '..']
                or '-TRAIN-' not in identifier or state['task'] not in task_count
                or type(state['sequence']) is not int or state['sequence'] < 0
                or state['tick'] != 200 * state['sequence']):
            raise PermissionError('FIXED_TRAIN_STUDENT_IDENTITY_AND_CLOCK')
        validate_observation(state['payload'])
        original = originals[identifier]
        request, action = unseal(original['request']), unseal(original['action'])
        receipt = original['receipt']
        if (request['episode_uuid'] != state['original_episode_uuid']
                or request['sequence'] != state['sequence'] or request['tick'] != state['tick']
                or request['payload'] != state['payload']
                or request['observation_sha256'] != state['observation_sha256']
                or request['anchor_sha256'] != digest(state['payload']['state_eef10'])
                or original['action']['sha256'] != state['original_action_sha256']
                or action['payload']['trajectory_eef10'] != state['M0_physical_H40']
                or action['payload']['query_uuid'] != request['query_uuid']
                or action['payload']['observation_sha256'] != request['observation_sha256']
                or receipt['seed'] != state['noise_seed']
                or receipt['action_sha256'] != original['action']['sha256']):
            raise ValueError('ORIGINAL_E0_ACTUAL_INPUT_ACTION_AND_SEED_BINDING')
        observation = dict(protocol=request['protocol'], kind='observation',
            run_uuid=request['run_uuid'], episode_uuid=request['episode_uuid'],
            sequence=request['sequence'], tick=request['tick'], payload=request['payload'])
        if digest(observation) != state['observation_sha256']:
            raise ValueError('ORIGINAL_E0_OBSERVATION_HASH')
        record = by_id[identifier]
        arrays = folder / (identifier + '.npz')
        if (arrays.is_symlink() or not arrays.resolve().is_relative_to(folder.resolve())
                or sha(arrays) != record['arrays_sha256']):
            raise ValueError('STUDENT_PROBE_ARRAY_HASH_OR_ESCAPE')
        with np.load(arrays, allow_pickle=False) as z:
            physical = z['physical_H40'].copy()
            signature = z['observation_fingerprint'].item()
            weight = z['checkpoint_files_fingerprint'].item()
            noise = json.loads(z['noise_fingerprint_json'].item())
            precision = json.loads(z['inference_precision_json'].item())
            seed = int(z['noise_seed'].item())
        expected_signature = digest({k:v for k,v in state['payload'].items() if k != 'snapshot_uuid'})
        if (record['new_query'] is not True or physical.shape != (40, 10)
                or not np.isfinite(physical).all() or physical.tolist() != record['physical_H40']
                or weight != checkpoint_fingerprint or signature != expected_signature
                or seed != state['noise_seed'] or record['noise_seed'] != seed
                or noise != record['noise'] or noise != receipt['noise']
                or precision != record['inference_precision']
                or any(precision.get(k) != v for k,v in PRECISION.items())):
            raise ValueError('STUDENT_PROBE_INPUT_NOISE_PRECISION_OR_CHECKPOINT_CHANGED')
        anchor = np.asarray(state['payload']['state_eef10'])
        gate = gate_action(physical, anchor, state['tick']*.002, identifier)
        if gate != record['gate']:
            raise ValueError('STUDENT_TRAJECTORY_GATE_RECOMPUTATION')
        poses = matrix10(physical)
        parents.add(state['original_episode_uuid'])
        task_count[state['task']] += 1
        rows.append(dict(id=identifier, task=state['task'], student_parent=state['original_episode_uuid'],
            source_sequence=state['sequence'], source_tick=state['tick'], trajectory_gate=gate['decision'],
            XYZ_first8_displacement_from_actual_anchor_m=(physical[7,:3]-anchor[:3]).tolist(),
            first_position_jump_m=float(np.linalg.norm(physical[0,:3]-anchor[:3])),
            minimum_predicted_TCP_Z_first8_m=float(np.min(poses[:8,2,3])),
            minimum_predicted_TCP_Z_is_collision_clearance=False,
            gripper_first8_minimum=float(np.min(physical[:8,9])),
            gripper_first8_maximum=float(np.max(physical[:8,9])),
            arrays_sha256=record['arrays_sha256'],
            expert_error='NOT_COMPUTED_WITHOUT_SAME_STATE_QUALIFIED_LABEL',
            current_model_original_S1='NOT_RUN_FOR_THIS_SAVED_STUDENT_PROBE',
            initial_anchor_old_label_exclusion_preserved=state['sequence']==0))
        if sha(arrays) != record['arrays_sha256']:
            raise ValueError('STUDENT_ARRAY_CHANGED_DURING_AUDIT')
    if len(parents) != 4 or task_count != dict(T1=6, T2=6) or path.read_bytes() != raw:
        raise ValueError('FIXED_FOUR_PARENTS_TWO_TASKS_OR_UNCHANGED_RESULT')
    return dict(status='PASS_TWELVE_SAVED_STUDENT_DRAWS_ORIGINAL_E0_INPUTS_NOISE_AND_CHECKPOINT',
        result_sha256=__import__('hashlib').sha256(raw).hexdigest(), rows=rows,
        probe_states=12, independent_historical_student_parents=4, task_probe_states=task_count,
        historical_E0_not_current_S_BC_induced_states=True,
        new_model_queries=0, new_physics=0, new_teacher_labels=0,
        same_state_RMSE_or_current_model_S1_imputed_from_old_models=False)
