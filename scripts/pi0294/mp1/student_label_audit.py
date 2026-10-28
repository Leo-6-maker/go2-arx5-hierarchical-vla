"""Join saved student actions to actual ACK-certified same-state Oracle labels."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read
from scripts.pi0292.eval_contract import error_metrics, summarize_records
from scripts.pi0292.relay_protocol import digest, unseal
from scripts.pi0294.native_dataset import sha
from .probe_compare import PRECISION


def qualified_label(root, state, pool, history):
    """No replay or model call; original initial-restoration exclusions persist."""
    root = Path(root)
    row = pool[state['id']]
    for key in ['original_episode_uuid', 'sequence', 'tick', 'task', 'observation_sha256']:
        if row[key] != state[key]:
            raise ValueError('SAME_SAVED_STUDENT_ANCHOR_IDENTITY')
    if row['split'] != 'TRAIN' or row['physically_safety_valid'] is not True:
        raise PermissionError('TRAIN_ONLY_PHYSICALLY_VALID_SAVED_STATE')
    packet = read(root/'evidence/TRAIN_STATES'/row['snapshot_relative']/'observation.json')
    actual = unseal(packet)
    if packet['sha256'] != state['observation_sha256'] or actual['payload'] != state['payload']:
        raise ValueError('ACTUAL_CORE_STUDENT_OBSERVATION_CHANGED')
    if state['sequence'] == 0:
        return None, dict(status='EXCLUDED_INVALID_INITIAL_STATIC_IK_HISTORY',
                          labels_used=0, new_teacher_labels=0)
    matches = [x for x in history['records'] if x['attempt_id'] == row['original_attempt_id']
               and x['sequence'] == state['sequence']]
    if (history['status'] != 'PASS_ALL_USED_NONINITIAL_CORRECTION_STATES_EXACT_STATIC_HISTORY'
            or len(matches) != 1 or matches[0]['full_state_identical'] is not True
            or matches[0]['tick'] != state['tick']):
        raise ValueError('EXACT_NONINITIAL_DYNAMICS_AND_CONTROL_HISTORY_REQUIRED')
    folder = root/'forks'/(state['id']+'-SHORT')
    result = read(folder/'RESULT.json')
    independent = read(root/'evidence/fork-audits'/folder.name/'AUDIT.private.json')
    if (independent['status'] != 'PASS_INDEPENDENT_SAVED_FORK_AUDIT'
            or independent['restoration'] != 'EXACT_NONINITIAL_HISTORY'
            or independent['task_qualification_evidence_eligible'] is not True
            or not all(independent['checks'].values()) or independent['result'] != result
            or result['source_student_episode'] != state['original_episode_uuid']
            or result['anchor_tick'] != state['tick'] or result['anchor_sequence'] != state['sequence']
            or result['status'] != 'SAFE_TWO_SECOND_CONTINUATION'
            or result['live_native_failures'] or not result['labels_ACK_certified']
            or result['actual_expert_commands'] != 40 or result['actual_integrations'] != 1000
            or not result['engine_history_copied'] or result['maximum_live_shadow_difference'] > 1e-10):
        raise ValueError('ACTUAL_SHORT_ORACLE_RESULT_AND_INDEPENDENT_AUDIT')
    label = folder/'expert_labels.npz'
    if label.is_symlink() or sha(label) != result['labels_sha256']:
        raise ValueError('ACTUAL_SAVED_EXPERT_LABEL_HASH')
    with np.load(label, allow_pickle=False) as z:
        target, actual_state = z['committed_absolute10'].copy(), z['actual_state10'][0].copy()
        valid, paths, hashes = z['valid_window'].copy(), z['rgb_paths'][0].copy(), z['rgb_sha256'][0].copy()
        representation = z['label_representation'].item()
    if (target.shape != (40,10) or not np.isfinite(target).all() or valid.sum() != 1
            or not valid[0] or representation != 'ABSOLUTE_COMMITTED_EEF10'
            or np.max(np.abs(actual_state-np.asarray(state['payload']['state_eef10']))) > 1e-10):
        raise ValueError('ACTUAL_ORACLE_STATE_OR_COMPLETE_ACKED_H40')
    commands, tick = [], state['tick']
    for i, ack in enumerate(read(folder/'PREFIX_ACKS.json')):
        p = folder/f'prefix-{i:03d}.npz'
        if (sha(p) != ack['physical_receipt_sha256'] or ack['tick_before'] != tick
                or ack['tick_after'] != tick+200 or ack['integrations'] != 200 or ack['commands'] != 8):
            raise ValueError('SAVED_ORACLE_ACK_HASH_AND_CONTINUOUS_CLOCK')
        with np.load(p, allow_pickle=False) as z:
            if int(z['tick_before']) != tick or int(z['tick_after']) != tick+200:
                raise ValueError('SAVED_ORACLE_NATIVE_PREFIX_CLOCK')
            commands.extend(z['expert_commands'].copy())
        tick += 200
    if tick != state['tick']+1000 or not np.array_equal(np.asarray(commands), target):
        raise ValueError('SAVED_ORACLE_LABELS_EQUAL_ACTUAL_ACK_COMMANDS')
    for name, checksum, image in zip(paths, hashes, state['payload']['images']):
        p = folder/str(name)
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha(p) != str(checksum):
            raise ValueError('SAVED_ORACLE_FIRST_RGB_BYTES')
        with Image.open(p) as a, Image.open(io.BytesIO(base64.b64decode(image['png_base64']))) as b:
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                raise ValueError('SAME_STUDENT_STATE_ORACLE_CAMERA_PIXELS')
    return target, dict(status='QUALIFIED_SAVED_SAME_STATE_ACKED_H40_DIAGNOSTIC_ONLY',
        label_sha256=result['labels_sha256'], source_parent=state['original_episode_uuid'],
        source_sequence=state['sequence'], actual_ACKs=5, actual_commands=40,
        full_recovery_proven_here=False, new_teacher_labels=0,
        T2_full_recovery_or_training_qualification_claimed=False)


def join(root, folder, plan, expected_checkpoint_files_fingerprint):
    root, folder = Path(root), Path(folder)
    saved = read(folder/'ON_STUDENT_RESULTS.private.json')
    states = plan['states']
    records = {r['id']:r for r in saved['records']}
    ids = [s['id'] for s in states]
    if len(ids) != 12 or len(set(ids)) != 12 or len(saved['records']) != 12 or set(records) != set(ids):
        raise ValueError('TWELVE_STATES_EXACT_LEFT_JOIN_NO_ROW_LOSS')
    pool = {r['id']:r for r in read(root/'plans/TRAIN_STATE_POOL.json')['states']}
    if set(pool) != set(ids):
        raise ValueError('EXACT_FIXED_CORE_STUDENT_POOL')
    history = read(root/'evidence/INITIAL_STATIC_HISTORY_AUDIT.json')
    output, comparable = [], []
    for state in states:
        r = records[state['id']]
        p = folder/(state['id']+'.npz')
        if p.is_symlink() or sha(p) != r['arrays_sha256']:
            raise ValueError('RECEIVED_S_BC_STUDENT_ARRAY_HASH')
        with np.load(p, allow_pickle=False) as z:
            action = z['physical_H40'].copy()
            signature = z['observation_fingerprint'].item()
            weights = z['checkpoint_files_fingerprint'].item()
            noise, precision = [json.loads(z[k].item()) for k in ['noise_fingerprint_json','inference_precision_json']]
            seed = int(z['noise_seed'].item())
        if (action.shape != (40,10) or not np.isfinite(action).all()
                or action.tolist() != r['physical_H40'] or seed != state['noise_seed']
                or noise != r['noise'] or precision != r['inference_precision']
                or any(precision.get(k) != v for k,v in PRECISION.items())
                or weights != expected_checkpoint_files_fingerprint
                or signature != digest({k:v for k,v in state['payload'].items() if k!='snapshot_uuid'})):
            raise ValueError('RECEIVED_S_BC_SAVED_INPUT_NOISE_AND_WEIGHT_IDENTITY')
        target, qualification = qualified_label(root,state,pool,history)
        row = dict(id=state['id'], task=state['task'], student_parent=state['original_episode_uuid'],
            sequence=state['sequence'], source_tick=state['tick'], label_qualification=qualification,
            original_S1_for_this_S_BC_probe='NOT_RUN', geometric_collision_clearance='NOT_MEASURED',
            actual_live_execution_of_this_probe=False)
        if target is not None:
            errors = error_metrics(action,target,np.asarray(state['payload']['state_eef10']))
            row['diagnostic_error_to_actual_ACKed_expert'] = errors
            row['signed_XYZ_first8_mean_m'] = np.mean(action[:8,:3]-target[:8,:3],axis=0).tolist()
            comparable.append(dict(id=state['original_episode_uuid'], split='TRAIN',
                errors=errors,gate=r['gate'], task=state['task']))
        output.append(row)
    return dict(status='PASS_SAVED_STUDENT_ACTIONS_TO_QUALIFIED_ACTUAL_EXPERT_H40_JOIN',
        fixed_states=12, rows=output, qualified_error_states=len(comparable),
        excluded_initial_states=sum(r['sequence']==0 for r in output),
        independent_historical_student_parents=len({r['student_parent'] for r in output}),
        task_summaries={task:summarize_records([r for r in comparable if r['task']==task])['TRAIN'] for task in ['T1','T2']},
        student_state_population='HISTORICAL_E0_NOT_CURRENT_S_BC_INDUCED',
        error_definition='ACTION_ERROR_NOT_RMSE_TO_ACHIEVED_EEF_OR_UNEXECUTED_OFFLINE_TEACHER',
        qualified_H40_is_complete_32s_recovery_success=False,
        new_queries=0,new_teacher_labels=0,new_physics=0)
