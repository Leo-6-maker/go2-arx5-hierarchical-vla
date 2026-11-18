"""Summarize native Teacher evidence using sealed parent and capture manifests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path, digest=None):
    raw = Path(path).read_bytes()
    if digest and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'Unexpected evidence SHA for {Path(path).name}')
    return json.loads(raw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError('Finite recorded values required')
    return {key: float(value) for key, value in zip(('min', 'median', 'max'),
                                                   np.quantile(array, [0, .5, 1]))}


def anchor_rotation_error(matrix, accepted_pose):
    matrix = np.asarray(matrix, float)
    x, y, z, w = accepted_pose[3:]
    target = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])
    trace = np.trace(target.T @ matrix[:3, :3])
    return float(np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1))))


def captured_episode(case_path, entry):
    result_path = case_path.with_name('RESULT.private.json')
    result = read(result_path, entry['result_sha256'])
    capture_path = case_path.parent / 'native-01/LIVE_DATA/MANIFEST.private.json'
    capture = read(capture_path)
    if capture['final_native_step'] != result['actual_native_integrations']:
        raise ValueError('Capture did not close on the recorded terminal tick')
    labels = capture['labels']
    rows = read(labels['path'], labels['sha256'])
    if len(rows) != capture['queries'] or capture['planned_suffix_is_executed']:
        raise ValueError('Full desired H40 and consumed prefix required')

    anchor_xyz = []
    anchor_rotation = []
    previous_jaw_error = []
    accepted_observation_age = []
    planned_width_changes = {'P10': [], 'tail30': []}
    executed = 0
    for i, row in enumerate(rows):
        tick = row['native_step']
        ref = row['recent_reference']
        actual = row['actual_state']
        time = row['native_time_s']
        if (row['query'] != i or actual['native_step'] != tick or ref['native_step'] != tick
                or abs(ref['receiver_knot_time_s'] - time) > 1e-8 or not ref['exact_receiver_knot']
                or 'NO_SELECTED_POLICY_ID_EXPORTED' not in ref['controller_selected_reference']):
            raise ValueError('Accepted reference must align to the actual recorded tick')
        actual_tcp = np.asarray(actual['tcp'])
        pose = ref['accepted_pose7']
        anchor_xyz.append(1000 * np.linalg.norm(actual_tcp[:3, 3] - pose[:3]))
        anchor_rotation.append(anchor_rotation_error(actual_tcp, pose))
        previous = row['previous_confirmed_command']
        if abs(previous['query_native_s'] - time) > 1e-8:
            raise ValueError('Recorded previous hand command belongs to a different query')
        previous_jaw_error.append(1000 * (np.mean(actual['finger_q']) - previous['width_m']))
        accepted_observation_age.append(1000 * (time - ref['receiver_observation_time_s']))
        desired = np.asarray(row['nominal_world_H40'])
        submitted = np.asarray(row['installed_reference_H40'])
        mask = row['consumed_mask']
        if desired.shape != (40, 10) or submitted.shape != (40, 10) or len(mask) != 40 or any(mask[10:]):
            raise ValueError('Recorded H40 or P10 scope changed')
        delta = 44 * np.abs(np.diff(np.r_[previous['width_m']/.044, desired[:, 9]]))
        planned_width_changes['P10'].append(delta[:10].max())
        planned_width_changes['tail30'].append(delta[10:].max())
        executed += sum(mask)
    return dict(condition=entry['id'], native_ticks=entry['native_ticks'],
        success=entry['task_and_mobile_success'], hard_STOP=entry['hard_STOP'],
        technical_failure=entry['technical_failure'], post_STOP_integrations=entry['integrations_after_STOP'],
        result_sha256=entry['result_sha256'], capture_sha256=sha(capture_path), labels_sha256=labels['sha256'],
        queries=len(rows), recorded_P10_atoms=executed, terminal_partial_ticks=rows[-1]['partial_interval_ticks'],
        phase_windows=dict(Counter(row['phase'] for row in rows)),
        actual_TCP_vs_accepted_receiver_xyz_mm=summary(anchor_xyz),
        actual_TCP_vs_accepted_receiver_rotation_deg=summary(anchor_rotation),
        actual_fingers_minus_previous_command_mm=summary(previous_jaw_error),
        accepted_receiver_observation_age_ms=summary(accepted_observation_age),
        planned_50ms_gripper_change_mm={key: summary(values) for key, values in planned_width_changes.items()},
        selected_controller_reference='UNKNOWN')


def inspect(parents):
    rows = []
    parent_shas = []
    for parent in map(Path, parents):
        spec = read(parent)
        result_path = parent.with_name('RESULT.private.json')
        result = read(result_path)
        parent_shas.append(sha(result_path))
        cases = {Path(path).parent.name: Path(path) for path in spec['cases']}
        experts = [entry for entry in result['records'] if entry['role']=='EXPERT']
        if result['new_verified_native_requests'] != len(experts):
            raise ValueError('Parent native denominator inconsistent')
        rows.extend(captured_episode(cases[entry['id']], entry) for entry in experts)
    if len({row['condition'] for row in rows}) != len(rows):
        raise ValueError('Duplicate Teacher condition')
    return dict(episodes=rows, parent_result_shas=parent_shas,
                new_updates=0, new_model_generations=0, new_native_requests=0, new_integrations=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-spec', action='append', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.parent_spec), allow_nan=False))
