"""Audit sealed student queries; never run a policy or advance a native world.

Optional joint diagnostics call the ORIGINAL certifier on cloned MuJoCo data.
Exception-frame locals supply rejected joint trajectories without changing the
certifier, its limits, or historical outcomes. This is not WBC state recovery.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0291_arx5_data.native_dataset import pose10
from go2arx5_vla.trajectory.so3 import (
    matrix_to_quat, rotation_matrix_to_rot6d_rows, rot6d_rows_to_matrix,
)
from go2arx5_vla.world_model import fusion_trajectory_adapter as envelope


LIMITS = {
    'maximum_translation_speed_m_s': envelope.MAX_TRANSLATION_SPEED_M_S,
    'maximum_angular_speed_rad_s': envelope.MAX_ANGULAR_SPEED_RAD_S,
    'maximum_finger_speed_m_s': envelope.MAX_FINGER_SPEED_M_S,
    'maximum_discrete_translation_acceleration_m_s2': envelope.MAX_TRANSLATION_ACCELERATION_M_S2,
    'maximum_discrete_angular_acceleration_rad_s2': envelope.MAX_ANGULAR_ACCELERATION_RAD_S2,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, pin=None):
    path = Path(path)
    raw = path.read_bytes()
    if pin is not None and hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError('SEALED_EVIDENCE_BYTES_CHANGED:' + str(path))
    return json.loads(raw)


def geometry_detail(rows, anchor):
    """Match FiftyMsProgram's existing derivatives, including zero end rates."""
    rows = np.asarray(rows, float)
    metrics = envelope._geometry(rows, np.asarray(anchor, float))
    velocity = np.diff(rows[:, :3], axis=0) / .05
    acceleration = np.diff(np.vstack((np.zeros(3), velocity, np.zeros(3))), axis=0) / .05
    rotations = rot6d_rows_to_matrix(rows[:, 3:9])
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    q = matrix_to_quat(relative)
    q = np.where(q[:, 3:] < 0, -q, q)
    sine = np.linalg.norm(q[:, :3], axis=1, keepdims=True)
    angular = q[:, :3] * (2 * np.arctan2(sine, q[:, 3:])) / np.where(sine > 0, sine, 1.) / .05
    angular_acceleration = np.diff(np.vstack((np.zeros(3), angular, np.zeros(3))), axis=0) / .05
    series = {
        'maximum_translation_speed_m_s': np.linalg.norm(velocity, axis=1),
        'maximum_angular_speed_rad_s': np.linalg.norm(angular, axis=1),
        'maximum_finger_speed_m_s': envelope.MECHANICAL_TRAVEL_M * np.abs(np.diff(rows[:, 9])) / .05,
        'maximum_discrete_translation_acceleration_m_s2': np.linalg.norm(acceleration, axis=1),
        'maximum_discrete_angular_acceleration_rad_s2': np.linalg.norm(angular_acceleration, axis=1),
    }
    failed = []
    for name, limit in LIMITS.items():
        # The aggregate from the original implementation decides rejection.
        if metrics[name] > limit * (1 + 1e-10):
            bad = np.flatnonzero(series[name] > limit * (1 + 1e-10))
            failed.append(dict(metric=name, maximum=metrics[name], limit=limit,
                               ratio=metrics[name] / limit,
                               first_difference_index=int(bad[0]) if len(bad) else None))
    return dict(metrics=metrics, failed=failed, passes=not failed,
                index_convention='ZERO_BASED; velocity index joins k,k+1; acceleration uses zero endpoint rates')


def joint_rates(command_q, prefix):
    command_q = np.asarray(command_q, float)
    velocity = np.diff(command_q, axis=0) / .05
    acceleration = np.diff(velocity, axis=0) / .05
    bad = np.argwhere(np.abs(acceleration) > 1. * (1 + 1e-10))
    peak = np.unravel_index(np.abs(acceleration).argmax(), acceleration.shape)
    first = None
    if len(bad):
        i, j = map(int, bad[0])
        knots = [i - prefix, i + 1 - prefix, i + 2 - prefix]
        first = dict(joint=j + 1, acceleration_rad_s2=float(acceleration[i, j]),
                     absolute_ratio=float(abs(acceleration[i, j])),
                     H40_knot_triple=knots,
                     location='REFERENCE_JOIN' if knots[0] < 0 else 'H40_INTERIOR')
    i, j = map(int, peak)
    return dict(command_q=command_q.tolist(), velocity_rad_s=velocity.tolist(),
                acceleration_rad_s2=acceleration.tolist(),
                maximum_speed_rad_s=float(np.max(np.abs(velocity))), speed_limit_rad_s=.7,
                maximum_acceleration_rad_s2=float(np.max(np.abs(acceleration))), acceleration_limit_rad_s2=1.,
                peak_joint=j + 1, peak_H40_knot_triple=[i-prefix, i+1-prefix, i+2-prefix],
                first_acceleration_exceedance=first,
                history_points=prefix, index_convention='ZERO_BASED_H40; negative indices are executed history')


def rejected_joint_diagnostic(model, sample, installed, history, stage):
    import mujoco
    from scripts.pi0294.task_path_model import certify_chunk, PlanningObjectMotion
    from go2arx5_vla.world_model.native_replay import forbid_integration

    motion = (PlanningObjectMotion(sample, mode='ACTUAL_GRASP_ATTACHMENT')
              if stage in ('close', 'lift', 'hold', 'release') and sample['native_grasp'] else None)
    result = dict(scope='ORIGINAL_CLONE_CERTIFIER_AT_SAVED_MEASURED_STATE_AND_REFERENCE_HISTORY',
                  full_WBC_state_restored=False, physical_requests=0, physics_integrations=0,
                  original_certifier_sha256=sha(inspect.getsourcefile(certify_chunk)))
    try:
        with forbid_integration(mujoco) as calls:
            certificate = certify_chunk(model, sample, matrix10(installed), stage=stage,
                                        history=history, planning_object_motion=motion)
        assert not calls
        result.update(status='PASS_OFFLINE_CLONE_ONLY', certificate=certificate)
    except ValueError as error:
        result.update(status='REJECT_OFFLINE_CLONE_ONLY', error=str(error))
        frame = error.__traceback__
        while frame is not None and frame.tb_frame.f_code is not certify_chunk.__code__:
            frame = frame.tb_next
        if frame is None:
            raise
        values = frame.tb_frame.f_locals
        if 'command_q' in values:
            result['joint_rates'] = joint_rates(values['command_q'], values['prefix'])
    return result


def run(monitor, expected_result_sha256, *, diagnose_joints=False):
    started, cpu = time.time(), time.process_time()
    monitor = Path(monitor)
    fixed = load(monitor / 'RESULT.private.json', expected_result_sha256)
    spec = load(monitor / 'EXECUTION_SPEC.json')
    records, episodes = [], []
    models, native_sources = {}, set()
    if diagnose_joints:
        import mujoco
        from scripts.pi0294.task_state import ActualState, parse_state
        from go2arx5_vla.world_model.native_replay import forbid_integration
        if mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None:
            raise ValueError('ORIGINAL_MUJOCO_WITHOUT_CONTROLLER_CALLBACK_REQUIRED')

    for population, panel in spec['evaluation']['panels'].items():
        groups = sorted(Path(panel).glob('*-group-*'))
        for outcome in fixed['panels'][population]['outcomes']:
            roots = [g / 'runs' / outcome['native_run'] for g in groups
                     if (g / 'runs' / outcome['native_run']).exists()]
            if len(roots) != 1:
                raise ValueError('UNIQUE_ORIGINAL_NATIVE_REQUEST_REQUIRED')
            root = roots[0]
            native, gpu = root / 'native-01', root.with_name(root.name + '-gpu') / 'fit-01'
            native_result = load(native / 'RESULT.json', outcome['result_sha256'])
            gpu_result = load(gpu / 'RESULT.private.json', outcome['GPU_result_sha256'])
            ns = load(root / 'EXECUTION_SPEC.json')
            native_sources.add(ns['source_sha'])
            if sha(ns['model_path']) != native_result['scene_sha256']:
                raise ValueError('ORIGINAL_COMPILED_SCENE_REQUIRED')
            for actor in (root, root.with_name(root.name + '-gpu')):
                if load(actor / 'control/LEASE.json')['status'] != 'CLOSED':
                    raise ValueError('AUDIT_CLOSED_REQUESTS_ONLY')
            captures = load(native / 'TASK_LEARNING_CAPTURES.private.json')['records']
            queries = gpu_result['records']
            assert len(captures) == len(queries) == outcome['actual_SLA_queries']
            commands_path = native / 'TASK_LEARNING_CONFIRMED_COMMANDS.private.json'
            commands = load(commands_path)['records'] if commands_path.exists() else []
            assert outcome['actual_native_integrations'] == 5703 + 25 * len(commands)
            by_decision = Counter(c['student_decision_sha256'] for c in commands)
            preparation_path = native / 'STABLE_TASK_CYCLES/-1/RESULT.private.json'
            preparation = load(preparation_path)
            preparation_sha = sha(preparation_path)
            history_rows = list(preparation['windows']) + [dict(c,
                actual_installed_consumed_EEF10_BASE7=c['installed_consumed_EEF10']) for c in commands]
            with (native / 'COMMANDS.private.jsonl').open() as stream:
                events = [json.loads(line) for line in stream]
            command_journal_sha = sha(native / 'COMMANDS.private.jsonl')
            episode_records = []
            for index, (query, capture) in enumerate(zip(queries, captures, strict=True)):
                query_path = gpu / 'queries' / f'{index:02d}'
                assert load(query_path / 'RESULT.json') == query
                assert sha(query_path / 'ACTION.npz') == query['action_sha256']
                request_path = native / 'GR00T_REQUESTS' / f'{index:02d}.json'
                request = load(request_path, query['request_sha256'])
                assert request['observation'] == capture
                observed = unseal(load(capture['path'], query['observation_sha256']))
                shadow = unseal(observed['original_shadow_ABI'])
                state_path = Path(capture['path']).parent / 'STATE.private.bin'
                assert sha(state_path) == observed['files']['STATE.private.bin'] == capture['state_sha256']
                step = capture['native_step']
                assert step == 5703 + 250 * index
                previous = [w for w in history_rows if w['native_end_step'] <= step][-2:]
                assert [w['native_end_step'] for w in previous] == [step - 25, step]
                history = [dict(end_native_step=w['native_end_step'],
                                pose=matrix10(w['actual_installed_consumed_EEF10_BASE7'][:10])) for w in previous]
                actual = matrix10(shadow['target_observation']['actual_state_eef10'])
                frame = np.asarray(shadow['source_policy_view']['T_source_from_target'])
                with np.load(query_path / 'ACTION.npz', allow_pickle=False) as packet:
                    raw = packet['physical_H40'].copy()
                    assert raw.shape == (40, 10) and packet['normalized'].shape == (1, 40, 32)
                target = pose10(np.linalg.inv(frame) @ matrix10(raw), raw[:, 9])
                desired = target.copy()
                desired[:, 9] = (envelope.WORKING_CLOSED_M +
                    (envelope.WORKING_OPEN_M-envelope.WORKING_CLOSED_M)*target[:, 9]) / envelope.MECHANICAL_TRAVEL_M
                installed = desired.copy()
                reference = history[-1]['pose']
                installed[:, :3] = reference[:3, 3] + target[:, :3] - actual[:3, 3]
                installed[:, 3:9] = rotation_matrix_to_rot6d_rows(
                    reference[:3, :3] @ actual[:3, :3].T @ rot6d_rows_to_matrix(target[:, 3:9]))
                desired_metrics = geometry_detail(desired, actual)
                reference_metrics = geometry_detail(installed, reference)
                decision_path = native / f'TASK_LEARNING_DECISION_{index:03d}.private.json'
                terminal = index == len(queries) - 1
                decision = load(decision_path) if decision_path.exists() else None
                consumed = by_decision.get(sha(decision_path), 0) if decision else 0
                classification = ('EEF_FULL_H40_REJECT' if terminal and not decision else
                                  'JOINT_ACCELERATION_REJECT' if terminal else 'ACCEPTED_AND_CONSUMED')
                if decision:
                    assert decision['index'] == index and decision['action_sha256'] == query['action_sha256']
                    assert np.max(np.abs(np.asarray(decision['mapped_target_H40']) - target)) < 1e-12
                    assert not decision['teacher_fallback'] and not decision['endpoint_basis_applied']
                    assert not decision['expert_segment_generator_applied'] and decision['command_clips'] == 0
                    assert desired_metrics['passes'] and reference_metrics['passes']
                else:
                    rejection = load(native / 'TASK_LEARNING_STUDENT_REJECTION.private.json')
                    assert terminal and rejection['index'] == index and not rejection['proposal_executed']
                    assert not (desired_metrics['passes'] and reference_metrics['passes'])
                assert consumed == (10 if not terminal else 0)
                cert_events = [e for e in events if e.get('native_step') == step
                               and e.get('phase') in ('proposed', 'reference-certificate-rejected')]
                cert_events = [e for e in cert_events if 'action_EEF10' in e
                               and np.allclose(e['action_EEF10'], installed, rtol=0, atol=1e-12)]
                if decision:
                    assert len(cert_events) == 1
                    if terminal:
                        assert cert_events[0]['reason'] == 'ORIGINAL_JOINT_ACCELERATION_REJECT'
                    else:
                        assert cert_events[0]['native_reference_certificate']['physics_integrations'] == 0
                row = dict(population=population, native_run=root.name, layout=outcome['layout'],
                    restart=outcome['restart'], query=index, native_step=step,
                    classification=classification, confirmed_commands=consumed,
                    execution_stage=decision.get('execution_stage') if decision else None,
                    execution_stage_is_expert_ground_truth=False,
                    EEF_desired=desired_metrics, EEF_reference=reference_metrics,
                    actual_TCP=actual.tolist(), previous_reference_TCP=reference.tolist(),
                    H40_start_to_actual_m=float(np.linalg.norm(target[0, :3]-actual[:3, 3])),
                    H40_reference_start_to_previous_reference_m=float(np.linalg.norm(installed[0, :3]-reference[:3, 3])),
                    original_native_certificate_event=cert_events[0] if cert_events else None,
                    original_expert_final_plan_at_this_state_available=False,
                    full_WBC_state_restored=False,
                    pins=dict(action=query['action_sha256'], request=query['request_sha256'],
                        observation=query['observation_sha256'], state=capture['state_sha256'],
                        decision=sha(decision_path) if decision else None,
                        preparation=preparation_sha, command_journal=command_journal_sha))
                if diagnose_joints:
                    key = (ns['model_path'], native_result['scene_sha256'])
                    if key not in models:
                        models[key] = mujoco.MjModel.from_binary_path(ns['model_path'])
                    model = models[key]
                    with forbid_integration(mujoco):
                        sample = ActualState(model).update(parse_state(state_path.read_bytes()))
                    assert np.max(np.abs(sample['tcp']-actual)) < 1e-9
                    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'x5_joint'+str(j)) for j in range(1, 7)]
                    row['actual_joint_qpos'] = sample['qpos'][model.jnt_qposadr[joints]].tolist()
                    row['actual_joint_qvel'] = sample['qvel'][model.jnt_dofadr[joints]].tolist()
                    row['actual_object_lift_from_first_query_m'] = float(sample['object_pose'][2, 3] -
                        episode_records[0]['actual_object_z_m']) if episode_records else 0.
                    row['actual_object_z_m'] = float(sample['object_pose'][2, 3])
                    row['actual_native_grasp'] = sample['native_grasp']
                    if classification == 'JOINT_ACCELERATION_REJECT':
                        row['offline_joint_diagnostic'] = rejected_joint_diagnostic(
                            model, sample, installed, history, cert_events[0]['stage'])
                        assert row['offline_joint_diagnostic']['error'] == cert_events[0]['reason']
                records.append(row)
                episode_records.append(row)
            episodes.append(dict(population=population, native_run=root.name, layout=outcome['layout'],
                restart=outcome['restart'], queries=len(queries), confirmed_commands=len(commands),
                task_success=False, first_failure=outcome['first_failure'],
                terminal_query=episode_records[-1], result_sha256=outcome['result_sha256']))
    return dict(schema='SLA_EXISTING80_QUERY_FEASIBILITY_AUDIT_V7',
        monitor_result_sha256=expected_result_sha256, limits=LIMITS,
        queries=len(records), native_requests=len(episodes), task_successes=0,
        classifications=dict(Counter(r['classification'] for r in records)),
        confirmed_commands=sum(r['confirmed_commands'] for r in records),
        original_expert_same_state_plan_pairs=0,
        same_state_expert_comparison_status='NOT_AVAILABLE_NO_HISTORICAL_EXPERT_PLAN_FABRICATED',
        full_WBC_state_restored=False, native_equivalent_dynamic_replay=False,
        model_loads=0, optimizer_updates=0, training_visits=0,
        additional_physical_requests=0, physics_integrations=0, new_model_queries=0,
        original_outcomes_modified=False, original_safety_limits_modified=False,
        original_native_source_shas=sorted(native_sources),
        records=records, episodes=episodes, wall_s=time.time()-started, CPU_s=time.process_time()-cpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('monitor', type=Path)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--diagnose-joints', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.monitor, args.result_sha256, diagnose_joints=args.diagnose_joints), allow_nan=False))
