"""One offline adapter variant on all original15 queries; no native replay."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import time

import numpy as np


def read(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recorded_stage(decision, event, native_spec):
    if (native_spec.get('phase') != 'collect-target-task-learning-grasp-v1'
            or native_spec.get('sla_pick_place_v5') is not None):
        raise ValueError('EXACT_ORIGINAL_GRASP_STAGE_MAPPING_REQUIRED')
    stage = decision.get('execution_stage')
    if stage not in ('PREGRASP', 'DESCEND', 'CLOSE', 'LIFT', 'HOLD'):
        raise ValueError('RECORDED_STUDENT_EXECUTION_STAGE_REQUIRED')
    mapped = {'CLOSE': 'close', 'LIFT': 'lift', 'HOLD': 'hold'}.get(stage, 'approach')
    if event.get('stage') is not None and event['stage'] != mapped:
        raise ValueError('RECORDED_NATIVE_AND_DECISION_STAGE_MISMATCH')
    return mapped


def run(spec, output):
    import mujoco
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0292.relay_protocol import unseal
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from scripts.pi0294.task_state import ActualState, parse_state
    from scripts.pi0294.task_path_model import certify_chunk
    from scripts.sla.audit_query_feasibility_v7 import geometry_detail, rejected_joint_diagnostic
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
    from go2arx5_vla.world_model import fusion_trajectory_adapter as envelope
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows, rot6d_rows_to_matrix
    from scripts.pi0292.durable import write

    assert spec['method'] == 'REFERENCE_JOIN10_TWO_HISTORY_QUINTIC_V71'
    reuse = spec.get('reuse_prefix')
    assert spec['variants'] == 15
    assert spec['maximum_new_clone_calls'] == (12 if reuse else 14)
    assert spec['physics'] is False and spec['optimizer_updates'] == spec['new_generations'] == 0
    assert mujoco.__version__ == '3.2.7' and mujoco.get_mjcb_control() is None
    assert sha(inspect.getsourcefile(certify_chunk)) == spec['original_certifier_sha256']
    helper = Path(__file__).with_name('reference_join10_v71.py')
    assert sha(helper) == spec['helper_sha256']
    module_spec = importlib.util.spec_from_file_location('offline_join10_adapter', helper)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    group = Path(spec['group']); prior = Path(spec['joint_analysis'])
    assert sha(group/'RESULT.private.json') == 'd9185b7dff07e364368efccd04a1e2d26c382900c6b02000c7a8904d6e6a2401'
    assert sha(prior) == '6d2f6d5fba81176055c7e083489d3cc5f30d77871694225da22b5b9d715b3980'
    prior_joints = {r['native_run']: r for r in read(prior)['rows']}
    assert read(group/'control/LEASE.json')['status'] == 'CLOSED'
    output = Path(output); output.mkdir(exist_ok=False)
    started, cpu = time.time(), time.process_time()
    rows = []; models = {}; attempts = calls = 0
    prior_calls = 0; ordinal = 0
    if reuse:
        prior_root = Path(reuse['root']); directory = prior_root/'analysis'
        assert read(prior_root/'control/LEASE.json')['status'] == 'CLOSED'
        assert read(prior_root/'WORKER_EXIT.private.json')['exit_code'] == 1
        assert sha(directory/'ROWS.private.jsonl') == reuse['rows_sha256']
        assert sha(directory/'CLONE_CALLS.private.jsonl') == reuse['calls_sha256']
        counts = read(directory/'COUNTERS.private.json')
        assert (counts['new_clone_attempts'], counts['new_clone_calls_completed'], counts['completed_variants']) == (3, 2, 3)
        assert "KeyError: 'stage'" in read(prior_root/'FAILED.private.json')['log_tail']
        old_rows = [json.loads(line) for line in (directory/'ROWS.private.jsonl').read_text().splitlines()]
        assert len(old_rows) == 3 and sum(r['candidate_diagnostic'] is not None for r in old_rows) == 2
        for row in old_rows:
            source = directory/row['file']; assert sha(source) == row['file_sha256']
            with (output/row['file']).open('xb') as stream: stream.write(source.read_bytes())
            rows.append(dict(row, reused_prior_variant=True))
        prior_calls = 2
    try:
        for outcome in read(group/'RESULT.private.json')['outcomes']:
            root = group/'runs'/outcome['native_run']; native = root/'native-01'
            gpu = root.with_name(root.name+'-gpu')/'fit-01'
            assert sha(native/'RESULT.json') == outcome['result_sha256']
            assert sha(gpu/'RESULT.private.json') == outcome['GPU_result_sha256']
            assert all(read(p/'control/LEASE.json')['status'] == 'CLOSED'
                       for p in (root, root.with_name(root.name+'-gpu')))
            queries = read(gpu/'RESULT.private.json')['records']
            ns = read(root/'EXECUTION_SPEC.json')
            captures = read(native/'TASK_LEARNING_CAPTURES.private.json')['records']
            commands_path = native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json'
            commands = read(commands_path)['records'] if commands_path.exists() else []
            past = list(read(native/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
            past.extend(dict(c, actual_installed_consumed_EEF10_BASE7=c['installed_consumed_EEF10'])
                        for c in commands)
            events = [json.loads(line) for line in (native/'COMMANDS.private.jsonl').read_text().splitlines()]
            for index, (query, capture) in enumerate(zip(queries, captures, strict=True)):
                ordinal += 1
                if reuse and ordinal <= 3:
                    row = rows[ordinal-1]
                    assert (row['native_run'], row['query'], row['raw_action_sha256']) == (outcome['native_run'], index, query['action_sha256'])
                    continue
                assert sha(capture['path']) == query['observation_sha256']
                observation = unseal(read(capture['path'])); shadow = unseal(observation['original_shadow_ABI'])
                actual = matrix10(shadow['target_observation']['actual_state_eef10'])
                frame = np.asarray(shadow['source_policy_view']['T_source_from_target'])
                action = gpu/'queries'/f'{index:02d}'/'ACTION.npz'
                assert sha(action) == query['action_sha256']
                with np.load(action, allow_pickle=False) as z: raw = z['physical_H40'].copy()
                target = pose10(np.linalg.inv(frame) @ matrix10(raw), raw[:, 9])
                previous = [c for c in past if c['native_end_step'] <= capture['native_step']][-2:]
                assert [c['native_end_step'] for c in previous] == [capture['native_step']-25, capture['native_step']]
                history = [dict(end_native_step=c['native_end_step'],
                    pose=matrix10(c['actual_installed_consumed_EEF10_BASE7'][:10])) for c in previous]
                reference = history[-1]['pose']
                baseline = target.copy()
                baseline[:, 9] = (envelope.WORKING_CLOSED_M+(envelope.WORKING_OPEN_M-envelope.WORKING_CLOSED_M)*target[:, 9])/envelope.MECHANICAL_TRAVEL_M
                baseline[:, :3] += reference[:3, 3]-actual[:3, 3]
                baseline[:, 3:9] = rotation_matrix_to_rot6d_rows(reference[:3, :3] @ actual[:3, :3].T @ rot6d_rows_to_matrix(target[:, 3:9]))
                terminal = index == len(queries)-1
                original_class = ('ACCEPTED_P10' if not terminal else 'JOINT_REJECT'
                                  if outcome['native_run'] in prior_joints else 'EEF_REJECT')
                candidate_events = [e for e in events if e.get('native_step') == capture['native_step']
                    and e.get('phase') in ('proposed', 'reference-certificate-rejected') and 'action_EEF10' in e]
                if original_class != 'EEF_REJECT':
                    assert len(candidate_events) == 1
                    assert np.allclose(candidate_events[0]['action_EEF10'], baseline, rtol=0, atol=1e-12)
                    decision = read(native/f'TASK_LEARNING_DECISION_{index:03d}.private.json')
                    stage = recorded_stage(decision, candidate_events[0], ns)
                candidate = module.connect(baseline, np.stack([c['pose'] for c in history]))
                assert np.array_equal(candidate[9:], baseline[9:]) and np.array_equal(candidate[:, 9], baseline[:, 9])
                desired = candidate.copy(); desired[:, :3] += actual[:3, 3]-reference[:3, 3]
                desired[:, 3:9] = rotation_matrix_to_rot6d_rows(actual[:3, :3] @ reference[:3, :3].T @ rot6d_rows_to_matrix(candidate[:, 3:9]))
                envelope_error = None
                try:
                    FiftyMsProgram.validate_envelope(SimpleNamespace(desired_path=desired, reference_path=candidate,
                        decision_TCP_target=actual, initial_reference_TCP_target=reference))
                except ValueError as error: envelope_error = str(error)
                if original_class == 'EEF_REJECT': assert envelope_error is not None
                identity = dict(native_run=outcome['native_run'], layout=outcome['layout'], view=outcome['method'],
                    query=index, native_step=capture['native_step'], raw_action_sha256=query['action_sha256'],
                    native_result_sha256=outcome['result_sha256'], observation_sha256=query['observation_sha256'])
                diagnostic = None
                if envelope_error is None:
                    native_result = read(native/'RESULT.json')
                    assert sha(ns['model_path']) == native_result['scene_sha256']
                    key = (ns['model_path'], native_result['scene_sha256'])
                    if key not in models: models[key] = mujoco.MjModel.from_binary_path(ns['model_path'])
                    model = models[key]; state = Path(capture['path']).parent/'STATE.private.bin'
                    assert sha(state) == capture['state_sha256']
                    with forbid_integration(mujoco) as forbidden:
                        sample = ActualState(model).update(parse_state(state.read_bytes()))
                    assert not forbidden and sample['native_step'] == capture['native_step']
                    assert np.allclose(sample['tcp'], actual, rtol=0, atol=1e-9)
                    attempts += 1; assert attempts <= spec['maximum_new_clone_calls']
                    with (output/'CLONE_CALLS.private.jsonl').open('a') as f:
                        f.write(json.dumps(dict(identity, event='INTENT', attempt=attempts))+'\n')
                    before = [sample[k].copy() for k in ('qpos', 'qvel', 'ctrl')]
                    diagnostic = rejected_joint_diagnostic(model, sample, candidate, history, stage); calls += 1
                    assert all(np.array_equal(v, sample[k]) for v, k in zip(before, ('qpos', 'qvel', 'ctrl')))
                    with (output/'CLONE_CALLS.private.jsonl').open('a') as f:
                        f.write(json.dumps(dict(identity, event='COMPLETED', call=calls, status=diagnostic['status']))+'\n')
                file = output/f'{len(rows):02d}.npz'
                with file.open('xb') as stream:
                    np.savez_compressed(stream, original_raw_policy_H40=raw, original_reference_H40=baseline,
                        candidate_reference_H40=candidate, previous_reference_poses=np.stack([c['pose'] for c in history]))
                angles = [np.linalg.norm(module._log(a @ b.T)) for a, b in zip(
                    rot6d_rows_to_matrix(candidate[:, 3:9]), rot6d_rows_to_matrix(baseline[:, 3:9]), strict=True)]
                row = dict(identity, original_classification=original_class, original_result_unchanged=True,
                    candidate_EEF_pass=envelope_error is None, candidate_EEF_failure=envelope_error,
                    candidate_diagnostic=diagnostic, candidate_EEF_and_clone_pass=bool(diagnostic and diagnostic['status']=='PASS_OFFLINE_CLONE_ONLY'),
                    maximum_translation_change_m=float(np.linalg.norm(candidate[:, :3]-baseline[:, :3], axis=1).max()),
                    maximum_rotation_change_rad=float(max(angles)), suffix_k9_through39_bit_exact=True,
                    nominal_grip_and_clock_unchanged=True, file=file.name, file_sha256=sha(file),
                    candidate_reference_geometry=geometry_detail(candidate, reference),
                    expert_training_eligible=False, native_execution_admitted=False)
                rows.append(row)
                with (output/'ROWS.private.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
        assert len(rows) == ordinal == 15 and attempts == calls <= spec['maximum_new_clone_calls']
        assert prior_calls+calls <= 14
        assert Counter(r['original_classification'] for r in rows) == dict(ACCEPTED_P10=7, JOINT_REJECT=7, EEF_REJECT=1)
        summary = {kind: dict(variants=len(group), candidate_EEF_pass=sum(r['candidate_EEF_pass'] for r in group),
            candidate_EEF_and_clone_pass=sum(r['candidate_EEF_and_clone_pass'] for r in group),
            outcomes=dict(Counter(r['candidate_diagnostic'].get('error', 'PASS_OFFLINE_CLONE_ONLY')
                if r['candidate_diagnostic'] else 'CANDIDATE_EEF_REJECT' for r in group)))
            for kind in ('ACCEPTED_P10', 'JOINT_REJECT', 'EEF_REJECT')
            for group in ([r for r in rows if r['original_classification']==kind],)}
        result = dict(schema='SLA_PD4_JOIN10_OFFLINE_SINGLE_VARIANT_V71', method=module.METHOD,
            helper_sha256=sha(helper), rows=rows, summaries=summary, variants=15,
            new_clone_attempts=attempts, new_clone_calls_completed=calls, original_clone_calls_repeated=0,
            prior_candidate_calls_reused=prior_calls, total_unique_candidate_calls=prior_calls+calls,
            prior_variants_reused=3 if reuse else 0, prior_aborted_before_callee_intents=1 if reuse else 0,
            native_kinematic_model_loads=len(models), SLA_parent_loads=0, new_model_generations=0,
            optimizer_updates=0, training_visits=0, physical_requests=0, physics_integrations=0,
            original_formal_results_changed=False, full_WBC_state_restored=False,
            same_student_state_expert_final_plan_pairs=0, native_execution_admitted=False,
            CPU_s=time.process_time()-cpu, wall_s=time.time()-started)
        write(output/'RESULT.private.json', result); return result
    finally:
        write(output/'COUNTERS.private.json', dict(new_clone_attempts=attempts, new_clone_calls_completed=calls,
            completed_variants=len(rows), native_kinematic_model_loads=len(models), new_model_generations=0,
            prior_candidate_calls_reused=prior_calls,
            optimizer_updates=0, training_visits=0, physical_requests=0, physics_integrations=0))


if __name__ == '__main__':
    def stop(*args): raise InterruptedError('FINITE_OWNED_OFFLINE_JOIN_CHECK_STOP')
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('spec'); parser.add_argument('output')
    args = parser.parse_args(); result = run(read(args.spec), args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
