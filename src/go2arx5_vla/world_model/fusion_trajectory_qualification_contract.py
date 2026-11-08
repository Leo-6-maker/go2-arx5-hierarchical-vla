"""Read-only recipe and lineage checks for one future trajectory qualification.

The archived target query supplies geometry, never a live reference grant or
an executed source outcome. Actual AUTO1/P1 admission remains independent.
"""
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from .storage import digest
from .fusion_reference_qualification_contract import INIT_STEP, DECISION_STEP, SCREEN, hold_preview
from .fusion_trajectory_adapter import TrajectoryAdaptation, GEOMETRY_SCALES, RECIPE

PHASE = 'collect-target-fusion-trajectory-qualification-v1'
SCHEMA = 'WM02_FUSION_TRAJECTORY_QUALIFICATION_EPISODE_V1'
SLOW_PHASE = 'collect-target-fusion-trajectory-slow-qualification-v2'
SLOW_SCHEMA = 'WM02_FUSION_TRAJECTORY_QUALIFICATION_EPISODE_V2'
SLOW_RECIPE = 'FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_2X_V2'
SLOW_FAILED_RESULT = '77bea49654a7d33007312d20d416f3796896a508d3e93dbce134a7d2ca5e1a8d'
TRACE_CAPACITY_FAILED_RESULT='cf06b2d90f08266b8158940e9914002d6e7d6fc24791a9319ed6874cf5719f1c'
SLOW_FIRST_PREDECESSOR = '2732f75836d8d89f5789d53bb0b9de4f28716728bf44819a5fc46513b76be9c0'
TAIL_ATOMS = 8
PINS = dict(
    checkpoint='94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38',
    spec='0988bdafe07c49fc0740d6075fd9e172314894adf477fad8e9e54070972c7689',
    result='140d4c0db91e5daa36bdc51557942e040fd0b7d14b6753f1bdba0c93dbc39d09',
    action='e447e92a9b63371a560efe01d7c5a6e6a1337c5adf22657012f591c7211fc090',
    observation='389753cd96756719d83eaf55a440f8de504612840ef893e80b12ca36423f59dc',
    frame_map='fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1',
    first_predecessor='fd163e866a1c82873c3a5e044e60c3295b7696655b4af28580321a7cfad9b9c0')
QUERY_SOURCE = '1ca3b833139e92019c29029ef6be5a4f368de717'
SOURCE_MEMBERS = (
    'src/go2arx5_vla/world_model/fusion_trajectory_adapter.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_qualification_contract.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_qualification_native.py',
    'scripts/wm02/fusion_trajectory_qualification_check.py',
    'src/go2arx5_vla/world_model/auto1_native.py',
    'src/go2arx5_vla/world_model/target_sequential.py',
    'src/go2arx5_vla/world_model/fusion_reference_qualification_native.py',
    'src/go2arx5_vla/world_model/fusion_reference_qualification_contract.py',
    'src/go2arx5_vla/world_model/fusion_observation.py',
    'src/go2arx5_vla/world_model/next2_contact.py',
    'src/go2arx5_vla/world_model/target_reference.py',
    'src/go2arx5_vla/world_model/target_stability.py',
    'src/go2arx5_vla/world_model/next2_episode.py',
    'src/go2arx5_vla/world_model/init_trace_reader.py',
    'src/go2arx5_vla/world_model/next1_episode.py',
    'src/go2arx5_vla/trajectory/so3.py',
    'scripts/wm02/infer_fusion_vla.py',
    'scripts/wm02/pilot1_native_worker.py',
    'scripts/wm02/commission_target.py',
    'scripts/pi0294/task_state.py',
    'scripts/pi0294/task_gripper_contract.py',
    'scripts/pi0292/eval_contract.py',
    'src/go2arx5_vla/target_mujoco/init_trace.cpp',
    'src/go2arx5_vla/target_mujoco/init_trace.hpp')


def need(condition, reason):
    if not condition:
        raise ValueError('TRAJECTORY_QUALIFICATION_' + reason)


def episode_contract(episode):
    slow = isinstance(episode, dict) and episode.get('phase') == SLOW_PHASE
    exact = dict(schema=SCHEMA, phase=PHASE, purpose='TRAJECTORY_GEOMETRY_QUALIFICATION',
        experiment_role='NON_VLA_CALIBRATION', split='TRAIN', reference_rule=RECIPE,
        reference_anchor_mode='INIT_TCP_2503_REFERENCE_V1', prehold_atoms=40,
        screen_atom=48, screen=SCREEN, tail_hold_atoms=TAIL_ATOMS,
        desired_outcome_provenance='CALIBRATION_INTENT', physics_dt_s=.002,
        base_program='MEASURED_INIT_BASE_POSE_HOLD', query_pins={k: v for k, v in PINS.items()
                                                              if k != 'first_predecessor'})
    if slow:
        exact.update(schema=SLOW_SCHEMA, phase=SLOW_PHASE, reference_rule=SLOW_RECIPE,
                     time_stretch_factor=2)
    need(isinstance(episode, dict) and all(episode.get(k) == v for k, v in exact.items()),
         'EXACT_VERSIONED_TRAIN_GEOMETRY_RECIPE_AND_ARCHIVED_QUERY')
    need(type(episode.get('geometry_scale')) in (int, float)
         and episode['geometry_scale'] in ((.5, 1.) if slow else GEOMETRY_SCALES),
         'PREDECLARED_GRADED_GEOMETRY_SCALE')
    if slow:
        need(type(episode.get('time_stretch_factor')) is int,
             'EXPLICIT_INTEGER_2X_TIME_STRETCH_NO_BOOL')
    need(all(isinstance(episode.get(k), str) and episode[k] for k in ('run', 'scene_id', 'parent')),
         'EXACT_RUN_SCENE_PARENT')
    need(type(episode.get('planned_end_atom')) is int and episode['planned_end_atom'] >= 40,
         'EXPLICIT_FINITE_FULL_H40_END_ATOM')
    need(not (set(episode) - set(exact) - {'run', 'scene_id', 'parent', 'geometry_scale', 'planned_end_atom'}),
         'NO_ARBITRARY_PROGRAM_OR_GRANT_FIELDS')
    end = DECISION_STEP + 25 * (episode['planned_end_atom'] + TAIL_ATOMS)
    need(end <= (33000 if slow else 17000), 'FINITE_NATIVE_ENDPOINT_WITH_VERSIONED_TRACE_CAPACITY')
    return episode


def load_query(spec):
    """Reuse the complete old read-only payload verifier; no model/Processor."""
    from scripts.wm02.infer_fusion_vla import recorded_payload
    from .fusion_observation import transform_hash
    root = Path(spec['trajectory_query_root'])
    query_path, result_path, action_path = (root/'EXECUTION_SPEC.json',
        root/'fit-01/RESULT.private.json', root/'fit-01/REAL_VLA_ACTION.npz')
    for path, pin in ((query_path, PINS['spec']), (result_path, PINS['result']), (action_path, PINS['action'])):
        need(digest(path) == pin, 'ORIGINAL_TARGET_QUERY_FILE_BYTES')
    query, result = json.loads(query_path.read_text()), json.loads(result_path.read_text())
    lease = json.loads((root/'control/LEASE.json').read_text())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_text())
    need(lease.get('status') == 'CLOSED' and lease.get('worker_exit_code') == 0
         and worker.get('exit_code') == 0 and lease.get('spec_sha256') == PINS['spec'],
         'ORIGINAL_TARGET_QUERY_CLOSED_ZERO_EXIT')
    need(query.get('source_sha') == result.get('source_sha') == QUERY_SOURCE
         and query.get('checkpoint_manifest_sha256') == result.get('checkpoint_manifest_sha256') == PINS['checkpoint']
         and query.get('recorded_observation_sha256') == result.get('recorded_observation_sha256') == PINS['observation']
         and result.get('action_sha256') == PINS['action']
         and query.get('seed') == 366102
         and result.get('status') == 'REAL_FROZEN_VLA_OFFLINE_INFERENCE_DONE'
         and result.get('actual_VLA_inferences') == result.get('actual_model_loads') == 1
         and result.get('native_integrations') == result.get('GR00T_updates') == result.get('WM_updates') == result.get('Actor_updates') == 0
         and query.get('observation_mode') == 'SAVED_TARGET_NATIVE_REPLAY_SHADOW'
         and query.get('target_capture_step') == 3703, 'ONE_PINNED_TARGET_SHADOW_QUERY_NOT_NEW_INFERENCE')
    source = Path(spec['source_root'])
    for name in ('scripts/wm02/infer_fusion_vla.py', 'src/go2arx5_vla/world_model/fusion_observation.py'):
        need(digest(source/name) == query['source_files'][name], 'UNCHANGED_ARCHIVED_PAYLOAD_VERIFIER_BYTES')
    payload, provenance = recorded_payload(query)
    body = unseal(json.loads(Path(query['recorded_observation']).read_text()))
    view = body['source_policy_view']
    need(view['transform_sha256'] == transform_hash(view['T_source_from_target']) == PINS['frame_map']
         and provenance['native_step'] == 3703, 'EXACT_OLD_FRAME_MAP_AND_CAPTURE_BOUNDARY')
    with np.load(action_path, allow_pickle=False) as archive:
        raw = archive['physical_H40'].copy()
    need(raw.shape == (40, 10) and raw.dtype.kind == 'f' and np.isfinite(raw).all(),
         'ALREADY_DECODED_PHYSICAL_H40_NO_SECOND_PROCESSOR_DECODE')
    return dict(raw=raw, query_tcp=matrix10(payload['state_eef10']),
                frame_map=np.asarray(view['T_source_from_target'], float), provenance=provenance)


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    kwargs = dict(time_stretch_factor=2) if episode['phase'] == SLOW_PHASE else {}
    program = TrajectoryAdaptation(inputs['raw'], inputs['query_tcp'], inputs['frame_map'],
        decision_tcp, initial_tcp, base, episode['geometry_scale'], **kwargs)
    need(program.end_atom == episode['planned_end_atom'], 'ACTUAL_RETIME_LENGTH_MUST_EQUAL_PREDECLARED_LENGTH')
    return program


def episode_config(spec):
    slow = spec.get('phase') == SLOW_PHASE
    need(spec.get('phase') in (PHASE, SLOW_PHASE) and spec.get('profile') == 'wm02-auto1-target'
         and spec.get('domain') == 'WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('physics') is True and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_P1_OWNER_REQUIRED')
    need(not any(spec.get(k) for k in ('fusion_binding', 'night1_binding', 'next1_binding', 'next2_binding',
        'parallel_batch_spec', 'standby_shutdown_diagnostic', 'reference_readiness_probe', 'diagnostic_debugger')),
         'SINGLE_NEW_WORLD_NO_OLD_FUSION_BINDING_OR_BYPASS')
    need(spec.get('WBC_changes') is False and spec.get('physics_dt_s') == .002
         and type(spec.get('wall_limit_s')) in (int, float)
         and 0 < spec['wall_limit_s'] <= (1200 if slow else 600),
         'UNCHANGED_WBC_2MS_AND_FINITE_WALL')
    source, pins = Path(spec['source_root']), spec.get('source_files', {})
    need(all(name in pins and digest(source/name) == pins[name] for name in SOURCE_MEMBERS),
         'NEW_CAPSULE_NATIVE_AND_ADAPTER_SOURCE_BYTES')
    build_path = Path(spec['trajectory_trace_capacity_build'])
    need(digest(build_path) == spec['trajectory_trace_capacity_build_sha256'], 'TRACE_CAPACITY_BUILD_RECEIPT_BYTES')
    build = json.loads(build_path.read_text())
    capacity = spec.get('trajectory_trace_capacity',262144) if slow else 131072
    need(type(capacity) is int and capacity in ((262144,524288) if slow else (131072,)),
         'EXPLICIT_SUPPORTED_PHASE_TRACE_CAPACITY')
    if capacity==524288:
        need(episode_contract(json.loads(Path(spec['episode_manifest']).read_text()))['geometry_scale']==1.,
             'CAPACITY_REPAIR_REQUALIFIES_FULL1_ONLY_NO_HALF_RERUN')
        validate_trace_capacity_failure(spec)
    need(build.get('status') == f'TRACE_ONLY_{capacity}_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
         and build.get('trace_capacity') == capacity
         and build.get('trace_library_sha256') == digest(spec['trace_library'])
         and build.get('native_binary_sha256') == digest(spec['native_binary'])
         and build.get('observer_sha256') == digest(spec['observer_library'])
         and build.get('new_target_integrations') == 0
         and all(build.get('source_files', {}).get(name) == pins[name] for name in (
             'src/go2arx5_vla/target_mujoco/init_trace.cpp',
             'src/go2arx5_vla/target_mujoco/init_trace.hpp')),
         'ACTUAL_TRACE_ONLY_CAPACITY_BUILD_FROZEN_SOURCE_AND_EXECUTOR_BYTES')
    path = Path(spec['episode_manifest'])
    need(digest(path) == spec['episode_manifest_sha256'], 'EXACT_EPISODE_BYTES')
    episode = episode_contract(json.loads(path.read_text()))
    need(episode['phase'] == spec['phase'] and episode['run'] == spec['registered_root']
         and spec.get('max_ticks') == endpoint(episode)
         and spec.get('reference_binding_version') == 'MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'EXACT_RUN_ENDPOINT_AND_ORIGINAL_BASE_HOLD_REPAIR')
    catalog = Path(spec['scene_allowlist'])
    need(digest(catalog) == spec['scene_allowlist_sha256'], 'SCENE_CATALOG_BYTES')
    assets = json.loads(catalog.read_text())
    scenes = [s for s in assets['scenes'] if s['scene_id'] == episode['scene_id']]
    need(len(scenes) == 1, 'ONE_ORIGINAL_TRAIN_SCENE')
    scene = scenes[0]
    need(scene['split'] == 'TRAIN' and scene['parent'] == episode['parent']
         and scene['model_path'] == spec['model_path'] and scene['model_sha256'] == digest(spec['model_path'])
         and scene.get('static_check') == 'PASSED_NO_INTEGRATION'
         and assets['native_binary_sha256'] == digest(spec['native_binary']), 'UNCHANGED_TRAIN_SCENE_NATIVE_BYTES')
    inputs = load_query(spec)
    # Rigid frame/decision relocation cannot alter translation/rotation interval lengths.
    mapped = np.linalg.inv(inputs['frame_map']) @ inputs['query_tcp']
    make_program(inputs, episode, mapped, mapped, [0., 0., 0., 0., 0., 0., 1.])
    return episode, inputs


def endpoint(episode):
    return DECISION_STEP + 25 * (episode['planned_end_atom'] + TAIL_ATOMS)


def episode_limit(spec, previous):
    episode, _ = episode_config(spec)
    path = Path(spec['previous_stage'])
    need(digest(path) == spec['previous_stage_sha256'] and json.loads(path.read_text()) == previous,
         'EXACT_PREDECESSOR_FILE_NOT_BOOLEAN')
    prior_root = path.parent.parent
    lease = json.loads((prior_root/'control/LEASE.json').read_text())
    worker = json.loads((prior_root/'WORKER_EXIT.private.json').read_text())
    prior_spec_path = prior_root/'EXECUTION_SPEC.json'
    prior_spec = json.loads(prior_spec_path.read_text())
    need(lease.get('status') == 'CLOSED' and lease.get('worker_exit_code') == worker.get('exit_code') == 0
         and lease.get('spec_sha256') == digest(prior_spec_path), 'PREDECESSOR_ACTUAL_CLOSED_ZERO_EXIT')
    trace = previous.get('trace_audit') or {}
    need(previous.get('status') == 'NATIVE_STAGE_PASSED' and previous.get('shutdown_clean') is True
         and previous.get('reached_exact_pause') is True and previous.get('failure') is None
         and previous.get('physics_stops') == [] and previous.get('original_joint_or_warning_samples') == 0
         and previous.get('native_outer_guard_rejection_count') == 0
         and previous.get('native_outer_guard_rejections') == {}
         and previous.get('trace_audit_error') is None
         and {p.get('name') for p in previous.get('owned_process_exits', [])} == {'sim', 'controller', 'solver', 'planner'}
         and all(p.get('exit_code') == 0 for p in previous['owned_process_exits'])
         and trace.get('status') == 'RECORDED_TRACE_CHECKS_PASS' and trace.get('errors') == []
         and trace.get('missing') == [] and trace.get('native_step_records') == previous.get('actual_target_integrations')
         and previous.get('candidate_index') == spec['candidate_index']
         and previous.get('campaign_uuid') == Path(spec['campaign_root']).name
         and previous.get('native_binary_sha256') == digest(spec['native_binary'])
         and previous.get('observer_sha256') == digest(spec['observer_library'])
         and previous.get('scene_sha256') == digest(spec['model_path']), 'PREDECESSOR_COMPLETE_NATIVE_TRACE_SAFETY_SHUTDOWN')
    scale_index = GEOMETRY_SCALES.index(episode['geometry_scale'])
    report = previous.get('reference_qualification') or {}
    need((report.get('episode') or {}).get('scene_id') == episode['scene_id']
         and (report.get('episode') or {}).get('parent') == episode['parent'],
         'SAME_ORIGINAL_TRAIN_SCENE_AND_PHYSICAL_PARENT')
    if scale_index == 0:
        need(spec['previous_stage_sha256'] == PINS['first_predecessor']
             and previous.get('phase') == 'collect-target-fusion-reference-qualification-v1'
             and previous.get('actual_target_integrations') == 4103
             and report.get('complete_task_atoms') == report.get('solver_target_confirmed_atoms') == 64
             and report.get('program', {}).get('candidate_id') == 'Z_PLUS_0P5MM', 'FIRST_SCALE_REQUIRES_ACTUAL_Z_PLUS_0P5_PASS')
    else:
        prior_episode = report.get('episode') or {}
        slow = episode['phase'] == SLOW_PHASE
        first_slow = slow and episode['geometry_scale'] == .5
        expected_prior_phase = PHASE if not slow or first_slow else SLOW_PHASE
        if first_slow:
            need(spec['previous_stage_sha256'] == SLOW_FIRST_PREDECESSOR,
                 'NEW_2X_HALF_SCALE_REQUIRES_ACTUAL_OLD_QUARTER_SCALE_PASS')
        if slow:
            validate_retiming_failure(spec)
        need(previous.get('phase') == expected_prior_phase
             and prior_spec.get('phase') == expected_prior_phase
             and prior_episode.get('geometry_scale') == GEOMETRY_SCALES[scale_index-1]
             and prior_episode.get('query_pins') == episode['query_pins']
             and report.get('all_40_knots_consumed') is True
             and report.get('complete_task_atoms') == report.get('solver_target_confirmed_atoms') == report.get('gripper_confirmed_atoms')
             and report.get('complete_task_atoms') == (previous['actual_target_integrations']-INIT_STEP)//25
             and previous['actual_target_integrations'] == prior_spec.get('max_ticks') == report.get('native_endpoint_step'),
             'IMMEDIATELY_PRECEDING_SAME_FAMILY_SCALE_COMPLETE_NATIVE_PASS')
        need(episode_contract(prior_episode) and endpoint(prior_episode) == previous['actual_target_integrations'],
             'EXACT_PREDECESSOR_EPISODE_AND_COMPLETE_ENDPOINT')
        windows = report.get('windows', [])
        need(len(windows) == report['complete_task_atoms'], 'PREDECESSOR_COMPLETE_ATOM_EVIDENCE')
        for index, row in enumerate(windows):
            grip = row.get('gripper') or {}
            submitted = grip.get('submitted') or {}
            memory = grip.get('consumed_wrapper') or {}
            values = memory.get('values') or {}
            need(row.get('native_start_step') == INIT_STEP+25*index
                 and row.get('native_end_step') == INIT_STEP+25*(index+1)
                 and row.get('actual_ticks') == 25
                 and bool(row.get('solver_policies_with_exact_installed_target'))
                 and grip.get('command_consumed') is True
                 and submitted.get('sequence') == values.get('sequence') == index
                 and values.get('steps') == row['native_end_step']
                 and memory.get('status') == 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
                 and values.get('opening') == submitted.get('opening')
                 and values.get('expiry') == submitted.get('expiry'),
                 'PREDECESSOR_EACH_ACTUAL_POSE_AND_GRIPPER_ATOM')
    return endpoint(episode)


def validate_retiming_failure(spec):
    """Keep the actual old-half rejection; it is never a successful predecessor."""
    path = Path(spec['retiming_failed_result'])
    need(spec.get('retiming_failed_result_sha256') == SLOW_FAILED_RESULT
         and digest(path) == SLOW_FAILED_RESULT, 'EXACT_RETAINED_ORIGINAL_HALF_SCALE_FAILURE')
    failed = json.loads(path.read_text())
    root = path.parent.parent
    lease = json.loads((root/'control/LEASE.json').read_text())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_text())
    trace = failed.get('trace_audit') or {}
    need(lease.get('status') == 'CLOSED'
         and lease.get('worker_exit_code') == worker.get('exit_code') == 1
         and lease.get('spec_sha256') == digest(root/'EXECUTION_SPEC.json')
         and failed.get('status') == 'NATIVE_STAGE_FAILED_PRESERVED'
         and failed.get('phase') == PHASE
         and failed.get('actual_target_integrations') == 10178
         and 'ORIGINAL_JOINT_ACCELERATION_REJECT' in (failed.get('failure') or '')
         and failed.get('shutdown_clean') is True and failed.get('physics_stops') == []
         and failed.get('original_joint_or_warning_samples') == 0
         and trace.get('status') == 'RECORDED_TRACE_CHECKS_PASS'
         and trace.get('native_step_records') == 10178
         and (trace.get('writer') or {}).get('dropped') == 0
         and {p.get('name') for p in failed.get('owned_process_exits', [])}
             == {'sim', 'controller', 'solver', 'planner'}
         and all(p.get('exit_code') == 0 for p in failed['owned_process_exits'])
         and failed.get('native_binary_sha256') == digest(spec['native_binary'])
         and failed.get('observer_sha256') == digest(spec['observer_library'])
         and failed.get('scene_sha256') == digest(spec['model_path']),
         'RETAINED_FAILURE_MUST_BE_TERMINAL_ORIGINAL_GUARD_REJECTION_WITH_CLEAN_CLOSEOUT')
    return failed


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    need(type(step) is int and INIT_STEP <= step < end_step and (step-INIT_STEP) % 25 == 0,
         'ACTUAL_ORIGINAL_ATOM_BOUNDARY')
    base, frozen_base = np.asarray(base, float), np.asarray(frozen_base, float)
    need(base.shape == frozen_base.shape == (7,) and np.isfinite([base, frozen_base]).all()
         and np.allclose(base, frozen_base, rtol=0, atol=1e-10), 'UNCHANGED_MEASURED_BASE')
    need((step < DECISION_STEP and program is None) or (step >= DECISION_STEP
         and isinstance(program, TrajectoryAdaptation)
         and np.allclose(program.initial_reference_TCP_target, initial_tcp, rtol=0, atol=1e-12)
         and np.allclose(program.base_pose_hold, frozen_base, rtol=0, atol=1e-10)),
         'ORIGINAL_INIT_AND_BASE_RETAINED_IN_REAL_DECISION_PROGRAM')
    expected = hold_preview(initial_tcp) if step < DECISION_STEP else program.preview((step-DECISION_STEP)//25)
    action = np.asarray(action, float)
    need(action.shape == (40, 10) and np.isfinite(action).all()
         and np.allclose(action, expected, rtol=0, atol=1e-12), 'ENTIRE_FULL_H40_EXACT_PERSISTENT_SUFFIX')
    return dict(status='EXACT_TRAJECTORY_SUFFIX_NOT_NATIVE_QUALIFIED', native_step=step,
                points_checked=40, original_certify_chunk_required=True,
                original_install_ACK_required=True, physics_advance_authorized=False)


def validate_trace_capacity_failure(spec,*,historical_scene_sha256=None):
    """Keep the complete but trace-unqualified overflowing run FAILED."""
    pin=TRACE_CAPACITY_FAILED_RESULT
    path=Path(spec['trace_capacity_failed_result']);root=path.parent.parent
    need(spec.get('trace_capacity_failed_result_sha256')==pin and digest(path)==pin,
         'EXACT_RETAINED_FULL1_TRACE_OVERFLOW_FAILURE')
    failed=json.loads(path.read_text());ps=json.loads((root/'EXECUTION_SPEC.json').read_text())
    if historical_scene_sha256 is not None:
        if spec.get('phase')in('collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1'):
            from .stable_task_scene_v1 import scene
            from .stable_task_layout_static_v1 import BASE_SCENE as CALIBRATION_SCENE
        else:
            from .fusion_source_phase_c01 import scene,CALIBRATION_SCENE
        scene(spec)
        need(historical_scene_sha256==CALIBRATION_SCENE==digest(ps['model_path']),
             'EXACT_HISTORICAL_T06_TRACE_FAILURE_NOT_NEW_SCENE_QUALIFICATION')
    lease=json.loads((root/'control/LEASE.json').read_text());worker=json.loads((root/'WORKER_EXIT.private.json').read_text())
    trace=failed.get('trace_audit')or{};writer=trace.get('writer')or{};report=failed.get('reference_qualification')or{}
    need(lease.get('status')=='CLOSED' and lease.get('worker_exit_code')==worker.get('exit_code')==1
         and lease.get('spec_sha256')==digest(root/'EXECUTION_SPEC.json')
         and failed.get('status')=='NATIVE_STAGE_FAILED_PRESERVED' and failed.get('phase')==SLOW_PHASE
         and failed.get('actual_target_integrations')==ps.get('max_ticks')==29753
         and failed.get('reached_exact_pause') is True and failed.get('shutdown_clean') is True
         and failed.get('failure') is None and failed.get('physics_stops')==[]
         and failed.get('original_joint_or_warning_samples')==0
         and failed.get('native_outer_guard_rejection_count')==0
         and trace.get('status')=='RECORDED_TRACE_CHECKS_FAIL'
         and set(trace.get('errors',[]))=={'WRITER_EVENT_COUNT_DIFFERS','WRITER_REPORTED_DROPPED_EVENTS'}
         and trace.get('missing')==[] and writer.get('events')==266763
         and writer.get('written')==262144 and writer.get('dropped')==4619
         and writer.get('sealed_at_explicit_pause') is True and writer.get('stop_latched') is False
         and report.get('all_40_knots_consumed') is True
         and report.get('complete_task_atoms')==report.get('solver_target_confirmed_atoms')==report.get('gripper_confirmed_atoms')==1090
         and failed.get('native_binary_sha256')==digest(spec['native_binary'])
         and failed.get('observer_sha256')==digest(spec['observer_library'])
         and failed.get('scene_sha256')==(digest(spec['model_path'])if historical_scene_sha256 is None else historical_scene_sha256)
         and {p.get('name')for p in failed.get('owned_process_exits',[])}=={'sim','controller','solver','planner'}
         and all(p.get('exit_code')==0 for p in failed['owned_process_exits']),
         'ACTUAL_FAILED_TRACE_EVIDENCE_NOT_A_PHYSICAL_OR_TASK_PASS')
    return failed
