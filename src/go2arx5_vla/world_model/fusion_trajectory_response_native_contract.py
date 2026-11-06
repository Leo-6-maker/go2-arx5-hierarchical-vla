"""Byte-bound native entry for one cell of a predeclared response panel.

This is an original-owner recipe check. It never registers workers, schedules
a panel, creates a grant, fits a model, or substitutes fixtures for native PASS.
"""
import json
from pathlib import Path

import numpy as np

from .storage import digest
from .fusion_trajectory_adapter import TrajectoryAdaptation
from . import fusion_trajectory_qualification_contract as geometry
from .fusion_trajectory_response_contract import (
    PHASE, INIT_STEP, DECISION_STEP, TAIL_ATOMS, ContextProgram, ResponseProgram,
    validate_panel, validate_episode)

SOURCE_MEMBERS = tuple(dict.fromkeys((*geometry.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_trajectory_response_contract.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_response_native_contract.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_response_native.py',
    'scripts/wm02/fusion_trajectory_response_check.py')))


def need(condition, reason):
    if not condition:
        raise ValueError('TRAJECTORY_RESPONSE_NATIVE_' + reason)


def endpoint(episode):
    value = episode['planned_end_atom']
    need(type(value) is int and value >= 40, 'FULL_COMMON_INTEGER_END_ATOM')
    end = DECISION_STEP + 25*(value+TAIL_ATOMS)
    need(end <= 33000, 'FINITE_ENDPOINT_WITHIN_PINNED_TRACE_CAPACITY')
    return end


def make_program(inputs, episode, panel, initial_tcp, decision_tcp, base):
    # Both roundtrip contexts have the same planned endpoint, original INIT.
    # Actual decision TCP remains a separate desired-motion anchor.
    nominal = TrajectoryAdaptation(inputs['raw'], inputs['query_tcp'], inputs['frame_map'],
                                  decision_tcp, initial_tcp, base, 1.,
                                  time_stretch_factor=panel['time_stretch_factor'])
    program = ResponseProgram(nominal, episode['theta_level'], panel)
    need(program.end_atom == episode['planned_end_atom'], 'ACTUAL_COMMON_CLOCK_EQUALS_PREDECLARED_ENDPOINT')
    return program


def episode_config(spec):
    need(spec.get('phase') == PHASE and spec.get('profile') == 'wm02-auto1-target'
         and spec.get('domain') == 'WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('physics') is True and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_P1_OWNER_ENTRY')
    need(not any(spec.get(k) for k in ('fusion_binding', 'night1_binding', 'next1_binding', 'next2_binding',
        'parallel_batch_spec', 'standby_shutdown_diagnostic', 'reference_readiness_probe', 'diagnostic_debugger')),
         'SINGLE_WORLD_NO_OLD_DOMAIN_BINDING_OR_BYPASS')
    need(spec.get('WBC_changes') is False and spec.get('physics_dt_s') == .002
         and type(spec.get('wall_limit_s')) in (int, float) and 0 < spec['wall_limit_s'] <= 1200,
         'FROZEN_WBC_2MS_AND_FINITE_ORIGINAL_WALL_BOUND')
    source, files = Path(spec['source_root']), spec.get('source_files', {})
    need(all(name in files and digest(source/name) == files[name] for name in SOURCE_MEMBERS),
         'NEW_CAPSULE_AND_ORIGINAL_HELPER_SOURCE_BYTES')
    panel_path = Path(spec['trajectory_response_panel'])
    need(digest(panel_path) == spec['trajectory_response_panel_sha256'], 'EXACT_PANEL_FILE_BYTES')
    panel = validate_panel(json.loads(panel_path.read_text()))
    need(panel['source_sha'] == spec['source_sha'] and panel['query_pins'] == {
         k:v for k,v in geometry.PINS.items() if k != 'first_predecessor'},
         'ONE_FROZEN_SOURCE_AND_ARCHIVED_QUERY_GEOMETRY')
    path = Path(spec['episode_manifest'])
    need(digest(path) == spec['episode_manifest_sha256'], 'EXACT_EPISODE_FILE_BYTES')
    episode = validate_episode(json.loads(path.read_text()), panel)
    need(episode['run'] == spec['registered_root'] and spec.get('max_ticks') == endpoint(episode)
         and spec.get('reference_binding_version') == 'MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'EXACT_REGISTERED_ROOT_ENDPOINT_AND_ORIGINAL_BASE_HOLD')
    collection = spec.get('trajectory_response_collection')
    need(collection in ('PILOT12', 'PANEL72'), 'EXPLICIT_PILOT_OR_COMPLETE_PANEL_COLLECTION')
    if collection == 'PILOT12':
        need(episode['split'] == 'TRAIN' and episode['parent'] == spec.get('trajectory_response_pilot_parent')
             and panel['physical_parents'].get(episode['parent']) == 'TRAIN',
             'FIRST12_ONE_PREDECLARED_TRAIN_PARENT_NO_DEV_SELECTION')
    catalog_path = Path(spec['scene_allowlist'])
    need(digest(catalog_path) == spec['scene_allowlist_sha256'], 'EXACT_ORIGINAL_SCENE_CATALOG')
    catalog = json.loads(catalog_path.read_text())
    matches = [s for s in catalog['scenes'] if s['scene_id'] == episode['scene_id']]
    need(len(matches) == 1, 'ONE_REGISTERED_PANEL_SCENE')
    scene = matches[0]
    need(scene['parent'] == episode['parent'] and scene['split'] == episode['split']
         and scene['model_path'] == spec['model_path'] and scene['model_sha256'] == digest(spec['model_path'])
         and scene.get('static_check') == 'PASSED_NO_INTEGRATION'
         and catalog['native_binary_sha256'] == digest(spec['native_binary']),
         'EXACT_PHYSICAL_PARENT_SPLIT_STATIC_SCENE_AND_FROZEN_NATIVE')
    build_path = Path(spec['trajectory_trace_capacity_build'])
    need(digest(build_path) == spec['trajectory_trace_capacity_build_sha256'], 'EXACT_TRACE_BUILD_RECEIPT')
    build = json.loads(build_path.read_text())
    capacity=spec.get('trajectory_trace_capacity',262144)
    need(type(capacity) is int and capacity in (262144,524288),'EXPLICIT_RESPONSE_TRACE_CAPACITY')
    if capacity==524288:geometry.validate_trace_capacity_failure(spec)
    need(build.get('status') == f'TRACE_ONLY_{capacity}_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
         and build.get('trace_capacity') == capacity and build.get('new_target_integrations') == 0
         and build.get('trace_library_sha256') == digest(spec['trace_library'])
         and build.get('native_binary_sha256') == digest(spec['native_binary'])
         and build.get('observer_sha256') == digest(spec['observer_library'])
         and all(build.get('source_files', {}).get(name) == files[name] for name in (
             'src/go2arx5_vla/target_mujoco/init_trace.cpp', 'src/go2arx5_vla/target_mujoco/init_trace.hpp')),
         'UNCHANGED_NATIVE_AND_BYTE_PINNED_262144_TRACE_BUILD')
    inputs = geometry.load_query(spec)
    mapped = np.linalg.inv(inputs['frame_map']) @ inputs['query_tcp']
    make_program(inputs, episode, panel, mapped, mapped, [0.,0.,0.,0.,0.,0.,1.])
    return episode, panel, inputs


def episode_limit(spec, previous):
    episode, panel, _ = episode_config(spec)
    path = Path(spec['previous_stage'])
    need(digest(path) == spec['previous_stage_sha256'] and json.loads(path.read_text()) == previous,
         'EXACT_PREDECESSOR_RESULT_BYTES')
    root = path.parent.parent
    prior_spec_path = root/'EXECUTION_SPEC.json'
    prior_spec = json.loads(prior_spec_path.read_text())
    lease = json.loads((root/'control/LEASE.json').read_text())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_text())
    need(lease.get('status') == 'CLOSED' and lease.get('worker_exit_code') == worker.get('exit_code') == 0
         and lease.get('spec_sha256') == digest(prior_spec_path), 'ACTUAL_FULL_GEOMETRY_PREDECESSOR_CLOSED0')
    trace = previous.get('trace_audit') or {}
    report = previous.get('reference_qualification') or {}
    prior_episode = report.get('episode') or {}
    geometry.episode_contract(prior_episode)
    need(previous.get('status') == 'NATIVE_STAGE_PASSED' and previous.get('phase') == geometry.SLOW_PHASE
         and prior_spec.get('phase') == geometry.SLOW_PHASE
         and previous.get('shutdown_clean') is True and previous.get('failure') is None
         and previous.get('reached_exact_pause') is True and previous.get('physics_stops') == []
         and previous.get('original_joint_or_warning_samples') == 0
         and previous.get('native_outer_guard_rejection_count') == 0
         and previous.get('native_outer_guard_rejections') == {} and previous.get('trace_audit_error') is None
         and {p.get('name') for p in previous.get('owned_process_exits', [])} == {'sim','controller','solver','planner'}
         and all(p.get('exit_code') == 0 for p in previous['owned_process_exits'])
         and trace.get('status') == 'RECORDED_TRACE_CHECKS_PASS' and trace.get('errors') == []
         and trace.get('missing') == [] and trace.get('native_step_records') == previous.get('actual_target_integrations')
         and previous.get('candidate_index') == spec['candidate_index']
         and previous.get('campaign_uuid') == Path(spec['campaign_root']).name
         and previous.get('native_binary_sha256') == digest(spec['native_binary'])
         and previous.get('observer_sha256') == digest(spec['observer_library']),
         'COMPLETE_ORIGINAL_NATIVE_TRACE_SAFETY_AND_ALL_PROCESS_EXITS')
    need(prior_episode.get('geometry_scale') == 1. and prior_episode.get('time_stretch_factor') == 2
         and prior_episode.get('reference_rule') == geometry.SLOW_RECIPE
         and (report.get('actual_trace_binding') or {}).get('actual_trace_capacity') == spec.get('trajectory_trace_capacity',262144)
         and prior_episode.get('query_pins') == panel['query_pins']
         and report.get('all_40_knots_consumed') is True
         and report.get('complete_task_atoms') == report.get('solver_target_confirmed_atoms') == report.get('gripper_confirmed_atoms')
         and report.get('complete_task_atoms') == (previous['actual_target_integrations']-INIT_STEP)//25
         and previous['actual_target_integrations'] == prior_spec.get('max_ticks') == report.get('native_endpoint_step')
         and previous['actual_target_integrations'] == geometry.endpoint(prior_episode),
         'ACTUAL_FULL_GEOMETRY_1P0_REQUIRED_SMALL_SCALE_OR_FAILED_STAGE_CANNOT_ADMIT_RESPONSE')
    rows = report.get('windows', [])
    need(len(rows) == report['complete_task_atoms'], 'EVERY_PREDECESSOR_NATIVE_ATOM_EVIDENCED')
    for index, row in enumerate(rows):
        grip = row.get('gripper') or {}; submitted = grip.get('submitted') or {}
        memory = grip.get('consumed_wrapper') or {}; values = memory.get('values') or {}
        need(row.get('native_start_step') == INIT_STEP+25*index
             and row.get('native_end_step') == INIT_STEP+25*(index+1) and row.get('actual_ticks') == 25
             and bool(row.get('solver_policies_with_exact_installed_target'))
             and grip.get('command_consumed') is True
             and submitted.get('sequence') == values.get('sequence') == index
             and values.get('steps') == row['native_end_step']
             and memory.get('status') == 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
             and values.get('opening') == submitted.get('opening') and values.get('expiry') == submitted.get('expiry'),
             'COMPLETE_FULL_GEOMETRY_POSE_AND_GRIPPER_CONSUMPTION')
    return endpoint(episode)


def validate_install_preview(action, *, context, program, step, base, frozen_base, end_step):
    need(type(step) is int and INIT_STEP <= step < end_step and (step-INIT_STEP) % 25 == 0,
         'ORIGINAL_ATOMIC_BOUNDARY')
    base, frozen = np.asarray(base, float), np.asarray(frozen_base, float)
    need(base.shape == frozen.shape == (7,) and np.isfinite([base, frozen]).all()
         and np.allclose(base, frozen, rtol=0, atol=1e-10), 'UNCHANGED_MEASURED_BASE')
    need(isinstance(context, ContextProgram) and np.allclose(context.base_pose_hold, frozen, rtol=0, atol=1e-10),
         'ACTUAL_FIXED_INITIAL_CONTEXT')
    if step < DECISION_STEP:
        need(program is None, 'NO_THETA_PROGRAM_BEFORE_REAL_PROBE')
        expected = context.preview((step-INIT_STEP)//25)
    else:
        need(isinstance(program, ResponseProgram)
             and np.allclose(program.initial_reference_TCP_target, context.initial_tcp, rtol=0, atol=1e-12)
             and np.allclose(program.base_pose_hold, frozen, rtol=0, atol=1e-10),
             'ORIGINAL_INIT_REFERENCE_WITH_SEPARATE_ACTUAL_DECISION_ANCHOR')
        expected = program.preview((step-DECISION_STEP)//25)
    action = np.asarray(action, float)
    need(action.shape == (40,10) and np.isfinite(action).all()
         and np.allclose(action, expected, rtol=0, atol=1e-12), 'ENTIRE_CORRECTED_H40_PERSISTENT_SUFFIX')
    return dict(status='EXACT_RESPONSE_PREVIEW_NOT_NATIVE_QUALIFIED', native_step=step,
                points_checked=40, original_certify_chunk_required=True,
                original_install_ACK_required=True, physics_advance_authorized=False)
