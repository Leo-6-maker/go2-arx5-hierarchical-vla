"""Versioned qualification of query02's frozen proposal, using the shared adapter.

Historical query completion is evidence, not a current grant. AUTO1/P1 admission,
the original certifier/ACK and every actual native safety check remain required.
This is trajectory replay with explicit adaptation, not task success or training.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from . import fusion_trajectory_qualification_contract as geometry
from .fusion_trajectory_adapter import TrajectoryAdaptation
from .fusion_paused_query_archive import read_archive
from .storage import digest

PHASE = 'collect-target-fusion-live-proposal-qualification-v3'
SCHEMA = 'WM02_FUSION_LIVE_PROPOSAL_TRAJECTORY_EPISODE_V3'
RECIPE = 'FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_2X_V2'
PURPOSE = 'FROZEN_VLA_TRAJECTORY_REPLAY_EXECUTION_DIAGNOSTIC'
ROLE = 'VLA_PLUS_EXPLICIT_TRAJECTORY_ADAPTATION'
SCALES = (.25, .5, 1.)
INIT_STEP, DECISION_STEP, TAIL_ATOMS = 2503, 3703, 8
SCENE_ID, PARENT = 'F2T06', 'FUSION_F2_T06_ROOT_XY_V1'
FIRST_PREDECESSOR = '673c12772c2120af9a456569dea3599704cf7fd4a99fe25c1a181cee1bbc10dc'
FIRST_PREDECESSOR_SPEC = '9e62e4448a771d8584ba5b0b16b3b303021c57c6891e81eea4e3f9ee1669fafb'
FIRST_PREDECESSOR_SOURCE = '67faba025f906c6cb8e1c5c5a176a0cc3322bcec'
TRACE_BUILD = '79a628072d1c1ef03bdd81b4c39c1eaaa422d8760bb0ffbf873daaed53db9f67'
# Actual independently audited query02, not query01 or the old archived query.
PINS = dict(
    native_spec_sha256='286fe39b61c7f18a995058bd5b4d453048e7c00b819eb0a7471973bf872e17ac',
    gpu_spec_sha256='3d19a1fa0913424afa5adcaf386883d84520021de907d5b05ef8d33b4be8aac7',
    gpu_result_sha256='c4e3d4d0cb0464a79cc0e7ec902340b20faa0bceda967b9aa78f30782daabe5c',
    action_sha256='ab106c8604e0a989d9c67570eb6eb7c2cf53317476c37ed9eb841244ba002566',
    observation_sha256='5d581d90ea956aab6259e4479c526b7ddfcc307fa87f78695bce6f803fde911a',
    pair_sha256='ca839422004a70d0c84c129c8f80c8ce9bc16d78fd49069b87d5a165274c555d',
    native_result_sha256='42d4bf3ed8283f6c73997fd4cb8ee22ce4d0a315f9549f703a3971be81fd7887',
    source_sha='d069fb8715b81523f7b6373a75097eb036676744',
    checkpoint_manifest_sha256='94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38',
    checkpoint_file_inventory_sha256='f4c9486006de1a58721a1fc723361264e406ac4910ba46f14a31233c7a28fe23',
    task_sha256='83fb721af5ce7fcbb3efcaaa2f560496912eb34cdff2abe4eda5939ab36b8c90',
    frame_map_sha256='fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1',
    query_uuid='78a8dddd-d311-4d4e-988c-fa25926e7e5b', seed=366104)
SOURCE_MEMBERS = tuple(dict.fromkeys((*geometry.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_paused_query_archive.py',
    'src/go2arx5_vla/world_model/fusion_paused_observation.py',
    'src/go2arx5_vla/world_model/fusion_live_proposal_qualification_contract.py',
    'src/go2arx5_vla/world_model/fusion_live_proposal_qualification_native.py',
    'scripts/wm02/fusion_live_proposal_qualification_check.py')))


def need(value, reason):
    if not value:
        raise ValueError('LIVE_PROPOSAL_QUALIFICATION_' + reason)


def _json(path, pin=None):
    path = Path(path)
    raw = path.read_bytes()
    need(not path.is_symlink() and (pin is None or hashlib.sha256(raw).hexdigest() == pin),
         'EXACT_JSON_FILE_BYTES:' + path.name)
    return json.loads(raw)


def _fixed_episode():
    return dict(schema=SCHEMA, phase=PHASE, purpose=PURPOSE, experiment_role=ROLE, split='TRAIN',
        reference_rule=RECIPE, reference_anchor_mode='INIT_TCP_2503_REFERENCE_V1',
        prehold_atoms=40, screen_atom=48, screen=geometry.SCREEN, tail_hold_atoms=TAIL_ATOMS,
        desired_outcome_provenance='FROZEN_VLA_PROPOSAL_SEMANTICS_UNRESOLVED',
        physics_dt_s=.002, base_program='MEASURED_INIT_BASE_POSE_HOLD', time_stretch_factor=2,
        query_pins=dict(PINS), scene_id=SCENE_ID, parent=PARENT,
        full_task_success_claim=False, zero_shot_claim=False, online_closed_loop_claim=False,
        GR00T_updates=0, WM_updates=0, Actor_updates=0)


def episode_contract(episode):
    exact = _fixed_episode()
    need(isinstance(episode, dict) and set(episode) == set(exact) | {'run', 'geometry_scale', 'planned_end_atom'}
         and all(episode.get(k) == v for k, v in exact.items()), 'EXACT_V3_QUERY02_REPLAY_EPISODE')
    need(type(episode['time_stretch_factor']) is int
         and type(episode['geometry_scale']) in (int, float) and episode['geometry_scale'] in SCALES
         and isinstance(episode['run'], str) and Path(episode['run']).is_absolute(),
         'EXPLICIT_NEW_GRADE_AND_REGISTERED_RUN')
    endpoint(episode)
    return episode


def endpoint(episode):
    n = episode['planned_end_atom']
    need(type(n) is int and n >= 40, 'EXPLICIT_FULL40_INTEGER_CLOCK')
    end = DECISION_STEP + 25 * (n + TAIL_ATOMS)
    need(end <= 33000, 'FINITE_V3_NATIVE_ENDPOINT_AT_MOST_33000')
    return end


def load_query(spec):
    """Read completed historical query02; never call the live payload reader."""
    need(spec.get('live_proposal_query_pins') == PINS, 'EXACT_14_REGISTERED_QUERY02_PINS')
    archive = read_archive(Path(spec['live_proposal_native_spec']), Path(spec['live_proposal_gpu_spec']),
                           pins=dict(PINS))
    need(archive.get('status') == 'PAIR_COMPLETED_AND_PAUSE_REVALIDATED'
         and (archive.get('paired_return') or {}).get('established') is True
         and archive.get('external_pins') == PINS and archive.get('historical_read') is True
         and archive.get('current_native_pause_checked') is False
         and archive.get('native_execution_authorized') is False, 'ACTUAL_QUERY02_COMPLETED_HISTORICAL_PAIR_ONLY')
    body = unseal(archive['observation'])
    view = unseal(body['original_shadow_ABI'])['source_policy_view']
    need(body['world']['scene_sha256'] == digest(spec['model_path'])
         and body['task_sha256'] == PINS['task_sha256'], 'SAME_T06_QUERY_TASK_AND_SCENE')
    return dict(raw=archive['physical_H40'], query_tcp=matrix10(archive['payload']['state_eef10']),
        frame_map=np.asarray(view['T_source_from_target'], float), archive=archive,
        task=body['task'], task_sha256=PINS['task_sha256'],
        provenance=dict(query_pins=dict(PINS), archive_status=archive['status'], historical_read=True,
            original_live_query_completed=True, current_native_pause_checked=False,
            native_execution_authorized=False, source_outcomes_supplied=False,
            execution_condition='FROZEN_QUERY02_REFERENCE_REPLAY_NOT_NEW_ONLINE_QUERY',
            task_intent_sufficient_for_task_success=False))


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    episode_contract(episode)
    program = TrajectoryAdaptation(inputs['raw'], inputs['query_tcp'], inputs['frame_map'],
        decision_tcp, initial_tcp, base, episode['geometry_scale'], time_stretch_factor=2)
    need(program.end_atom == episode['planned_end_atom'], 'ACTUAL_RETIME_LENGTH_EQUALS_PREDECLARED_END')
    return program


def episode_config(spec):
    need(spec.get('phase') == PHASE and spec.get('profile') == 'wm02-auto1-target'
         and spec.get('domain') == 'WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('site_profile') == '4080-simulation' and spec.get('physics') is True
         and spec.get('scope') == 'WM02_PILOT1_SIM_ONLY' and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_P1_NATIVE_BACKEND_REQUIRED_NOT_A_GRANT')
    need(not any(spec.get(k) for k in ('fusion_binding', 'night1_binding', 'next1_binding', 'next2_binding',
        'parallel_batch_spec', 'standby_shutdown_diagnostic', 'reference_readiness_probe', 'diagnostic_debugger')),
         'SINGLE_WORLD_NO_FUSION_OR_OLD_SCOPE_BYPASS')
    need(spec.get('WBC_changes') is False and spec.get('physics_dt_s') == .002
         and type(spec.get('wall_limit_s')) in (int, float) and 0 < spec['wall_limit_s'] <= 1200,
         'FROZEN_WBC_2MS_FINITE_NATIVE_WALL')
    source, files = Path(spec['source_root']), spec.get('source_files', {})
    need(all(name in files and digest(source / name) == files[name] for name in SOURCE_MEMBERS),
         'NEW_V3_CAPSULE_AND_SHARED_ORIGINAL_HELPERS')
    episode = episode_contract(_json(spec['episode_manifest'], spec['episode_manifest_sha256']))
    need(episode['run'] == spec['registered_root'] and type(spec.get('max_ticks')) is int
         and spec['max_ticks'] == endpoint(episode)
         and spec.get('reference_binding_version') == 'MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'EXACT_REGISTERED_RUN_CLOCK_AND_INIT_REFERENCE_BINDING')
    catalog = _json(spec['scene_allowlist'], spec['scene_allowlist_sha256'])
    matches = [s for s in catalog['scenes'] if s['scene_id'] == SCENE_ID]
    need(len(matches) == 1 and matches[0]['parent'] == PARENT and matches[0]['split'] == 'TRAIN'
         and matches[0]['model_path'] == spec['model_path']
         and matches[0]['model_sha256'] == digest(spec['model_path'])
         and matches[0].get('static_check') == 'PASSED_NO_INTEGRATION'
         and catalog['native_binary_sha256'] == digest(spec['native_binary']), 'EXACT_STATIC_T06_AND_FROZEN_BINARY')
    need(spec.get('trajectory_trace_capacity') == 524288
         and type(spec.get('trajectory_trace_capacity')) is int
         and spec.get('trajectory_trace_capacity_build_sha256') == TRACE_BUILD,
         'EXACT_BUILD03_TRACE524_RECEIPT_PIN')
    geometry.validate_trace_capacity_failure(spec)
    build_path = Path(spec['trajectory_trace_capacity_build'])
    build = _json(build_path, TRACE_BUILD)
    need(build.get('status') == 'TRACE_ONLY_524288_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
         and build.get('trace_capacity') == 524288 and build.get('new_target_integrations') == 0
         and build.get('trace_library_sha256') == digest(spec['trace_library'])
         and build.get('native_binary_sha256') == digest(spec['native_binary'])
         and build.get('observer_sha256') == digest(spec['observer_library'])
         and all(build.get('source_files', {}).get(name) == files[name] for name in (
             'src/go2arx5_vla/target_mujoco/init_trace.cpp', 'src/go2arx5_vla/target_mujoco/init_trace.hpp')),
         'ACTUAL_TRACE_ONLY_BUILD03_UNCHANGED_NATIVE_BYTES')
    _closed(build_path.parent.parent)
    inputs = load_query(spec)
    mapped = np.linalg.inv(inputs['frame_map']) @ inputs['query_tcp']
    make_program(inputs, episode, mapped, mapped, [0., 0., 0., 0., 0., 0., 1.])
    return episode, inputs


def _closed(root):
    path = root / 'EXECUTION_SPEC.json'
    raw = path.read_bytes(); spec = json.loads(raw)
    lease, worker = _json(root / 'control/LEASE.json'), _json(root / 'WORKER_EXIT.private.json')
    need(spec.get('registered_root') == str(root) and lease.get('status') == 'CLOSED'
         and lease.get('run_uuid') == root.name and lease.get('worker_exit_code') == worker.get('exit_code') == 0
         and worker.get('failure') is None
         and lease.get('spec_sha256') == hashlib.sha256(raw).hexdigest(), 'ACTUAL_CLOSED0_SPEC_WORKER_JOIN')
    return spec, hashlib.sha256(raw).hexdigest()


def _complete_previous(spec, previous, prior_spec, report, expected_end):
    trace = previous.get('trace_audit') or {}; writer = trace.get('writer') or {}
    need(previous.get('status') == 'NATIVE_STAGE_PASSED' and previous.get('shutdown_clean') is True
         and previous.get('reached_exact_pause') is True and previous.get('failure') is None
         and previous.get('physics_stops') == [] and previous.get('original_joint_or_warning_samples') == 0
         and previous.get('native_outer_guard_rejection_count') == 0
         and previous.get('native_outer_guard_rejections') == {} and previous.get('trace_audit_error') is None
         and {p.get('name') for p in previous.get('owned_process_exits', [])} == {'sim','controller','solver','planner'}
         and all(p.get('exit_code') == 0 for p in previous['owned_process_exits'])
         and trace.get('status') == 'RECORDED_TRACE_CHECKS_PASS' and trace.get('errors') == [] and trace.get('missing') == []
         and writer.get('dropped') == 0 and writer.get('sealed_at_explicit_pause') is True
         and writer.get('stop_latched') is False and type(writer.get('written')) is int
         and type(writer.get('events')) is int and writer['written'] == writer['events'] >= expected_end
         and trace.get('native_step_records') == expected_end
         and previous.get('actual_target_integrations') == prior_spec.get('max_ticks') == report.get('native_endpoint_step') == expected_end
         and previous.get('source_sha') == prior_spec.get('source_sha')
         and previous.get('candidate_index') == spec['candidate_index']
         and previous.get('campaign_uuid') == Path(spec['campaign_root']).name
         and previous.get('native_binary_sha256') == digest(spec['native_binary'])
         and previous.get('observer_sha256') == digest(spec['observer_library'])
         and previous.get('scene_sha256') == digest(spec['model_path']), 'COMPLETE_PREDECESSOR_NATIVE_SAFETY_AND_TRACE')
    atoms = (expected_end - INIT_STEP) // 25
    need(report.get('all_40_knots_consumed') is True and report.get('raw_knot_consumed') == [True] * 40
         and report.get('complete_task_atoms') == report.get('solver_target_confirmed_atoms') == report.get('gripper_confirmed_atoms') == atoms
         and (report.get('actual_trace_binding') or {}).get('actual_trace_capacity') == 524288,
         'FULL40_ALL_POSE_AND_GRIPPER_CONSUMPTION_WITH_TRACE524')
    windows = report.get('windows') or []
    need(len(windows) == atoms, 'EVERY_PREDECESSOR_ATOM_RETAINED')
    for i, row in enumerate(windows):
        grip = row.get('gripper') or {}; sent = grip.get('submitted') or {}
        memory = grip.get('consumed_wrapper') or {}; values = memory.get('values') or {}
        pair = [sent.get('opening'), sent.get('expiry'), values.get('opening'), values.get('expiry')]
        source_time = sent.get('source_mpc_s')
        need(row.get('native_start_step') == INIT_STEP + 25*i and row.get('native_end_step') == INIT_STEP + 25*(i+1)
             and row.get('actual_ticks') == 25 and bool(row.get('solver_policies_with_exact_installed_target'))
             and grip.get('command_consumed') is True and sent.get('sequence') == values.get('sequence') == i
             and memory.get('status') == 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
             and values.get('initialized') is True and values.get('has_command') is True
             and values.get('steps') == row['native_end_step']
             and all(type(x) in (int, float) and np.isfinite(x) for x in pair)
             and type(source_time) in (int, float) and np.isfinite(source_time)
             and sent.get('native_start_step') == row['native_start_step']
             and abs(pair[0] - pair[2]) <= 1e-12 and abs(pair[1] - pair[3]) <= 1e-12
             and abs(pair[1] - source_time - .4) <= 1e-9
             and row['native_end_step'] * .002 <= pair[1] + 1e-9,
             'EXACT_PREDECESSOR_POSE_GRIPPER_ATOM:' + str(i))


def episode_limit(spec, previous):
    episode, _ = episode_config(spec)
    path = Path(spec['previous_stage'])
    need(_json(path, spec['previous_stage_sha256']) == previous, 'ACTUAL_PINNED_PREDECESSOR_RESULT')
    prior_spec, prior_pin = _closed(path.parent.parent)
    report = previous.get('reference_qualification') or {}; prior_episode = report.get('episode') or {}
    need(_json(prior_spec['episode_manifest'], prior_spec['episode_manifest_sha256']) == prior_episode,
         'PREDECESSOR_SPEC_EPISODE_REPORT_JOIN')
    index = SCALES.index(episode['geometry_scale'])
    if index == 0:
        need(spec['previous_stage_sha256'] == FIRST_PREDECESSOR and prior_pin == FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
             and previous.get('phase') == prior_spec.get('phase') == geometry.SLOW_PHASE,
             'FIRST_NEW_PROPOSAL_GRADE_REQUIRES_EXACT_OLD_FULL1_BOOTSTRAP_ONLY')
        geometry.episode_contract(prior_episode)
        need(prior_episode['geometry_scale'] == 1. and prior_episode['time_stretch_factor'] == 2,
             'COMPLETE_OLD_FULL1_IS_NOT_NEW_PROPOSAL_QUALIFICATION')
        expected_end = geometry.endpoint(prior_episode)
    else:
        episode_contract(prior_episode)
        need(previous.get('phase') == prior_spec.get('phase') == PHASE
             and prior_episode['geometry_scale'] == SCALES[index-1]
             and prior_episode['query_pins'] == episode['query_pins'],
             'IMMEDIATELY_PRECEDING_V3_SAME_QUERY_GRADE')
        expected_end = endpoint(prior_episode)
    need(prior_episode['scene_id'] == SCENE_ID and prior_episode['parent'] == PARENT,
         'SAME_T06_PHYSICAL_PARENT')
    _complete_previous(spec, previous, prior_spec, report, expected_end)
    return endpoint(episode)
