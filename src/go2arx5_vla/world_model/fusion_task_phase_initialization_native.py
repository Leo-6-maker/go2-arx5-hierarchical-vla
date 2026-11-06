"""Thin task-phase calibration on the unchanged original native consumption loop.

At the real 3703 pause, record the original ActualState FK witness and freeze a
separate cube-derived goal binding before any generated trajectory is installed.
The end diagnostic consumes nine real samples; it is never a permission grant.
"""
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.pi0294.task_state import parse_state
from . import fusion_task_phase_initialization as recipe
from .fusion_task_phase_initialization_contract import PHASE, need
from .storage import digest

GOAL_SCHEMA = 'WM02_NATIVE_TASK_PHASE_GOAL_BINDING_V1'
REPORT_SCHEMA = 'WM02_NATIVE_TASK_PHASE_INITIALIZATION_V1'


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _verify_actual_fk(adapter, packet, measured, *, expected_step=recipe.DECISION_STEP):
    """Join the original reader's current FK data to the exact saved packet.

    No forward or step is performed here. TargetSequentialAdapter's unchanged
    measured_task_state has already called original ActualState.update under
    forbid_integration. Its current data and poses must match this witness.
    """
    from scripts.pi0294.task_state import ActualState
    state = getattr(adapter, 'actual_state', None)
    need(type(state) is ActualState, 'ORIGINAL_ACTUALSTATE_READER_REQUIRED')
    need((state.model.nq, state.model.nv, state.model.nu) == (34, 32, 20)
         and digest(adapter.spec['model_path']) == recipe.SCENE,
         'ORIGINAL_SCENE_AND_NATIVE_STATE_ABI')
    need(type(expected_step) is int and measured['native_step'] == packet['step'] == expected_step
         and measured['time_s'] == state.data.time == packet['time_s'],
         'ACTUAL_FK_EXACT_DECISION_CLOCK')
    for key in ('qpos', 'qvel', 'ctrl'):
        need(np.array_equal(getattr(state.data, key), packet[key])
             and np.array_equal(measured[key], packet[key]), 'ACTUAL_FK_PACKET_' + key)
    for key, name in (('tcp', 'tool'), ('object_pose', 'cube')):
        need(np.array_equal(state.pose(name), measured[key]), 'ACTUAL_FK_MEASURED_' + key)
    return dict(schema='WM02_ORIGINAL_ACTUALSTATE_SAME_PACKET_FK_WITNESS_V1',
        status='ORIGINAL_ACTUALSTATE_DATA_AND_FK_MATCH_SAVED_PACKET',
        reader='scripts.pi0294.task_state.ActualState.update',
        reader_sha256=adapter.spec['source_files']['scripts/pi0294/task_state.py'],
        model_path=adapter.spec['model_path'], model_sha256=recipe.SCENE,
        native_step=packet['step'], native_time_s=packet['time_s'],
        packet_qpos_qvel_ctrl_exact=True, original_reader_instance_retained=True,
        original_forward_only_reader=True, wrapper_mj_forward_calls=0,
        wrapper_mj_step_calls=0, wrapper_model_loads=0, models_or_queries=0,
        object_truth_use='CALIBRATION_GOAL_AND_DIAGNOSTIC_ONLY_NOT_POLICY_INPUT_OR_EXPERT_LABEL')


def validate_adapter_install(adapter, action):
    need(adapter.spec.get('phase') == PHASE and hasattr(adapter, 'trajectory_initial_tcp'),
         'NEW_PHASE_AND_ORIGINAL_MEASURED_INIT_REQUIRED')
    return recipe.validate_install_preview(action, initial_tcp=adapter.trajectory_initial_tcp,
        program=adapter.trajectory_program, step=adapter.observe()['step'],
        base=adapter.base_pose_hold, frozen_base=adapter.trajectory_initial_base,
        end_step=adapter.spec['max_ticks'])


def run(adapter):
    from .fusion_task_phase_initialization_contract import episode_config
    episode, source = episode_config(adapter.spec)
    return run_initialization_program(adapter, episode, source)


def run_initialization_program(adapter, episode, source, *, recipe_module=recipe, actual_phase=PHASE,
                               report_schema=REPORT_SCHEMA, goal_schema=GOAL_SCHEMA,
                               reference_calibration_factory=None, report_context=None,
                               before_preview=None, report_extension=None, continuation=None):
    """Run a checked grade; only an explicit V8 caller may continue before seal."""
    from scripts.pi0292.durable import write
    from .fusion_trajectory_qualification_native import run_program
    recipe_module.episode_contract(episode)
    boundary = recipe_module.endpoint(episode)
    registered_end = adapter.spec['max_ticks']
    continuation_bound = (continuation is not None
        and (actual_phase,adapter.spec.get('kind')) in (
            ('collect-target-fusion-task-phase-live-skill-v8','FUSION_TASK_PHASE_LIVE_SKILL_V8'),
            ('collect-target-fusion-task-phase-live-skill-v9','FUSION_TASK_PHASE_LIVE_SKILL_V9'))
        and adapter.spec.get('live_skill_start_step') == boundary
        and type(registered_end) is int and boundary < registered_end <= 66503
        and (registered_end-boundary) % 25 == 0)
    need(adapter.spec.get('phase') == actual_phase
         and ((continuation is None and registered_end == boundary) or continuation_bound),
         'ACTUAL_NEW_PHASE_AND_STATIC_ENDPOINT')
    need(set(source) == {'source_observation_bytes', 'T_source_from_target', 'scene_sha256'},
         'EXACT_FROZEN_SOURCE_CONFIGURATION')
    paths = {name: adapter.output / name for name in (
        'TASK_PHASE_GOAL_BINDING.private.json', 'TASK_PHASE_DECISION_STATE.bin',
        'TASK_PHASE_DECISION_MEASURED.private.json', 'TASK_PHASE_HOLD_SAMPLES.private.json',
        'TASK_PHASE_READINESS.private.json')}
    need(not any(path.exists() or path.is_symlink() for path in paths.values()),
         'FRESH_RUNTIME_GOAL_BINDING_ONLY_NO_OVERWRITE')
    inputs, goal_receipt = {}, {}
    actual_tail = deque(maxlen=9)

    def record_sample(sample):
        if actual_tail:
            need(sample['native_step'] == actual_tail[-1]['native_step'] + recipe.TICKS_PER_ATOM,
                 'REAL_CONTIGUOUS_PER_ATOM_READINESS_SAMPLES')
        actual_tail.append(copy.deepcopy(sample))

    def factory(destination, actual_episode, initial_tcp, decision_tcp, base):
        need(destination is inputs and actual_episode is episode and not inputs,
             'ONE_RUNTIME_GOAL_BINDING_AT_FIRST_DECISION_ONLY')
        state_path = adapter.output / 'state.bin'
        raw = state_path.read_bytes()
        packet = parse_state(raw)
        original_reader = getattr(adapter, 'actual_state', None)
        measured = adapter.measured_task_state()
        need(getattr(adapter, 'actual_state', None) is original_reader
             and np.array_equal(measured['tcp'], decision_tcp),
             'UNCHANGED_ORIGINAL_READER_AND_DECISION_TCP')
        witness = _verify_actual_fk(adapter, packet, measured)
        loaded = recipe.load_inputs(**source, decision_packet_bytes=raw,
                                    measured_decision_state=measured)
        need(state_path.read_bytes() == raw and adapter.observe()['step'] == recipe.DECISION_STEP,
             'ORIGINAL_PACKET_UNCHANGED_DURING_GOAL_FREEZE')
        # Separate runtime evidence: do not edit the preregistered static episode
        # or replace original state.bin. Save before the shared loop installs.
        saved_packet = paths['TASK_PHASE_DECISION_STATE.bin']
        with saved_packet.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        saved_state = paths['TASK_PHASE_DECISION_MEASURED.private.json']
        write(saved_state, _plain(measured))
        goal_path = paths['TASK_PHASE_GOAL_BINDING.private.json']
        calibrated_program, reference_binding = None, None
        if reference_calibration_factory is not None:
            calibrated_program, reference_binding = reference_calibration_factory(
                loaded, actual_episode, initial_tcp, decision_tcp, base)
        goal = dict(schema=goal_schema, episode_sha256=adapter.spec['episode_manifest_sha256'],
            actual_spec_phase=actual_phase, actual_spec_source_sha=adapter.spec['source_sha'],
            frozen_input=loaded['frozen'], actual_fk_witness=witness,
            original_packet_path=str(state_path), saved_packet_path=str(saved_packet),
            saved_packet_sha256=digest(saved_packet), measured_state_path=str(saved_state),
            measured_state_sha256=digest(saved_state),
            decision_step=recipe.DECISION_STEP, frozen_before_generated_reference_install=True,
            experiment_role='NON_VLA_CALIBRATION', actual_VLA_commands=0, VLA_queries=0,
            source_pose_is_expert_action_label=False, object_and_goal_are_policy_inputs=False,
            native_execution_authorized_by_binding=False)
        if reference_binding is not None:
            goal['frozen_predecision_reference_calibration'] = reference_binding
        write(goal_path, goal)
        need(json.loads(goal_path.read_bytes()) == goal and saved_packet.read_bytes() == raw
             and digest(saved_packet) == loaded['provenance']['decision_packet_sha256']
             and state_path.read_bytes() == raw and adapter.observe()['step'] == recipe.DECISION_STEP,
             'SAVED_READBACK_GOAL_AND_PACKET_BINDING_BEFORE_INSTALL')
        goal_receipt.update(goal_binding_path=str(goal_path), goal_binding_sha256=digest(goal_path),
            goal_binding_body_sha256=loaded['frozen']['sha256'],
            decision_packet_sha256=digest(saved_packet), actual_fk_witness=witness)
        if reference_binding is not None:
            goal_receipt['frozen_predecision_reference_calibration'] = reference_binding
        inputs.update(loaded)
        return (recipe_module.make_program(inputs, episode, initial_tcp, decision_tcp, base)
                if calibrated_program is None else calibrated_program)

    def finalizer(report):
        need(bool(goal_receipt) and len(actual_tail) == 9,
             'ACTUAL_SAVED_GOAL_AND_NINE_DISTINCT_ENDPOINT_SAMPLES')
        samples_path = paths['TASK_PHASE_HOLD_SAMPLES.private.json']
        samples = _plain(list(actual_tail))
        write(samples_path, samples)
        saved_samples = json.loads(samples_path.read_bytes())
        need(saved_samples == samples, 'NINE_REAL_COMPLETE_SAMPLES_SAVED_READBACK')
        readiness = recipe_module.query_readiness(saved_samples, episode, inputs)
        path = paths['TASK_PHASE_READINESS.private.json']
        write(path, readiness)
        need(json.loads(path.read_bytes()) == readiness, 'ACTUAL_READINESS_SAVED_READBACK')
        report['task_phase_initialization'] = dict(goal_receipt,
            readiness_path=str(path), readiness_sha256=digest(path), readiness=readiness,
            hold_samples_path=str(samples_path), hold_samples_sha256=digest(samples_path),
            readiness_sample_native_steps=[s['native_step'] for s in actual_tail],
            experiment_role='NON_VLA_CALIBRATION', desired_outcome_provenance='CALIBRATION_INTENT',
            actual_VLA_commands=0, VLA_queries=0, actual_VLA_action_installations=0,
            object_and_goal_are_policy_inputs=False, source_pose_is_expert_action_label=False,
            query_readiness_is_native_or_GPU_grant=False, generated_calibration_waypoints=40,
            raw_VLA_knots=0, full_task_success=False)
        if report_extension is not None:
            report_extension(report)

    context = dict(schema=report_schema, actual_spec_phase=actual_phase,
            experiment_role='NON_VLA_CALIBRATION', desired_outcome_provenance='CALIBRATION_INTENT',
            generated_reference_is_GR00T_action=False, source_pose_is_expert_action_label=False,
            object_and_goal_are_policy_inputs=False, actual_VLA_commands=0, VLA_commands=0,
            VLA_reference_replays=0, VLA_queries=0, original_Processor_decodes=0,
            calibration_goal='ACTUAL_3703_CUBE_CENTER_PLUS_WORLD_Z_0P14M_SOURCE_ROTATION',
            query_readiness_is_permission=False)
    if report_context is not None:
        context.update(report_context)
    return run_program(adapter, episode, inputs, program_factory=factory,
        report_context=context, report_finalizer=finalizer,
        sample_observer=record_sample, before_preview=before_preview, continuation=continuation)
