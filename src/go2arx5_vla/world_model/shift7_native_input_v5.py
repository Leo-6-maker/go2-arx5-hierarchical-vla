"""Prepare SHIFT7 input on an already owned paused adapter, without motion.

Candidate only. This does not construct/start an adapter, dispatch a model,
install an action, resume physics or grant READMISSION. Its caller still needs
an independently admitted native scope and a separately authorized GPU scope.
"""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.durable import write
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0294.e1.artifacts import atomic_npz
from .close_reference_input_v1 import (
    SHIFT7_METHOD, SHIFT7_NATIVE_VERSION, SHIFT7_SIM_VERSION, query_reference, reference_arrays,
)
from .sim_paused_contract_v5 import SELECTED_SHIFT7_MANIFEST_SHA256
from .storage import digest
from .verified_observation_v1 import VerifiedObservation
from .hybrid_nonphysical_v3 import read


def _identity(adapter):
    spec = adapter.spec
    if spec.get('mobile_experiment_version')=='MFREE_SHIFT7_SIM_ONLY_V52':
        from .mobile_scope_v52 import query_identity
        query_identity(spec)
        spec_path=Path(spec['registered_root'])/'EXECUTION_SPEC.json'
        if read(spec_path)!=spec or not Path(adapter.output).resolve().is_relative_to(Path(spec['registered_root']).resolve()):
            raise ValueError('SAME_OWNED_V52_SCOPE_SPEC_AND_OUTPUT_REQUIRED')
        return dict(path=str(spec_path),sha256=digest(spec_path))
    sim_only = spec.get('mobile_experiment_version') == SHIFT7_SIM_VERSION
    max_ticks = 35703 if sim_only else 30000
    if (spec.get('mobile_experiment_version') not in (SHIFT7_NATIVE_VERSION, SHIFT7_SIM_VERSION)
            or spec['mobile_condition'].get('method') != SHIFT7_METHOD
            or spec['mobile_condition'].get('mode') != 'M_FREE'
            or spec['mobile_condition'].get('family') not in ('F1', 'F2')
            or spec.get('execution_clock_profile') != 'SIM_PAUSED_NATIVE'
            or spec.get('SHIFT7_package', {}).get('sha256') != SELECTED_SHIFT7_MANIFEST_SHA256
            or type(spec.get('max_ticks')) is not int or not 0 < spec['max_ticks'] <= max_ticks):
        raise ValueError('NEW_PAUSED_SHIFT7_SINGLE_MFREE_IDENTITY_REQUIRED')
    spec_path = Path(spec['registered_root'])/'EXECUTION_SPEC.json'
    if (read(spec_path) != spec
            or not Path(adapter.output).resolve().is_relative_to(Path(spec['registered_root']).resolve())):
        raise ValueError('SAME_OWNED_NATIVE_SCOPE_SPEC_AND_OUTPUT_REQUIRED')
    return dict(path=str(spec_path), sha256=digest(spec_path))


def capture(adapter, sample, camera, fixed_task_frame, accepted_reference_TCP, query_index):
    """Export a whitelisted current input using the existing native pause ABI.

    Object/Teacher data in scoring-side sample dictionaries are never serialized
    into the model input. The accepted-reference matrix is compiler metadata;
    it is not a claim about the controller's selected MPC policy.
    """
    from .fusion_paused_observation import _pause, _after_pause
    from .fusion_observation import rigid
    scope_pin = _identity(adapter)
    if type(query_index) is not int or query_index < 0:
        raise ValueError('NONNEGATIVE_UNIQUE_QUERY_INDEX')
    before = _pause(adapter)  # Real packet, ACK, quota, physics/FSM/leg clocks.
    packet = before['packet']
    if (sample['native_step'] != packet['step']
            or abs(sample['time_s'] - packet['time_s']) > 1e-8
            or any(not np.array_equal(sample[k], packet[k]) for k in ('qpos', 'qvel', 'ctrl'))):
        raise ValueError('CURRENT_SAMPLE_AND_NATIVE_PACKET_CLOCK_MUST_MATCH')
    available_prefix_atoms = min(10, (adapter.spec['max_ticks'] - packet['step']) // 25)
    if available_prefix_atoms <= 0:
        raise ValueError('NO_COMPLETE_NATIVE_ATOM_LEFT_WITHIN_TOTAL_EPISODE_CAP')
    previous = query_reference(adapter, packet, sample)
    frame = rigid(adapter.spec['policy_from_world'])
    task_frame = rigid(fixed_task_frame)
    accepted = rigid(accepted_reference_TCP)
    # The task frame must have been fixed at episode initialization, rather
    # than recomputed from the moving base or an object pose on each query.
    if not np.array_equal(task_frame, np.asarray(adapter.shift7_fixed_task_frame)):
        raise ValueError('EPISODE_FIXED_TASK_FRAME_CHANGED')
    if (Path(camera.model_path) != Path(adapter.spec['model_path'])
            or camera.model_sha != adapter.spec['frozen_files'][adapter.spec['model_path']]):
        raise ValueError('SAME_FROZEN_SCENE_CAMERA_SESSION_REQUIRED')
    adapter.guard()
    images, body, _, calibration = camera.render_memory(packet)
    body, wrist = body.copy(), images['wrist_rgb'].copy()
    available = time.monotonic()
    for image in (body, wrist):
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError('ORIGINAL_NATIVE_BODY_WRIST_UINT8_256_REQUIRED')
    state = pose10(frame @ sample['tcp'], sample['gripper_fraction'])
    if not np.allclose(matrix10(state), frame @ sample['tcp'], rtol=0, atol=1e-9):
        raise ValueError('ONE_OBSERVABLE_EEF_FRAME_MAPPING')
    folder = Path(adapter.output)/'shift7-queries'/('%04d' % query_index)
    folder.mkdir(parents=True, exist_ok=False)
    output = folder/'INPUT.npz'
    atomic_npz(output, state_eef10=state, body_rgb=body, wrist_rgb=wrist,
               T_policy_from_mount=task_frame, policy_from_world=frame,
               accepted_reference_TCP=accepted, **reference_arrays(previous))
    pin = dict(path=str(output), sha256=digest(output))
    observation = VerifiedObservation.from_file(pin)
    after = _after_pause(adapter, before)
    if (_identity(adapter) != scope_pin
            or query_reference(adapter, after['packet'], sample) != previous):
        raise ValueError('SCOPE_OR_EMITTED_HISTORY_CHANGED_DURING_CAPTURE')
    # Copy/hash the exact arrays retained above, not a mutable renderer alias.
    data = observation.values_for(pin)
    record = dict(schema='MFREE_SHIFT7_NATIVE_INPUT_CANDIDATE_V5',
        query_index=query_index, input=pin,
        native_scope_spec=scope_pin, native_output=str(adapter.output),
        source_kind='CALLER_OWNED_NATIVE_PAUSE_ABI_NOT_A_NATIVE_ADMISSION',
        native_step=packet['step'], native_time_s=packet['time_s'],
        available_prefix_atoms=available_prefix_atoms, max_total_episode_ticks=adapter.spec['max_ticks'],
        observation_available_monotonic=available,
        input_ready_monotonic=time.monotonic(),
        native_state_sha256=hashlib.sha256(before['raw']).hexdigest(),
        step_quota_sha256=hashlib.sha256(before['quota']).hexdigest(),
        pause_before=before['pause'], pause_after=after['pause'],
        previous_reference=copy.deepcopy(previous),
        previous_reference_basis='CONFIRMED_COMPLETE_NATIVE_ATOM_COMMAND_NOT_MEASURED_APERTURE',
        model_fields=['state_eef10', 'body_rgb', 'wrist_rgb', 'T_policy_from_mount'],
        provider_context_fields=['previous_reference'],
        compiler_only_fields=['policy_from_world', 'accepted_reference_TCP'],
        array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()},
        camera_calibration=calibration, teacher_or_object_truth_in_model_input=False,
        controller_selected_reference_known=False, model_generations_requested=0,
        native_execution_admitted=False, physics_integrations_requested=0)
    write(folder/'INPUT_META.private.json', record)
    return dict(observation=observation, previous_reference=copy.deepcopy(previous),
                metadata=record, pause_snapshot=before)


def verify_query_return(adapter, prepared):
    """Read existing feedback again after any separate external model wait."""
    from .fusion_paused_observation import _pause, _after_pause, _same_pause
    scope_pin = _identity(adapter)
    record = prepared['metadata']
    if scope_pin != record['native_scope_spec'] or str(adapter.output) != record['native_output']:
        raise ValueError('EPISODE_OR_NATIVE_SCOPE_CHANGED_DURING_QUERY')
    adapter.guard()
    current = _pause(adapter)
    _same_pause(prepared['pause_snapshot'], current)
    after = _after_pause(adapter, current)
    if (query_reference(adapter, after['packet'], dict(native_step=after['packet']['step']))
            != prepared['previous_reference']
            or digest(record['input']['path']) != record['input']['sha256']):
        raise ValueError('INPUT_OR_PREVIOUS_ACTUALLY_EMITTED_COMMAND_CHANGED_DURING_QUERY')
    return dict(unchanged_native_step=after['packet']['step'],
        observation_age_wall_s=time.monotonic()-record['observation_available_monotonic'],
        native_execution_admitted=False, physics_integrations_requested=0)
