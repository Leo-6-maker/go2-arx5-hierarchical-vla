"""Read-only bindings for one frozen query on an owned paused capture.

The GPU process reads retained files, never the simulator. Only the actual
native owner can revalidate its pause after the query, before using a result.
"""
import json
from pathlib import Path
import re

import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, unseal, uuid_string
from scripts.pi0294.task_state import parse_state
from .fusion_paused_observation import INPUT_MODE, paused_inference_payload, verify_still_paused
from .storage import digest

KIND = 'FROZEN_VLA_LIVE_PAUSED_QUERY'
MODE = 'ACTUAL_LIVE_PAUSED_VLA_INFERENCE'
PROFILE = 'wm02-fusion-vla-shadow'
BINDING_MODE = 'SEALED_SNAPSHOT_EXACT_SHA'
PAIR_BINDING_MODE = 'PRE_REGISTERED_SINGLE_SNAPSHOT_V1'
POSTURE_PAIR_BINDING_MODE = 'PRE_REGISTERED_SOURCE_POSTURE_SNAPSHOT_V2'
LIVE_SKILL_PAIR_BINDING_MODE = 'PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V8'
SLOW_LIVE_SKILL_PAIR_BINDING_MODE = 'PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V9'
LIVE_SKILL_PAIR_BINDING_MODES = (LIVE_SKILL_PAIR_BINDING_MODE,SLOW_LIVE_SKILL_PAIR_BINDING_MODE)
PAIR_BINDING_MODES = (PAIR_BINDING_MODE, POSTURE_PAIR_BINDING_MODE, *LIVE_SKILL_PAIR_BINDING_MODES)
RESULT_SCHEMA = 'WM02_FUSION_LIVE_PAUSED_INFERENCE_RESULT_V1'
CHECKPOINT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
# Exact full-file inventory of the actually delivered CORR source/target draws.
# Every member is reread before model loading, including all saved processors.
CHECKPOINT_INVENTORY = 'f4c9486006de1a58721a1fc723361264e406ac4910ba46f14a31233c7a28fe23'
SOURCE_MEMBERS = (
    'scripts/wm02/infer_fusion_paused_vla.py',
    'src/go2arx5_vla/world_model/fusion_paused_inference.py',
    'src/go2arx5_vla/world_model/fusion_paused_query_bridge.py',
    'src/go2arx5_vla/world_model/fusion_paused_observation.py',
    'src/go2arx5_vla/world_model/fusion_observation.py',
    'src/go2arx5_vla/world_model/fusion_scope.py',
    'src/go2arx5_vla/world_model/runtime.py',
    'scripts/wm02/pilot1_neural_worker.py',
    'scripts/pi0294/resident.py', 'scripts/pi0294/e1/model.py',
    'scripts/pi0294/local_backbone.py', 'scripts/pi0294/task_state.py',
    'scripts/pi0292/relay_protocol.py', 'scripts/pi0292/live_inputs.py',
    'scripts/pi0292/train_adapter.py')


def need(condition, reason):
    if not condition:
        raise ValueError('FUSION_PAUSED_INFERENCE_' + reason)


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_spec(spec):
    """Cheap exact identity checks, safe before CUDA import or a load intent."""
    need(spec.get('kind') == KIND and spec.get('profile') == PROFILE
         and spec.get('execution_site') == 'CORE_FUSION_NEURAL'
         and spec.get('inference_mode') == MODE and spec.get('observation_mode') == INPUT_MODE
         and spec.get('physics') is False and spec.get('device') == 'cuda'
         and type(spec.get('max_queries')) is int and spec['max_queries'] == 1
         and spec.get('numeric_mode') == 'ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
         and spec.get('GR00T_updates') == spec.get('WM_updates') == spec.get('Actor_updates') == 0,
         'EXACT_ONE_FROZEN_GPU_QUERY_NO_PHYSICS_OR_UPDATES')
    need(spec.get('checkpoint_manifest_sha256') == CHECKPOINT
         and isinstance(spec.get('checkpoint_files'), dict)
         and document_hash(spec['checkpoint_files']) == CHECKPOINT_INVENTORY
         and all(not Path(name).is_absolute() and '..' not in Path(name).parts
                 and _hash(pin) for name, pin in spec['checkpoint_files'].items()),
         'EXACT_CORR2048_FULL_WEIGHT_AND_PROCESSOR_INVENTORY')
    need(type(spec.get('seed')) is int and 0 <= spec['seed'] < 2**63
         and _hash(spec.get('paused_task_sha256'))
         and isinstance(spec.get('source_sha'), str) and re.fullmatch('[0-9a-f]{40}',spec['source_sha'])
         and isinstance(spec.get('query_uuid'), str), 'PREDECLARED_SEED_QUERY_AND_INPUT_IDENTITY')
    uuid_string(spec['query_uuid'])
    if spec.get('snapshot_binding_mode') == BINDING_MODE:
        need(_hash(spec.get('recorded_observation_sha256')), 'PREEXISTING_EXACT_SNAPSHOT_SHA')
    elif spec.get('snapshot_binding_mode') in PAIR_BINDING_MODES:
        from .fusion_paused_query_bridge import validate_pair_spec
        validate_pair_spec(spec)
    else:
        need(False, 'EXPLICIT_EXACT_SNAPSHOT_OR_PREREGISTERED_PAIR_MODE')
    source = Path(spec['source_root']); files = spec.get('source_files', {})
    need(all(name in files and digest(source/name) == files[name] for name in SOURCE_MEMBERS),
         'NEW_CAPSULE_AND_ORIGINAL_INFERENCE_PRIMITIVE_BYTES')


def read_payload(spec):
    """Verify every versioned actual capture member and return the original ABI."""
    validate_spec(spec)
    path = Path(spec['recorded_observation'])
    observation_hash = digest(path)
    need(path.name == 'PAUSED_OBSERVATION.private.json' and not path.is_symlink()
         and (spec['snapshot_binding_mode'] in PAIR_BINDING_MODES
              or observation_hash == spec['recorded_observation_sha256']), 'EXACT_SEALED_PAUSED_ENVELOPE_BYTES')
    record = json.loads(path.read_text()); body = unseal(record)
    payload = paused_inference_payload(record)
    shadow = unseal(body['original_shadow_ABI'])
    source_sensor = shadow['target_observation'].get('sensor_adapter')
    need(source_sensor is None or spec['snapshot_binding_mode'] in (
             POSTURE_PAIR_BINDING_MODE,*LIVE_SKILL_PAIR_BINDING_MODES),
         'SOURCE_CAMERA_SENSOR_REQUIRES_EXPLICIT_POSTURE_PAIR_V2')
    if spec['snapshot_binding_mode'] in PAIR_BINDING_MODES:
        from .fusion_paused_query_bridge import validate_pair_capture
        pair = validate_pair_capture(spec, record)
    root = path.parent
    need(root.name == body['query_uuid'] == spec['query_uuid']
         and root.parent.name == 'PAUSED_OBSERVATIONS'
         and root.parent.parent == Path(body['world']['native_output'])
         and body['task_sha256'] == spec['paused_task_sha256']
         and not (root/'CAPTURE_FAILED.private.json').exists(), 'EXACT_QUERY_TASK_AND_SUCCESSFUL_CAPTURE_DIRECTORY')
    index = body['capture']['index']
    expected = {'STATE.private.bin', 'BOUND_SHADOW_ABI.private.json',
                f'images/{index:06d}-calibration.json',
                *(f'images/{index:06d}-{camera}.png' for camera in ('body_rgb','wrist_rgb'))}
    if source_sensor is not None:
        expected |= {f'native-camera/images/{index:06d}-{suffix}' for suffix in
                     ('body_rgb.png','wrist_rgb.png','calibration.json')}
    need(set(body['files']) == expected, 'EXACT_VERSIONED_CAPTURE_MEMBERS_NO_DROPPED_SIDECAR'
         if source_sensor is not None else 'EXACT_FIVE_CAPTURE_MEMBERS_NO_DROPPED_SIDECAR')
    for name, pin in body['files'].items():
        member = root/name
        need(not member.is_symlink() and member.resolve().is_relative_to(root.resolve())
             and digest(member) == pin, 'ACTUAL_CAPTURE_MEMBER_CHANGED:'+name)
    nested = json.loads((root/'BOUND_SHADOW_ABI.private.json').read_text())
    need(nested == body['original_shadow_ABI'], 'NESTED_SHADOW_FILE_DIFFERS_FROM_ENVELOPE')
    raw = parse_state((root/'STATE.private.bin').read_bytes())
    need(raw['step'] == body['native_step'] and raw['time_s'] == body['native_time_s']
         and abs(raw['time_s']-raw['step']*.002) <= 1e-8, 'ORIGINAL_ABSOLUTE_RAW_STATE_CLOCK_NO_RETIMING')
    identity = dict(query_uuid=body['query_uuid'], snapshot_uuid=body['snapshot_uuid'],
        observation_sha256=observation_hash, sealed_body_sha256=record['sha256'],
        task_sha256=body['task_sha256'], task=body['task'], world=body['world'],
        native_step=body['native_step'], native_time_s=body['native_time_s'],
        files=body['files'], renderer=body['renderer'], input_mode=body['input_mode'],
        mapped_payload_sha256=document_hash(payload), target_camera_provenance=shadow['provenance'],
        source_policy_view={k: v for k, v in shadow['source_policy_view'].items() if k != 'payload'},
        snapshot_binding_mode=spec['snapshot_binding_mode'], pause_before=body['pause_before'],
        pause_after_capture=body['pause_after'], native_pause_after_query='NOT_CHECKED_BY_GPU_PROCESS',
        native_caller_must_verify_still_paused=True, native_execution_authorized=False,
        target_embodiment_qualified=False, target_task_qualified=False,
        online_closed_loop=False, source_outcomes_supplied=False)
    if spec['snapshot_binding_mode'] == POSTURE_PAIR_BINDING_MODE:
        identity.update(producer_phase='collect-target-fusion-source-posture-query-v3',
            physical_recipe_phase='collect-target-fusion-source-posture-qualification-v1',
            capture_condition='FULL_SOURCE_POSTURE_AFTER_CANCEL_HOLD_AND_TRACE_SEAL',
            capture_camera_recipe=pair['camera_recipe'],
            source_camera_identity_sha256=pair['source_camera_identity_sha256'],
            returned_VLA_action_installations=0,camera_equivalence=False)
    elif spec['snapshot_binding_mode'] in LIVE_SKILL_PAIR_BINDING_MODES:
        from .fusion_paused_query_bridge import _live_pair_module
        module = _live_pair_module(spec['snapshot_binding_mode'])
        initialization = ('collect-target-fusion-task-phase-pose-slow-feedback-v9'
            if spec['snapshot_binding_mode']==SLOW_LIVE_SKILL_PAIR_BINDING_MODE
            else 'collect-target-fusion-task-phase-pose-continuation-v7')
        identity.update(producer_phase=module.PHASE, experiment_role=module.EXPERIMENT_ROLE,
            physical_recipe_phase=initialization,
            initialization_recipe_phase=initialization,
            initialization_geometry_scale=.5, native_boundary_step=pair['native_boundary_step'],
            native_endpoint_step=pair['native_endpoint_step'], task_condition=pair['task_condition'],
            capture_condition='PRESEAL_WBC_ORIGINAL_QUOTA_ACK_PAUSED',
            capture_camera_recipe=pair['camera_recipe'],
            source_camera_identity_sha256=pair['source_camera_identity_sha256'],
            returned_VLA_action_installations=0, camera_equivalence=False,
            old_initialization_readiness_relabelled=False, source_task_phase_equivalence=False,
            native_trace_seal_checked_by_GPU=False)
    return payload, identity, record


def verify_paused_query_return(adapter, spec, result_path):
    """Native-owner readback only; this never certifies or installs an action."""
    if spec['snapshot_binding_mode'] in LIVE_SKILL_PAIR_BINDING_MODES:
        from .fusion_paused_query_bridge import _live_pair_module
        _live_pair_module(spec['snapshot_binding_mode']).validate_native_pause(adapter)
    _, identity, record = read_payload(spec)
    result_path = Path(result_path)
    need(result_path == Path(spec['registered_root'])/'fit-01/RESULT.private.json'
         and not result_path.is_symlink(), 'EXACT_REGISTERED_GPU_RESULT_PATH')
    result = json.loads(result_path.read_text())
    need(result.get('schema') == RESULT_SCHEMA
         and result.get('status') == 'REAL_FROZEN_VLA_LIVE_PAUSED_QUERY_DONE'
         and result.get('query_uuid') == identity['query_uuid']
         and result.get('recorded_observation_sha256') == identity['observation_sha256']
         and result.get('source_sha') == spec['source_sha']
         and result.get('execution_spec_sha256') == digest(Path(spec['registered_root'])/'EXECUTION_SPEC.json')
         and result.get('pair_sha256') == spec.get('paused_query_pair_sha256')
         and result.get('producer_native_spec_sha256') == spec.get('producer_native_spec_sha256')
         and result.get('inference_mode') == MODE and result.get('numeric_mode') == spec['numeric_mode']
         and result.get('checkpoint_manifest_sha256') == CHECKPOINT
         and result.get('checkpoint_file_inventory_sha256') == CHECKPOINT_INVENTORY
         and result.get('seed') == spec['seed'] and result.get('actual_VLA_inferences') == 1
         and result.get('native_integrations') == result.get('GR00T_updates')
             == result.get('WM_updates') == result.get('Actor_updates') == 0
         and result.get('actual_model_loads') == 1
         and result.get('native_execution_authorized') is False
         and result.get('observation_binding') == identity,
         'RESULT_MUST_BIND_THIS_EXACT_PAUSED_QUERY')
    action = result_path.parent/'REAL_VLA_ACTION.npz'
    need(digest(action) == result['action_sha256'], 'ACTUAL_RETURNED_FULL_ACTION_BYTES')
    with np.load(action, allow_pickle=False) as arrays:
        physical = arrays['physical_H40']
        normalized = arrays['normalized']
        need(physical.shape == (40,10) and np.isfinite(physical).all()
             and normalized.shape == (1,40,132) and np.isfinite(normalized).all(), 'COMPLETE_FINITE_H40_RETURN')
    # A successful GPU result alone does not prove that its native world is
    # still the captured world. This is the existing actual-owner barrier.
    pause = verify_still_paused(adapter, record)
    if spec['snapshot_binding_mode'] in LIVE_SKILL_PAIR_BINDING_MODES:
        _live_pair_module(spec['snapshot_binding_mode']).validate_native_pause(adapter)
    return dict(schema=RESULT_SCHEMA, query_uuid=identity['query_uuid'],
        result_sha256=digest(result_path), action_sha256=result['action_sha256'],
        original_pause_revalidation=pause, model_result_checked=True,
        native_execution_authorized=False, original_certificate_and_ACK_still_required=True)
