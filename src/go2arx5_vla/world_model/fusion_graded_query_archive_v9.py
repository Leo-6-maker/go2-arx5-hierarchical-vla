"""Read an externally pinned CLOSED V9 draw without promoting its native FAIL.

Only saved identities and pause proofs are checked. No current clock, lease
admission, model, FK, Processor, or original prefix replay is invoked. The
large prefix report is streamed for its hash once per unchanged file identity.
"""
import copy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, unseal, uuid_string, validate_observation
from scripts.pi0294.task_clock import require_pause
from scripts.pi0294.task_state import parse_state
from .fusion_observation import transform_hash
from .fusion_paused_observation import paused_inference_payload

SCHEMA = 'WM02_FUSION_GRADED_QUERY_ARCHIVE_V9'
PHASE = 'collect-target-fusion-task-phase-live-skill-v9'
RECIPE_PHASE = 'collect-target-fusion-task-phase-pose-slow-feedback-v9'
PAIR_SCHEMA = 'WM02_FUSION_TASK_PHASE_LIVE_SKILL_PAIR_V9'
PAIR_MODE = 'PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V9'
B, E, SEED = 34303, 66503, 366104
CHECKPOINT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
INVENTORY = 'f4c9486006de1a58721a1fc723361264e406ac4910ba46f14a31233c7a28fe23'
SOURCE_OBSERVATION = '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
SOURCE_CAMERA_IDENTITY = '0fe9090e469e7b864213b23c3b98b82bd2bd0162a2c4afc4cea7820536091858'
PIN_KEYS = {
    'native_spec_sha256', 'gpu_spec_sha256', 'gpu_result_sha256', 'action_sha256',
    'observation_sha256', 'pair_sha256', 'native_result_sha256', 'source_sha',
    'checkpoint_manifest_sha256', 'checkpoint_file_inventory_sha256',
    'task_sha256', 'frame_map_sha256', 'query_uuid', 'seed',
    'native_lease_sha256', 'native_worker_sha256', 'gpu_lease_sha256', 'gpu_worker_sha256',
    'episode_sha256', 'initialization_sha256', 'proposal_sha256', 'failure_sha256',
    'noise_sha256', 'source_observation_sha256', 'source_camera_identity_sha256',
}
SOURCE_MEMBERS = (
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_slow_feedback_v9.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_native_v9.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_native_v8.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_qualification_native.py',
    'src/go2arx5_vla/world_model/fusion_paused_observation.py',
    'src/go2arx5_vla/world_model/fusion_paused_inference.py',
    'src/go2arx5_vla/world_model/fusion_source_camera_capture.py',
    'src/go2arx5_vla/world_model/fusion_source_camera_rig.py',
    'scripts/wm02/infer_fusion_paused_vla.py', 'scripts/pi0294/resident.py',
)
DURATION_FAILURE = 'WHOLE_H40_AND_TAIL_MUST_FIT_PREREGISTERED_E_NO_PREFIX_SELECTION'


def _need(ok, reason):
    if not ok:
        raise ValueError('GRADED_QUERY_ARCHIVE_V9_' + reason)


def _hash(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def _path(value):
    path = Path(value)
    _need(path.is_absolute() and not path.is_symlink() and path.resolve() == path and path.is_file(), 'CANONICAL_FILE')
    return path


def _bytes(value, pin=None, cap=16*1024**2):
    path = _path(value)
    _need(path.stat().st_size <= cap, 'BOUNDED_SMALL_ARTIFACT')
    raw = path.read_bytes()
    _need(pin is None or hashlib.sha256(raw).hexdigest() == pin, 'EXACT_FILE_BYTES_'+path.name)
    return raw


def _read(value, pin=None):
    return json.loads(_bytes(value, pin))


def _stat(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@lru_cache(maxsize=8)
def _stream_pin(value, pin, identity):
    path = _path(value)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    _need(_stat(path) == identity and digest.hexdigest() == pin, 'EXACT_IMMUTABLE_PREFIX_HASH')
    return identity[2]


def _closed(spec, side, pin, pins, code):
    root = Path(spec['registered_root'])
    _need(spec['lease'] == str(root/'control/LEASE.json'), 'CANONICAL_CLOSED_LEASE_PATH')
    lease = _read(spec['lease'], pins[side+'_lease_sha256'])
    worker = _read(root/'WORKER_EXIT.private.json', pins[side+'_worker_sha256'])
    _need(lease.get('status') == 'CLOSED' and lease.get('spec_sha256') == pin
          and lease.get('run_uuid') == root.name and lease.get('source_sha') == spec['source_sha']
          and lease.get('owner') == spec['owner'] and lease.get('scope') == spec['scope']
          and type(lease.get('worker_exit_code')) is int and lease['worker_exit_code'] == code
          and type(worker.get('exit_code')) is int and worker['exit_code'] == code
          and worker.get('failure') is None, 'EXACT_CLOSED_'+side)
    return dict(exit_code=code, lease_sha256=pins[side+'_lease_sha256'], worker_sha256=pins[side+'_worker_sha256'])


def _binding(record, payload, shadow, pins):
    body = unseal(record)
    value = dict(query_uuid=body['query_uuid'], snapshot_uuid=body['snapshot_uuid'],
        observation_sha256=pins['observation_sha256'], sealed_body_sha256=record['sha256'],
        task_sha256=body['task_sha256'], task=body['task'], world=body['world'],
        native_step=body['native_step'], native_time_s=body['native_time_s'], files=body['files'],
        renderer=body['renderer'], input_mode=body['input_mode'], mapped_payload_sha256=document_hash(payload),
        target_camera_provenance=shadow['provenance'],
        source_policy_view={k:v for k,v in shadow['source_policy_view'].items() if k != 'payload'},
        snapshot_binding_mode=PAIR_MODE, pause_before=body['pause_before'], pause_after_capture=body['pause_after'],
        native_pause_after_query='NOT_CHECKED_BY_GPU_PROCESS', native_caller_must_verify_still_paused=True,
        native_execution_authorized=False, target_embodiment_qualified=False, target_task_qualified=False,
        online_closed_loop=False, source_outcomes_supplied=False,
        producer_phase=PHASE, experiment_role='GRADED_NONCONTACT_DIAGNOSTIC',
        physical_recipe_phase=RECIPE_PHASE, initialization_recipe_phase=RECIPE_PHASE,
        initialization_geometry_scale=.5, native_boundary_step=B, native_endpoint_step=E,
        task_condition='SOURCE71558_ORIGINAL_FULL_TASK_NO_PARAPHRASE',
        capture_condition='PRESEAL_WBC_ORIGINAL_QUOTA_ACK_PAUSED',
        capture_camera_recipe='SIM_EXTERNAL_WORLD_SOURCE_RIG',
        source_camera_identity_sha256=pins['source_camera_identity_sha256'],
        returned_VLA_action_installations=0, camera_equivalence=False,
        old_initialization_readiness_relabelled=False, source_task_phase_equivalence=False,
        native_trace_seal_checked_by_GPU=False)
    return value


def read_archive(native_spec_path, gpu_spec_path, *, pins):
    """Resolve one exact historical V9 duration failure plus its valid GPU draw."""
    _need(isinstance(pins, dict) and set(pins) == PIN_KEYS, 'EXACT_EXTERNAL_25_PIN_DESCRIPTOR')
    _need(all(_hash(pins[k]) for k in PIN_KEYS-{'source_sha','query_uuid','seed'})
          and _hash(pins['source_sha'], 40) and type(pins['seed']) is int and pins['seed'] == SEED,
          'EXTERNAL_PIN_TYPES_AND_MATCHED_SEED')
    uuid_string(pins['query_uuid'])
    _need(pins['checkpoint_manifest_sha256'] == CHECKPOINT and pins['checkpoint_file_inventory_sha256'] == INVENTORY
          and pins['source_observation_sha256'] == SOURCE_OBSERVATION
          and pins['source_camera_identity_sha256'] == SOURCE_CAMERA_IDENTITY, 'FROZEN_ASSET_IDENTITIES')
    npth, gpth = _path(native_spec_path), _path(gpu_spec_path)
    n, g = _read(npth, pins['native_spec_sha256']), _read(gpth, pins['gpu_spec_sha256'])
    nr, gr = Path(n['registered_root']), Path(g['registered_root'])
    _need(npth == nr/'EXECUTION_SPEC.json' and gpth == gr/'EXECUTION_SPEC.json' and nr != gr
          and not nr.is_relative_to(gr) and not gr.is_relative_to(nr)
          and n['source_sha'] == g['source_sha'] == pins['source_sha'] and n['source_root'] == g['source_root']
          and n['owner'] == g['owner'], 'SAME_ARCHIVED_SOURCE_AND_DISTINCT_ROOTS')
    closed = dict(native=_closed(n, 'native', pins['native_spec_sha256'], pins, 1),
                  gpu=_closed(g, 'gpu', pins['gpu_spec_sha256'], pins, 0))
    _need(n.get('phase') == PHASE and n.get('kind') == 'FUSION_TASK_PHASE_LIVE_SKILL_V9'
          and n.get('max_ticks') == E and n.get('live_skill_start_step') == B
          and n.get('physics') is True and n.get('physics_dt_s') == .002 and n.get('WBC_changes') is False
          and g.get('kind') == 'FROZEN_VLA_LIVE_PAUSED_QUERY' and g.get('physics') is False
          and g.get('profile') == 'wm02-fusion-vla-shadow' and g.get('device') == 'cuda'
          and g.get('execution_site') == 'CORE_FUSION_NEURAL'
          and g.get('inference_mode') == 'ACTUAL_LIVE_PAUSED_VLA_INFERENCE'
          and g.get('snapshot_binding_mode') == PAIR_MODE and type(g.get('max_queries')) is int and g['max_queries'] == 1
          and g.get('seed') == SEED and g.get('query_uuid') == pins['query_uuid']
          and g.get('numeric_mode') == 'ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
          and g.get('checkpoint_manifest_sha256') == CHECKPOINT and document_hash(g['checkpoint_files']) == INVENTORY,
          'EXACT_RECORDED_V9_COMPOSITE_AND_ONE_FROZEN_DRAW')
    files = dict(n['source_files'])
    _need(all(k not in files or files[k] == v for k,v in g['source_files'].items()), 'SAME_RECORDED_MEMBERS')
    files.update(g['source_files'])
    for member in SOURCE_MEMBERS:
        _bytes(Path(n['source_root'])/member, files[member])
    episode = _read(n['episode_manifest'], pins['episode_sha256'])
    _need(n['episode_manifest_sha256'] == pins['episode_sha256'] and episode.get('run') == str(nr)
          and episode.get('phase') == RECIPE_PHASE and episode.get('geometry_scale') == .5
          and episode.get('ramp_atoms') == 768 and episode.get('planned_end_atom') == 1216
          and episode.get('feedback_decision_native_steps') == [23103+900*i for i in range(12)], 'ORIGINAL_HALF_PREFIX_IDENTITY')
    pair_path = nr/'QUERY_PAIR.private.json'
    _need(n.get('paused_query_pair') == g.get('paused_query_pair') == str(pair_path)
          and n.get('paused_query_pair_sha256') == g.get('paused_query_pair_sha256') == pins['pair_sha256']
          and n.get('paused_query_gpu_spec') == str(gpth) and g.get('producer_native_spec') == str(npth)
          and g.get('producer_native_spec_sha256') == pins['native_spec_sha256'], 'ORIGINAL_PAIR_SPEC_BACKREFERENCES')
    pair = _read(pair_path, pins['pair_sha256'])
    _need(pair.get('schema') == PAIR_SCHEMA and pair.get('producer_phase') == PHASE
          and pair.get('native_root') == str(nr) and pair.get('gpu_root') == str(gr)
          and pair.get('native_run_id') == 'native-01' and pair.get('native_boundary_step') == B
          and pair.get('native_endpoint_step') == E and pair.get('capture_index') == 0
          and pair.get('max_queries') == 1 and pair.get('native_integrations_by_bridge') == 0
          and all(pair.get(k) == pins[k] for k in ('source_sha','query_uuid','seed','task_sha256','frame_map_sha256','checkpoint_manifest_sha256'))
          and pair.get('source_camera_identity_sha256') == SOURCE_CAMERA_IDENTITY
          and pair.get('camera_recipe') == 'SIM_EXTERNAL_WORLD_SOURCE_RIG'
          and pair.get('task_condition') == 'SOURCE71558_ORIGINAL_FULL_TASK_NO_PARAPHRASE', 'EXACT_HISTORICAL_V9_PAIR')
    path = nr/'native-01/PAUSED_OBSERVATIONS'/pins['query_uuid']/'PAUSED_OBSERVATION.private.json'
    _need(g.get('recorded_observation') == str(path), 'EXACT_ARCHIVED_CAPTURE_PATH')
    record = _read(path, pins['observation_sha256']); body = unseal(record)
    payload = paused_inference_payload(record); shadow = unseal(body['original_shadow_ABI'])
    sensor = shadow['target_observation'].get('sensor_adapter') or {}
    _need(sensor.get('source_identity_audit_sha256') == SOURCE_CAMERA_IDENTITY
          and sensor.get('frame_map_sha256') == pins['frame_map_sha256'], 'EXPLICIT_RETAINED_SOURCE_SENSOR')
    names = {'STATE.private.bin','BOUND_SHADOW_ABI.private.json','images/000000-calibration.json',
             'images/000000-body_rgb.png','images/000000-wrist_rgb.png',
             'native-camera/images/000000-calibration.json','native-camera/images/000000-body_rgb.png',
             'native-camera/images/000000-wrist_rgb.png'}
    _need(set(body['files']) == names and body['capture']['index'] == 0
          and body['query_uuid'] == pins['query_uuid'] and body['native_step'] == B
          and body['task_sha256'] == pins['task_sha256'] and abs(body['native_time_s']-B*.002) <= 1e-8
          and body['world']['execution_spec_sha256'] == pins['native_spec_sha256']
          and body['world']['native_output'] == str(nr/'native-01')
          and body['world']['scene_sha256'] == pair['scene_sha256']
          and transform_hash(n['paused_source_from_target']) == pins['frame_map_sha256'], 'EIGHT_MEMBER_ABSOLUTE_B_CAPTURE')
    retained = {name:_bytes(path.parent/name, pin) for name,pin in body['files'].items()}
    _need(json.loads(retained['BOUND_SHADOW_ABI.private.json']) == body['original_shadow_ABI'], 'NESTED_SHADOW_FILE')
    packet = parse_state(retained['STATE.private.bin'])
    _need(packet['step'] == B and packet['time_s'] == body['native_time_s'], 'EXACT_B_RAW_STATE')
    _bytes(nr/'native-01/state.bin', body['files']['STATE.private.bin'])
    source = unseal(_read(n['task_phase_source_observation'], SOURCE_OBSERVATION))['payload']
    validate_observation(source)
    task = _read(n['paused_target_task'], n['paused_target_task_file_sha256'])
    score = _read(n['paused_score_contract'], n['paused_score_contract_file_sha256'])
    _need(source['language'] == payload['language'] == task['language'] and task == body['task']
          and document_hash(task) == n['paused_target_task_sha256'] == g['paused_task_sha256'] == pins['task_sha256']
          and document_hash(score) == task['score_contract_sha256'], 'ORIGINAL71558_VERBATIM_TASK_AND_SCORE')
    native = _read(nr/'native-01/RESULT.json', pins['native_result_sha256'])
    _need(native.get('status') == 'NATIVE_STAGE_FAILED_PRESERVED' and native.get('phase') == PHASE
          and native.get('source_sha') == pins['source_sha'] and native.get('actual_target_integrations') == B
          and native.get('complete_atomic_transitions') == 1372 and native.get('shutdown_clean') is True
          and DURATION_FAILURE in (native.get('failure') or ''), 'PRESERVE_ORIGINAL_NATIVE_DURATION_FAIL')
    prefix_path = _path(nr/'native-01/LIVE_SKILL_INITIALIZATION_STAGE.private.json')
    _need(prefix_path.stat().st_size <= 128*1024**2, 'BOUNDED_PREFIX_REPORT')
    prefix_bytes = _stream_pin(str(prefix_path), pins['initialization_sha256'], _stat(prefix_path))
    proposal = _read(nr/'native-01/LIVE_SKILL_GENERATED_PROPOSAL.private.json', pins['proposal_sha256'])
    failure = _read(nr/'native-01/LIVE_SKILL_FAILED.private.json', pins['failure_sha256'])
    _need(proposal.get('schema') == 'WM02_ACTUAL_GENERATED_LIVE_PROPOSAL_V9'
          and proposal.get('native_step') == B and proposal.get('pre_effect') is True
          and proposal.get('action_npz_sha256') == pins['action_sha256'] and proposal.get('raw_knots') == 40
          and proposal.get('clips') == proposal.get('original_Processor_extra_decodes') == 0
          and failure.get('schema') == 'WM02_GRADED_LIVE_SKILL_FAILURE_V9'
          and failure.get('phase') == PHASE and failure.get('native_step') == B
          and failure.get('proposal_saved') is True and DURATION_FAILURE in (failure.get('failure') or '')
          and failure.get('query') == proposal.get('query'), 'SAVED_PROPOSAL_AND_FAILURE_SAME_RETURN')
    result = _read(gr/'fit-01/RESULT.private.json', pins['gpu_result_sha256'])
    expected = _binding(record, payload, shadow, pins)
    _need(result.get('observation_binding') == expected == proposal.get('observation_identity')
          and result.get('schema') == 'WM02_FUSION_LIVE_PAUSED_INFERENCE_RESULT_V1'
          and result.get('status') == 'REAL_FROZEN_VLA_LIVE_PAUSED_QUERY_DONE'
          and all(result.get(k) == pins[k] for k in ('source_sha','query_uuid','seed','checkpoint_manifest_sha256',
                    'checkpoint_file_inventory_sha256','pair_sha256','action_sha256'))
          and result.get('execution_spec_sha256') == pins['gpu_spec_sha256']
          and result.get('producer_native_spec_sha256') == pins['native_spec_sha256']
          and result.get('recorded_observation_sha256') == pins['observation_sha256']
          and result.get('actual_model_loads') == result.get('actual_VLA_inferences') == 1
          and result.get('native_integrations') == result.get('GR00T_updates') == result.get('WM_updates') == result.get('Actor_updates') == 0
          and result.get('numeric_mode') == g['numeric_mode'] and result.get('inference_mode') == g['inference_mode']
          and result.get('native_execution_authorized') is False and result.get('returned_VLA_action_installations') == 0,
          'EXACT_SAVED_ONE_QUERY_BINDING')
    precision, noise = result.get('precision') or {}, result.get('noise') or {}
    _need(precision.get('mode') == 'ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST'
          and precision.get('bf16_autocast') is True and precision.get('cuda_matmul_allow_tf32') is False
          and noise.get('shape') == [1,40,132] and noise.get('dtype') == 'torch.float32'
          and noise.get('sha256') == pins['noise_sha256'], 'ORIGINAL_PRECISION_AND_NOISE')
    with np.load(io.BytesIO(_bytes(gr/'fit-01/REAL_VLA_ACTION.npz', pins['action_sha256'])), allow_pickle=False) as arrays:
        _need(set(arrays.files) == {'physical_H40','normalized'} and arrays['physical_H40'].shape == (40,10)
              and arrays['normalized'].shape == (1,40,132)
              and arrays['physical_H40'].dtype.kind == 'f' and arrays['normalized'].dtype == np.dtype('float32')
              and all(np.isfinite(arrays[k]).all() for k in arrays.files)
              and np.array_equal(arrays['physical_H40'], proposal['original_decoded_physical_H40']), 'FULL_ORIGINAL_DECODED_ACTION')
    query = proposal['query']; checked = query.get('query_result') or {}; pause = checked.get('original_pause_revalidation') or {}
    _need(query.get('schema') == PAIR_SCHEMA and query.get('query_uuid') == pins['query_uuid']
          and query.get('pair_sha256') == pins['pair_sha256'] and query.get('gpu_spec_sha256') == pins['gpu_spec_sha256']
          and query.get('capture_sha256') == pins['observation_sha256'] and query.get('native_integrations_by_bridge') == 0
          and query.get('native_execution_authorized') is False and query.get('online_action_execution') is False
          and checked.get('schema') == result['schema']
          and checked.get('query_uuid') == pins['query_uuid'] and checked.get('result_sha256') == pins['gpu_result_sha256']
          and checked.get('action_sha256') == pins['action_sha256'] and checked.get('model_result_checked') is True
          and checked.get('native_execution_authorized') is False and checked.get('original_certificate_and_ACK_still_required') is True,
          'ORIGINAL_NATIVE_RECORDED_PAIR_RETURN')
    _need(pause.get('schema') == body['schema'] and pause.get('query_uuid') == pins['query_uuid']
          and pause.get('native_step') == B and pause.get('native_time_s') == body['native_time_s']
          and pause.get('unchanged_measured_packet_sha256') == body['files']['STATE.private.bin']
          and pause.get('native_execution_authorized') is False and pause.get('model_result_checked') is False,
          'SAME_ARCHIVED_PACKET_AFTER_QUERY')
    require_pause(body['pause_after'], pause['pause_after_query'], B)
    provenance = dict(external_pins=copy.deepcopy(pins), closed_workers=closed,
        native_status=native['status'], native_failure_preserved=native['failure'], native_endpoint_step=B,
        original_task=copy.deepcopy(task), original_score=copy.deepcopy(score),
        original_task_sha256=pins['task_sha256'], capture_sha256=pins['observation_sha256'],
        capture_members=copy.deepcopy(body['files']), actual_capture=copy.deepcopy(body['capture']),
        world=copy.deepcopy(body['world']), original_shadow_provenance=copy.deepcopy(shadow['provenance']),
        observation_binding=copy.deepcopy(expected), paired_return=copy.deepcopy(query),
        existing_original_query=dict(gpu_result_sha256=pins['gpu_result_sha256'], action_npz_sha256=pins['action_sha256'],
            proposal_sha256=pins['proposal_sha256'], noise_sha256=pins['noise_sha256'], new_draw_required=False),
        initialization_sha256=pins['initialization_sha256'], initialization_bytes=prefix_bytes,
        prefix_windows_parsed=False, prefix_math_replayed=False, native_qualification_established=False,
        recorded_source_members_verified=len(SOURCE_MEMBERS), checkpoint_bytes_reread=False,
        historical_read=True, current_deadline_validation_performed=False, current_native_pause_checked=False,
        native_execution_authorized=False, target_task_success=False, new_queries=0, new_integrations=0, processor_decodes=0)
    return dict(schema=SCHEMA, status='CLOSED_V9_DURATION_FAIL_WITH_VALID_SAVED_QUERY_AND_PAUSE_RETURN',
                payload=copy.deepcopy(payload), provenance=provenance, external_pins=copy.deepcopy(pins))
