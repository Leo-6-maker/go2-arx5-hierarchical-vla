"""One preregistered query at a graded, unsealed native initialization boundary.

This is a file/pause binding, not an initialization or action certificate. The
actual V9 producer composes a slower V9 half-grade manifest with a fixed
future endpoint. Old full-grade/readiness results are never relabelled here.
No registration, model call, Processor decode, integration or install occurs.
"""
import hashlib
import json
from pathlib import Path

from scripts.pi0292.relay_protocol import digest as document_hash, unseal, validate_observation
from .storage import digest

PHASE = 'collect-target-fusion-task-phase-live-skill-v9'
KIND = 'FUSION_TASK_PHASE_LIVE_SKILL_V9'
PAIR_MODE = 'PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V9'
PAIR_SCHEMA = 'WM02_FUSION_TASK_PHASE_LIVE_SKILL_PAIR_V9'
BOUNDARY_STEP, ENDPOINT_STEP = 34303, 66503
EXPERIMENT_ROLE = 'GRADED_NONCONTACT_DIAGNOSTIC'
TASK_CONDITION = 'SOURCE71558_ORIGINAL_FULL_TASK_NO_PARAPHRASE'
SOURCE_CAMERA_RECIPE = 'SIM_EXTERNAL_WORLD_SOURCE_RIG'
SOURCE_CAMERA_IDENTITY = '0fe9090e469e7b864213b23c3b98b82bd2bd0162a2c4afc4cea7820536091858'
FRAME_MAP = 'fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1'
SOURCE_OBSERVATION = '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
SOURCE_PAYLOAD = '867e229119dcade9183e0090705492e80429c6c32474d633b2d5fe389b42bb96'
PAIR_EXTRA_KEYS = {'native_endpoint_step', 'producer_phase', 'camera_recipe',
                   'source_camera_identity_sha256', 'task_condition'}
SOURCE_MEMBERS = (
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_slow_feedback_v9.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_live_skill_pair_v9.py',
    'src/go2arx5_vla/world_model/fusion_source_camera_capture.py',
    'src/go2arx5_vla/world_model/fusion_source_camera_rig.py')


def need(value, reason):
    if not value:
        raise ValueError('TASK_PHASE_LIVE_SKILL_PAIR_V9_' + reason)


def _bytes(path, pin):
    path = Path(path)
    need(path.is_absolute() and not path.is_symlink() and path.resolve() == path,
         'CANONICAL_PINNED_INPUT_PATH')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == pin, 'PINNED_INPUT_BYTES_' + path.name)
    return raw


def make_camera_recipe(producer):
    """Construct the original typed source-rig declaration; no rendering/FK."""
    from .fusion_source_camera_capture import SourceCameraCapture
    from .fusion_observation import transform_hash
    need(producer.get('source_camera_identity_sha256') == SOURCE_CAMERA_IDENTITY
         and producer.get('paused_frame_map_sha256') == FRAME_MAP
         and transform_hash(producer['paused_source_from_target']) == FRAME_MAP,
         'EXACT_SOURCE_SENSOR_AUDIT_AND_FC347_FRAME')
    return SourceCameraCapture(_bytes(producer['source_camera_identity'], SOURCE_CAMERA_IDENTITY),
                               producer['paused_source_from_target'])


def _task(producer):
    need(producer.get('task_phase_source_observation_sha256') == SOURCE_OBSERVATION,
         'EXACT_ORIGINAL_SOURCE_OBSERVATION')
    source = unseal(json.loads(_bytes(producer['task_phase_source_observation'],
                                     SOURCE_OBSERVATION)))['payload']
    validate_observation(source)
    need(document_hash(source) == SOURCE_PAYLOAD, 'EXACT_ORIGINAL_SOURCE_POLICY_PAYLOAD')
    task = json.loads(_bytes(producer['paused_target_task'],
                             producer['paused_target_task_file_sha256']))
    need(document_hash(task) == producer['paused_target_task_sha256']
         and task.get('scene_sha256') == digest(producer['model_path'])
         and task.get('language') == source['language'],
         'SOURCE71558_FULL_TASK_LANGUAGE_UNCHANGED')


def validate_pair_spec(spec):
    """Validate a V9 pair without importing a future native wrapper/contract."""
    from . import fusion_paused_query_bridge as bridge
    from . import fusion_task_phase_pose_slow_feedback_v9 as initialization
    pair, producer = bridge._validate_pair_spec(spec, pair_schema=PAIR_SCHEMA,
        binding_mode=PAIR_MODE, boundary_step=BOUNDARY_STEP,
        producer_endpoint_step=ENDPOINT_STEP, pair_keys=bridge.KEYS | PAIR_EXTRA_KEYS)
    need(pair['producer_phase'] == producer.get('phase') == PHASE
         and producer.get('kind') == KIND
         and producer.get('snapshot_binding_mode') == PAIR_MODE
         and producer.get('profile') == 'wm02-auto1-target'
         and producer.get('scope') == 'WM02_PILOT1_SIM_ONLY'
         and not any(producer.get(key) for key in ('fusion_binding', 'night1_binding',
             'next1_binding', 'next2_binding'))
         and type(pair['native_endpoint_step']) is int
         and pair['native_endpoint_step'] == producer.get('max_ticks') == ENDPOINT_STEP
         and type(producer.get('live_skill_start_step')) is int
         and producer['live_skill_start_step'] == BOUNDARY_STEP
         and pair['capture_index'] == 0
         and pair['task_condition'] == producer.get('task_condition') == TASK_CONDITION
         and producer.get('experiment_role') == EXPERIMENT_ROLE
         and type(producer.get('live_skill_time_stretch_factor')) is int
         and producer['live_skill_time_stretch_factor'] in (1,2)
         and producer.get('physics_dt_s') == .002 and producer.get('WBC_changes') is False,
         'EXACT_GRADED_DIAGNOSTIC_V9_PRODUCER_AND_DISTINCT_B_E')
    need(pair['camera_recipe'] == SOURCE_CAMERA_RECIPE
         and producer.get('capture_camera_recipe') == SOURCE_CAMERA_RECIPE
         and pair['source_camera_identity_sha256'] == SOURCE_CAMERA_IDENTITY
         and pair['frame_map_sha256'] == FRAME_MAP
         and not any(producer.get(key) for key in ('camera_override','paused_camera_override',
             'posture_camera_override','source_camera_rig')),
         'EXACT_EXPLICIT_SOURCE_SENSOR_NO_SILENT_OVERRIDE')
    need(type(spec.get('wall_limit_s')) in (int, float)
         and 0 < spec['wall_limit_s'] <= 1200,
         'FINITE_ONE_QUERY_WALL_AT_MOST_1200_SECONDS')
    # The manifest retains its real V9 slow-feedback phase. The actual spec is LIVE V9.
    episode = json.loads(_bytes(producer['episode_manifest'], producer['episode_manifest_sha256']))
    initialization.episode_contract(episode)
    need(episode['run'] == producer['registered_root'] and episode['geometry_scale'] == .5
         and initialization.endpoint(episode) == BOUNDARY_STEP
         and episode['scene_sha256'] == pair['scene_sha256']
         and episode['frame_map_sha256'] == FRAME_MAP
         and episode['source_observation_sha256'] == SOURCE_OBSERVATION,
         'EXACT_V9_SLOW_HALF_INITIALIZATION_MANIFEST_NOT_FULL_GRADE')
    for member in SOURCE_MEMBERS:
        pin = producer.get('source_files', {}).get(member)
        need(pin is not None and spec.get('source_files', {}).get(member) == pin
             and digest(Path(producer['source_root']) / member) == pin,
             'PINNED_NEW_PAIR_AND_ORIGINAL_SOURCE_SENSOR_' + member)
    make_camera_recipe(producer)
    _task(producer)
    return pair, producer


def validate_capture_camera(pair, shadow):
    sensor = shadow['target_observation'].get('sensor_adapter')
    need(pair['camera_recipe'] == SOURCE_CAMERA_RECIPE
         and isinstance(sensor, dict)
         and sensor.get('schema') == 'WM02_FUSION_SOURCE_CAMERA_CAPTURE_V1'
         and sensor.get('status') == 'ACTUAL_TARGET_PACKET_CPU_RENDERED'
         and sensor.get('rig') == SOURCE_CAMERA_RECIPE
         and sensor.get('source_identity_audit_sha256') == pair['source_camera_identity_sha256']
             == SOURCE_CAMERA_IDENTITY
         and sensor.get('frame_map_sha256') == pair['frame_map_sha256'] == FRAME_MAP,
         'ACTUAL_TARGET_PACKET_DECLARED_SOURCE_RIG_WITH_NATIVE_SIDECARS')


def validate_native_pause(adapter):
    """Native-owner check alongside the original raw-packet/quota/fresh ACK ABI."""
    mode = (getattr(adapter, 'latest', {}) or {}).get('mode')
    need(adapter.spec.get('phase') == PHASE and adapter.spec.get('kind') == KIND
         and adapter.spec.get('live_skill_start_step') == BOUNDARY_STEP
         and adapter.spec.get('max_ticks') == ENDPOINT_STEP
         and getattr(adapter, 'trace_sealed', None) is False
         and all(not (Path(adapter.output) / name).exists()
                 and not (Path(adapter.output) / name).is_symlink()
                 for name in ('trace_seal_request', 'init.trace.summary'))
         and getattr(mode, 'data', None) == 'WBC'
         and adapter.observe()['step'] == BOUNDARY_STEP,
         'ACTUAL_PRESEAL_WBC_BOUNDARY_REQUIRED_NO_CANCEL_SEAL_OR_ADVANCE')
