"""Finite file handoff between two already registered original-owner scopes.

This module neither registers nor launches a worker, queries a model, advances
physics, renews a reference, nor certifies/installs returned actions.
"""
import json
import os
from pathlib import Path
import re
import time

from scripts.pi0292.relay_protocol import unseal, uuid_string
from .fusion_paused_observation import capture_paused_observation, verify_still_paused
from .storage import digest, write_json

SCHEMA = 'WM02_FUSION_PAUSED_QUERY_PAIR_V1'
MODE = 'PRE_REGISTERED_SINGLE_SNAPSHOT_V1'
POSTURE_MODE = 'PRE_REGISTERED_SOURCE_POSTURE_SNAPSHOT_V2'
LIVE_SKILL_MODE = 'PRE_REGISTERED_TASK_PHASE_LIVE_SKILL_V8'
KEYS = {'schema', 'pair_uuid', 'query_uuid', 'native_root', 'native_run_id',
        'native_boundary_step', 'capture_index', 'gpu_root', 'task_sha256', 'scene_sha256',
        'frame_map_sha256', 'seed', 'checkpoint_manifest_sha256', 'source_sha',
        'deadline_epoch', 'max_queries', 'native_integrations_by_bridge'}


def need(condition, reason):
    if not condition:
        raise ValueError('FUSION_PAUSED_QUERY_BRIDGE_'+reason)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _json(path):
    return json.loads(Path(path).read_text())


def _snapshot(pair):
    return (Path(pair['native_root'])/pair['native_run_id']/'PAUSED_OBSERVATIONS'
            /pair['query_uuid']/'PAUSED_OBSERVATION.private.json')


def validate_pair_spec(spec):
    """Acyclic pins: pair -> producer spec -> consumer spec, never the reverse."""
    if spec.get('snapshot_binding_mode') == POSTURE_MODE:
        from .fusion_source_posture_paused_query_v3 import validate_pair_spec as posture_pair
        return posture_pair(spec)
    if spec.get('snapshot_binding_mode') == LIVE_SKILL_MODE:
        from .fusion_task_phase_live_skill_pair_v8 import validate_pair_spec as skill_pair
        return skill_pair(spec)
    return _validate_pair_spec(spec)


def _validate_pair_spec(spec, *, pair_schema=SCHEMA, binding_mode=MODE,
                        boundary_step=3703, pair_keys=None, producer_endpoint_step=None):
    """Common owner checks; explicit typed modes may separate pause and end."""
    from .fusion_paused_inference import CHECKPOINT, KIND
    need(producer_endpoint_step is None or (binding_mode == LIVE_SKILL_MODE
         and type(producer_endpoint_step) is int and producer_endpoint_step > boundary_step),
         'DISTINCT_PRODUCER_ENDPOINT_REQUIRES_EXPLICIT_LIVE_SKILL_MODE')
    pair_path = Path(spec['paused_query_pair'])
    need(not pair_path.is_symlink() and digest(pair_path) == spec['paused_query_pair_sha256'],
         'IMMUTABLE_PREDECLARED_PAIR_BYTES')
    pair = _json(pair_path)
    need(set(pair) == (KEYS if pair_keys is None else pair_keys)
         and pair['schema'] == pair_schema, 'EXACT_VERSIONED_PAIR_SCHEMA')
    uuid_string(pair['pair_uuid']); uuid_string(pair['query_uuid'])
    need(type(pair['native_boundary_step']) is int and pair['native_boundary_step'] == boundary_step
         and type(pair['capture_index']) is int and pair['capture_index'] >= 0
         and pair['native_run_id'] == 'native-01'
         and type(pair['max_queries']) is int and pair['max_queries'] == 1
         and type(pair['native_integrations_by_bridge']) is int
         and pair['native_integrations_by_bridge'] == 0
         and type(pair['seed']) is int and 0 <= pair['seed'] < 2**63
         and all(_sha(pair[k]) for k in ('task_sha256','scene_sha256','frame_map_sha256'))
         and pair['checkpoint_manifest_sha256'] == CHECKPOINT,
         'ONE_PREDECLARED_BOUNDARY_TASK_FRAME_MODEL_AND_NO_MOTION')
    native_root, gpu_root = Path(pair['native_root']), Path(pair['gpu_root'])
    need(all(p.is_absolute() and not p.is_symlink() and p.resolve() == p for p in (native_root,gpu_root))
         and native_root != gpu_root and not native_root.is_relative_to(gpu_root)
         and not gpu_root.is_relative_to(native_root), 'DISTINCT_CANONICAL_SCOPE_ROOTS')
    producer_path = native_root/'EXECUTION_SPEC.json'
    producer = _json(producer_path)
    need(not producer_path.is_symlink()
         and str(producer_path) == spec['producer_native_spec']
         and digest(producer_path) == spec['producer_native_spec_sha256']
         and producer.get('registered_root') == str(native_root)
         and producer.get('paused_query_pair') == str(pair_path)
         and producer.get('paused_query_pair_sha256') == spec['paused_query_pair_sha256']
         and producer.get('paused_query_gpu_spec') == str(gpu_root/'EXECUTION_SPEC.json')
         and producer.get('source_sha') == spec['source_sha'] == pair['source_sha']
         and producer.get('source_root') == spec['source_root']
         and producer.get('physics') is True
         and producer.get('max_ticks') == (pair['native_boundary_step']
             if producer_endpoint_step is None else producer_endpoint_step),
         'ACTUAL_PRODUCER_SPEC_AND_COMMON_PAIR_NO_CIRCULAR_SHA')
    need(spec.get('kind') == KIND and spec.get('snapshot_binding_mode') == binding_mode
         and spec.get('registered_root') == str(gpu_root)
         and spec.get('recorded_observation') == str(_snapshot(pair))
         and spec.get('recorded_observation_sha256') is None
         and spec.get('query_uuid') == pair['query_uuid']
         and spec.get('seed') == pair['seed']
         and spec.get('paused_task_sha256') == pair['task_sha256']
         and producer.get('paused_target_task_sha256') == pair['task_sha256']
         and producer.get('paused_frame_map_sha256') == pair['frame_map_sha256']
         and digest(producer['model_path']) == pair['scene_sha256']
         and type(spec.get('snapshot_wait_limit_s')) in (int,float)
         and 0 < spec['snapshot_wait_limit_s'] <= spec['wall_limit_s']
         and type(pair['deadline_epoch']) in (int,float)
         and time.time() < pair['deadline_epoch']
         and pair['deadline_epoch'] == spec['deadline_epoch'] <= producer['deadline_epoch']
         and all(producer.get(k) == spec.get(k) and spec.get(k) is not None
                 for k in ('owner','host','uid','campaign_root','registry','pilot1_binding','pilot1_binding_sha256')),
         'EXACT_ONE_SNAPSHOT_IDENTITY_AND_SHARED_ORIGINAL_OWNER_DEADLINE')
    return pair, producer


def validate_pair_capture(spec, record):
    pair, _ = validate_pair_spec(spec)
    body = unseal(record)
    shadow = unseal(body['original_shadow_ABI'])
    need(body['query_uuid'] == pair['query_uuid']
         and body['world']['registered_root'] == pair['native_root']
         and body['world']['native_output'] == str(Path(pair['native_root'])/pair['native_run_id'])
         and body['world']['execution_spec_sha256'] == spec['producer_native_spec_sha256']
         and body['world']['scene_sha256'] == pair['scene_sha256']
         and body['native_step'] == pair['native_boundary_step']
         and body['capture']['index'] == pair['capture_index']
         and body['task_sha256'] == pair['task_sha256']
         and shadow['source_policy_view']['transform_sha256'] == pair['frame_map_sha256'],
         'CAPTURE_MUST_BE_THE_PREDECLARED_ACTUAL_NATIVE_BOUNDARY')
    if spec['snapshot_binding_mode']==POSTURE_MODE:
        from .fusion_source_posture_paused_query_v3 import validate_capture_camera
        validate_capture_camera(pair,shadow)
    elif spec['snapshot_binding_mode']==LIVE_SKILL_MODE:
        from .fusion_task_phase_live_skill_pair_v8 import validate_capture_camera
        validate_capture_camera(pair,shadow)
    else:
        need('sensor_adapter' not in shadow['target_observation'],
             'PAIR_V1_ORIGINAL_CAMERA_IDENTITY_UNCHANGED')
    return pair


def _registered(spec, spec_path, *, closed=False):
    """Original actual registry/lease only; this creates or modifies neither."""
    spec_path = Path(spec_path)
    root = Path(spec['registered_root'])
    lease_path = root/'control/LEASE.json'
    need(Path(spec['lease']) == lease_path and not lease_path.is_symlink()
         and not spec_path.is_symlink(), 'ORIGINAL_EXACT_SCOPE_SPEC_AND_LEASE_PATH')
    lease = _json(lease_path)
    need(lease.get('status') == ('CLOSED' if closed else 'ACTIVE')
         and lease.get('spec_sha256') == digest(spec_path)
         and lease.get('run_uuid') == root.name, 'EXACT_REGISTERED_SCOPE_LEASE_STATE')
    entries = _json(spec['registry'])['entries']
    need(any(e.get('path') == str(root) and e.get('role') == spec['role']
             and e.get('source_sha') == spec['source_sha'] and e.get('owner') == spec['owner']
             and e.get('local_lease') == str(lease_path) for e in entries),
         'ACTUAL_ORIGINAL_REGISTRY_PAIR_MEMBER')
    if closed:
        worker = _json(root/'WORKER_EXIT.private.json')
        need(lease.get('worker_exit_code') == 0 and worker.get('exit_code') == 0
             and worker.get('failure') is None,
             'ORIGINAL_GPU_WORKER_TERMINAL_ZERO')
    return lease


def _live(spec, producer):
    now = time.time()
    need(now < min(spec['deadline_epoch'], producer['deadline_epoch'])
         and not any((Path(root)/'STOP.json').exists() for root in (
             spec['registered_root'], producer['registered_root'], spec['campaign_root'])),
         'ORIGINAL_STOP_AND_FINITE_PAIR_DEADLINE')
    # The existing scope validator rechecks P1/FUSION leases and scope STOP files;
    # it acquires no campaign/physical lock and grants no native motion.
    from .fusion_scope import validate
    validate(spec)


def preflight_pair(gpu_spec_path, native_spec):
    """Read-only check before either already registered scope starts physics.

    No snapshot, running GPU process, owner-lock acquisition or new registration
    is required. Existing active leases and original scope validators are real
    dependencies; the return value grants neither computation nor motion.
    """
    gpu_spec_path = Path(gpu_spec_path)
    spec = _json(gpu_spec_path)
    pair, producer = validate_pair_spec(spec)
    need(gpu_spec_path == Path(pair['gpu_root'])/'EXECUTION_SPEC.json'
         and native_spec == producer, 'PREFLIGHT_EXACT_GPU_PATH_AND_NATIVE_SPEC')
    _live(spec, producer)
    _registered(producer, spec['producer_native_spec'])
    _registered(spec, gpu_spec_path)
    folder = _snapshot(pair).parent
    need(not folder.exists() and not folder.is_symlink()
         and not any((Path(pair['gpu_root'])/'fit-01'/name).exists() for name in (
             'MODEL_LOAD_INTENT.private.json','GENERATION_INTENT.private.json',
             'RESULT.private.json','FAILED.private.json')),
         'FRESH_PAIR_WITHOUT_PREVIOUS_CAPTURE_LOAD_OR_QUERY')
    proof = dict(schema=pair['schema'],status='PRE_REGISTERED_PAIR_READ_ONLY_PREFLIGHT_PASSED',
        pair_sha256=spec['paused_query_pair_sha256'],gpu_spec_sha256=digest(gpu_spec_path),
        producer_native_spec_sha256=spec['producer_native_spec_sha256'],
        query_uuid=pair['query_uuid'],native_boundary_step=pair['native_boundary_step'],
        recorded_observation=str(_snapshot(pair)),snapshot_exists=False,
        source_sha=pair['source_sha'],seed=pair['seed'],
        registered_scope_leases='BOTH_ACTIVE_EXACT_SPEC_SHA',
        native_execution_authorized=False,GPU_execution_authorized=False)
    if spec['snapshot_binding_mode'] == LIVE_SKILL_MODE:
        proof['native_endpoint_step'] = pair['native_endpoint_step']
    return proof


def wait_for_snapshot(spec):
    """GPU worker's one bounded wait, after original admission, before CUDA."""
    if spec['snapshot_binding_mode'] not in (MODE,POSTURE_MODE,LIVE_SKILL_MODE):
        return
    pair, producer = validate_pair_spec(spec)
    producer_path = Path(spec['producer_native_spec'])
    end = min(time.monotonic()+spec['snapshot_wait_limit_s'],
              time.monotonic()+max(0.,spec['deadline_epoch']-time.time()))
    while True:
        _live(spec, producer)
        _registered(producer, producer_path)
        folder = _snapshot(pair).parent
        need(not (folder/'CAPTURE_FAILED.private.json').exists(), 'PRODUCER_CAPTURE_FAILED_NO_RETRY')
        ready_path = folder/'PAIR_SNAPSHOT_READY.private.json'
        if ready_path.exists():
            ready = _json(ready_path)
            need(ready == dict(schema=pair['schema'], pair_sha256=spec['paused_query_pair_sha256'],
                producer_native_spec_sha256=spec['producer_native_spec_sha256'],
                observation_sha256=digest(_snapshot(pair)), query_uuid=pair['query_uuid']),
                'EXACT_CAPTURE_COMPLETION_AND_OBSERVATION_BYTES')
            validate_pair_capture(spec, _json(_snapshot(pair)))
            return
        need(time.monotonic() < end, 'PREDECLARED_SNAPSHOT_WAIT_TIMEOUT_NO_QUERY')
        time.sleep(.05)


def _waiting_pause(adapter, record):
    # Reuse the actual raw/quota/fresh ACK checks while pumping the original
    # adapter. The full executable/library identity readback is deferred to
    # query return, rather than writing one loaded-identity snapshot per poll.
    from .fusion_paused_observation import _pause, _after_pause
    import hashlib
    if adapter.spec.get('phase') == 'collect-target-fusion-task-phase-live-skill-v8':
        from .fusion_task_phase_live_skill_pair_v8 import validate_native_pause
        validate_native_pause(adapter)
    body = unseal(record)
    current = _pause(adapter)
    need(hashlib.sha256(current['raw']).hexdigest() == body['files']['STATE.private.bin']
         and hashlib.sha256(current['quota']).hexdigest() == body['step_quota_sha256'],
         'UNCHANGED_ACTUAL_PACKET_AND_QUOTA_WHILE_WAITING')
    _after_pause(adapter, dict(current, pause=body['pause_after']))


def capture_and_wait_query(adapter, *, binding, task, task_sha256, gpu_spec_path, capture_index=0,
                           camera_recipe=None):
    """Native-owner callable: one capture + one existing GPU job, no motion.

    Both jobs must already be registered; this function never competes for the
    campaign OWNER lock held by the actual native adapter.
    """
    from .fusion_paused_inference import read_payload, verify_paused_query_return
    gpu_spec_path = Path(gpu_spec_path)
    spec = _json(gpu_spec_path)
    pair, producer = validate_pair_spec(spec)
    preflight_pair(gpu_spec_path, adapter.spec)
    need(gpu_spec_path == Path(pair['gpu_root'])/'EXECUTION_SPEC.json'
         and adapter.spec == producer and Path(adapter.output) == Path(pair['native_root'])/pair['native_run_id']
         and capture_index == pair['capture_index'] and task_sha256 == pair['task_sha256']
         and binding.scene_sha256 == pair['scene_sha256']
         and binding.expected_transform_sha256 == pair['frame_map_sha256'],
         'ACTUAL_OWNER_ADAPTER_TASK_AND_FIXED_FRAME_BINDING')
    camera_kwargs = {}
    if spec['snapshot_binding_mode'] in (POSTURE_MODE,LIVE_SKILL_MODE):
        if spec['snapshot_binding_mode'] == LIVE_SKILL_MODE:
            from .fusion_task_phase_live_skill_pair_v8 import SOURCE_CAMERA_RECIPE
        else:
            from .fusion_source_posture_paused_query_v3 import SOURCE_CAMERA_RECIPE
        if pair['camera_recipe'] == SOURCE_CAMERA_RECIPE:
            from .fusion_source_camera_capture import SourceCameraCapture
            need(isinstance(camera_recipe,SourceCameraCapture) and camera_recipe.name == SOURCE_CAMERA_RECIPE
                 and (spec['snapshot_binding_mode'] != LIVE_SKILL_MODE or type(camera_recipe) is SourceCameraCapture),
                 'EXACT_TYPED_PREDECLARED_SOURCE_CAMERA_CAPTURE_REQUIRED')
            camera_kwargs['camera_recipe'] = camera_recipe
        else:
            need(camera_recipe is None,'ORIGINAL_CAMERA_RECIPE_CANNOT_ACCEPT_SILENT_OVERRIDE')
    else:
        need(camera_recipe is None,'PAIR_V1_ORIGINAL_CAMERA_DEFAULT_UNCHANGED')
    if spec['snapshot_binding_mode'] == LIVE_SKILL_MODE:
        from .fusion_task_phase_live_skill_pair_v8 import validate_native_pause
        validate_native_pause(adapter)
    record = capture_paused_observation(adapter, binding=binding, task=task,
        task_sha256=task_sha256, query_uuid=pair['query_uuid'], capture_index=capture_index,**camera_kwargs)
    if spec['snapshot_binding_mode'] == LIVE_SKILL_MODE:
        validate_native_pause(adapter)
    read_payload(spec)
    # The original capture writer fsyncs its final envelope. Publish readiness
    # only after it returns and all actual members are checked, so the GPU never
    # mistakes an exclusive file being written for a completed JSON envelope.
    folder = _snapshot(pair).parent
    pending = folder/'PAIR_SNAPSHOT_READY.pending.private.json'
    write_json(pending, dict(
        schema=pair['schema'], pair_sha256=spec['paused_query_pair_sha256'],
        producer_native_spec_sha256=spec['producer_native_spec_sha256'],
        observation_sha256=digest(_snapshot(pair)), query_uuid=pair['query_uuid']))
    ready = folder/'PAIR_SNAPSHOT_READY.private.json'
    # Original scope ownership and the unique capture directory give one writer.
    # Use the existing qualified replace/fsync filesystem ABI, not hard links.
    need(not ready.exists() and not ready.is_symlink(), 'DO_NOT_REPLACE_EXISTING_READY_MARKER')
    os.replace(pending, ready)
    directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    result_path = Path(pair['gpu_root'])/'fit-01/RESULT.private.json'
    while True:
        adapter.guard()
        _live(spec, producer)
        _registered(producer, spec['producer_native_spec'])
        # Keep checking actual raw packet/quota/process and fresh native ACKs;
        # no reference TTL is extended while wall-clock inference runs.
        _waiting_pause(adapter, record)
        output = result_path.parent
        need(not (output/'FAILED.private.json').exists(), 'GPU_QUERY_FAILED_NO_RETRY')
        lease = _json(spec['lease'])
        if lease.get('status') == 'CLOSED':
            _registered(spec, gpu_spec_path, closed=True)
            checked = verify_paused_query_return(adapter, spec, result_path)
            return dict(schema=pair['schema'], pair_sha256=spec['paused_query_pair_sha256'],
                query_uuid=pair['query_uuid'], gpu_spec_sha256=digest(gpu_spec_path),
                capture_sha256=digest(_snapshot(pair)), query_result=checked,
                native_integrations_by_bridge=0, native_execution_authorized=False,
                online_action_execution=False)
        _registered(spec, gpu_spec_path)
        worker_path = Path(pair['gpu_root'])/'WORKER_EXIT.private.json'
        if worker_path.exists():
            worker = _json(worker_path)
            need(worker.get('exit_code') == 0 and worker.get('failure') is None,
                 'GPU_WORKER_FAILED_NO_RETRY')
        # Successful WORKER_EXIT precedes lease closeout in the original wrapper;
        # allow that short handoff without treating it as another query.
        time.sleep(.05)
