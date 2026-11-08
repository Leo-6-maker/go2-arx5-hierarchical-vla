"""Synthetic pre-seal pair transport; no model, renderer, ROS or integrations."""
import base64
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal
from scripts.wm02 import infer_fusion_paused_vla as infer
from go2arx5_vla.world_model import fusion_paused_inference as inference
from go2arx5_vla.world_model import fusion_paused_observation as observation
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model import fusion_source_camera_capture as camera_capture
from go2arx5_vla.world_model import fusion_source_camera_rig as camera_rig
from go2arx5_vla.world_model import fusion_task_phase_initialization as original
from go2arx5_vla.world_model import fusion_task_phase_live_skill_pair_v8 as live
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_v7 as v7
from go2arx5_vla.world_model.fusion_observation import transform_hash
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_observation import png
from tests.world_model.test_fusion_paused_inference import install_fake_draw
from tests.world_model.test_fusion_paused_query_bridge import setup_pair, close_gpu
from tests.world_model.test_fusion_source_camera_rig import fixture as rig_fixture
from test_fusion_task_phase_initialization import setup as pure_setup


PROJECT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    # These are disposable synthetic documents, rehashed before validation.
    Path(path).write_text(json.dumps(value, allow_nan=False))


def bind(gpu, adapter, pair):
    """Rehash synthetic immutable specs, their real leases and registry entries."""
    producer = adapter.spec
    write_json(gpu['paused_query_pair'], pair)
    pin = digest(gpu['paused_query_pair'])
    gpu['paused_query_pair_sha256'] = producer['paused_query_pair_sha256'] = pin
    native_path = Path(producer['registered_root'])/'EXECUTION_SPEC.json'
    write_json(native_path, producer)
    gpu['producer_native_spec_sha256'] = digest(native_path)
    gpu_path = Path(gpu['registered_root'])/'EXECUTION_SPEC.json'
    write_json(gpu_path, gpu)
    entries = []
    for spec, path in ((producer, native_path), (gpu, gpu_path)):
        root = Path(spec['registered_root'])
        write_json(spec['lease'], dict(status='ACTIVE', spec_sha256=digest(path), run_uuid=root.name))
        entries.append(dict(path=str(root), role=spec['role'], source_sha=spec['source_sha'],
            owner=spec['owner'], local_lease=spec['lease']))
    write_json(gpu['registry'], dict(entries=entries))
    adapter.fixture_binding = replace(adapter.fixture_binding,
        target_runtime_sha256=digest(native_path))


@pytest.fixture
def pair_fixture(tmp_path, monkeypatch):
    gpu, adapter, pair, events, admissions = setup_pair(tmp_path, monkeypatch)
    root = Path(adapter.spec['registered_root'])
    monkeypatch.setattr(original, 'SCENE', digest(adapter.spec['model_path']))
    source_setup = pure_setup.__wrapped__(monkeypatch)
    frame = source_setup['frame']
    frame_pin = transform_hash(frame)
    raw_source = source_setup['args']['source_observation_bytes']
    source_payload = unseal(json.loads(raw_source))['payload']
    source_path = root/'SYNTHETIC_SOURCE_ORIGINAL_LANGUAGE.json'
    source_path.write_bytes(raw_source)
    monkeypatch.setattr(live, 'SOURCE_OBSERVATION', digest(source_path))
    monkeypatch.setattr(live, 'SOURCE_PAYLOAD', document_hash(source_payload))
    monkeypatch.setattr(live, 'FRAME_MAP', frame_pin)
    monkeypatch.setattr(camera_rig, 'FRAME_MAP_SHA256', frame_pin)
    monkeypatch.setattr(camera_capture, 'FRAME_MAP_SHA256', frame_pin)
    audit, native_cameras = rig_fixture()
    audit_raw = json.dumps(audit, allow_nan=False).encode()
    audit_path = root/'SYNTHETIC_SOURCE_CAMERA_AUDIT.json'
    audit_path.write_bytes(audit_raw)
    audit_pin = digest(audit_path)
    monkeypatch.setattr(live, 'SOURCE_CAMERA_IDENTITY', audit_pin)
    monkeypatch.setattr(camera_rig, 'SOURCE_IDENTITY_SHA256', audit_pin)
    monkeypatch.setattr(camera_capture, 'SOURCE_IDENTITY_SHA256', audit_pin)
    camera = camera_capture.SourceCameraCapture(audit_raw, frame)
    rig = camera_rig.SourceCameraRig(audit_raw, frame, native_cameras,
        expected_native_camera_records_sha256=camera_rig.record_hash(native_cameras))

    producer = adapter.spec
    episode = v7.make_episode(run=str(root), geometry_scale=.5)
    manifest = root/'SYNTHETIC_ORIGINAL_V7_HALF_EPISODE.json'
    write_json(manifest, episode)
    task = copy.deepcopy(adapter.fixture_task)
    task.update(language=source_payload['language'],
        language_provenance='EXACT_SOURCE_TASK_PROMPT')
    task_path = root/'SYNTHETIC_ORIGINAL_FULL_TASK.json'
    write_json(task_path, task)
    capsule = Path(gpu['source_root'])
    for name in set(inference.SOURCE_MEMBERS) | set(live.SOURCE_MEMBERS):
        target = capsule/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes())
        gpu['source_files'][name] = digest(target)
    producer.update(phase=live.PHASE, kind=live.KIND,
        profile='wm02-auto1-target', scope='WM02_PILOT1_SIM_ONLY',
        snapshot_binding_mode=live.PAIR_MODE,
        live_skill_start_step=live.BOUNDARY_STEP, max_ticks=live.ENDPOINT_STEP,
        physics_dt_s=.002, WBC_changes=False,
        task_condition=live.TASK_CONDITION, experiment_role=live.EXPERIMENT_ROLE,
        capture_camera_recipe=live.SOURCE_CAMERA_RECIPE,
        episode_manifest=str(manifest), episode_manifest_sha256=digest(manifest),
        task_phase_source_observation=str(source_path),
        task_phase_source_observation_sha256=digest(source_path),
        paused_target_task=str(task_path), paused_target_task_file_sha256=digest(task_path),
        paused_target_task_sha256=document_hash(task), paused_source_from_target=frame.tolist(),
        paused_frame_map_sha256=frame_pin, source_camera_identity=str(audit_path),
        source_camera_identity_sha256=audit_pin,
        source_files=gpu['source_files'].copy())
    gpu.update(snapshot_binding_mode=live.PAIR_MODE, paused_task_sha256=document_hash(task),
        experiment_role=live.EXPERIMENT_ROLE)
    pair.update(schema=live.PAIR_SCHEMA, native_boundary_step=live.BOUNDARY_STEP,
        native_endpoint_step=live.ENDPOINT_STEP, producer_phase=live.PHASE,
        camera_recipe=live.SOURCE_CAMERA_RECIPE, source_camera_identity_sha256=audit_pin,
        task_condition=live.TASK_CONDITION, task_sha256=document_hash(task), frame_map_sha256=frame_pin)
    adapter.fixture_task = task
    adapter.fixture_binding = replace(adapter.fixture_binding,
        T_source_from_target=frame, expected_transform_sha256=frame_pin)
    adapter.trace_sealed = False
    adapter.latest = dict(mode=SimpleNamespace(data='WBC'))
    raw = bytearray((adapter.output/'state.bin').read_bytes())
    struct.pack_into('<Qd', raw, 8, live.BOUNDARY_STEP, live.BOUNDARY_STEP*.002)
    (adapter.output/'state.bin').write_bytes(raw)
    ack_path = adapter.output/'pause_ack.json'
    ack = json.loads(ack_path.read_bytes())
    ack.update(step=live.BOUNDARY_STEP, time_s=live.BOUNDARY_STEP*.002)
    write_json(ack_path, ack)
    (adapter.output/'step_quota').write_text(str(live.BOUNDARY_STEP)+'\n')
    quota = (adapter.output/'quota.binding').read_text().splitlines()
    quota[4] = str(live.ENDPOINT_STEP)
    (adapter.output/'quota.binding').write_text('\n'.join(quota)+'\n')
    # The existing fake recorder writes the calibration saved at its initial
    # fixture clock. Keep its synthetic camera geometry, with the new packet's
    # real absolute timestamp; no model or renderer is constructed.
    old_recorder = observation.FrameRecorder
    class Recorder(old_recorder):
        def save(self, index, packet):
            value = super().save(index, packet)
            path = self.folder/'images'/f'{index:06d}-calibration.json'
            cal = json.loads(path.read_bytes())
            for item in cal.values():
                item['timestamp_s'] = packet['time_s']
            write_json(path, cal)
            return value
    monkeypatch.setattr(observation, 'FrameRecorder', Recorder)
    bind(gpu, adapter, pair)
    return dict(gpu=gpu, adapter=adapter, pair=pair, events=events, admissions=admissions,
        source_payload=source_payload, source_path=source_path, task=task, episode=episode,
        camera=camera, rig=rig)


def capture_source_archive(fixture):
    """Build hermetic eight-file sensor records, never renderer qualification.

    The native packet/pause capture and every archive reader stay real. The
    pixels and sensor geometry are explicit pure test data, as in the existing
    fake recorder; no FK, source image or achieved outcome supplies policy data.
    """
    gpu, adapter, rig = fixture['gpu'], fixture['adapter'], fixture['rig']
    record = observation.capture_paused_observation(adapter, binding=adapter.fixture_binding,
        task=fixture['task'], task_sha256=gpu['paused_task_sha256'], query_uuid=gpu['query_uuid'])
    body = unseal(record)
    shadow = unseal(body['original_shadow_ABI'])
    folder = Path(gpu['recorded_observation']).parent
    target = shadow['target_observation']
    native_images = dict(body_rgb=png(20), wrist_rgb=(folder/'images/000000-wrist_rgb.png').read_bytes())
    native = rig.declaration()['native_target_camera_records']
    native_cal = {name: dict(timestamp_s=body['native_time_s'], K=item['K'],
        T_world_optical=item['T_world_optical']) for name, item in native.items()}
    retained = folder/'native-camera/images'
    retained.mkdir(parents=True)
    for name, raw in native_images.items():
        (retained/f'000000-{name}.png').write_bytes(raw)
    write_json(retained/'000000-calibration.json', native_cal)
    (folder/'images/000000-body_rgb.png').write_bytes(png(55))
    sensor = rig.declaration()
    sensor.update(schema='WM02_FUSION_SOURCE_CAMERA_CAPTURE_V1', status='ACTUAL_TARGET_PACKET_CPU_RENDERED',
        native_calibration_sha256=digest(retained/'000000-calibration.json'),
        native_image_sha256={name:digest(retained/f'000000-{name}.png') for name in native_images},
        rendered_external_image_sha256=digest(folder/'images/000000-body_rgb.png'),
        original_compiled_model_sha256=target['scene_sha256'],
        packet_identity=dict(native_step=body['native_step'], time_s=body['native_time_s']),
        rendering_slot_alias={'body_rgb':camera_rig.OVERLAY_CAMERA, 'wrist_rgb':'wrist_rgb'},
        effects=dict(renders=3, render_copy_FK_calls=2, integrations=0, model_queries=0),
        source_images_supplied=False, observation_payload_created=True)
    external = copy.deepcopy(sensor['declared_rendering_sensors'][camera_rig.OVERLAY_CAMERA])
    external.update(native_camera=False, rendering_copy_slot='body_rgb')
    target['cameras'] = {camera_rig.OVERLAY_CAMERA:external, 'wrist_rgb':native['wrist_rgb']}
    target['sensor_adapter'] = sensor
    shadow['provenance']['policy_camera_sensor_mapping'] = sensor['policy_camera_role_mapping']
    cal = dict(body_rgb=dict(timestamp_s=body['native_time_s'], K=external['K'],
        T_world_optical=external['T_world_optical']), wrist_rgb=native_cal['wrist_rgb'])
    write_json(folder/'images/000000-calibration.json', cal)
    target['calibration_sha256'] = digest(folder/'images/000000-calibration.json')
    target['image_sha256'] = {name:digest(folder/f'images/000000-{name}.png') for name in native_images}
    for image in shadow['source_policy_view']['payload']['images']:
        slot = 'body_rgb' if image['camera'] == 'external_rgb' else 'wrist_rgb'
        image['sha256'] = target['image_sha256'][slot]
        image['png_base64'] = base64.b64encode((folder/f'images/000000-{slot}.png').read_bytes()).decode()
    body['original_shadow_ABI'] = sealed(shadow)
    write_json(folder/'BOUND_SHADOW_ABI.private.json', body['original_shadow_ABI'])
    for path in [folder/'BOUND_SHADOW_ABI.private.json', *sorted((folder/'images').iterdir()), *sorted(retained.iterdir())]:
        body['files'][str(path.relative_to(folder))] = digest(path)
    record = sealed(body)
    write_json(gpu['recorded_observation'], record)
    return record


def test_predeclared_pause_and_final_endpoint_are_distinct_with_original_v7_manifest(pair_fixture):
    f = pair_fixture
    gpu, adapter, pair = f['gpu'], f['adapter'], f['pair']
    assert live.BOUNDARY_STEP == 34303 and live.ENDPOINT_STEP == 66503
    assert live.PAIR_EXTRA_KEYS == {'native_endpoint_step', 'producer_phase', 'camera_recipe',
        'source_camera_identity_sha256', 'task_condition'}
    assert f['episode']['phase'] == v7.PHASE and f['episode']['geometry_scale'] == .5
    assert v7.endpoint(f['episode']) == pair['native_boundary_step']
    assert adapter.spec['max_ticks'] == pair['native_endpoint_step'] > pair['native_boundary_step']
    raw = Path(adapter.spec['episode_manifest']).read_bytes()
    inference.validate_spec(gpu)
    checked, producer = bridge.validate_pair_spec(gpu)
    assert checked == pair and producer == adapter.spec
    assert live.validate_pair_spec(gpu) == (pair, adapter.spec)
    camera = live.make_camera_recipe(adapter.spec)
    assert type(camera) is camera_capture.SourceCameraCapture
    assert camera.source_identity_bytes == f['camera'].source_identity_bytes
    np.testing.assert_array_equal(camera.T_source_from_target, f['camera'].T_source_from_target)
    proof = bridge.preflight_pair(Path(gpu['registered_root'])/'EXECUTION_SPEC.json', adapter.spec)
    assert proof['schema'] == live.PAIR_SCHEMA and proof['native_boundary_step'] == 34303
    assert proof['native_endpoint_step'] == 66503
    assert proof['snapshot_exists'] is False and proof['native_execution_authorized'] is False
    assert Path(adapter.spec['episode_manifest']).read_bytes() == raw
    assert not Path(gpu['recorded_observation']).exists() and f['events'].saves == []
    assert 'producer_native_spec_sha256' not in pair
    assert 'recorded_observation_sha256' not in adapter.spec
    assert live.validate_native_pause(adapter) is None


def test_exact_1200_second_finite_query_wall_and_wait_are_accepted(pair_fixture):
    gpu = copy.deepcopy(pair_fixture['gpu'])
    gpu.update(wall_limit_s=1200, snapshot_wait_limit_s=1200)
    assert live.validate_pair_spec(gpu)[0] == pair_fixture['pair']


@pytest.mark.parametrize('fault', ['over-wall', 'nan-wall', 'wait-over-wall'])
def test_nonfinite_or_excess_query_wall_and_wait_reject_before_capture(pair_fixture, fault):
    gpu = copy.deepcopy(pair_fixture['gpu'])
    if fault == 'over-wall':
        gpu['wall_limit_s'] = 1200.001
    elif fault == 'nan-wall':
        gpu['wall_limit_s'] = float('nan')
    else:
        gpu['snapshot_wait_limit_s'] = gpu['wall_limit_s']+1
    with pytest.raises(ValueError):
        live.validate_pair_spec(gpu)
    assert pair_fixture['events'].saves == []


@pytest.mark.parametrize('fault', ['native-closed', 'gpu-closed', 'registry', 'STOP'])
def test_new_mode_preflight_keeps_both_actual_active_registrations_and_stop_guard(pair_fixture, fault):
    gpu, adapter = pair_fixture['gpu'], pair_fixture['adapter']
    if fault.endswith('-closed'):
        spec = adapter.spec if fault == 'native-closed' else gpu
        lease = json.loads(Path(spec['lease']).read_bytes())
        lease['status'] = 'CLOSED'
        write_json(spec['lease'], lease)
    elif fault == 'registry':
        write_json(gpu['registry'], dict(entries=[]))
    else:
        (Path(gpu['registered_root'])/'STOP.json').write_text('{}')
    with pytest.raises(ValueError):
        bridge.preflight_pair(Path(gpu['registered_root'])/'EXECUTION_SPEC.json', adapter.spec)
    assert not Path(gpu['recorded_observation']).exists() and pair_fixture['events'].saves == []


@pytest.mark.parametrize('fault', ['old-boundary', 'collapsed-endpoint', 'producer-endpoint',
    'producer-phase', 'producer-start', 'full-episode', 'paraphrase', 'source-bytes', 'audit-pin',
    'frame', 'new-source-bytes', 'extra-v2-key', 'old-mode', 'owner', 'profile', 'scope', 'hidden-fusion'])
def test_rehashed_wrong_pair_cannot_relabel_a_short_task_or_terminal_query(pair_fixture, fault):
    f = pair_fixture
    gpu, adapter, pair = f['gpu'], f['adapter'], f['pair']
    producer = adapter.spec
    if fault == 'old-boundary':
        pair['native_boundary_step'] = 3703
    elif fault == 'collapsed-endpoint':
        pair['native_endpoint_step'] = producer['max_ticks'] = live.BOUNDARY_STEP
    elif fault == 'producer-endpoint':
        producer['max_ticks'] += 25
    elif fault == 'producer-phase':
        producer['phase'] = v7.PHASE
    elif fault == 'producer-start':
        producer['live_skill_start_step'] += 25
    elif fault == 'full-episode':
        value = v7.make_episode(run=producer['registered_root'], geometry_scale=1.)
        write_json(producer['episode_manifest'], value)
        producer['episode_manifest_sha256'] = digest(producer['episode_manifest'])
    elif fault == 'paraphrase':
        task = dict(f['task'], language='Move only a little toward the cube.')
        write_json(producer['paused_target_task'], task)
        producer['paused_target_task_file_sha256'] = digest(producer['paused_target_task'])
        producer['paused_target_task_sha256'] = gpu['paused_task_sha256'] = pair['task_sha256'] = document_hash(task)
    elif fault == 'source-bytes':
        f['source_path'].write_text('{}')
    elif fault == 'audit-pin':
        pair['source_camera_identity_sha256'] = '0'*64
    elif fault == 'frame':
        pair['frame_map_sha256'] = '0'*64
    elif fault == 'new-source-bytes':
        (Path(gpu['source_root'])/live.SOURCE_MEMBERS[0]).write_text('changed capsule')
    elif fault == 'extra-v2-key':
        pair['returned_VLA_action_installations'] = 1
    elif fault == 'old-mode':
        gpu['snapshot_binding_mode'] = bridge.MODE
    elif fault == 'profile':
        producer['profile'] = inference.PROFILE
    elif fault == 'scope':
        producer['scope'] = 'WM02_FUSION_NIGHT1'
    elif fault == 'hidden-fusion':
        producer['fusion_binding'] = '/SYNTHETIC_OTHER_BINDING.json'
    else:
        gpu['owner'] = 'OTHER_OWNER'
    bind(gpu, adapter, pair)
    with pytest.raises((ValueError, KeyError)):
        inference.validate_spec(gpu)
    assert f['events'].saves == [] and not Path(gpu['recorded_observation']).exists()


@pytest.mark.parametrize('fault', ['sealed', 'seal-request', 'summary', 'mode', 'boundary'])
def test_preseal_means_actual_unsealed_trace_and_original_wbc_at_b(pair_fixture, fault):
    adapter = pair_fixture['adapter']
    if fault == 'sealed':
        adapter.trace_sealed = True
    elif fault == 'seal-request':
        (adapter.output/'trace_seal_request').write_text('1\n')
    elif fault == 'summary':
        (adapter.output/'init.trace.summary').write_text('terminal\n')
    elif fault == 'mode':
        adapter.latest['mode'].data = 'HOLD'
    else:
        raw = bytearray((adapter.output/'state.bin').read_bytes())
        struct.pack_into('<Qd', raw, 8, live.BOUNDARY_STEP+25, (live.BOUNDARY_STEP+25)*.002)
        (adapter.output/'state.bin').write_bytes(raw)
    with pytest.raises(ValueError):
        live.validate_native_pause(adapter)
    assert pair_fixture['events'].saves == []


@pytest.mark.parametrize('kind', ['missing', 'untyped', 'subclass'])
def test_bridge_requires_the_exact_original_source_capture_type_before_writing(pair_fixture, kind):
    f = pair_fixture
    if kind == 'subclass':
        class OtherCapture(camera_capture.SourceCameraCapture):
            pass
        camera = OtherCapture(f['camera'].source_identity_bytes, f['camera'].T_source_from_target)
    else:
        camera = None if kind == 'missing' else object()
    with pytest.raises(ValueError, match='TYPED_PREDECLARED_SOURCE_CAMERA_CAPTURE_REQUIRED'):
        bridge.capture_and_wait_query(f['adapter'], binding=f['adapter'].fixture_binding,
            task=f['task'], task_sha256=f['gpu']['paused_task_sha256'],
            gpu_spec_path=Path(f['gpu']['registered_root'])/'EXECUTION_SPEC.json', camera_recipe=camera)
    assert f['events'].saves == [] and not Path(f['gpu']['recorded_observation']).exists()


def test_eight_file_payload_retains_full_language_and_honest_preseal_identity(pair_fixture):
    f = pair_fixture
    record = capture_source_archive(f)
    payload, identity, returned = inference.read_payload(f['gpu'])
    body = unseal(record)
    shadow = unseal(body['original_shadow_ABI'])
    live.validate_capture_camera(f['pair'], shadow)
    assert returned == record and len(body['files']) == 8
    assert payload['language'] == f['source_payload']['language'] == f['task']['language']
    assert payload['language'] != 'Move the open gripper a little toward the red cube, then hold without touching it.'
    assert identity['producer_phase'] == live.PHASE
    assert identity['physical_recipe_phase'] == v7.PHASE
    assert identity['capture_condition'] == 'PRESEAL_WBC_ORIGINAL_QUOTA_ACK_PAUSED'
    assert identity['initialization_recipe_phase'] == v7.PHASE
    assert identity['initialization_geometry_scale'] == .5
    assert identity['native_boundary_step'] == 34303 and identity['native_endpoint_step'] == 66503
    assert identity['task_condition'] == live.TASK_CONDITION
    assert identity['experiment_role'] == live.EXPERIMENT_ROLE
    assert identity['old_initialization_readiness_relabelled'] is False
    assert identity['source_task_phase_equivalence'] is False
    assert identity['native_trace_seal_checked_by_GPU'] is False
    assert identity['capture_camera_recipe'] == live.SOURCE_CAMERA_RECIPE
    assert identity['source_camera_identity_sha256'] == f['pair']['source_camera_identity_sha256']
    assert identity['returned_VLA_action_installations'] == 0
    assert identity['native_step'] == live.BOUNDARY_STEP and identity['files'] == body['files']
    assert not identity['online_closed_loop'] and not identity['native_execution_authorized']
    assert not identity['camera_equivalence'] and identity['native_caller_must_verify_still_paused']
    assert 'payload' not in identity['source_policy_view']
    assert f['adapter'].trace_sealed is False and f['adapter'].latest['mode'].data == 'WBC'
    assert f['events'].saves == [(0, live.BOUNDARY_STEP)]
    folder = Path(f['gpu']['recorded_observation']).parent
    assert (folder/'STATE.private.bin').read_bytes() == (f['adapter'].output/'state.bin').read_bytes()
    assert (folder/'images/000000-wrist_rgb.png').read_bytes() == (folder/'native-camera/images/000000-wrist_rgb.png').read_bytes()


@pytest.mark.parametrize('member', ['native-camera/images/000000-body_rgb.png',
    'native-camera/images/000000-wrist_rgb.png', 'native-camera/images/000000-calibration.json'])
def test_retained_original_camera_member_cannot_be_dropped_or_changed(pair_fixture, member):
    capture_source_archive(pair_fixture)
    path = Path(pair_fixture['gpu']['recorded_observation']).parent/member
    path.write_bytes(b'changed original native camera evidence')
    with pytest.raises(ValueError):
        inference.read_payload(pair_fixture['gpu'])


@pytest.mark.parametrize('fault', ['missing', 'declaration-only', 'rig', 'source-audit', 'frame'])
def test_new_pair_never_accepts_original_views_or_an_uncaptured_source_rig(pair_fixture, fault):
    sensor = dict(schema='WM02_FUSION_SOURCE_CAMERA_CAPTURE_V1', status='ACTUAL_TARGET_PACKET_CPU_RENDERED',
        rig=live.SOURCE_CAMERA_RECIPE, source_identity_audit_sha256=live.SOURCE_CAMERA_IDENTITY,
        frame_map_sha256=live.FRAME_MAP)
    if fault == 'missing':
        sensor = None
    elif fault == 'declaration-only':
        sensor['status'] = 'DECLARATION_ONLY_NOT_RENDERED'
    elif fault == 'rig':
        sensor['rig'] = 'ORIGINAL_TARGET_NATIVE_DUAL_CAMERA_V1'
    elif fault == 'source-audit':
        sensor['source_identity_audit_sha256'] = '0'*64
    else:
        sensor['frame_map_sha256'] = '0'*64
    with pytest.raises(ValueError):
        live.validate_capture_camera(pair_fixture['pair'], dict(target_observation=dict(sensor_adapter=sensor)))


def test_one_original_generate_and_return_check_do_not_install_or_rebase_reference(pair_fixture, monkeypatch):
    f = pair_fixture
    gpu, adapter = f['gpu'], f['adapter']
    draw = install_fake_draw(monkeypatch)
    def capture_writer(actual_adapter, **kwargs):
        assert actual_adapter is adapter and kwargs['camera_recipe'] is f['camera']
        return capture_source_archive(f)
    monkeypatch.setattr(bridge, 'capture_paused_observation', capture_writer)
    def execute_once(seconds):
        assert seconds == .05
        ready = Path(gpu['recorded_observation']).parent/'PAIR_SNAPSHOT_READY.private.json'
        assert ready.is_file()
        output = Path(gpu['registered_root'])/'fit-01'
        output.mkdir()
        infer.infer(gpu, output)
        close_gpu(gpu)
    monkeypatch.setattr(bridge.time, 'sleep', execute_once)
    result = bridge.capture_and_wait_query(adapter, binding=adapter.fixture_binding,
        task=f['task'], task_sha256=gpu['paused_task_sha256'],
        gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json', camera_recipe=f['camera'])
    assert draw.loads == draw.queries == 1
    assert draw.payloads[0]['language'] == f['source_payload']['language']
    assert f['events'].saves == [(0, live.BOUNDARY_STEP)] and adapter.trace_sealed is False
    assert result['query_result']['model_result_checked'] is True
    assert result['native_integrations_by_bridge'] == 0 and not result['online_action_execution']
    output = Path(gpu['registered_root'])/'fit-01'
    raw_result = json.loads((output/'RESULT.private.json').read_bytes())
    assert raw_result['producer_phase'] == live.PHASE
    assert raw_result['physical_recipe_phase'] == v7.PHASE
    assert raw_result['noise']['sha256'] == '1'*64 and raw_result['seed'] == gpu['seed']
    assert raw_result['returned_action_is_live_reference'] is False and raw_result['reference_TTL_rebased'] is False
    assert raw_result['observation_binding']['returned_VLA_action_installations'] == 0
    with np.load(output/'REAL_VLA_ACTION.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['physical_H40'], np.arange(400).reshape(40, 10)/1000)
        assert np.all(arrays['normalized'] == 1.25)
    f['events'].pid += 1
    with pytest.raises(ValueError, match='SAME_ACTUAL_OWNED_SIMULATOR'):
        inference.verify_paused_query_return(adapter, gpu, output/'RESULT.private.json')
    f['events'].pid -= 1
    adapter.latest['mode'].data = 'HOLD'
    with pytest.raises(ValueError, match='PRESEAL_WBC'):
        inference.verify_paused_query_return(adapter, gpu, output/'RESULT.private.json')
    adapter.latest['mode'].data = 'WBC'
    packet_path = adapter.output/'state.bin'
    packet = packet_path.read_bytes()
    changed = bytearray(packet)
    struct.pack_into('<d', changed, 36, .005)
    packet_path.write_bytes(changed)
    with pytest.raises(ValueError, match='QUERY_RETURN'):
        inference.verify_paused_query_return(adapter, gpu, output/'RESULT.private.json')
    packet_path.write_bytes(packet)
    (adapter.output/'trace_seal_request').write_text('1\n')
    with pytest.raises(ValueError):
        inference.verify_paused_query_return(adapter, gpu, output/'RESULT.private.json')
    with pytest.raises((FileExistsError, ValueError)):
        infer.infer(gpu, output)
    assert draw.loads == draw.queries == 1


@pytest.mark.parametrize('fault', ['trace-sealed', 'wbc-lost'])
def test_losing_preseal_wbc_during_capture_stops_before_any_model_load(pair_fixture, monkeypatch, fault):
    f = pair_fixture
    draw = install_fake_draw(monkeypatch)
    def capture_writer(adapter, **kwargs):
        record = capture_source_archive(f)
        if fault == 'trace-sealed':
            adapter.trace_sealed = True
        else:
            adapter.latest['mode'].data = 'HOLD'
        return record
    monkeypatch.setattr(bridge, 'capture_paused_observation', capture_writer)
    with pytest.raises(ValueError, match='PRESEAL_WBC'):
        bridge.capture_and_wait_query(f['adapter'], binding=f['adapter'].fixture_binding,
            task=f['task'], task_sha256=f['gpu']['paused_task_sha256'],
            gpu_spec_path=Path(f['gpu']['registered_root'])/'EXECUTION_SPEC.json', camera_recipe=f['camera'])
    assert draw.loads == draw.queries == 0
    assert not (Path(f['gpu']['registered_root'])/'fit-01/GENERATION_INTENT.private.json').exists()
    assert not (Path(f['gpu']['recorded_observation']).parent/'PAIR_SNAPSHOT_READY.private.json').exists()


def test_original_pair_v1_keeps_its_3703_terminal_boundary(tmp_path, monkeypatch):
    gpu, adapter, pair, events, _ = setup_pair(tmp_path, monkeypatch)
    inference.validate_spec(gpu)
    assert bridge.validate_pair_spec(gpu) == (pair, adapter.spec)
    assert pair['schema'] == bridge.SCHEMA and pair['native_boundary_step'] == adapter.spec['max_ticks'] == 3703
    assert set(pair) == bridge.KEYS and not Path(gpu['recorded_observation']).exists()
    assert events.saves == []
    with pytest.raises(ValueError, match='DISTINCT_PRODUCER_ENDPOINT_REQUIRES_EXPLICIT_LIVE_SKILL_MODE'):
        bridge._validate_pair_spec(gpu, producer_endpoint_step=live.ENDPOINT_STEP)


def test_new_leaf_import_does_not_load_model_ros_or_physics_runtime():
    source = '''
import sys
from go2arx5_vla.world_model import fusion_task_phase_live_skill_pair_v8
from go2arx5_vla.world_model import fusion_paused_query_bridge, fusion_paused_inference
loaded = {'torch', 'gr00t', 'mujoco', 'rclpy', 'transformers'}.intersection(
    name.split('.')[0] for name in sys.modules)
if loaded:
    raise AssertionError('runtime backend imported: '+repr(loaded))
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT/'src'), str(PROJECT),
        os.environ.get('PYTHONPATH', '')]))
    command = [sys.executable, '-B']+(['-O'] if sys.flags.optimize else [])+['-c', source]
    result = subprocess.run(command, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
