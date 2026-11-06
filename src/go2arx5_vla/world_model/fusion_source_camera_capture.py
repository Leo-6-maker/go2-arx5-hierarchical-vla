"""An explicit source-world sensor rendered from an actual target packet.

Only a private render copy's existing external camera slot is changed. The
original native cameras and bytes are retained separately. This never steps
physics, changes the owned reader/model, queries a policy, or supplies source
pixels. The policy's external_rgb wire slot names this new sensor in a sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.pi0294.task_native_worker import FrameRecorder
from .fusion_source_camera_rig import (SourceCameraRig, SOURCE_IDENTITY_SHA256,
    FRAME_MAP_SHA256, RIG, OVERLAY_CAMERA, record_hash)

SCHEMA = 'WM02_FUSION_SOURCE_CAMERA_CAPTURE_V1'
_CAMERA_PATCH_FIELDS = {'cam_bodyid', 'cam_pos', 'cam_quat', 'cam_fovy'}


def need(ok, reason):
    if not ok:
        raise ValueError('FUSION_SOURCE_CAMERA_CAPTURE_' + reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _arrays(model):
    return {name: value.copy() for name in dir(model)
            if isinstance((value := getattr(model, name)), np.ndarray)}


def _settings(value, prefix='', depth=0):
    result = {}
    for name in dir(value):
        if name.startswith('_'):
            continue
        item = getattr(value, name)
        key = prefix + name
        if isinstance(item, np.ndarray):
            result[key] = item.copy()
        elif isinstance(item, (int, float, bool, np.number)):
            result[key] = np.asarray(item)
        elif depth < 2 and type(item).__module__.startswith('mujoco'):
            result.update(_settings(item, key + '.', depth + 1))
    return result


def _same_model_except_external(native, rendered):
    """Compare every exposed model array; only one camera slot may differ."""
    left, right = _arrays(native), _arrays(rendered)
    need(set(left) == set(right), 'IDENTICAL_RENDER_MODEL_ARRAY_MEMBERS')
    cid = int(native.camera('body_rgb').id)
    need(cid == int(rendered.camera('body_rgb').id), 'SAME_COPY_CAMERA_SLOT')
    for name, value in left.items():
        actual = right[name]
        need(value.shape == actual.shape and value.dtype == actual.dtype,
             'IDENTICAL_MODEL_ARRAY_SHAPES_' + name)
        if name in _CAMERA_PATCH_FIELDS:
            value, actual = np.delete(value, cid, axis=0), np.delete(actual, cid, axis=0)
        need(np.array_equal(value, actual), 'RENDER_COPY_CHANGED_NONDECLARED_ARRAY_' + name)
    need(native is not rendered, 'INDEPENDENT_RENDER_MODEL_REQUIRED')
    for field in ('opt', 'vis', 'stat'):
        left_settings = _settings(getattr(native, field))
        right_settings = _settings(getattr(rendered, field))
        need(set(left_settings) == set(right_settings)
             and all(np.array_equal(value, right_settings[name])
                     for name, value in left_settings.items()),
             'RENDER_COPY_CHANGED_NATIVE_SETTINGS_' + field)


def _packet_data(data, packet):
    need(float(data.time) == packet['time_s']
         and all(np.array_equal(getattr(data, key), packet[key]) for key in ('qpos', 'qvel', 'ctrl')),
         'EXACT_CURRENT_PACKET_RENDER_STATE')


def _load_packet(model, data, packet):
    import mujoco
    for key in ('qpos', 'qvel', 'ctrl'):
        getattr(data, key)[:] = packet[key]
    data.time = packet['time_s']
    mujoco.mj_forward(model, data)
    _packet_data(data, packet)


def apply_source_rig_to_render_copy(native_model, render_model, render_data, packet, rig):
    """Patch one independent camera slot; FK only, never integrate a world."""
    import mujoco
    need(isinstance(rig, SourceCameraRig), 'PINNED_SOURCE_RIG_DECLARATION')
    need(native_model is not render_model, 'INDEPENDENT_RENDER_MODEL_REQUIRED')
    before = _arrays(native_model)
    for name, value in before.items():
        need(np.array_equal(value, getattr(render_model, name)), 'FRESH_NATIVE_BINARY_RENDER_COPY_' + name)
    cid = int(render_model.camera('body_rgb').id)
    need(int(render_model.cam_mode[cid]) == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED),
         'FIXED_CAMERA_SLOT_REQUIRED')
    transform = rig.external_T_target_world_mujoco
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, transform[:3, :3].reshape(9))
    render_model.cam_bodyid[cid] = 0
    render_model.cam_pos[cid] = transform[:3, 3]
    render_model.cam_quat[cid] = quat
    render_model.cam_fovy[cid] = rig.external_fovy_deg
    _load_packet(render_model, render_data, packet)
    _same_model_except_external(native_model, render_model)
    need(all(np.array_equal(value, getattr(native_model, name)) for name, value in before.items()),
         'ORIGINAL_NATIVE_MODEL_UNCHANGED')
    actual = np.eye(4)
    actual[:3, :3] = render_data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
    actual[:3, 3] = render_data.cam_xpos[cid]
    need(np.allclose(actual, rig.external_T_target_world_optical, rtol=0, atol=1e-9),
         'ACTUAL_RENDER_COPY_WORLD_CAMERA_FK')


def _calibration(model, data, stamp):
    result = {}
    for name in ('body_rgb', 'wrist_rgb'):
        cid = int(model.camera(name).id)
        transform = np.eye(4)
        transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = data.cam_xpos[cid]
        focal = 128 / np.tan(np.deg2rad(float(model.cam_fovy[cid])) / 2)
        result[name] = dict(timestamp_s=stamp, T_world_optical=transform.tolist(),
            K=[[focal, 0, 128], [0, focal, 128], [0, 0, 1]])
    return result


@dataclass(frozen=True)
class SourceCameraCapture:
    """Typed, pinned recipe; constructing it does not render or load MuJoCo."""
    source_identity_bytes: bytes
    T_source_from_target: object
    name: str = RIG

    def __post_init__(self):
        need(type(self.source_identity_bytes) is bytes
             and sha(self.source_identity_bytes) == SOURCE_IDENTITY_SHA256,
             'EXACT_SOURCE_CAMERA_IDENTITY_BYTES')
        need(not any(isinstance(v, (bool, np.bool_)) for v in
                     np.asarray(self.T_source_from_target, dtype=object).flat), 'NUMERIC_FRAME_NOT_BOOLEAN')
        frame = np.asarray(self.T_source_from_target, dtype='<f8')
        need(frame.shape == (4, 4) and np.isfinite(frame).all()
             and sha(frame.tobytes()) == FRAME_MAP_SHA256 and self.name == RIG,
             'EXACT_DECLARED_SOURCE_FRAME_AND_RECIPE')
        object.__setattr__(self, 'T_source_from_target', np.frombuffer(frame.tobytes(), dtype='<f8').reshape(4, 4))

    def recorder(self, model_path, folder):
        return SourceCameraRecorder(model_path, folder, self)


class SourceCameraRecorder:
    def __init__(self, model_path, folder, recipe):
        need(isinstance(recipe, SourceCameraCapture), 'TYPED_PINNED_SOURCE_CAMERA_RECIPE')
        self.model_path, self.folder, self.recipe = Path(model_path), Path(folder), recipe
        self.native_folder = self.folder / 'native-camera'
        (self.native_folder / 'images').mkdir(parents=True, exist_ok=False)
        self.native = FrameRecorder(self.model_path, self.native_folder)
        self.renderer = None
        self.camera_adapter = None

    def save(self, index, packet):
        import mujoco
        from PIL import Image
        from .fusion_observation import camera_records
        need(self.camera_adapter is None, 'ONE_CAPTURE_PER_RECIPE_RECORDER')
        model_sha = sha(self.model_path.read_bytes())
        result = self.native.save(index, packet)
        native_cal_path = self.native_folder / 'images' / f'{index:06d}-calibration.json'
        native_cal_raw = native_cal_path.read_bytes()
        native_cal = json.loads(native_cal_raw)
        native_state = SimpleNamespace(model=self.native.model, data=self.native.data)
        records = camera_records(native_state, native_cal, packet['time_s'])
        rig = SourceCameraRig(self.recipe.source_identity_bytes, self.recipe.T_source_from_target,
            records, expected_native_camera_records_sha256=record_hash(records))
        model = mujoco.MjModel.from_binary_path(str(self.model_path))
        data = mujoco.MjData(model)
        apply_source_rig_to_render_copy(self.native.model, model, data, packet, rig)
        self.renderer = mujoco.Renderer(model, height=256, width=256)
        self.renderer.update_scene(data, camera='body_rgb')
        external = self.renderer.render().copy()
        Image.fromarray(external).save(self.folder / 'images' / f'{index:06d}-body_rgb.png')
        rendered_external_sha256 = sha((self.folder / 'images' / f'{index:06d}-body_rgb.png').read_bytes())
        native_images = {name: (self.native_folder / 'images' / f'{index:06d}-{name}.png').read_bytes()
                         for name in ('body_rgb', 'wrist_rgb')}
        (self.folder / 'images' / f'{index:06d}-wrist_rgb.png').write_bytes(native_images['wrist_rgb'])
        calibration = _calibration(model, data, packet['time_s'])
        need(calibration['wrist_rgb'] == native_cal['wrist_rgb'], 'ORIGINAL_WRIST_CALIBRATION_UNCHANGED')
        cal_path = self.folder / 'images' / f'{index:06d}-calibration.json'
        cal_path.write_text(json.dumps(calibration, allow_nan=False))
        need(sha(self.model_path.read_bytes()) == model_sha, 'ORIGINAL_COMPILED_MJB_BYTES_UNCHANGED')
        self.camera_adapter = CapturedSourceCameraAdapter(rig, SimpleNamespace(model=model, data=data),
            native_cal_raw, native_images, model_sha, rendered_external_sha256)
        return result

    def retained_files(self, index):
        return [self.native_folder / 'images' / f'{index:06d}-{suffix}' for suffix in
                ('body_rgb.png', 'wrist_rgb.png', 'calibration.json')]

    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        finally:
            self.native.close()


@dataclass(frozen=True)
class CapturedSourceCameraAdapter:
    rig: SourceCameraRig
    camera_state: object
    native_calibration_bytes: bytes
    native_image_bytes: dict
    original_model_sha256: str
    rendered_external_image_sha256: str

    def validate(self, *, actual_state, packet, calibration, image_bytes):
        """Validate both rigs against the original measured packet at binding."""
        from .fusion_observation import camera_records, checked_bytes, json_object
        from PIL import Image
        need(isinstance(self.rig, SourceCameraRig), 'ACTUAL_SOURCE_RIG_TYPE')
        original = camera_records(actual_state, json_object(self.native_calibration_bytes), packet['time_s'])
        need(record_hash(original) == self.rig.native_camera_records_sha256,
             'ORIGINAL_NATIVE_SAME_PACKET_CAMERA_RECORDS')
        _packet_data(actual_state.data, packet)
        _packet_data(self.camera_state.data, packet)
        _same_model_except_external(actual_state.model, self.camera_state.model)
        for name, raw in self.native_image_bytes.items():
            checked_bytes(raw, sha(raw), 'RETAINED_NATIVE_' + name.upper())
            with Image.open(io.BytesIO(raw)) as picture:
                need(picture.format == 'PNG' and picture.mode == 'RGB' and picture.size == (256, 256),
                     'RETAINED_ORIGINAL_256_RGB_NATIVE_PNG')
                picture.verify()
        need(set(self.native_image_bytes) == {'body_rgb', 'wrist_rgb'}
             and image_bytes['wrist_rgb'] == self.native_image_bytes['wrist_rgb'],
             'ORIGINAL_NATIVE_WRIST_PNG_UNCHANGED')
        need(sha(image_bytes['body_rgb']) == self.rendered_external_image_sha256,
             'ACTUAL_SOURCE_RIG_RENDERED_EXTERNAL_PNG_UNCHANGED')
        adapted = camera_records(self.camera_state, calibration, packet['time_s'])
        need(adapted['wrist_rgb'] == original['wrist_rgb'], 'ORIGINAL_NATIVE_WRIST_FK_UNCHANGED')
        external = adapted['body_rgb']
        need(int(self.camera_state.model.cam_bodyid[int(self.camera_state.model.camera('body_rgb').id)]) == 0
             and external['fovy_deg'] == self.rig.external_fovy_deg
             and np.allclose(external['T_world_optical'], self.rig.external_T_target_world_optical, rtol=0, atol=1e-9)
             and np.allclose(external['K'], self.rig.external_K, rtol=0, atol=1e-9),
             'EXACT_DECLARED_EXTERNAL_CAMERA_ACTUAL_RENDER_FK')
        declaration = self.rig.declaration()
        declaration.update(status='ACTUAL_TARGET_PACKET_CPU_RENDERED', schema=SCHEMA,
            native_calibration_sha256=sha(self.native_calibration_bytes),
            native_image_sha256={k: sha(v) for k, v in self.native_image_bytes.items()},
            rendered_external_image_sha256=self.rendered_external_image_sha256,
            original_compiled_model_sha256=self.original_model_sha256,
            packet_identity=dict(native_step=packet['step'], time_s=packet['time_s']),
            rendering_slot_alias={'body_rgb': OVERLAY_CAMERA, 'wrist_rgb': 'wrist_rgb'},
            effects=dict(renders=3, render_copy_FK_calls=2, integrations=0, model_queries=0),
            source_images_supplied=False, observation_payload_created=True)
        sensor = copy.deepcopy(external)
        sensor.update(name=OVERLAY_CAMERA, native_camera=False, rendering_copy_slot='body_rgb')
        cameras = {OVERLAY_CAMERA: sensor, 'wrist_rgb': copy.deepcopy(adapted['wrist_rgb'])}
        return cameras, declaration


def validate_sensor_sidecar(sensor, *, target, native_step, native_time_s, transform_sha256):
    """Check the sealed, explicit new sensor identity without rereading a world."""
    need(isinstance(sensor, dict) and sensor.get('schema') == SCHEMA
         and sensor.get('status') == 'ACTUAL_TARGET_PACKET_CPU_RENDERED'
         and sensor.get('rig') == RIG
         and sensor.get('source_identity_audit_sha256') == SOURCE_IDENTITY_SHA256
         and sensor.get('frame_map_sha256') == transform_sha256 == FRAME_MAP_SHA256,
         'SEALED_EXACT_SOURCE_SENSOR_RECIPE')
    need(sensor.get('policy_camera_role_mapping') == {'external_rgb': OVERLAY_CAMERA, 'wrist_rgb': 'wrist_rgb'}
         and sensor.get('rendering_slot_alias') == {'body_rgb': OVERLAY_CAMERA, 'wrist_rgb': 'wrist_rgb'}
         and set(target['cameras']) == {OVERLAY_CAMERA, 'wrist_rgb'}
         and sensor.get('original_compiled_model_sha256') == target['scene_sha256']
         and sensor.get('packet_identity') == dict(native_step=native_step, time_s=native_time_s),
         'SEALED_RENDER_COPY_CAMERA_PACKET_AND_SCENE')
    native = sensor['native_target_camera_records']
    need(record_hash(native) == sensor['native_camera_records_sha256']
         and native['wrist_rgb'] == target['cameras']['wrist_rgb']
         and target['image_sha256']['wrist_rgb'] == sensor['native_image_sha256']['wrist_rgb']
         and target['image_sha256']['body_rgb'] == sensor['rendered_external_image_sha256'],
         'SEALED_ORIGINAL_NATIVE_CAMERA_RETENTION_AND_WRIST')
    external = target['cameras'][OVERLAY_CAMERA]
    declared = sensor['declared_rendering_sensors'][OVERLAY_CAMERA]
    need(external['name'] == OVERLAY_CAMERA and external.get('native_camera') is False
         and external.get('rendering_copy_slot') == 'body_rgb'
         and external['fovy_deg'] == declared['fovy_deg'] == 55.
         and np.allclose(external['T_world_optical'], declared['T_world_optical'], rtol=0, atol=1e-9)
         and np.allclose(external['K'], declared['K'], rtol=0, atol=1e-9),
         'SEALED_EXTERNAL_SENSOR_ACTUAL_CALIBRATION')
    need(all(sensor.get(key) is False for key in ('native_camera_equivalence', 'image_domain_equivalence',
             'occlusion_equivalence', 'target_embodiment_qualified', 'target_task_qualified',
             'native_execution_authorized', 'source_images_supplied'))
         and sensor.get('effects') == dict(renders=3, render_copy_FK_calls=2, integrations=0, model_queries=0),
         'SOURCE_SENSOR_IS_NOT_CAMERA_EQUIVALENCE_OR_EXECUTION_GRANT')
