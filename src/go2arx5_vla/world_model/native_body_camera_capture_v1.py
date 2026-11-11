"""Independent fixed BASE observation recipe, retaining the original wrist.

It supplies pixels and custody only. Numerical/physical policy admission is a
separate versioned contract; source-camera equivalence is never claimed.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import copy
import io
import json
import numpy as np

from .native_camera_transfer_v1 import CAMERA_ID, POSITION, LOOK_AT, FOVY, camera_rotation, FixedBodyRenderer
from .fusion_source_camera_capture import sha, need, _packet_data, _same_model_except_external
from .fusion_source_camera_rig import FRAME_MAP_SHA256
from scripts.pi0294.task_native_worker import FrameRecorder

SCHEMA = 'NATIVE_BODY_CAMERA_CAPTURE_V1'


@dataclass(frozen=True)
class BodyBaseCameraCapture:
    T_source_from_target: object
    name: str = CAMERA_ID

    def __post_init__(self):
        frame = np.asarray(self.T_source_from_target, dtype='<f8')
        need(self.name == CAMERA_ID and frame.shape == (4, 4)
             and sha(frame.tobytes()) == FRAME_MAP_SHA256, 'BODY_CAMERA_SAME_DECLARED_POLICY_FRAME')
        object.__setattr__(self, 'T_source_from_target', np.frombuffer(frame.tobytes(), dtype='<f8').reshape(4, 4))

    def recorder(self, model_path, folder):
        return BodyBaseCameraRecorder(model_path, folder)


class BodyBaseCameraRecorder:
    def __init__(self, model_path, folder):
        self.model_path, self.folder = Path(model_path), Path(folder)
        self.native_folder = self.folder/'native-camera'
        (self.native_folder/'images').mkdir(parents=True, exist_ok=False)
        self.native = FrameRecorder(self.model_path, self.native_folder)
        self.render_copy = None
        self.camera_adapter = None

    def save(self, index, packet):
        from PIL import Image
        need(self.camera_adapter is None, 'ONE_FRESH_BODY_CAPTURE_PER_RECORDER')
        model_sha = sha(self.model_path.read_bytes())
        result = self.native.save(index, packet)
        self.render_copy = FixedBodyRenderer(self.model_path)
        rgb, _, calibration, _ = self.render_copy.render(packet, audit_masks=False)
        prefix = f'{index:06d}-'
        native_cal = (self.native_folder/'images'/(prefix+'calibration.json')).read_bytes()
        need(calibration['wrist_rgb'] == json.loads(native_cal)['wrist_rgb'],
             'BODY_RECIPE_RETAINS_NATIVE_WRIST_CALIBRATION')
        images = {k:(self.native_folder/'images'/(prefix+k+'.png')).read_bytes()
                  for k in ('body_rgb', 'wrist_rgb')}
        Image.fromarray(rgb).save(self.folder/'images'/(prefix+'body_rgb.png'))
        (self.folder/'images'/(prefix+'wrist_rgb.png')).write_bytes(images['wrist_rgb'])
        (self.folder/'images'/(prefix+'calibration.json')).write_text(json.dumps(calibration, allow_nan=False))
        need(sha(self.model_path.read_bytes()) == model_sha, 'BODY_RECIPE_ORIGINAL_MJB_BYTES_UNCHANGED')
        self.camera_adapter = CapturedBodyBaseCameraAdapter(
            SimpleNamespace(model=self.render_copy.model, data=self.render_copy.data),
            native_cal, images, model_sha,
            sha((self.folder/'images'/(prefix+'body_rgb.png')).read_bytes()))
        return result

    def retained_files(self, index):
        return [self.native_folder/'images'/f'{index:06d}-{s}' for s in
                ('body_rgb.png', 'wrist_rgb.png', 'calibration.json')]

    def close(self):
        try:
            if self.render_copy is not None:
                self.render_copy.close()
        finally:
            self.native.close()


@dataclass(frozen=True)
class CapturedBodyBaseCameraAdapter:
    camera_state: object
    native_calibration_bytes: bytes
    native_image_bytes: dict
    original_model_sha256: str
    rendered_body_image_sha256: str

    def validate(self, *, actual_state, packet, calibration, image_bytes):
        from .fusion_observation import camera_records
        from PIL import Image
        _packet_data(actual_state.data, packet)
        _packet_data(self.camera_state.data, packet)
        _same_model_except_external(actual_state.model, self.camera_state.model)
        original = camera_records(actual_state, json.loads(self.native_calibration_bytes), packet['time_s'])
        adapted = camera_records(self.camera_state, calibration, packet['time_s'])
        need(adapted['wrist_rgb'] == original['wrist_rgb']
             and image_bytes['wrist_rgb'] == self.native_image_bytes['wrist_rgb']
             and sha(image_bytes['body_rgb']) == self.rendered_body_image_sha256,
             'BODY_RECIPE_EXACT_WRIST_BYTES_AND_ACTUAL_NEW_BODY')
        for raw in self.native_image_bytes.values():
            with Image.open(io.BytesIO(raw)) as im:
                need(im.mode == 'RGB' and im.size == (256, 256) and im.format == 'PNG', 'BODY_NATIVE_RETAINED_PNG')
        base = np.eye(4)
        b = self.camera_state.model.body('base_link').id
        base[:3, :3] = self.camera_state.data.xmat[b].reshape(3, 3)
        base[:3, 3] = self.camera_state.data.xpos[b]
        sensor = copy.deepcopy(adapted['body_rgb'])
        sensor.update(name=CAMERA_ID, native_camera=False, rendering_copy_slot='body_rgb')
        cameras = {CAMERA_ID:sensor, 'wrist_rgb':copy.deepcopy(adapted['wrist_rgb'])}
        declaration = dict(schema=SCHEMA, rig=CAMERA_ID, status='ACTUAL_TARGET_PACKET_CPU_RENDERED',
            policy_camera_role_mapping={'external_rgb':CAMERA_ID, 'wrist_rgb':'wrist_rgb'},
            position_BASE_m=POSITION.tolist(), look_at_BASE_m=LOOK_AT.tolist(), up_BASE=[0, 0, 1],
            vertical_fovy_deg=FOVY, T_world_BASE=base.tolist(),
            original_compiled_model_sha256=self.original_model_sha256,
            native_target_camera_records=original,
            native_calibration_sha256=sha(self.native_calibration_bytes),
            native_image_sha256={k:sha(v) for k, v in self.native_image_bytes.items()},
            rendered_body_image_sha256=self.rendered_body_image_sha256,
            packet_identity=dict(native_step=packet['step'], time_s=packet['time_s']),
            source_images_supplied=False, native_execution_authorized=False,
            image_domain_equivalence=False, integrations=0, model_queries=0)
        # Reuse exactly the sealed-sidecar validation also used after transport.
        validate_sensor_sidecar(declaration, target=dict(cameras=cameras,
            scene_sha256=self.original_model_sha256,
            image_sha256={k:sha(v) for k, v in image_bytes.items()}),
            native_step=packet['step'], native_time_s=packet['time_s'], transform_sha256=FRAME_MAP_SHA256)
        return cameras, declaration


def validate_sensor_sidecar(sensor, *, target, native_step, native_time_s, transform_sha256):
    from .fusion_observation import rigid
    need(sensor.get('schema') == SCHEMA and sensor.get('rig') == CAMERA_ID
         and sensor.get('status') == 'ACTUAL_TARGET_PACKET_CPU_RENDERED'
         and transform_sha256 == FRAME_MAP_SHA256
         and sensor.get('position_BASE_m') == POSITION.tolist()
         and sensor.get('look_at_BASE_m') == LOOK_AT.tolist()
         and sensor.get('up_BASE') == [0, 0, 1] and sensor.get('vertical_fovy_deg') == FOVY,
         'SEALED_FIXED_BODY_RECIPE')
    need(sensor['policy_camera_role_mapping'] == {'external_rgb':CAMERA_ID, 'wrist_rgb':'wrist_rgb'}
         and set(target['cameras']) == {CAMERA_ID, 'wrist_rgb'}
         and sensor['original_compiled_model_sha256'] == target['scene_sha256']
         and sensor['packet_identity'] == dict(native_step=native_step, time_s=native_time_s),
         'SEALED_BODY_PACKET_SCENE_AND_ROLES')
    base = rigid(sensor['T_world_BASE'])
    expected = np.eye(4)
    expected[:3, :3] = base[:3, :3] @ camera_rotation() @ np.diag([1., -1., -1.])
    expected[:3, 3] = base[:3, 3] + base[:3, :3] @ POSITION
    camera = target['cameras'][CAMERA_ID]
    focal = 128 / np.tan(np.deg2rad(FOVY)/2)
    need(camera['parent_body'] == 'base_link' and camera['position_local_m'] == POSITION.tolist()
         and camera['fovy_deg'] == FOVY and camera['name'] == CAMERA_ID
         and camera['width'] == camera['height'] == 256
         and np.allclose(camera['T_world_optical'], expected, rtol=0, atol=1e-9)
         and np.allclose(camera['K'], [[focal, 0, 128], [0, focal, 128], [0, 0, 1]], rtol=0, atol=1e-9)
         and target['cameras']['wrist_rgb'] == sensor['native_target_camera_records']['wrist_rgb']
         and target['image_sha256']['wrist_rgb'] == sensor['native_image_sha256']['wrist_rgb']
         and target['image_sha256']['body_rgb'] == sensor['rendered_body_image_sha256'],
         'SEALED_FULL_BASE_SE3_AND_ORIGINAL_WRIST')
    need(all(sensor[k] is False for k in ('native_execution_authorized', 'source_images_supplied', 'image_domain_equivalence'))
         and sensor['integrations'] == sensor['model_queries'] == 0, 'BODY_OBSERVATION_GRANTS_NO_EXECUTION')
