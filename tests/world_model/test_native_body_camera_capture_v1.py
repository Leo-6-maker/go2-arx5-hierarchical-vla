"""Real MuJoCo FK/ABI with renderer doubles; not native pixel qualification."""
import copy
import hashlib
import json
import numpy as np
import pytest

from tests.world_model.test_fusion_source_camera_capture import scene, forbid_step
from tests.world_model.test_fusion_source_camera_rig import FRAME
from scripts.pi0292.relay_protocol import sealed, unseal
from scripts.pi0294.task_state import ActualState
from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding, shadow_inference_payload, transform_hash
from go2arx5_vla.world_model.native_body_camera_capture_v1 import BodyBaseCameraCapture
from go2arx5_vla.world_model.native_camera_transfer_v1 import CAMERA_ID


def test_new_typed_body_recipe_binds_full_base_pose_without_changing_wrist_or_packet(tmp_path, monkeypatch):
    import mujoco
    path, model, _, packet, raw = scene(tmp_path)
    cube = model.geom('task_cube_geom').id
    model.geom_type[cube] = mujoco.mjtGeom.mjGEOM_BOX
    model.geom_size[cube] = [.03, .03, .03]
    mujoco.mj_saveModel(model, str(path), None)
    renderers = []
    class Renderer:
        def __init__(self, model, **kwargs):
            self.model = model
            self.closed = False
            renderers.append(self)
        def update_scene(self, data, *, camera):
            self.camera = camera
        def disable_segmentation_rendering(self):
            pass
        def render(self):
            return np.full((256,256,3), 40 if self.camera=='body_rgb' else 80, np.uint8)
        def close(self):
            self.closed = True
    monkeypatch.setattr(mujoco, 'Renderer', Renderer)
    forbid_step(monkeypatch)
    root = tmp_path/'capture'
    (root/'images').mkdir(parents=True)
    camera = BodyBaseCameraCapture(FRAME)
    recorder = camera.recorder(path, root)
    try:
        capture = recorder.save(0, packet)
        calibration = (root/'images/000000-calibration.json').read_bytes()
        images = {k:(root/'images'/('000000-'+k+'.png')).read_bytes() for k in ('body_rgb','wrist_rgb')}
        binding = TargetShadowBinding('MUJOCO_GO2_ARX5_WORLD', hashlib.sha256(path.read_bytes()).hexdigest(),
                                      '1'*64, FRAME, transform_hash(FRAME))
        record = binding.bind(actual_state=ActualState(model), native_state_bytes=raw,
            expected_state_sha256=hashlib.sha256(raw).hexdigest(), capture=capture,
            calibration_bytes=calibration, expected_calibration_sha256=hashlib.sha256(calibration).hexdigest(),
            image_bytes=images, expected_image_sha256={k:hashlib.sha256(v).hexdigest() for k,v in images.items()},
            language='Grasp the red cube.', snapshot_uuid='12345678-1234-4234-8234-123456789abc',
            camera_adapter=recorder.camera_adapter)
        payload = shadow_inference_payload(record)
        body = unseal(record)
        assert body['target_observation']['sensor_adapter']['rig'] == CAMERA_ID
        assert [i['camera'] for i in payload['images']] == ['external_rgb', 'wrist_rgb']
        assert payload['images'][1]['sha256'] == hashlib.sha256(
            (root/'native-camera/images/000000-wrist_rgb.png').read_bytes()).hexdigest()
        assert not body['provenance']['native_execution_authorized']
        assert model.cam_pos[model.camera('body_rgb').id, 2] == .28
        # Re-sealing is not sufficient to turn a changed geometry into the
        # fixed recipe. Downstream transport must recheck its camera contract.
        for field, value in [('position_BASE_m',[.3,0,.5]), ('vertical_fovy_deg',55)]:
            altered = copy.deepcopy(body)
            altered['target_observation']['sensor_adapter'][field] = value
            with pytest.raises(ValueError, match='SEALED_FIXED_BODY_RECIPE'):
                shadow_inference_payload(sealed(altered))
    finally:
        recorder.close()
    assert all(r.closed for r in renderers)


def test_body_recipe_rejects_undeclared_policy_frame_and_identity():
    with pytest.raises(ValueError):
        BodyBaseCameraCapture(np.eye(4))
    with pytest.raises(ValueError):
        BodyBaseCameraCapture(FRAME, name='OBJECT_TRACKED_CAMERA')
