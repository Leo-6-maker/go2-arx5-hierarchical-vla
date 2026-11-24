"""Keep the original Body and Wrist recipes on the fruit scene."""

from pathlib import Path

from go2arx5_vla.world_model.fusion_source_camera_capture import _same_model_except_external
from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
from go2arx5_vla.world_model.native_body_camera_capture_v1 import BodyBaseCameraSession
from go2arx5_vla.world_model.native_camera_transfer_v1 import FOVY, POSITION, camera_rotation
from scripts.pi0294.task_native_worker import FrameRecorder

from sea_three_view import AuxRGBRenderer
from source_custody import pin, require


def fruit_body_model(path):
    import mujoco as mj

    require(mj.__version__ == "3.2.7" and mj.get_mjcb_control() is None,
            "pinned camera runtime without control callback")
    original = mj.MjModel.from_binary_path(str(path))
    m = mj.MjModel.from_binary_path(str(path))
    require((m.nq, m.nv, m.nu, m.na, m.nplugin, m.nmocap) == TwoObjectBodyRenderer.MODEL_ABI
            and m.geom_type[m.geom("task_cube_geom").id] == mj.mjtGeom.mjGEOM_ELLIPSOID
            and m.geom_type[m.geom("task_blue_cube_geom").id] == mj.mjtGeom.mjGEOM_BOX,
            "registered fruit and passive distractor required")
    c = m.camera("body_rgb").id
    require(m.cam_mode[c] == mj.mjtCamLight.mjCAMLIGHT_FIXED, "fixed body camera required")
    m.cam_bodyid[c] = m.body("base_link").id
    m.cam_pos[c] = POSITION
    mj.mju_mat2Quat(m.cam_quat[c], camera_rotation().reshape(9))
    m.cam_fovy[c] = FOVY
    _same_model_except_external(original, m)
    return m


class FruitBodyRenderer(TwoObjectBodyRenderer):
    def __init__(self, model_path):
        import mujoco as mj

        # Inherited rendering and visible masks use the true ellipsoid. The
        # inherited optional corner projection is its oriented bounding box.
        self.model = fruit_body_model(model_path)
        self.cube = self.model.geom("task_cube_geom").id
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, height=256, width=256)


class FruitCameraSession(BodyBaseCameraSession):
    def __init__(self, path, scene_sha256):
        super().__init__(pin(Path(path), scene_sha256))
        self.auxiliary = None

    def acquire(self, model_path, folder):
        require(not self.busy and not self.closed and Path(model_path) == self.model_path,
                "exclusive same-scene fruit capture required")
        pin(self.model_path, self.model_sha)
        if self.native is None:
            self.render_copy = FruitBodyRenderer(self.model_path)
            self.native = FrameRecorder(self.model_path, folder)
            self.native.reset_each_packet = True
        else:
            self.native.folder = folder
        self.busy = True
        return self.native, self.render_copy

    def capture_three(self, episode, packet):
        require(not self.closed and not self.busy, "camera session is unavailable")
        if self.auxiliary is None:
            self.auxiliary = AuxRGBRenderer(self.model_path, self.model_sha)
        originals, body, _, calibration = self.render_memory(packet)
        auxiliary = self.auxiliary.capture(episode, packet)
        return body, originals["wrist_rgb"], auxiliary, calibration

    def close(self):
        super().close()
        if self.auxiliary is not None:
            self.auxiliary.close()
            self.auxiliary = None
