"""One fixed body-mounted camera and its synchronous route to the official left slot."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock
import time

import numpy as np
from scipy.spatial.transform import Rotation

from sea_observation import AUXILIARY_CAMERA, THREE_VIEW_V1, Frame, SensorObservation, prepare_sensor
from source_custody import BASE, pin, require
from policy_query import input_identity, tensor_sha

import sys
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from observation_stream import ObservationStream, rigid

H, W = 480, 640
FOVY_DEG = 75.
# Static left/front bracket and nominal right workspace in base_link coordinates.
# The optical pose is independent of the scene, observed object, or policy output.
POSITION_M = (.20, .22, .42)
FIXED_SIGHT_M = (.42, 0., .31)


def camera_local():
    front = np.subtract(FIXED_SIGHT_M, POSITION_M)
    front /= np.linalg.norm(front)
    right = np.cross(front, [0.,0.,1.])
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3,:3] = np.column_stack((right, np.cross(right,front), -front))
    transform[:3,3] = POSITION_M
    return rigid(transform)


def calibration():
    local = camera_local()
    wxyz = Rotation.from_matrix(local[:3,:3]).as_quat()[[3,0,1,2]]
    f = (H/2.)/np.tan(np.radians(FOVY_DEG)/2.)
    return dict(
        profile=THREE_VIEW_V1, physical_camera=AUXILIARY_CAMERA, physical_parent="base_link",
        policy_slot="cam_left_wrist", raw_hwc=[H,W,3], encoding="RGB_UINT8",
        base_from_camera=local.tolist(), camera_wxyz=wxyz.tolist(), fovy_deg=FOVY_DEG,
        K=[[f,0.,W/2.],[0.,f,H/2.],[0.,0.,1.]], mujoco_axes="right/up/back",
        source_recording_extrinsics="UNKNOWN",
    )


def calibration_sha256():
    return hashlib.sha256(json.dumps(calibration(),sort_keys=True,separators=(",",":")).encode()).hexdigest()


@dataclass(frozen=True)
class AuxCapture:
    frame: Frame
    calibration_sha256: str
    render_wall_s: float
    native_state_sha256: str

    def __post_init__(self):
        require(type(self.frame) is Frame and self.frame.camera == AUXILIARY_CAMERA
                and self.frame.rgb.shape == (H,W,3) and self.frame.rgb.any()
                and self.calibration_sha256 == calibration_sha256()
                and np.isfinite(self.render_wall_s) and self.render_wall_s >= 0
                and len(self.native_state_sha256) == 64, "invalid auxiliary capture")


def native_sample_identity(episode, packet):
    value = dict(episode=episode,tick=packet["step"],time_s=packet["time_s"],
                 state={k:tensor_sha(packet[k]) for k in ("qpos","qvel","ctrl")})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def camera_copy(path):
    """Use a separate optical render copy of an existing sealed MJB."""
    import mujoco as mj
    from go2arx5_vla.world_model.fusion_source_camera_capture import _same_model_except_external

    require(mj.__version__ == "3.2.7" and mj.get_mjcb_control() is None, "camera runtime identity")
    source = mj.MjModel.from_binary_path(str(path))
    copy = mj.MjModel.from_binary_path(str(path))
    optical = copy.camera("body_rgb").id
    require(
        copy.cam_mode[optical] == mj.mjtCamLight.mjCAMLIGHT_FIXED
        and copy.cam_bodyid[optical] == copy.body("base_link").id
        and copy.vis.global_.offheight >= H and copy.vis.global_.offwidth >= W,
        "sealed model has no compatible fixed render camera or framebuffer",
    )
    mount = calibration()
    copy.cam_pos[optical] = POSITION_M
    copy.cam_quat[optical] = mount["camera_wxyz"]
    copy.cam_fovy[optical] = FOVY_DEG
    _same_model_except_external(source,copy)
    return copy, optical


class AuxRGBRenderer:
    """One auxiliary optical view. The original body/wrist producer is separate."""

    def __init__(self, scene, sha256):
        import mujoco as mj

        self.model, self.optical = camera_copy(pin(Path(scene),sha256))
        self.scene_sha256 = sha256
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model,height=H,width=W)

    def capture(self, episode, packet):
        import mujoco as mj
        from go2arx5_vla.world_model.native_replay import apply_frame, forbid_integration

        t0 = time.monotonic()
        with forbid_integration(mj) as calls:
            mj.mj_resetData(self.model,self.data)
            apply_frame(mj,self.model,self.data,packet)
            base = self.data.body("base_link")
            b = np.eye(4)
            b[:3,:3], b[:3,3] = base.xmat.reshape(3,3), base.xpos
            world = b @ camera_local()
            require(np.allclose(self.data.cam_xpos[self.optical],world[:3,3],rtol=0,atol=1e-9)
                    and np.allclose(self.data.cam_xmat[self.optical].reshape(3,3),world[:3,:3],rtol=0,atol=1e-9),
                    "aux camera does not follow fixed base transform")
            self.renderer.update_scene(self.data,camera=self.optical)
            captured = self.renderer.render()
        require(not calls,"camera capture integrated the render model")
        return AuxCapture(Frame(episode,AUXILIARY_CAMERA,packet["step"],captured),calibration_sha256(),
                          time.monotonic()-t0,native_sample_identity(episode,packet))

    def close(self):
        self.renderer.close()


class ThreeViewStream(ObservationStream):
    """Keep the original state, memory and query-binding code with a mandatory auxiliary capture."""

    def __init__(self, episode, prompt, virtual_left, source_origin, link6_tcp, virtual_left_sha256):
        left = np.asarray(virtual_left,dtype=np.float32)
        require(tensor_sha(left) == virtual_left_sha256, "virtual left arm does not match registration")
        super().__init__(episode,prompt,left,source_origin,link6_tcp)
        self._lock = RLock()
        self._observed_aux = None

    def observe(self, episode, tick, time_s, body, wrist, armbase, link6, tcp, finger_q_m,
                *, auxiliary, sample_sha256):
        with self._lock:
            require(type(auxiliary) is AuxCapture and auxiliary.frame.episode == episode
                    and auxiliary.frame.tick == tick and auxiliary.native_state_sha256 == sample_sha256,
                    "missing, stale or differently sampled auxiliary camera")
            super().observe(episode,tick,time_s,body,wrist,armbase,link6,tcp,finger_q_m)
            self._observed_aux = auxiliary

    def _sensor_locked(self, query):
        old = super()._input_locked()
        tick = self._ticks[-1]
        history = [0,*[max(0,tick-500*i) for i in range(8,0,-1)],tick]
        images = old["images"]
        require(self._observed_aux is not None and images["cam_high"][-1].any() and images["cam_right_wrist"].any(),
                "missing live original or third view")
        return SensorObservation(
            self.episode,query,tick,"GO2_NATIVE_2MS",
            tuple(Frame(self.episode,"cam_high",t,rgb) for t,rgb in zip(history,images["cam_high"],strict=True)),
            Frame(self.episode,"cam_right_wrist",tick,images["cam_right_wrist"]),
            self._observed_aux.frame,old["state"],old["prompt"],THREE_VIEW_V1,
        )

    def _input_locked(self):
        return self._sensor_locked("latest").legacy_input()

    def begin_query_with_receipt(self):
        with self._lock:
            key, observation = super().begin_query()
            try:
                sensor = self._sensor_locked(key)
                ident, fields = input_identity(observation)
                require(ident == input_identity(sensor.legacy_input())[0],"three-view payload changed during one query")
                aux = self._observed_aux
                receipt = dict(
                    profile=THREE_VIEW_V1, episode=self.episode, query=key, tick=sensor.tick,
                    time_s=.002*sensor.tick, input_sha256=ident, input_fields=fields,
                    current_physical_cameras=("body_rgb","wrist_rgb",AUXILIARY_CAMERA),
                    history_ticks=[f.tick for f in sensor.head[:-1]], current_aux_tick=aux.frame.tick,
                    aux_calibration_sha256=aux.calibration_sha256, aux_rgb_sha256=tensor_sha(aux.frame.rgb),
                    native_sample_sha256=aux.native_state_sha256, aux_render_wall_s=aux.render_wall_s,
                )
            except BaseException:
                self.discard_query(key)
                raise
            return key,observation,receipt

    def prepared_latest(self, official_prepare):
        with self._lock:
            return prepare_sensor(self._sensor_locked("latest"),official_prepare)
