"""Fixed BASE camera and offline coverage; no controller or training admission.

Object geometry and masks are audit outputs only. The RGB policy input never
contains either. Legacy source-camera profiles remain unchanged.
"""
from __future__ import annotations

from itertools import product
import math
import numpy as np

CAMERA_ID = 'BODY_BASE_FIXED_V1_20261111'
POSITION = np.array([.25, -.24, .45])
LOOK_AT = np.array([.38, 0., .34])
FOVY = 75.
PHASES = ('PREGRASP', 'DESCEND', 'CLOSE', 'LIFT', 'HOLD')


def camera_rotation(position=POSITION, look_at=LOOK_AT, up=(0., 0., 1.)):
    """MuJoCo camera columns: right, up, backwards, expressed in BASE."""
    forward = np.asarray(look_at, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-10:
        raise ValueError('CAMERA_UP_PARALLEL_TO_VIEW')
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def project_box(center, rotation, half_sizes, world_optical, K, *, margin=8, size=256):
    """Amodal bbox from ALL eight box corners, including occluded corners."""
    corners = np.array(list(product((-1., 1.), repeat=3))) * half_sizes
    world = corners @ np.asarray(rotation).T + center
    optical = (world - world_optical[:3, 3]) @ world_optical[:3, :3]
    if not np.isfinite(optical).all() or (optical[:, 2] <= 0).any():
        return dict(all_in_front=False, bbox=None, margin_pass=False)
    image = optical @ np.asarray(K).T
    image = image[:, :2] / image[:, 2:3]
    low, high = image.min(0), image.max(0)
    return dict(all_in_front=True, bbox=[*low.tolist(), *high.tolist()],
                margin_pass=bool((low >= margin).all() and (high <= size-margin).all()))


def coverage(rows, *, processed=False):
    """Never drop excluded/intervention captures from the separate denominator."""
    selected = [r for r in rows if r['supervised']]
    field = 'processed_visible_pixels' if processed else 'visible_pixels'
    phases = {}
    for phase in PHASES:
        part = [r for r in selected if r['phase'] == phase]
        good = sum(max(r[field].values()) >= 64 for r in part if field in r)
        phases[phase] = dict(n=len(part), passed=good, required=math.ceil(.9*len(part)),
                             passes=bool(part) and good >= math.ceil(.9*len(part)))
    amodal = sum(r['body_projection']['margin_pass'] for r in selected)
    required = math.ceil(.95*len(selected))
    complete = len(rows) == 1500 and len(selected) == 1450
    return dict(captures=len(rows), supervised=len(selected), interventions=len(rows)-len(selected),
                amodal_passed=amodal, amodal_required=required, phases=phases,
                passes=complete and amodal >= required and all(v['passes'] for v in phases.values()))


def unchanged_packet(original, exported):
    """Every array's bytes, shape, and dtype are invariant except body RGB pins."""
    if set(original) != set(exported):
        raise ValueError('PACKET_KEYS_CHANGED')
    for key, a in original.items():
        b = exported[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('PACKET_ARRAY_METADATA_CHANGED_' + key)
        if key == 'rgb_sha256':
            a, b = a[:, 1], b[:, 1]
        if a.tobytes() != b.tobytes():
            raise ValueError('NON_BODY_PACKET_BYTES_CHANGED_' + key)


class FixedBodyRenderer:
    """One independent native MJB copy, FK only; body camera alone is patched."""
    def __init__(self, model_path):
        import mujoco
        from .fusion_source_camera_capture import _same_model_except_external
        if mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None:
            raise ValueError('PINNED_MUJOCO_WITHOUT_CONTROL_CALLBACK')
        native = mujoco.MjModel.from_binary_path(str(model_path))
        self.model = m = mujoco.MjModel.from_binary_path(str(model_path))
        if (m.nq, m.nv, m.nu, m.na, m.nplugin, m.nmocap) != (34, 32, 20, 0, 0, 0):
            raise ValueError('RECORDED_STATE_ABI_CANNOT_RECONSTRUCT_THIS_MODEL')
        c = m.camera('body_rgb').id
        if m.cam_mode[c] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            raise ValueError('FIXED_CAMERA_MODE_REQUIRED')
        m.cam_bodyid[c] = m.body('base_link').id
        m.cam_pos[c] = POSITION
        mujoco.mju_mat2Quat(m.cam_quat[c], camera_rotation().reshape(9))
        m.cam_fovy[c] = FOVY
        _same_model_except_external(native, m)
        self.data = mujoco.MjData(m)
        self.renderer = mujoco.Renderer(m, height=256, width=256)
        self.cube = m.geom('task_cube_geom').id
        if m.geom_type[self.cube] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError('AMODAL_AUDIT_REQUIRES_DECLARED_BOX_GEOMETRY')

    def render(self, packet):
        import mujoco
        from .native_replay import apply_frame, forbid_integration
        from .fusion_source_camera_capture import _calibration
        m, d, renderer = self.model, self.data, self.renderer
        with forbid_integration(mujoco) as integrations:
            mujoco.mj_resetData(m, d)
            apply_frame(mujoco, m, d, packet)
            calibration = _calibration(m, d, packet['time_s'])
            c = m.camera('body_rgb').id
            b = m.body('base_link').id
            if not (np.allclose(d.cam_xpos[c], d.xpos[b] + d.xmat[b].reshape(3, 3) @ POSITION,
                                rtol=0, atol=1e-12) and
                    np.allclose(d.cam_xmat[c].reshape(3, 3),
                                d.xmat[b].reshape(3, 3) @ camera_rotation(), rtol=0, atol=1e-12)):
                raise ValueError('CAMERA_MUST_FOLLOW_FULL_BASE_SE3')
            renderer.disable_segmentation_rendering()
            renderer.update_scene(d, camera='body_rgb')
            rgb = renderer.render().copy()
            masks = {}
            try:
                renderer.enable_segmentation_rendering()
                for camera in ('body_rgb', 'wrist_rgb'):
                    renderer.update_scene(d, camera=camera)
                    segmentation = renderer.render().copy()
                    masks[camera] = ((segmentation[:, :, 0] == self.cube) &
                                     (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
            finally:
                renderer.disable_segmentation_rendering()
            cal = calibration['body_rgb']
            box = project_box(d.geom_xpos[self.cube], d.geom_xmat[self.cube].reshape(3, 3),
                              m.geom_size[self.cube], np.array(cal['T_world_optical']), cal['K'])
            if integrations:
                raise RuntimeError('OFFLINE_RENDER_MUST_NEVER_INTEGRATE')
        return rgb, masks, calibration, box

    def close(self):
        self.renderer.close()
