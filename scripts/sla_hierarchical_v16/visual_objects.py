"""Causal, untrained two-view object observations for the stationary tabletop tasks."""

import numpy as np
from scipy import ndimage

from observation_stream import camera, rigid


COLORS = ("red", "blue")
VIEWS = ("body_rgb", "wrist_rgb")


def color_component(rgb, channel, *, reject_spanning):
    dominant = rgb[:, :, channel]
    other = np.maximum(rgb[:, :, 1], rgb[:, :, 2 - channel])
    mask = (dominant - other >= 25) & (10 * dominant >= 13 * other)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    if reject_spanning:
        # This exploratory small-object selector excludes frame-spanning components.
        for first, last in ((labels[0], labels[-1]), (labels[:, 0], labels[:, -1])):
            spanning = np.intersect1d(first, last)
            sizes[spanning[spanning > 0] - 1] = 0
    index = int(np.argmax(sizes))
    if sizes[index] < 16:
        return None
    y, x = np.nonzero(labels == index + 1)
    bbox = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    return dict(
        pixels=int(sizes[index]),
        bbox_xyxy=bbox,
        center_px=[(bbox[0] + bbox[2] + 1) / 2, (bbox[1] + bbox[3] + 1) / 2],
    )


def optical_frame(calibration, timestamp):
    frame = rigid(calibration["T_world_optical"])
    k = np.asarray(calibration["K"], dtype=float)
    if (
        not np.isfinite(timestamp)
        or not np.isclose(calibration["timestamp_s"], timestamp, rtol=0, atol=1e-8)
        or k.shape != (3, 3)
        or not np.isfinite(k).all()
        or not np.allclose(k[2], [0, 0, 1])
        or k[0, 0] <= 0
        or k[1, 1] <= 0
    ):
        raise ValueError("invalid or asynchronous camera calibration")
    return frame, np.linalg.inv(k)


def optical_ray(optical, uv):
    frame, inverse_k = optical
    direction = frame[:3, :3] @ inverse_k @ [*uv, 1]
    return frame[:3, 3], direction / np.linalg.norm(direction)


def triangulate(body, wrist):
    a, u = body
    b, v = wrist
    angle = float(np.rad2deg(np.arccos(np.clip(u @ v, -1, 1))))
    result = dict(center_world_m=None, angle_deg=angle, separation_mm=None)
    if min(angle, 180 - angle) < 2:
        return dict(**result, status="near_parallel")
    depths = np.linalg.lstsq(np.column_stack((u, -v)), b - a, rcond=None)[0]
    pa, pb = a + depths[0] * u, b + depths[1] * v
    gap = float(1000 * np.linalg.norm(pa - pb))
    result["separation_mm"] = gap
    if min(depths) <= 0:
        return dict(**result, status="behind_camera")
    if gap > 15:
        return dict(**result, status="ray_separation")
    result["center_world_m"] = ((pa + pb) / 2).tolist()
    return dict(**result, status="measured")


def estimate_objects(
    body_rgb, wrist_rgb, calibration, timestamp, *, reject_spanning=False
):
    """Both centers from simultaneous pixels and calibrated optical poses only."""
    frames = [optical_frame(calibration[k], timestamp) for k in VIEWS]
    rgb = [camera(image).astype(np.int16) for image in (body_rgb, wrist_rgb)]
    result = {}
    for color, channel in zip(COLORS, (0, 2)):
        components = [
            color_component(view, channel, reject_spanning=reject_spanning)
            for view in rgb
        ]
        diagnostics = dict(zip(VIEWS, components))
        if any(c is None for c in components):
            missing = [v for v, c in diagnostics.items() if c is None]
            estimate = dict(
                status="missing_" + "_and_".join(missing), center_world_m=None
            )
        else:
            estimate = triangulate(
                *(optical_ray(f, c["center_px"]) for f, c in zip(frames, components))
            )
        result[color] = dict(**estimate, views=diagnostics)
    return result


class VisualLiftProgress:
    """One instance per episode, anchored exclusively at its first scheduled query."""

    def __init__(self):
        self.initial_z = {}
        self.last_time = None

    def observe(self, timestamp, objects):
        if not np.isfinite(timestamp) or (
            self.last_time is not None and timestamp <= self.last_time
        ):
            raise ValueError("chronological observations required")
        z = {
            c: (
                None
                if objects[c]["center_world_m"] is None
                else objects[c]["center_world_m"][2]
            )
            for c in COLORS
        }
        if self.last_time is None:
            self.initial_z = z.copy()
        self.last_time = timestamp
        result = {}
        for c in COLORS:
            dz = (
                None
                if self.initial_z[c] is None or z[c] is None
                else z[c] - self.initial_z[c]
            )
            result[c] = dict(
                delta_z_m=dz, sampled_height20=None if dz is None else dz >= 0.020
            )
        return result
