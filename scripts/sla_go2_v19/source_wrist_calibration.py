"""Audit static table landmarks and cross-episode wrist calibration on the fixed source videos."""

import argparse
from collections import Counter
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from source_custody import SOURCE_H5, VIEWS, array_hash, pin, require, write_new
from source_head_calibration import CALIBRATION_MEDIAN_PX, CALIBRATION_P95_PX, SIZE

STRIDE = 15
OPTICAL_AT_REST = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
WARM_STARTS = (-.4, 0., .4)
PARAMETERS = (
    "mount_rx_rad", "mount_ry_rad", "mount_rz_rad", "mount_x_m", "mount_y_m", "mount_z_m",
    "focal_px", "cx_px", "cy_px", "corner_x_m", "corner_y_m", "corner_z_m",
    "edge_x_per_abs_y", "edge_z_per_abs_y",
)


def tabletop_edge(rgb):
    """Extract the upper light-table contour only where both sides support an edge."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = ((gray >= 140) & (hsv[..., 1] <= 55)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    if n_labels < 2:
        return np.empty((0, 2), dtype=int)
    area = stats[1:, cv2.CC_STAT_AREA]
    if max(area) < 10_000:
        return np.empty((0, 2), dtype=int)
    table = labels == (1 + np.argmax(area))
    x = np.arange(12, gray.shape[1] - 12)
    top = table[:, x].argmax(axis=0)
    valid = (top >= 12) & (top <= 334)
    x, top = x[valid], top[valid]
    valid = (gray[top - 9, x] < 110) & (gray[top + 9, x] >= 140)
    return np.column_stack([x[valid], top[valid]])


def line_segments(points):
    canvas = np.zeros(SIZE[::-1], dtype=np.uint8)
    canvas[points[:, 1], points[:, 0]] = 255
    candidates = cv2.HoughLinesP(canvas, 1, np.pi / 360, threshold=24,
                                 minLineLength=34, maxLineGap=10)
    if candidates is None:
        return []
    models = []
    for x1, y1, x2, y2 in candidates[:, 0]:
        if abs(x2 - x1) < 24:
            continue
        slope = (y2 - y1) / (x2 - x1)
        line = slope * points[:, 0] + y1 - slope * x1
        support = points[np.abs(line - points[:, 1]) <= 2.0]
        if len(support) < 27:
            continue
        # Use a single contiguous physical edge rather than distant collinear fragments.
        groups = np.split(support, np.flatnonzero(np.diff(support[:, 0]) > 12) + 1)
        segment = max(groups, key=len)
        if len(segment) < 27 or np.ptp(segment[:, 0]) < 32:
            continue
        slope, intercept = np.polyfit(segment[:, 0], segment[:, 1], 1)
        models.append((slope, intercept, int(segment[0, 0]), int(segment[-1, 0]), len(segment)))
    return models


def table_corners(rgb):
    models = line_segments(tabletop_edge(rgb))
    far = [line for line in models if abs(line[0]) <= .29 and line[4] >= 92]
    if not far:
        return {}, "FAR_EDGE_UNOBSERVED", None
    m, b, left, right, _ = max(far, key=lambda line: line[4])
    result = {}
    for label, end, sign in (("left", left, -1), ("right", right, 1)):
        for slope, intercept, low, high, support in sorted(models, key=lambda line: -line[4]):
            if sign * (slope - m) < .23:
                continue
            meeting = (b - intercept) / (slope - m)
            proximal, distal = (high, low) if label == "left" else (low, high)
            if (abs(meeting - end) <= 16 and abs(meeting - proximal) <= 16
                    and sign * (distal - meeting) >= 30 and 22 <= meeting <= SIZE[0] - 22):
                result[label] = (float(meeting), float(m * meeting + b), support)
                break
    return result, "OBSERVED" if result else "NO_TWO_EDGE_INTERSECTION", (float(m), float(b))


def self_check():
    rgb = np.full((480, 640, 3), 18, dtype=np.uint8)
    require(table_corners(rgb)[0] == {}, "flat background produced a table corner")
    outline = np.array([[151, 102], [487, 116], [639, 258], [639, 479], [0, 479], [0, 283]])
    cv2.fillConvexPoly(rgb, outline, (225, 225, 225))
    corners, _, _ = table_corners(rgb)
    require(set(corners) == {"left", "right"} and all(
        np.linalg.norm(np.asarray(corners[side][:2]) - vertex) < 2. for side, vertex in
        zip(("left", "right"), outline[:2], strict=True)), "known corner recovery changed")


def extract(path):
    from XPolicyLab.utils.process_data import decode_image_bit

    with h5py.File(path) as h:
        require(int(h["additional_info/frequency"][()]) == 30, "archive clock changed")
        result = {}
        for view in VIEWS:
            frames = h[f"vision/{view}/colors"]
            rows = []
            for frame in range(0, len(frames), STRIDE):
                rgb = decode_image_bit(frames[frame])
                require(rgb.shape == (*SIZE[::-1], 3) and rgb.dtype == np.uint8,
                        "original RGB dimensions changed")
                corners, status, edge = table_corners(rgb)
                rows.append(dict(frame=frame, status=status, corners=corners, far_edge=edge,
                                 rgb_sha256=array_hash(rgb)))
            entry = dict(rows=rows, coverage=dict(Counter(row["status"] for row in rows)))
            if view != "cam_head":
                side = view.split("_")[1]
                poses = h[f"state/{side}_ee_poses"][:]
                require(poses.shape == (len(frames), 7), "wrist/state frame count mismatch")
                entry["link6"] = poses
            result[view] = entry
    return result


def observations(view, side):
    rows = [row for row in view["rows"] if side in row["corners"]]
    frames = np.array([row["frame"] for row in rows], dtype=int)
    poses = view["link6"][frames]
    rot = Rotation.from_quat(poses[:, [4, 5, 6, 3]]) if len(frames) else None
    return dict(frames=frames, pixels=np.array([row["corners"][side][:2] for row in rows]),
                edge=np.array([row["far_edge"] for row in rows]), xyz=poses[:, :3], rot=rot)


def camera_fit(train, holdout, side):
    """Infer a rigid mount, pinhole intrinsics, and a stationary point and line."""
    if min(len(d["frames"]) for d in (train, holdout)) < 10:
        return dict(status="INSUFFICIENT_CORNER_FRAMES")
    lower = [-1.2] * 3 + [-.14] * 3 + [240., 250., 175., .35, -.8, -.55, -.5, -.5]
    upper = [1.2] * 3 + [.14] * 3 + [900., 390., 305., 1.7, .8, .4, .5, .5]
    sign = -1 if side == "left" else 1

    def project(x, data):
        mount = Rotation.from_rotvec(x[:3]).as_matrix() @ OPTICAL_AT_REST
        direction = np.array([x[12], sign, x[13]])
        # This arbitrary point on the 3-D line supplies direction, not a measured edge length.
        points = np.stack([x[9:12], x[9:12] + .25 * direction / np.linalg.norm(direction)])
        delta = points[None, :, :] - data["xyz"][:, None, :]
        link = np.einsum("nji,nkj->nki", data["rot"].as_matrix(), delta) - x[3:6]
        xyz_optical = link @ mount
        depth = xyz_optical[..., 2]
        uv = x[6] * xyz_optical[..., :2] / np.maximum(depth[..., None], .02) + x[7:9]
        slope, intercept = data["edge"].T
        to_line = (uv[:, 1, 1] - slope * uv[:, 1, 0] - intercept) / np.hypot(slope, 1.)
        return uv[:, 0], to_line, depth

    def residual(x):
        px, to_line, depth = project(x, train)
        return np.r_[(px - train["pixels"]).ravel(), to_line, 500 * np.minimum(depth - .02, 0.).ravel()]

    best = None
    for down in WARM_STARTS:
        start = [0., down, 0., 0., 0., .05, 500., 320., 240.,
                 .9, -.23 * sign, -.1, 0., 0.]
        fit = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1",
                            f_scale=2., x_scale="jac", max_nfev=240)
        if best is None or fit.cost < best.cost:
            best = fit

    def score(data):
        predicted, to_line, depth = project(best.x, data)
        error = np.linalg.norm(predicted - data["pixels"], axis=1)
        motion = data["xyz"] - data["xyz"][0]
        angle = (data["rot"][0].inv() * data["rot"]).magnitude()
        return dict(n=len(error), median_px=float(np.median(error)),
                    p95_px=float(np.percentile(error, 95)), positive_depth=bool(depth.min() > 0),
                    line_p95_px=float(np.percentile(np.abs(to_line), 95)),
                    from_first_max_translation_mm=float(np.linalg.norm(motion, axis=1).max() * 1000),
                    from_first_max_rotation_deg=float(np.rad2deg(angle.max())),
                    residuals=[dict(frame=int(frame), observed=uv.tolist(), projected=prediction.tolist(),
                                    error_px=float(value), line_error_px=float(line))
                               for frame, uv, prediction, value, line in
                               zip(data["frames"], data["pixels"], predicted, error, to_line, strict=True)])

    singular = np.linalg.svd(best.jac, compute_uv=False)
    training, check = score(train), score(holdout)
    near_bound = np.flatnonzero(np.minimum(best.x - lower, upper - best.x)
                                <= .001 * (np.array(upper) - lower))
    return dict(status="MATCHED_CORNER_AND_LINE_DIAGNOSTIC", converged=bool(best.success),
                meets_pixel_gate=check["positive_depth"] and check["median_px"] <= CALIBRATION_MEDIAN_PX
                and check["p95_px"] <= CALIBRATION_P95_PX and check["line_p95_px"] <= CALIBRATION_P95_PX,
                parameters=best.x.tolist(),
                training=training, validation=check, near_parameter_bounds=near_bound.tolist(),
                jacobian_singular_values=singular.tolist())


def self_check_projection():
    rng = np.random.default_rng(110)
    n = 40
    xyz = rng.uniform([-.07, -.07, -.04], [.24, .07, .13], size=(n, 3))
    rot = Rotation.from_euler("xyz", rng.uniform(-.28, .28, size=(n, 3)))
    mount = Rotation.from_rotvec([.04, -.08, .025]).as_matrix() @ OPTICAL_AT_REST
    mount_position = np.array([.015, -.02, .06])
    intrinsics = np.array([[504., 0., 316.], [0., 504., 242.], [0., 0., 1.]])
    world = np.array([[.84, .20, -.08], [.84, -.05, -.08]])
    pixels, edges = [], []
    for r, p in zip(rot.as_matrix(), xyz, strict=True):
        world_to_camera = (r @ mount).T
        camera_position = p + r @ mount_position
        rvec, _ = cv2.Rodrigues(world_to_camera)
        projected, _ = cv2.projectPoints(world, rvec, -world_to_camera @ camera_position, intrinsics, None)
        corner, along_edge = projected[:, 0]
        slope = (along_edge[1] - corner[1]) / (along_edge[0] - corner[0])
        pixels.append(corner)
        edges.append([slope, corner[1] - slope * corner[0]])
    all_frames = dict(frames=np.arange(n), pixels=np.array(pixels), edge=np.array(edges), xyz=xyz, rot=rot)
    train, holdout = ({k: v[i::2] for k, v in all_frames.items()} for i in (0, 1))
    trial = camera_fit(train, holdout, "left")
    error = np.linalg.norm(np.array(trial["parameters"][3:6]) - mount_position)
    require(trial["converged"] and error < 1e-5 and trial["validation"]["p95_px"] < 1e-5,
            "camera pose or projection direction failed the independent OpenCV geometry control")
    return dict(mount_position_error_m=float(error), heldout_corner_p95_px=trial["validation"]["p95_px"])


def fold_disagreement(pair):
    if any("parameters" not in row for row in pair):
        return None
    a, b = [np.asarray(row["parameters"]) for row in pair]
    angle = (Rotation.from_rotvec(a[:3]).inv() * Rotation.from_rotvec(b[:3])).magnitude()
    return dict(mount_position_delta_mm=1000 * float(np.linalg.norm(a[3:6] - b[3:6])),
                mount_rotation_delta_deg=float(np.rad2deg(angle)), focal_delta_px=float(abs(a[6] - b[6])))


def run(root, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only source camera diagnostic")
    cv2.setNumThreads(1)
    self_check()
    control = self_check_projection()
    episodes = [extract(pin(root / "data" / name, sha)) for name, sha in SOURCE_H5.items()]
    fits, disagreements = [], {}
    for view in VIEWS[1:]:
        side = view.split("_")[1]
        data = [observations(episode[view], side) for episode in episodes]
        for i in (0, 1):
            fits.append(dict(view=view, fitted_corner=side, fit_episode=i, **camera_fit(data[i], data[1 - i], side)))
        disagreements[view] = fold_disagreement(fits[-2:])
    result = []
    for (name, sha), episode in zip(SOURCE_H5.items(), episodes, strict=True):
        views = {}
        for view, entry in episode.items():
            coverage = Counter(side for row in entry["rows"] for side in row["corners"])
            views[view] = dict(rows=entry["rows"], frame_status=entry["coverage"], corner_counts=dict(coverage))
        result.append(dict(episode=name, sha256=sha, views=views))
    write_new(output, dict(
        status="STATIC_SCENE_CORNER_DIAGNOSTIC", fps=30, stride=STRIDE, image_size=SIZE,
        heldout_median_threshold_px=CALIBRATION_MEDIAN_PX, heldout_p95_threshold_px=CALIBRATION_P95_PX,
        opencv_version=cv2.__version__, synthetic_geometry_control=control,
        episodes=result, wrist_fits=fits, parameter_order=PARAMETERS,
        fold_disagreement=disagreements, authorized_panel_cells_qualified=0,
        sampler_forwards=0, physics_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve())
