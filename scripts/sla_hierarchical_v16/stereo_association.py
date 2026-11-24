"""Sensor-only stereo cube association with size, reprojection, visibility, and temporal checks."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require
from observation_stream import camera
from scipy import ndimage
from train_visual_progress import INDEX_SHA, predict, score
from visual_lift_panel import write_new
from visual_objects import COLORS, VIEWS, optical_frame, optical_ray, triangulate

# Catalogued cube dimensions are optical priors, with no online object pose or task phase.
CORNERS = 0.02 * np.array(list(product((-1, 1), repeat=3)))


def components(im, channel):
    primary = im[:, :, channel]
    other = np.maximum(im[:, :, 1], im[:, :, 2 - channel])
    labels, n = ndimage.label(
        (primary - other >= 25) & (10 * primary >= 13 * other), np.ones((3, 3), bool)
    )
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    boxes = []
    for i, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None or sizes[i] < 16:
            continue
        y, x = region
        height, width = y.stop - y.start, x.stop - x.start
        if (
            min(x.start, y.start) == 0
            or x.stop == im.shape[1]
            or y.stop == im.shape[0]
            or sizes[i] / (height * width) < 0.35
            or max(height, width) > 3 * min(height, width)
        ):
            continue
        boxes.append(
            (
                np.array([(x.start + x.stop) / 2, (y.start + y.stop) / 2]),
                np.array([width, height]),
            )
        )
    return boxes


def project_size(center, frame, k):
    xyz = (center + CORNERS - frame[:3, 3]) @ frame[:3, :3]
    if xyz[:, 2].min() <= 0:
        return None
    pixels = xyz @ k.T
    return np.ptp(pixels[:, :2] / pixels[:, 2:], axis=0)


def associate(body, wrist, calibration, timestamp, *, reject_spanning=True):
    views = [camera(v).astype(np.int16) for v in (body, wrist)]
    frames = [optical_frame(calibration[v], timestamp) for v in VIEWS]
    ks = [np.asarray(calibration[v]["K"]) for v in VIEWS]
    result = {}
    for col, ch in zip(COLORS, (0, 2)):
        pairs = [components(im, ch) for im in views]
        candidates = []
        for left, right in product(*pairs):
            est = triangulate(
                *(optical_ray(frame, c[0]) for frame, c in zip(frames, (left, right)))
            )
            if est["status"] != "measured" or est["separation_mm"] > 8:
                continue
            pos = np.asarray(est["center_world_m"])
            sizes = [project_size(pos, f[0], k) for f, k in zip(frames, ks)]
            if any(v is None for v in sizes):
                continue
            err = max(
                float(np.abs(np.log(box[1] / v)).max())
                for box, v in zip((left, right), sizes)
            )
            if err <= np.log(1.75):
                candidates.append(
                    (
                        err / np.log(1.75) + est["separation_mm"] / 8,
                        pos,
                        [
                            (box[0] - box[1] / 2).tolist()
                            + (box[0] + box[1] / 2).tolist()
                            for box in (left, right)
                        ],
                    )
                )
        candidates.sort(key=lambda v: v[0])
        status = "missing_or_geometry_rejected"
        center = None
        selected = None
        if candidates:
            if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.35:
                status = "ambiguous"
            else:
                status = "measured"
                center = candidates[0][1].tolist()
                selected = candidates[0][2]
        result[col] = {"status": status, "center_world_m": center, "boxes": selected}
    return result


class PersistentLift:
    def __init__(self):
        self.t0 = self.last = None
        self.base = {}
        self.last_seen = {}
        self.previous_high = {}

    def observe(self, t, objects):
        require(self.last is None or t > self.last, "unordered images")
        if self.t0 is None:
            self.t0 = t
        self.last = t
        result = {}
        for col, observation in objects.items():
            value = None
            pos = observation["center_world_m"]
            prev = self.last_seen.get(col)
            if pos is None:
                self.previous_high[col] = False
            else:
                xyz = np.array(pos)
                if (
                    prev
                    and (t - prev[0]) <= 1
                    and np.linalg.norm(xyz - prev[1]) > 0.10 * (t - prev[0]) + 0.005
                ):
                    observation["status"] = "discontinuous"
                    self.previous_high[col] = False
                else:
                    if (
                        col not in self.base
                        and t - self.t0 <= 1.00001
                        and prev
                        and t - prev[0] <= 0.50001
                        and np.linalg.norm(xyz - prev[1]) <= 0.005
                    ):
                        self.base[col] = (xyz[2] + prev[1][2]) / 2
                    if col in self.base:
                        dz = xyz[2] - self.base[col]
                        high = dz >= 0.020
                        if (
                            high
                            and self.previous_high.get(col)
                            and prev
                            and t - prev[0] <= 0.50001
                        ):
                            value = True
                        elif dz < 0.015:
                            value = False
                        self.previous_high[col] = high
                    self.last_seen[col] = (t, xyz)
            result[col] = {"sampled_height20": value}
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["predict", "score"])
    p.add_argument("index", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--seal")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    panel = read(pin(a.index, INDEX_SHA))
    if a.mode == "predict":
        predict(
            panel,
            a.predictions,
            estimator=associate,
            observer_factory=PersistentLift,
            capture_identity=True,
        )
    else:
        write_new(a.out, score(panel, a.predictions, a.seal, reproduce_original=False))
