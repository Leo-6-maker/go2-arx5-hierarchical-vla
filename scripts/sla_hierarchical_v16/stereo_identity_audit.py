"""Offline identity readback from sealed RGB estimates against both original object poses."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require
from stereo_association import CORNERS
from train_visual_progress import INDEX_SHA, source
from visual_lift_panel import write_new
from visual_objects import COLORS, VIEWS, optical_frame


def box(pose, cal, t):
    optical, _ = optical_frame(cal, t)
    world = CORNERS @ pose[:3, :3].T + pose[:3, 3]
    xyz = (world - optical[:3, 3]) @ optical[:3, :3]
    if xyz[:, 2].min() <= 0:
        return None
    px = xyz @ np.asarray(cal["K"]).T
    uv = px[:, :2] / px[:, 2:]
    return np.r_[uv.min(axis=0), uv.max(axis=0)]


def iou(a, b):
    if b is None:
        return 0.0
    a, b = np.array(a), np.array(b)
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter
    return float(inter / union)


def audit(index, original, augmented):
    panel = read(pin(index, INDEX_SHA))
    require(len(panel) == 87, "archive roster")
    bins = {c: Counter() for c in COLORS}
    per_episode = {}
    count = 0
    with original.open() as left, augmented.open() as right:
        for ll, rr in zip(left, right, strict=True):
            a, b = json.loads(ll), json.loads(rr)
            name = b["episode"]
            require(name == a["episode"] and name not in per_episode, "episode join")
            item = panel[name]
            target = item["task"].split("-")[1]
            other = next(c for c in COLORS if c != target)
            dst = Counter()
            per_episode[name] = dst
            if item["archive_status"] != "recorded":
                require(not a["frames"] and not b["frames"], "missing archive")
                continue
            _, _, qs = source(item)
            for old, new, q in zip(a["frames"], b["frames"], qs, strict=True):
                count += 1
                require(
                    old["tick"] == new["tick"] == q["native_step"],
                    "changed query clock",
                )
                for c in COLORS:
                    require(
                        old["colors"][c]
                        == {k: new["colors"][c][k] for k in ("seen", "status")},
                        "prediction changed during instrumentation",
                    )
                obs = new["colors"][target]
                dst["queries"] += 1
                if obs["status"] != "measured":
                    dst["unknown"] += 1
                    continue
                poses = {
                    c: np.array(q["actual_state"]["objects"][c]["pose"]) for c in COLORS
                }
                pos = np.array(obs["center_world_m"])
                diff = {c: np.linalg.norm(pos - poses[c][:3, 3]) * 1000 for c in COLORS}
                t = q["native_time_s"]
                overlaps = {
                    c: [
                        iou(pr, box(poses[c], q["camera_calibration"][v], t))
                        for v, pr in zip(VIEWS, obs["boxes"])
                    ]
                    for c in COLORS
                }
                if (
                    diff[target] <= 20
                    and diff[target] < diff[other]
                    and min(overlaps[target]) >= 0.30
                    and all(x > y for x, y in zip(overlaps[target], overlaps[other]))
                ):
                    key = "both_views_and_3d_match"
                elif diff[other] < diff[target] and min(overlaps[other]) >= 0.30:
                    key = "wrong_object"
                else:
                    key = "unresolved_match"
                dst[key] += 1
                if obs["seen"] is True:
                    dst["lift_positive_" + key] += 1
            bins[target].update(dst)
    require(panel.keys() == per_episode.keys() and count == 7713, "incomplete readback")
    return {
        "old_frame_decisions_reproduced": count,
        "episodes": per_episode,
        "target": bins,
        "audit_geometry": "original pose and nominal cube vertices for offline amodal bounding boxes only",
        "native_integrations": 0,
        "model_forwards": 0,
        "optimizer_updates": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("index", "original", "augmented", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("original_sha")
    p.add_argument("augmented_sha")
    x = p.parse_args()
    result = audit(
        x.index, pin(x.original, x.original_sha), pin(x.augmented, x.augmented_sha)
    )
    result.update(prediction_sha256=x.original_sha, instrumented_sha256=x.augmented_sha)
    write_new(x.out, result)
