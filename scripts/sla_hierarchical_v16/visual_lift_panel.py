"""Replay registered stereo RGB first, then independently score the sealed observations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from demonstrated_grip import pin, read, require, rows
from visual_objects import COLORS, VisualLiftProgress, estimate_objects

P1 = "1d496553b347a43c6f5c45c99d736896060ca9d71bf6cd5a72fef3e767b1b900"
S0 = "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9"


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def panel(root):
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0))["inputs"]
    path = root / "v12/A2_O2_DENSE_CAPTURE.json"
    capture = read(pin(path, pins[str(path)]))
    roster = read(pin(root / "evidence/P1_REGISTRATION.json", P1))["target_windows"]
    meta = {r["id"]: r for r in capture["episodes"]}
    require(len(roster) == len({g["episode"] for g in roster}) == 6, "episode roster")
    for group in roster:
        name = group["episode"]
        ep = pin(Path(group["episode_pin"]["path"]), group["episode_pin"]["sha256"])
        info = read(ep)
        require(
            info["split"] == "TRAIN"
            and info["id"] == name
            and info["task"] == group["instruction"],
            "TRAIN task",
        )
        receipt = read(
            pin(Path(meta[name]["receipt_path"]), meta[name]["receipt_sha256"])
        )
        queries = (
            ep.parents[3] / "cases" / name / "native-01/LIVE_DATA/QUERIES.private.json"
        )
        qpin = receipt["source_files"]["LIVE_DATA/QUERIES.private.json"]
        require(
            pins[str(queries)] == qpin and receipt["id"] == name, "native query source"
        )
        native = read(pin(queries, qpin))
        require(
            len(native) == info["windows"] == group["episode_pin"]["windows"],
            "full native denominator",
        )
        windows = {
            w["tick"]: w for w in capture["windows"].values() if w["episode"] == name
        }
        yield ep, info, native, receipt, windows


def predict(root, *, reject_spanning=False):
    result = {}
    for ep, info, queries, receipt, windows in panel(root):
        name = info["id"]
        dense = root / "v12/DENSE" / name / "frames.jsonl"
        overlap = {
            f["tick"]: f
            for f in rows(pin(dense, receipt["frames_sha256"]))
            if f["query_clock"]
        }
        progress = VisualLiftProgress()
        records = []
        verified_policy = 0
        for shard in info["frame_shards"]:
            require(shard["first_frame"] == len(records), "discontinuous image shards")
            with np.load(
                pin(ep.parent.parent / shard["file"], shard["sha256"]),
                allow_pickle=False,
            ) as src:
                rgb, steps = src["rgb"], src["native_step"]
            require(
                rgb.shape == (len(steps), 2, 256, 256, 3)
                and rgb.dtype == np.uint8
                and len(steps) == shard["frames"],
                "stereo image count/type",
            )
            for pair, tick in zip(rgb, steps):
                i = len(records)
                q = queries[i]
                tick = int(tick)
                t = q["native_time_s"]
                cal = q["camera_calibration"]
                require(
                    q["query"] == i
                    and q["native_step"] == tick == 5703 + 250 * i
                    and abs(t - 0.002 * tick) < 1e-8,
                    "query clock",
                )
                require(
                    q["rgb_provenance"] == "CURRENT_OWNED_PAUSED_NATIVE_PACKET"
                    and cal["wrist_rgb"] == q["native_camera_calibration"]["wrist_rgb"],
                    "source camera identity",
                )
                if tick in overlap:
                    f = overlap[tick]
                    require(
                        cal == f["calibration"]
                        and [sha256(v.tobytes()) for v in pair] == f["rgb_sha256"],
                        "dense stereo disagreement",
                    )
                if tick in windows:
                    window = windows[tick]
                    path = root / "v12/STUDENT_INPUT" / (window["id"] + ".npz")
                    with np.load(
                        pin(path, window["input_sha256"]), allow_pickle=False
                    ) as student:
                        require(
                            np.array_equal(pair[0], student["head"][-1])
                            and np.array_equal(pair[1], student["right_wrist"]),
                            "actual frozen-policy image disagreement",
                        )
                    verified_policy += 1
                objects = estimate_objects(
                    *pair, cal, t, reject_spanning=reject_spanning
                )
                records.append(
                    dict(
                        tick=tick,
                        objects=objects,
                        progress=progress.observe(t, objects),
                    )
                )
        require(
            len(records) == len(queries) and verified_policy == len(windows),
            "incomplete replay",
        )
        verified_dense = sum(r["tick"] in overlap for r in records)
        require(
            verified_dense > 0 and windows.keys() <= overlap.keys(), "dense frame join"
        )
        result[name] = dict(
            queries=records, dense_checks=verified_dense, policy_checks=verified_policy
        )
    require(
        sum(len(e["queries"]) for e in result.values()) == 527
        and sum(e["policy_checks"] for e in result.values()) == 16,
        "fixed cohort",
    )
    return result


def summary(records):
    bins = Counter()
    first = {}
    for r in records:
        if r["sampled_height20"] is None:
            key = "abstain_positive" if r["actual_height20"] else "abstain_negative"
        elif r["sampled_height20"]:
            key = "tp" if r["actual_height20"] else "fp"
        else:
            key = "fn" if r["actual_height20"] else "tn"
        bins[key] += 1
        for field in ("sampled_height20", "actual_height20"):
            if r[field]:
                first.setdefault(field, r["tick"])
    known = [r for r in records if r["delta_mm"] is not None]

    def stats(vals):
        if not vals:
            return None
        return dict(
            n=len(vals),
            median=float(np.median(vals)),
            p95=float(np.percentile(vals, 95)),
            maximum=float(max(vals)),
        )

    return dict(
        n=len(records),
        status=dict(Counter(r["status"] for r in records)),
        confusion={
            k: bins[k]
            for k in ("tp", "fp", "tn", "fn", "abstain_positive", "abstain_negative")
        },
        position_abs_error_mm=stats(
            [
                r["position_error_mm"]
                for r in records
                if r["position_error_mm"] is not None
            ]
        ),
        height_delta_abs_error_mm=stats(
            [abs(r["delta_mm"] - r["actual_delta_mm"]) for r in known]
        ),
        height_delta_median_signed_error_mm=float(
            np.median([r["delta_mm"] - r["actual_delta_mm"] for r in known])
        )
        if known
        else None,
        maximum_delta_mm=max((r["delta_mm"] for r in known), default=None),
        maximum_actual_delta_mm=max(r["actual_delta_mm"] for r in records),
        first_visual_tick=first.get("sampled_height20"),
        first_physical_tick=first.get("actual_height20"),
    )


def score(root, predictions):
    groups = {}
    buckets = {"target": [], "distractor": []}
    for _, info, native, _, _ in panel(root):
        name = info["id"]
        saved = predictions[name]
        require(len(saved["queries"]) == len(native), "prediction denominator")
        role_results = {}
        for color in COLORS:
            role = "target" if info["task"].split("-")[1] == color else "distractor"
            initial = native[0]["actual_state"]["objects"][color]["pose"][2][3]
            lines = []
            for q, pred in zip(native, saved["queries"]):
                actual = q["actual_state"]
                require(
                    pred["tick"] == q["native_step"] == actual["native_step"],
                    "paired scoring rows",
                )
                truth = np.asarray(actual["objects"][color]["pose"])[:3, 3]
                measured = pred["objects"][color]
                center = measured["center_world_m"]
                prog = pred["progress"][color]
                dz = truth[2] - initial
                lines.append(
                    dict(
                        tick=pred["tick"],
                        status=measured["status"],
                        actual_delta_mm=float(1000 * dz),
                        actual_height20=bool(dz >= 0.020),
                        delta_mm=None
                        if prog["delta_z_m"] is None
                        else 1000 * prog["delta_z_m"],
                        sampled_height20=prog["sampled_height20"],
                        position_error_mm=None
                        if center is None
                        else float(1000 * np.linalg.norm(center - truth)),
                    )
                )
            buckets[role].extend(lines)
            role_results[role] = dict(color=color, **summary(lines))
        groups[name] = dict(
            task=info["task"],
            dense_checks=saved["dense_checks"],
            policy_checks=saved["policy_checks"],
            **role_results,
        )
    require(predictions.keys() == groups.keys(), "unexpected episodes")
    totals = {}
    for role, records in buckets.items():
        total = summary(records)
        # Different epochs have unrelated clocks and distinct initial positions.
        for k in ("first_visual_tick", "first_physical_tick"):
            del total[k]
        totals[role] = total
    return dict(
        groups=groups,
        totals=totals,
        resource_counts=dict(
            new_native_starts=0, new_gpu_forwards=0, training_updates=0
        ),
    )


def bbox_iou(a, b):
    if a is None or b is None:
        return None
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(hi - lo + 1, 0))

    def area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    return float(intersection / (area(a) + area(b) - intersection))


def mask_audit(predictions):
    path = (
        Path(__file__).resolve().parents[2]
        / "reports/SLA_HIERARCHICAL_V16/OBSERVATION_VISIBILITY_ACTUAL.json"
    )
    reference = read(
        pin(path, "f41b2a9a368220dd3ff491401c23dd01404accc9f5d545b47a1dabdd3f619e39")
    )["target"]
    require(
        len({r["id"] for r in reference}) == len(reference) == 16,
        "independent mask window count",
    )
    index = {
        (name, q["tick"]): q for name, e in predictions.items() for q in e["queries"]
    }
    rows = []
    for r in reference:
        obs = index[r["episode"], r["tick"]]["objects"]
        for color in COLORS:
            for view, ref in (
                ("body_rgb", r["body_history"][-1]),
                ("wrist_rgb", r["wrist"]),
            ):
                actual = ref[color]["bbox_xyxy"]
                selector = obs[color]["views"][view]
                bbox = None if selector is None else selector["bbox_xyxy"]
                rows.append(
                    dict(
                        id=r["id"],
                        color=color,
                        view=view,
                        visible=actual is not None,
                        selected=bbox is not None,
                        iou=bbox_iou(bbox, actual),
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["predict", "score", "mask"])
    parser.add_argument("root", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--reject-spanning",
        action="store_true",
        help="use exploratory component selection when predicting",
    )
    parser.add_argument("--sealed-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "predict":
        write_new(args.predictions, predict(root, reject_spanning=args.reject_spanning))
        print(sha256(args.predictions.read_bytes()))
    else:
        if not args.sealed_sha256 or not args.output:
            parser.error("scoring requires --sealed-sha256 and --output")
        predictions = read(pin(args.predictions, args.sealed_sha256))
        result = (
            score(root, predictions)
            if args.mode == "score"
            else dict(mask_audit=mask_audit(predictions))
        )
        write_new(args.output, dict(predictions_sha256=args.sealed_sha256, **result))


if __name__ == "__main__":
    main()
