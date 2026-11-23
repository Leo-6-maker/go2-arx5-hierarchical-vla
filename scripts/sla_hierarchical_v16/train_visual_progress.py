"""Evaluate a frozen stereo lift observer on the original TRAIN campaign, including failed cases."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require
from target_labels import checksum
from visual_lift_panel import P1, write_new
from visual_objects import COLORS, VisualLiftProgress, estimate_objects

RECORDS = "aca8606636db476df4d14f6e9532a05f8b3460ee6b3fced17c0717bed2a07ceb"
PLANS = "62daa720bf9e131106792946ae0dae1022037a3252fe93f17420db305bc94271"
INDEX_SHA = "b985985bf2d63b313e59831af19abe3fca4ea3e07e1350d0eac5258c1ceb1f93"
COLS = ("tp", "fp", "tn", "fn", "abstain_positive", "abstain_negative")


def roster(root):
    old = read(pin(root / "evidence/P1_REGISTRATION.json", P1))["target_windows"]
    seen = {r["episode"] for r in old}
    campaign = Path(old[0]["episode_pin"]["path"]).parents[3]
    plans = {
        r["id"]: r for r in read(pin(campaign / "REGISTERED_PANEL.private.json", PLANS))
    }
    output = {}
    for r in read(pin(campaign / "RECORDS.private.json", RECORDS)):
        if r["split"] != "TRAIN":
            continue
        name = r["condition"]
        plan = plans[name]
        condition = plan["condition"]
        require(
            name not in output
            and condition["collection_episode"] == name
            and plan["role"] == "EXPERT"
            and "-TRAIN-" in name,
            "wrong TRAIN record",
        )
        result = pin(Path(r["result"]["path"]), r["result"]["sha256"])
        require(
            result.parent.name == name and result.parent.parent == campaign / "cases",
            "case path",
        )
        native = read(result)
        sources = list(result.parent.glob("native-*/LIVE_DATA/MANIFEST.private.json"))
        item = {
            "group": "original_six"
            if name in seen
            else ("other_admitted" if r["admitted"] else "nonadmitted"),
            "task": plan["instruction"],
            "family": condition["family"],
            "layout": condition["layout"],
            "reset": condition["reset"],
            "seed": condition["seed"],
            "result_sha256": r["result"]["sha256"],
            "archive_status": "no_unique_recording",
            "producer_capture_link": False,
            "original_requests": r["requests"],
            "teacher_success": r["task_and_mobile_success"],
        }
        output[name] = item
        if len(sources) != 1:
            continue
        manifest = sources[0]
        link = (native.get("detail") or {}).get("capture")
        if link:
            require(manifest == Path(link["path"]), "ambiguous producer recording")
            pin(manifest, link["sha256"])
        m = read(manifest)
        require(
            m["schema"] == "MFREE_LIVE_FINAL_PLAN_CAPTURE_V1"
            and m["writer_error"] is None
            and Path(m["labels"]["path"]).parent == manifest.parent,
            "invalid archived recording",
        )
        item.update(
            archive_status="recorded" if m["queries"] else "no_queries",
            manifest_sha256=checksum(manifest),
            manifest=str(manifest),
            queries=m["queries"],
            scene=m["frozen_scene_sha256"],
            producer_capture_link=bool(link),
        )
        if r["admitted"]:
            e = read(pin(Path(r["episode"]["path"]), r["episode"]["sha256"]))
            require(
                e["split"] == "TRAIN"
                and e["id"] == name
                and e["task"] == item["task"]
                and e["native_result_sha256"] == item["result_sha256"]
                and e["reset"] == item["reset"]
                and e["windows"] == m["queries"]
                and e["scene_sha256"] == item["scene"]
                and [s["sha256"] for s in e["frame_shards"]]
                == [s["sha256"] for s in m["frame_shards"]],
                "admitted archive differs",
            )
    require(
        Counter(r["group"] for r in output.values())
        == {"original_six": 6, "other_admitted": 51, "nonadmitted": 30},
        "TRAIN roster changed",
    )
    return output


def source(item):
    path = pin(Path(item["manifest"]), item["manifest_sha256"])
    m = read(path)
    queries = read(pin(Path(m["labels"]["path"]), m["labels"]["sha256"]))
    require(len(queries) == item["queries"], "query count")
    return path, m, queries


def predict(index, out):
    pin(
        Path(__file__).with_name("visual_objects.py"),
        "6cac1d53a2d5aa8fa97cefad7ea4c4a18ab55edf9864088b0176bdd7c1f7f15d",
    )
    with out.open("x") as f:
        for name, item in index.items():
            preds = []
            if item["archive_status"] == "recorded":
                path, m, queries = source(item)
                observer = VisualLiftProgress()
                for s in m["frame_shards"]:
                    p = Path(s["path"])
                    require(
                        p.parent == path.parent and s["first_query"] == len(preds),
                        "cross-episode RGB or gap",
                    )
                    with np.load(pin(p, s["sha256"])) as z:
                        ticks, rgb = z["native_step"], z["rgb"]
                    require(
                        len(ticks) == s["count"]
                        and rgb.shape == (len(ticks), 2, 256, 256, 3)
                        and rgb.dtype == np.uint8,
                        "raw stereo",
                    )
                    for tick, pair in zip(ticks, rgb):
                        q = queries[len(preds)]
                        t = q["native_time_s"]
                        cal = q["camera_calibration"]
                        require(
                            q["query"] == len(preds)
                            and q["native_step"] == tick
                            and abs(t - tick * 0.002) < 1e-8
                            and q["rgb_provenance"]
                            == "CURRENT_OWNED_PAUSED_NATIVE_PACKET"
                            and cal["wrist_rgb"]
                            == q["native_camera_calibration"]["wrist_rgb"],
                            "observation identity",
                        )
                        obj = estimate_objects(*pair, cal, t, reject_spanning=True)
                        prog = observer.observe(t, obj)
                        preds.append(
                            {
                                "tick": int(tick),
                                "colors": {
                                    col: {
                                        "status": obj[col]["status"],
                                        "seen": prog[col]["sampled_height20"],
                                    }
                                    for col in COLORS
                                },
                            }
                        )
                require(len(preds) == len(queries), "incomplete sample")
            f.write(
                json.dumps(
                    {"episode": name, "frames": preds},
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def summarize_color(preds, queries, color):
    bins = Counter()
    statuses = Counter()
    false_positives = []
    first_true = first_pred = first_tp = None
    initial = queries[0]["actual_state"]["objects"][color]["pose"][2][3]
    for p, q in zip(preds, queries, strict=True):
        require(
            p["tick"] == q["actual_state"]["native_step"] == q["native_step"],
            "truth clock",
        )
        height_m = q["actual_state"]["objects"][color]["pose"][2][3] - initial
        y = height_m >= 0.020
        pred = p["colors"][color]
        v = pred["seen"]
        if y and first_true is None:
            first_true = p["tick"]
        if v and first_pred is None:
            first_pred = p["tick"]
        if y and v and first_tp is None:
            first_tp = p["tick"]
        if v is None:
            key = "abstain_positive" if y else "abstain_negative"
        elif v:
            key = "tp" if y else "fp"
        else:
            key = "fn" if y else "tn"
        if key == "fp":
            false_positives.append(
                {"tick": p["tick"], "actual_height_delta_mm": 1000 * height_m}
            )
        bins[key] += 1
        statuses[pred["status"]] += 1
    return {
        "confusion": {k: bins[k] for k in COLS},
        "status": dict(statuses),
        "first_actual": first_true,
        "first_visual": first_pred,
        "first_same_tick_true_positive": first_tp,
        "false_positive_frames": false_positives,
    }


def aggregate(items, role):
    colors = [item[role] for item in items if role in item]
    counts = {k: sum(c["confusion"][k] for c in colors) for k in COLS}
    return {
        "episodes": len(items),
        "scored": len(colors),
        "frames": sum(counts.values()),
        "confusion": counts,
        "physically_lifted": sum(c["first_actual"] is not None for c in colors),
        "detected_while_lifted": sum(
            c["first_same_tick_true_positive"] is not None for c in colors
        ),
        "any_false_positive": sum(c["confusion"]["fp"] > 0 for c in colors),
        "first_visual_precedes_lift_or_no_lift": sum(
            c["first_visual"] is not None
            and (c["first_actual"] is None or c["first_visual"] < c["first_actual"])
            for c in colors
        ),
        "all_unknown": sum(
            not any(c["confusion"][k] for k in ("tp", "fp", "tn", "fn")) for c in colors
        ),
    }


def matched_hold_targets(results):
    cells = {}
    for name, item in results.items():
        if (
            item["group"] != "other_admitted"
            or item["family"] != "F1"
            or item["task"] not in ("HOLD-red", "HOLD-blue")
            or "target" not in item
        ):
            continue
        key = item["layout"], item["scene"], item["reset"], item["seed"]
        bucket = cells.setdefault(key, {})
        require(item["task"] not in bucket, "duplicate task in matched F1 cell")
        bucket[item["task"]] = name
    return [
        {
            "layout": layout,
            "scene_sha256": scene,
            "reset": reset,
            "seed": seed,
            "episodes": tasks,
        }
        for (layout, scene, reset, seed), tasks in sorted(cells.items())
        if len(tasks) == 2
    ]


def score(index, predfile, predsha):
    old = read(
        Path(__file__).resolve().parents[2]
        / "reports/SLA_HIERARCHICAL_V16/VISUAL_LIFT_BORDER_ACTUAL.json"
    )["groups"]
    results = {}
    with pin(predfile, predsha).open() as f:
        for line in f:
            p = json.loads(line)
            name = p["episode"]
            require(name not in results, "duplicate prediction")
            rec = index[name]
            item = {k: v for k, v in rec.items() if k != "manifest"}
            results[name] = item
            if rec["archive_status"] != "recorded":
                require(p["frames"] == [], "prediction without recording")
                continue
            _, _, qs = source(rec)
            target = rec["task"].split("-")[1]
            require(all(q["instruction"] == rec["task"] for q in qs), "task identity")
            for color in COLORS:
                role = "target" if color == target else "distractor"
                item[role] = summarize_color(p["frames"], qs, color)
                if rec["group"] == "original_six":
                    require(
                        item[role]["confusion"] == old[name][role]["confusion"]
                        and item[role]["status"] == old[name][role]["status"],
                        "old control differs",
                    )
    require(results.keys() == index.keys(), "TRAIN denominator")
    strata = {}
    for group in ("original_six", "other_admitted", "nonadmitted"):
        g = [v for v in results.values() if v["group"] == group]
        for family, task in sorted({(r["family"], r["task"]) for r in g}):
            select = [r for r in g if r["task"] == task and r["family"] == family]
            strata[f"{group} / {family} / {task}"] = {
                role: aggregate(select, role) for role in ("target", "distractor")
            }
    return {
        "predictions_sha256": predsha,
        "roster_sha256": INDEX_SHA,
        "matched_F1_HOLD": matched_hold_targets(results),
        "strata": strata,
        "episodes": results,
        "native_integrations": 0,
        "forwards": 0,
        "updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["roster", "predict", "score"])
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--sealed-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "roster":
        write_new(args.index, roster(args.root.resolve()))
    elif args.mode == "predict":
        predict(read(pin(args.index, INDEX_SHA)), args.predictions)
    else:
        write_new(
            args.output,
            score(
                read(pin(args.index, INDEX_SHA)), args.predictions, args.sealed_sha256
            ),
        )
