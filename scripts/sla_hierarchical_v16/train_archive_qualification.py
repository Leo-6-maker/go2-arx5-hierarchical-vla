"""Qualify all 87 registered TRAIN records, package raw examples and pass the frozen processor."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from demonstrated_grip import pin, read, require
from target_labels import checksum, prepare_episode, validate_episode
from train_visual_progress import PLANS, RECORDS, roster, source
from visual_lift_panel import P1, S0, write_new

SEED_REPORT = "7447bb6a9ec9df817c49478bf6c110d04926b7d0d7f33900f833bab9f836a253"
TASKS = ["HOLD-blue", "HOLD-red", "PLACE-red-right"]


def qualify(row, result, info):
    score = row["score"]
    require(
        row["requests"] == 1
        and row["admitted"]
        and row["task_and_mobile_success"]
        and not row["technical_failure"]
        and not row["hard_STOP"]
        and row["post_STOP_ticks"] in (None, 0)
        and info["individual_pass"]
        and result["shutdown_clean"]
        and not result["unsafe"]
        and not result["physics_stops"]
        and result["actual_native_integrations"] == row["ticks"]
        and result["trace_audit"]["status"] == "RECORDED_TRACE_CHECKS_PASS"
        and result["detail"]["score"] == score
        and score["instruction"] == info["task"]
        and score["clock"] == "ACTUAL_2MS_NATIVE_STREAM"
        and score["task_success"]
        and score["missing_native_intervals"] == score["unsafe_states"] == 0
        and not score["wrong_object_grasp_ever"]
        and not score["wrong_object_contact_ever"]
        and score["ever_target_grasp"]
        and score["ever_target_lift20mm"]
        and score["base_directional_motion_m"] >= 0.08,
        "teacher outcome or safety record",
    )
    if info["task"].startswith("HOLD"):
        require(
            score["target_grasp"] and score["longest_grasp_lift_hold_s"] >= 1,
            "incomplete hold",
        )
    else:
        require(
            score["inside_selected_region"]
            and score["released"]
            and score["stable_release_condition"]
            and "retreated50mm" in score["events"],
            "incomplete placement",
        )


def run(root, out):
    from checkpoint_inputs import configured_transforms
    from openpi import transforms

    require(
        os.environ["CUDA_VISIBLE_DEVICES"] == ""
        and os.environ["JAX_PLATFORMS"] == "cpu",
        "CPU only",
    )
    out.mkdir(parents=True, exist_ok=True)
    records = roster(root)
    reg0 = read(pin(root / "evidence/P1_REGISTRATION.json", P1))
    campaign = Path(reg0["target_windows"][0]["episode_pin"]["path"]).parents[3]
    originals = {
        r["condition"]: r for r in read(pin(campaign / "RECORDS.private.json", RECORDS))
    }
    plans = read(pin(campaign / "REGISTERED_PANEL.private.json", PLANS))
    require(all(name in originals for name in records), "missing campaign record")
    s0 = read(pin(root / "v13/run01/S0_ACTUAL.json", S0))["inputs"]
    regfile = root / "v12/TARGET_REGISTRATION.json"
    reg = read(pin(regfile, s0[str(regfile)]))
    capfile = root / "v12/A2_O2_DENSE_CAPTURE.json"
    capture = read(pin(capfile, s0[str(capfile)]))
    surrogates = {tuple(v["state"][:8]) for v in capture["windows"].values()}
    require(len(surrogates) == 1, "virtual arm identity changed")
    surrogate = next(iter(surrogates))
    seed = read(pin(root / "v16/target-labels04/result.json", SEED_REPORT))["episodes"]
    cfg, data, stats = configured_transforms(root)
    require(
        cfg.model.action_horizon == 50 and not cfg.model.extra_frames_first,
        "official clock and history",
    )
    decoder = transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=True),
            *data.data_transforms.outputs,
        ]
    )
    result = {}
    for name, item in sorted(records.items()):
        row = originals[name]
        summary = {
            k: item.get(k)
            for k in (
                "group",
                "family",
                "layout",
                "seed",
                "reset",
                "task",
                "scene",
                "queries",
                "manifest_sha256",
                "result_sha256",
            )
        }
        summary.update(
            original_reason=row["first_failure"],
            original_ticks=row["ticks"],
            original_requests=row["requests"],
            original_admitted=row["admitted"],
        )
        result[name] = summary
        if not row["admitted"]:
            summary["status"] = "excluded_original_failure"
            continue
        case = campaign / "cases" / name
        ep = pin(Path(row["episode"]["path"]), row["episode"]["sha256"])
        info = read(ep)
        native = read(pin(case / "RESULT.private.json", item["result_sha256"]))
        try:
            qualify(row, native, info)
            summary["native_event_ticks"] = {
                k: v["native_step"] for k, v in row["score"]["events"].items()
            }
            summary["episode_sha256"] = row["episode"]["sha256"]
            if name in seed:
                folder = root / "v16/target-labels04" / name
                for file, digest in seed[name]["artifacts"].items():
                    pin(folder / file, digest)
                summary.update(status="verified_seed", prepared=seed[name])
                continue
            require(item["producer_capture_link"], "missing producer seal")
            dest = out / name
            if not dest.exists():
                _, manifest, queries = source(item)
                spec = read(case / "EXECUTION_SPEC.json")
                keys = [
                    "MUJOCO_GL",
                    "PYOPENGL_PLATFORM",
                    "LIBGL_ALWAYS_SOFTWARE",
                    "__EGL_VENDOR_LIBRARY_FILENAMES",
                    "LP_NUM_THREADS",
                    "MESA_GLTHREAD",
                    "PYTHONPATH",
                    "OMP_NUM_THREADS",
                ]
                env = dict(
                    os.environ,
                    **{
                        k: spec["native_environment"][k]
                        for k in keys
                        if k in spec["native_environment"]
                    },
                )
                env.update(
                    PYTHONDONTWRITEBYTECODE="1",
                    OPENBLAS_NUM_THREADS="1",
                    PYTHONPATH=str(root / "v12/CPU_LIB")
                    + os.pathsep
                    + env["PYTHONPATH"],
                )
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    work = Path(tmp)
                    helper = Path(__file__).with_name("train_history_replay.py")
                    subprocess.run(
                        [spec["python"], str(helper), str(case), str(work / "replay")],
                        env=env,
                        check=True,
                    )
                    receipt = read(work / "replay/receipt.json")
                    atomfile = case / "native-01/MOBILE_SAMPLES.private.jsonl"
                    digest = checksum(atomfile)
                    bundle = ep, info, queries, receipt, {}
                    datafolder = work / "examples"
                    report = prepare_episode(
                        root,
                        datafolder,
                        bundle,
                        {str(atomfile): digest},
                        reg,
                        TASKS.index(info["task"]),
                        dense=work / "replay",
                        virtual_left=surrogate,
                    )
                    report.update(validate_episode(datafolder, data, stats, decoder))
                    require(
                        report["official_rows"]
                        == manifest["queries"]
                        == info["windows"]
                        and report["confirmed_native_intervals"]
                        == row["episode"]["confirmed_P10_atoms"],
                        "query or consumption disagreement",
                    )
                    report["sensor_replay"] = receipt
                    write_new(datafolder / "qualified.json", report)
                    datafolder.rename(dest)
            report = read(dest / "qualified.json")
            for file, digest in report["artifacts"].items():
                pin(dest / file, digest)
            require(
                read(dest / "episode.json")["episode_sha256"]
                == row["episode"]["sha256"],
                "wrong cached case",
            )
            summary.update(status="verified_expansion", prepared=report)
        except (
            ValueError,
            KeyError,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ) as exc:
            summary.update(status="preparation_failed", diagnostic=str(exc))
        print(name, summary["status"], summary.get("diagnostic", ""), flush=True)

    # Only inspect split identities from the pre-existing registration. No selection by VAL outcomes.
    split_keys = ("layout", "seed", "reset", "scene")
    groups = {s: {k: set() for k in split_keys} for s in ("TRAIN", "VAL", "TEST")}
    for p in plans:
        split = p["id"].split("-")[1]
        if split in groups and (split != "TRAIN" or p["id"] in records):
            for k in split_keys:
                groups[split][k].add(
                    p["scene"]["sha256"] if k == "scene" else p["condition"][k]
                )
    audit = {
        s: {
            k: {
                "distinct": len(v),
                "common_with_audited_train": len(v & groups["TRAIN"][k]),
            }
            for k, v in groups[s].items()
        }
        for s in groups
    }
    write_new(
        out / "result.json",
        {
            "episodes": result,
            "split_identity": audit,
            "campaign_records_sha256": RECORDS,
            "registered_panel_sha256": PLANS,
            "seed_report_sha256": SEED_REPORT,
            "optimizer_updates": 0,
            "model_forwards": 0,
            "new_integrations": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())
