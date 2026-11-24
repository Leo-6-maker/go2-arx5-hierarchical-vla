"""Build official SLA inputs from closed validation Teacher captures with no sampling."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from checkpoint_inputs import configured_transforms
from demonstrated_grip import pin, read, require
from target_labels import checksum, prepare_episode, validate_episode
from train_archive_qualification import TASKS, qualify
from visual_lift_panel import S0, write_new

REPORTS = BASE / "reports/SLA_GO2_V18/P1_P2"
SCOPE = "SLA_GO2_V18_R5_VAL_TEACHER"
LEGACY = "CHV1-VAL-F1-PLACE-red-right-00"
LEGACY_RESULT = "147b7d2aa3f5e7f5bb784e0a55eb36bd844b1af3814fb08986f6460ab1542752"
LEGACY_EPISODE = "0c84358b9e9a35668323afb54e844ea7b52d2b2ed72647fdff4026a167b67482"


def partition(campaign, digest):
    closed = read(pin(campaign / "RESULT.private.json", digest))
    expected = read(REPORTS / "VAL_CANDIDATES.json")["rows"]
    runtime = read(REPORTS / "VAL_RUNTIME_PREFLIGHT.json")
    accepted = {row["slot"]: row for row in expected}
    rows = closed["records"]
    ids = [r.get("id") for r in rows]
    require(
        ids == [row["slot"] for row in expected[:len(rows)]]
        and closed.get("allocated_requests") == len(expected)
        and closed["allocated_ticks"] == sum(row["prospective_max_ticks"] for row in expected)
        and closed["source_sha"] == runtime["source_commit"]
        and closed["unreconciled"] is None
        and closed["verified_requests"] == len(rows)
        and closed["verified_ticks"] == sum(row["ticks"] for row in rows)
        and closed["admitted_VAL"] == sum(row["admitted"] for row in rows)
        and closed["unstarted"] == [row["slot"] for row in expected[len(rows):]],
        "sealed VAL prefix and complete accounting required",
    )
    captures = []
    for row in rows:
        slot = row["id"]
        require(
            row["split"] == "VAL" and row["requests"] == 1
            and 0 <= row["ticks"] <= accepted[slot]["prospective_max_ticks"]
            and row["shutdown_clean"] and not row["technical_failure"]
            and (not row["hard_STOP"] or row["post_STOP_ticks"] == 0),
            "invalid or unresolved validation terminal",
        )
        if row["admitted"]:
            captures.append((slot, row["result"]["sha256"], row["episode"]["sha256"], accepted[slot]))
    return captures, dict(
        started=ids,
        missing_admission=[row["id"] for row in rows if not row["admitted"]],
        unstarted=closed["unstarted"],
    )


def approved_capture(root, slot, result_digest, episode_digest, expected):
    location = root / "cases" / slot
    case = read(location / "EXECUTION_SPEC.json")
    terminal = read(pin(location / "RESULT.private.json", result_digest))
    receipt = read(location / "control" / "LEASE.json")
    ep_path = pin(root / "admission" / "VAL" / slot / "EPISODE.private.json", episode_digest)
    info = read(ep_path)
    score = terminal["detail"]["score"]
    require(
        case["collection_split"] == info["split"] == "VAL"
        and case["expert_training_eligible"] is False
        and receipt["status"] == "CLOSED"
        and receipt["spec_sha256"] == checksum(location / "EXECUTION_SPEC.json")
        and case["source_sha"] == terminal["source_sha"]
        and score["native_step"] == terminal["actual_native_integrations"]
        and info["native_result_sha256"] == result_digest
        and info["id"] == case["panel_entry_id"] == slot
        and info["scene_sha256"] == case["frozen_files"][case["model_path"]]
        and info["reset"] == case["mobile_condition"]["reset"],
        "validation capture and native receipt identity mismatch",
    )
    if expected:
        require(
            case["scope"] == SCOPE
            and case["registered_panel"]["sha256"] == read(REPORTS / "VAL_RUNTIME_PREFLIGHT.json")["panel_sha256"]
            and info["scene_sha256"] == expected["geometry_sha256"]
            and info["reset"] == expected["reset"]
            and case["mobile_condition"]["seed"] == expected["seed"]
            and info["task"] == expected["task"] and info["family"] == expected["family"],
            "independent VAL identity changed",
        )
    qualify(
        dict(requests=terminal["physical_requests"], admitted=True,
             task_and_mobile_success=terminal["metrics"]["task_and_required_motion_success"],
             technical_failure=terminal["technical_failure"], hard_STOP=terminal["unsafe"],
             post_STOP_ticks=terminal["trace_audit"]["integrations_after_first_STOP"],
             score=score, ticks=terminal["actual_native_integrations"]),
        terminal, info,
    )
    link = terminal["detail"]["capture"]
    manifest = read(pin(Path(link["path"]), link["sha256"]))
    labels = manifest["labels"]
    queries = read(pin(Path(labels["path"]), labels["sha256"]))
    require(info["windows"] == manifest["queries"] == len(queries), "native query count mismatch")
    return location, ep_path, info, queries


def run(root, campaign, out, *, digest, compatibility):
    from openpi import transforms

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("JAX_PLATFORMS") == "cpu",
            "processor-only CPU route")
    if compatibility:
        require(digest is None, "legacy fixture does not select a validation result")
        captures = [(LEGACY, LEGACY_RESULT, LEGACY_EPISODE, None)]
        denominators = dict(compatibility_episodes=[LEGACY], started=[], missing_admission=[], unstarted=[])
    else:
        captures, denominators = partition(campaign, digest)
    require(captures and not out.exists(), "need original validation receipts and new output")
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0))["inputs"]
    reg_path = root / "v12/TARGET_REGISTRATION.json"
    registration = read(pin(reg_path, pins[str(reg_path)]))
    source = root / "v12/A2_O2_DENSE_CAPTURE.json"
    captures_old = read(pin(source, pins[str(source)]))["windows"]
    virtual = {tuple(item["state"][:8]) for item in captures_old.values()}
    require(len(virtual) == 1, "virtual arm calibration changed")

    cfg, data, stats = configured_transforms(root)
    require(cfg.model.action_horizon == 50 and cfg.model.extra_frames_first is False, "wrong official clock")
    decoder = transforms.compose(
        [*data.model_transforms.outputs, transforms.Unnormalize(stats, use_quantiles=True),
         *data.data_transforms.outputs]
    )
    results = []
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=out.name + "-") as tmp:
        staging = Path(tmp) / "sealed"
        staging.mkdir()
        for slot, result_digest, episode_digest, expected in captures:
            case, ep, info, queries = approved_capture(campaign, slot, result_digest, episode_digest, expected)
            dense = staging / (slot + "-camera")
            spec = read(case / "EXECUTION_SPEC.json")
            replay_script = BASE / "scripts/sla_hierarchical_v16/train_history_replay.py"
            replay_env = dict(os.environ, **spec["native_environment"])
            with tempfile.TemporaryDirectory(dir=staging, prefix="render-") as scratch:
                replay_env.update(
                    CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1",
                    LIBGL_ALWAYS_SOFTWARE="1", MESA_SHADER_CACHE_DISABLE="true",
                    XDG_CACHE_HOME=scratch, TMPDIR=scratch, ROS_LOG_DIR=scratch,
                    PYTHONPATH=os.pathsep.join((str(root / "v12/CPU_LIB"), replay_env["PYTHONPATH"])),
                )
                subprocess.run(
                    [spec["python"], "-B", str(replay_script), str(case), str(dense), "--split", "VAL"],
                    env=replay_env, check=True, timeout=300,
                )
            replay_receipt = read(dense / "receipt.json")
            atomfile = case / "native-01" / "MOBILE_SAMPLES.private.jsonl"
            prepared = staging / slot
            report = prepare_episode(
                root, prepared, (ep, info, queries, replay_receipt, {}),
                {str(atomfile): checksum(atomfile)}, registration, TASKS.index(info["task"]),
                dense=dense, virtual_left=np.array(next(iter(virtual))),
            )
            report.update(validate_episode(prepared, data, stats, decoder))
            results.append(dict(slot=slot, terminal_sha256=result_digest, admission_sha256=episode_digest,
                                original_scene_sha256=info["scene_sha256"],
                                historical_pixels=replay_receipt["missing_count"], **report))
        write_new(staging / "RESULT.json", dict(
            scope="LEGACY_VAL_READER_COMPATIBILITY" if compatibility else "INDEPENDENT_VAL_MODEL_INPUTS",
            sample_accounting=denominators, examples=results, training_authorized=False,
            independent_VAL_episodes=0 if compatibility else len(results),
            physics_ticks=0, optimizer_updates=0, model_forwards=0, TEST_episodes=0))
        staging.rename(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("campaign", type=Path)
    p.add_argument("out", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--campaign-result-sha256")
    mode.add_argument("--legacy-only", action="store_true")
    args = p.parse_args()
    run(args.root.resolve(), args.campaign.resolve(), args.out.resolve(),
        digest=args.campaign_result_sha256, compatibility=args.legacy_only)
