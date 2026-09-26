#!/usr/bin/env python3
"""Run PI0248's fixed C/R checkpoint screens through the sealed M64 evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import pi0247_m64_screen as m64


AUTH = "PI0248_M64_POSE_REPAIR_GO"
REVISION = "093c450fd49052c8fd3285bf38111a30a1f59009"
TREE_SHA256 = "1811bfa5c29fb7ce2e911ade0a0b476607e32ac1c1322cce68b8dff04449d776"
STEPS = (1000, 2000)
ROOT = Path("/var/cache/users/liuyu-nfs/pi0248_m64_pose_repair_20260927")
SCREEN_ROOT = ROOT / "paired_screens"
NOISES = ("B0", "B1")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    m64.write(path, value)


def check_card(card_path: Path, arm: str) -> tuple[dict, str]:
    card = read(card_path)
    card_sha = sha(card_path)
    seal = read(card_path.with_name("run_card.seal.json"))
    family = f"PI0248_{arm}_" + ("U_CONTINUATION_M64_2K_V1" if arm == "C" else "ROTATION_ENDPOINT_M64_2K_V1")
    assert arm in ("C", "R")
    assert seal.get("sha256") == card_sha
    assert card.get("schema") == "PI0248_CR_RUN_CARD_V1"
    assert card.get("status") == "SEALED_READY_NOT_STARTED"
    assert card.get("authorization") == AUTH and card.get("execution_authorization") == AUTH
    assert card.get("arm") == arm and card.get("family") == family
    assert card.get("selected_checkpoint") == "NONE" and card.get("d1_runtime_id") == "S4"
    assert card.get("stage") == {"name": "M64_REPAIR", "steps": 2000,
                                  "checkpoint_steps": [1000, 2000], "source_revision": REVISION}
    assert card.get("loss_coefficients", {}).get("ell_pos") == 0.1
    assert card.get("loss_coefficients", {}).get("ell_delta") == 0.1
    assert (arm == "R") == (card.get("loss_coefficients", {}).get("ell_rot") == 0.1)
    assert card.get("archive", {}).get("full_trainer_checkpoints") is True
    return card, card_sha


def check_evidence(card: dict, card_sha: str, metadata_root: Path, step: int) -> tuple[dict, dict, dict, dict, list[dict], str]:
    preflight = read(metadata_root / "hf_preflight_receipt.json")
    manifest_path = metadata_root / "training_manifest.json"
    integrity_path = metadata_root / "pi0248_training_integrity.json"
    receipt_path = metadata_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
    readback_path = metadata_root / "hf_readbacks" / f"checkpoint-{step}.json"
    manifest, integrity, receipt, readback = map(read, (manifest_path, integrity_path, receipt_path, readback_path))
    assert preflight.get("status") == "PASS"
    assert preflight.get("repo_id") == card["archive"]["repo_id"]
    assert preflight.get("arm") == card["arm"] and preflight.get("run_card_sha256") == card_sha
    assert preflight.get("repo_public") is True and preflight.get("identity_verified") is True
    assert preflight.get("target_prefix_empty") is True
    assert preflight.get("write_probe", {}).get("readback") == "PASS"
    assert manifest.get("status") == "PASS_TRAINING_INTEGRITY"
    assert manifest.get("global_step") == 2000 and manifest.get("selected_checkpoint") == "NONE"
    assert manifest.get("run_card_sha256") == card_sha
    assert manifest.get("arm") == card["arm"] and manifest.get("family") == card["family"]
    assert integrity.get("status") == "PASS_PI0248_CR_TRAINING_INTEGRITY"
    assert integrity.get("arm") == card["arm"] and integrity.get("run_card_sha256") == card_sha
    assert integrity.get("training_manifest_sha256") == sha(manifest_path)
    assert integrity.get("initializer_revision") == REVISION
    assert integrity.get("initializer_tree_sha256") == TREE_SHA256
    assert integrity.get("updates") == 2000 and integrity.get("checkpoint_steps") == [1000, 2000]
    assert integrity.get("selected_checkpoint") == "NONE"
    if card["arm"] == "R":
        assert integrity.get("paired_noise_identity") == "PASS_ALL_2000_STEPS"
    assert receipt.get("status") == "DURABLE_VERIFIED"
    assert receipt.get("family") == card["family"]
    assert receipt.get("checkpoint_step") == step and receipt.get("max_steps") == 2000
    inventory = receipt["source_record"]["inventory"]
    assert receipt.get("hf_manifest_commit_id")
    assert readback.get("status") == "PASS_PT_FULL_HF_READBACK"
    assert readback.get("pi0247_status") == "PASS_PI0247_LU_FULL_HF_READBACK"
    assert readback.get("revision") == receipt["hf_manifest_commit_id"]
    assert readback.get("repo_id") == card["archive"]["repo_id"]
    assert readback.get("source_tree_sha256") == inventory["tree_sha256"]
    repo_id = card["archive"]["repo_id"]
    durable_path = receipt["durable_path"]
    assert durable_path.startswith(repo_id + "/")
    prefix = durable_path[len(repo_id) + 1:]
    suffix = f"/checkpoint-{step}"
    assert prefix.endswith(suffix)
    archive_prefix = prefix[:-len(suffix)]
    assert archive_prefix == readback.get("path_prefix")
    assert archive_prefix == card["archive"]["path_prefix"] or archive_prefix.startswith(card["archive"]["path_prefix"] + "/recovery/")
    expected = {prefix + "/" + row["path"]: row for row in inventory["entries"]}
    observed = {row["path"]: row for row in readback["files"]}
    assert set(expected) == set(observed)
    assert all(observed[name]["sha256"] == row["sha256"] and observed[name]["bytes"] == row["bytes"]
               for name, row in expected.items())
    model_files = [row for row in inventory["entries"]
                   if row["path"] in ("config.json", "model.safetensors.index.json")
                   or row["path"].endswith(".safetensors")]
    assert len(model_files) >= 4
    return manifest, integrity, receipt, readback, model_files, prefix


def materialize(card_path: Path, metadata_root: Path, arm: str, step: int) -> None:
    from huggingface_hub import snapshot_download

    card, card_sha = check_card(card_path, arm)
    _, integrity, receipt, readback, files, prefix = check_evidence(card, card_sha, metadata_root, step)
    download = SCREEN_ROOT / arm / "weights" / f"checkpoint-{step}"
    snapshot_download(card["archive"]["repo_id"], revision=receipt["hf_manifest_commit_id"],
                      allow_patterns=[prefix + "/" + row["path"] for row in files],
                      local_dir=str(download), token=False)
    view = SCREEN_ROOT / arm / "model_views" / f"checkpoint-{step}"
    view.mkdir(parents=True, exist_ok=True)
    for entry in files:
        source = download / prefix / entry["path"]
        assert source.is_file() and source.stat().st_size == entry["bytes"] and sha(source) == entry["sha256"]
        target = view / entry["path"]
        if target.is_symlink():
            assert target.resolve() == source.resolve()
        elif target.exists():
            assert target.is_file() and target.stat().st_size == entry["bytes"] and sha(target) == entry["sha256"]
        else:
            target.symlink_to(source)
    restore = {
        "schema": "PI0248_CR_INFERENCE_VIEW_V1", "arm": arm, "checkpoint_step": step,
        "optimizer_restored": False, "run_card_sha256": card_sha,
        "prefix": prefix, "revision": receipt["hf_manifest_commit_id"],
        "checkpoint_tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
        "durable_receipt_sha256": sha(metadata_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"),
        "hf_readback_sha256": sha(metadata_root / "hf_readbacks" / f"checkpoint-{step}.json"),
        "training_integrity_sha256": sha(metadata_root / "pi0248_training_integrity.json"),
        "files": files,
    }
    write(view / "pi0245_inference_restore.json", restore)
    print(json.dumps({"status": "PASS_PI0248_PINNED_INFERENCE_VIEW", "arm": arm,
                      "checkpoint_step": step, "revision": restore["revision"],
                      "tree_sha256": restore["checkpoint_tree_sha256"]}))


def screen(card_path: Path, metadata_root: Path, arm: str, step: int) -> None:
    import torch
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from transformers import AutoModel

    card, card_sha = check_card(card_path, arm)
    _, integrity, receipt, readback, _, prefix = check_evidence(card, card_sha, metadata_root, step)
    base_card, fit, ctx = m64.inputs()
    root = SCREEN_ROOT / arm / f"screen_step{step}"
    root.mkdir(parents=True, exist_ok=True)
    model_root = SCREEN_ROOT / arm / "model_views" / f"checkpoint-{step}"
    model_files = ctx["p1"].verify_model_files(model_root)
    restore = read(model_root / "pi0245_inference_restore.json")
    assert restore["run_card_sha256"] == card_sha and restore["arm"] == arm and restore["checkpoint_step"] == step
    assert restore["revision"] == receipt["hf_manifest_commit_id"]
    assert restore["checkpoint_tree_sha256"] == receipt["source_record"]["inventory"]["tree_sha256"]
    score = {
        "schema": "PI0248_CR_SCREEN_CARD_V1", "stage": "M64_REPAIR", "arm": arm,
        "checkpoint_step": step, "runtime_id": "S4", "selected_checkpoint": "NONE",
        "run_card_sha256": card_sha, "m64_parent_run_card_sha256": m64.CARD_SHA,
        "authorization": AUTH, "input_manifest_sha256": sha(m64.M64 / "screen_inputs_manifest.json"),
        "training_manifest_sha256": integrity["training_manifest_sha256"],
        "training_integrity_sha256": sha(metadata_root / "pi0248_training_integrity.json"),
        "checkpoint_binding": {"revision": receipt["hf_manifest_commit_id"], "prefix": prefix,
                                "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
                                "durable_receipt_sha256": restore["durable_receipt_sha256"],
                                "hf_readback_sha256": restore["hf_readback_sha256"]},
        "inference_restore_sha256": model_files["manifest_sha256"],
        "runner_sha256": sha(Path(__file__).resolve()),
        "loss_contract": card["loss_contract"], "processor_identity": fit["processor_identity"],
        "resource": {"device": "cuda:0"},
    }
    score_path = root / "score_card.json"
    write(score_path, score)
    score_sha = sha(score_path)
    model = AutoModel.from_pretrained(model_root, local_files_only=True, trust_remote_code=True)
    model.eval().to(device="cuda:0", dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps == 4
    old16 = set(base_card["support"]["old16_source_ids"])
    rows = []
    for source_row in ctx["rows"]:
        source = source_row["source_episode_id"]
        for noise in NOISES:
            ctx["noise"][noise] = ctx["m64_noise"][source][noise]
            ctx["facts"]["noise_files_sha256"][noise] = sha(Path(ctx["m64_packets"][source][noise]))
            query_root = root / "observations" / f"source-{source_row['episode_index']:06d}"
            record = d1.run_one(
                args=argparse.Namespace(output_root=query_root), card=score, card_sha=score_sha,
                context=ctx, model_name=f"PI0248_{arm}{step}",
                runtime={"id": "S4", "steps": 4, "inactive": "stock"}, source=source,
                noise_name=noise, model=model, model_files=model_files,
                bridge=bridge, selector=selector, stage43=stage43, gate=TrajectorySafetyGateV1(),
            )
            cell = query_root / "cells" / record["cell_key"]
            rows.append({**record, "noise_name": noise, "source_cell": str(cell),
                         "record_sha256": sha(cell / "record.json"), "score_card_path": str(score_path)})
    assert len(rows) == 128 and len({(r["source_episode_id"], r["noise_name"]) for r in rows}) == 128
    groups = {n: m64.group([r for r in rows if r["noise_name"] == n], 64, step == 2000) for n in NOISES}
    subsets = {n: m64.group([r for r in rows if r["noise_name"] == n and r["source_episode_id"] in old16],
                            16, step == 2000) for n in NOISES}
    passed = step == 2000 and all(g["quality_pass"] for g in list(groups.values()) + list(subsets.values()))
    report = {
        "schema": "PI0248_CR_SCREEN_REPORT_V1",
        "status": ("PASS_PI0248_CR_FINAL2000" if passed else "SCIENCE_FAIL_PI0248_CR_FINAL2000")
        if step == 2000 else "OBSERVATION_ONLY",
        "arm": arm, "checkpoint_step": step, "selected_checkpoint": "NONE",
        "run_card_sha256": card_sha, "score_card_sha256": score_sha,
        "score_card": score, "group_results": groups, "old16_group_results": subsets,
        "execution_cards": [{"path": str(score_path), "sha256": score_sha}],
        "counts": {"logical_cells": len(rows), "model_loads": 1, "model_forwards": len(rows),
                   "head_denoising_forwards": 4 * len(rows)},
        "cells": rows,
    }
    write(root / "screen_report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "arm", "checkpoint_step", "counts",
                                              "group_results", "old16_group_results")}))


def same(a: object, b: object) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def cpu_replay(card_path: Path, metadata_root: Path, arm: str, step: int) -> None:
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    card, card_sha = check_card(card_path, arm)
    _, integrity, receipt, _, _, _ = check_evidence(card, card_sha, metadata_root, step)
    _, fit, ctx = m64.inputs()
    root = SCREEN_ROOT / arm / f"screen_step{step}"
    report_path = root / "screen_report.json"
    report = read(report_path)
    score = report["score_card"]
    if step == 1000:
        assert report["status"] == "OBSERVATION_ONLY"
    else:
        assert report["status"] in ("PASS_PI0248_CR_FINAL2000", "SCIENCE_FAIL_PI0248_CR_FINAL2000")
    assert report["arm"] == arm and report["checkpoint_step"] == step
    assert report["selected_checkpoint"] == "NONE" and report["run_card_sha256"] == card_sha
    assert score["run_card_sha256"] == card_sha and score["runtime_id"] == "S4"
    assert score["training_manifest_sha256"] == integrity["training_manifest_sha256"]
    assert score["input_manifest_sha256"] == sha(m64.M64 / "screen_inputs_manifest.json")
    assert score["processor_identity"] == fit["processor_identity"]
    for entry in report["execution_cards"]:
        assert sha(Path(entry["path"])) == entry["sha256"]
    old16 = set(m64.read(m64.START / "run_card.json")["support"]["old16_source_ids"])
    sources = {r["source_episode_id"] for r in ctx["rows"]}
    expected = {(source, noise) for source in sources for noise in NOISES}
    seen, replay = set(), []
    for row in report["cells"]:
        source, noise = row["source_episode_id"], row["noise_name"]
        assert (source, noise) in expected and (source, noise) not in seen
        seen.add((source, noise))
        cell = Path(row["source_cell"])
        record = d1.collect_record(cell / "record.json", ctx["p1"])
        assert record["source_episode_id"] == source and record["noise_name"] == noise
        assert record["model_name"] == f"PI0248_{arm}{step}" and record["runtime"] == "S4" and record["steps"] == 4
        assert record["target_xyz_in_model_input"] is False
        assert record["model_manifest_sha256"] == score["inference_restore_sha256"]
        assert record["run_card_sha256"] == sha(Path(row["score_card_path"])) == report["score_card_sha256"]
        source_parent = Path(ctx["source_cells"][source][noise]["cell_path"])
        parent = m64.read(source_parent / "record.json")
        assert record["condition_identity"] == ctx["source_cells"][source][noise]["condition_identity"]
        assert record["condition_sha256"] == sha(source_parent / "condition.pt") == parent["condition_sha256"]
        assert record["source_tensor_sha256"] == parent["source_tensor_sha256"]
        tensor_path = cell / ("tensors.npz" if record["status"] == "VALID" else "invalid_output.npz")
        with np.load(tensor_path, allow_pickle=False) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files if k not in ("raw_native", "raw_scored"))
            raw = data["raw_scored"].copy()
            fixed = ctx["m64_noise"][source][noise]
            native = torch.from_numpy(fixed).to(torch.bfloat16).float().numpy()
            assert np.array_equal(data["noise_source_float32"], fixed)
            assert np.array_equal(data["noise_initial"], native)
            assert np.array_equal(data["trace_x"][0], native)
            assert data["trace_x"].shape == (4, 1, 40, 132)
            assert data["trace_v"].shape == (4, 1, 40, 132)
            if record["status"] in ("INVALID_RAW_NONFINITE", "INVALID_DECODE"):
                if record["status"] == "INVALID_RAW_NONFINITE":
                    assert not np.isfinite(data["raw_native"]).all()
                else:
                    assert np.isfinite(data["raw_native"]).all()
                    try:
                        bridge._decode_physical(ctx["processor"], ctx["embodiment"], data["raw_scored"], data["state"])
                    except Exception:
                        pass
                    else:
                        raise AssertionError("saved invalid decode succeeded on independent CPU replay")
                assert record["gate_decision"] == "UNSCORABLE" and record.get("metrics") is None
                replay.append({"source_episode_id": source, "noise_name": noise, "status": record["status"],
                               "metrics": None, "gate_decision": "UNSCORABLE", "tensor_sha256": record["tensor_sha256"]})
                continue
            assert record["status"] == "VALID"
            assert np.array_equal(data["raw_scored"], data["raw_native"])
            assert not record["m_path"]["applicable"]
            decoded_saved, state, expert = [data[k].copy() for k in ("decoded", "state", "expert")]
        decoded = bridge._decode_physical(ctx["processor"], ctx["embodiment"], raw, state)
        error = float(np.max(np.abs(decoded - decoded_saved)))
        assert error <= 1e-7
        metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
        residual = decoded[0, :, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
        squared = np.sum(residual * residual, axis=1)
        metrics["physical_residual_delta_max_mps"] = float(np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / .05)
        metrics["tail35_39_pooled_sse_fraction"] = float(squared[35:].sum() / squared.sum()) if squared.sum() else 0.0
        assert set(metrics) == set(row["metrics"]) and all(same(v, row["metrics"][k]) for k, v in metrics.items())
        gate = TrajectorySafetyGateV1().evaluate(
            selector._build_chunk(decoded, chunk_id=f"pi0247-S4-{record['model_name']}-{source}-{noise}"),
            now=0.0, achieved_position=state[:3],
            achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0],
        )
        assert gate.decision.value == record["gate_decision"] == row["gate_decision"]
        issues = gate.as_dict()["issues"]
        assert len(issues) == len(record["gate"]["issues"])
        for actual, saved in zip(issues, record["gate"]["issues"]):
            assert set(actual) == set(saved) and all(
                same(value, saved[key]) if isinstance(value, (int, float)) else value == saved[key]
                for key, value in actual.items())
        replay.append({"source_episode_id": source, "noise_name": noise, "status": "VALID",
                       "metrics": metrics, "gate_decision": gate.decision.value,
                       "tensor_sha256": record["tensor_sha256"], "decoded_max_abs": error,
                       "model_source_card_noise_binding_verified": True})
    assert seen == expected
    groups = {n: m64.group([r for r in replay if r["noise_name"] == n], 64, step == 2000) for n in NOISES}
    subsets = {n: m64.group([r for r in replay if r["noise_name"] == n and r["source_episode_id"] in old16],
                            16, step == 2000) for n in NOISES}
    assert groups == report["group_results"] and subsets == report["old16_group_results"]
    result = {
        "schema": "PI0248_CR_INDEPENDENT_CPU_REPLAY_V1", "status": "PASS_INDEPENDENT_CPU_REPLAY",
        "arm": arm, "checkpoint_step": step, "selected_checkpoint": "NONE",
        "run_card_sha256": card_sha, "training_integrity_sha256": sha(metadata_root / "pi0248_training_integrity.json"),
        "screen_report_sha256": sha(report_path),
        "counts": {"cells_replayed": len(replay), "model_loads": 0, "model_forwards": 0},
        "group_results": groups, "old16_group_results": subsets, "cells": replay,
    }
    write(root / "cpu_replay.json", result)
    print(json.dumps({"status": result["status"], "arm": arm, "step": step,
                      "cells": len(replay), "sha256": sha(root / "cpu_replay.json")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare-inputs", "materialize", "screen", "cpu-replay"), required=True)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--arm", choices=("C", "R"))
    parser.add_argument("--step", type=int, choices=STEPS)
    args = parser.parse_args()
    if args.mode == "prepare-inputs":
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        m64.inputs()
        print(json.dumps({"status": "PASS_PI0248_SHARED_M64_INPUTS", "manifest_sha256": sha(m64.M64 / "screen_inputs_manifest.json")}))
        return
    if not all((args.card, args.metadata_root, args.arm, args.step)):
        parser.error("materialize/screen/cpu-replay require --card, --metadata-root, --arm, and --step")
    if args.mode == "materialize":
        os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="1")
        materialize(args.card, args.metadata_root, args.arm, args.step)
    elif args.mode == "screen":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        screen(args.card, args.metadata_root, args.arm, args.step)
    else:
        cpu_replay(args.card, args.metadata_root, args.arm, args.step)


if __name__ == "__main__":
    main()
