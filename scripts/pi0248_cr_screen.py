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
ROOT = Path(os.environ.get("PI0248_AS_EXECUTED_ROOT", "/var/cache/users/liuyu-nfs/pi0248_as_executed_qualification_20260927"))
SCREEN_ROOT = ROOT / "as_executed_screens"
AS_EXECUTED_AUTH = "PI0248_AS_EXECUTED_QUALIFICATION_GO"
AS_EXECUTED_PROTOCOL = "2f75aec82f9d0fe4d72c154615f339f50c7eb41c"
AS_EXECUTED_ISSUE_COMMENT = 5851408869
PINNED_CHECKPOINTS = {
    ("C", 1000): ("3d817c29f972735475c2e5fde3b244aede6f6d79", "8c0cfdd6116cdfc0e295091900539b438fcd56d87423a4cbcc7a28a5fc8c5e46"),
    ("C", 2000): ("001d86cc3492304e4df2816b3182a5e826daa0b7", "d72cb37db3f2f3931a7adacf24c2ff8736e4c1dc4836e3a4c75dd3087758425b"),
    ("R", 1000): ("b32f423d6dc58f00a05c246d409ac594f02fd78b", "b3077d1c25c40271ec27fdd031b7bcc192df39b10f4801ab0153015a702ed3fd"),
    ("R", 2000): ("d089c09380c568e720c3dfa6bec5aa064acd7c4f", "4422a24d2117206681c423a0f8d3137dff7eb614c990cb0334a8cdaf73900298"),
}
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
    check_strict_pair_integrity(integrity, card["arm"])
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


def validate_deviation_manifest(deviation: dict) -> None:
    assert deviation.get("schema") == "PI0248_AS_EXECUTED_PAIR_DEVIATION_V1"
    assert deviation.get("protocol_commit") == AS_EXECUTED_PROTOCOL
    assert deviation.get("canonical_issue_comment_id") == AS_EXECUTED_ISSUE_COMMENT
    assert deviation.get("authorization") == AS_EXECUTED_AUTH
    expected = {
        "original_paired_execution_status": "STOP_UNPAIRED_NATIVE_TAU_IDENTITY_MISMATCH",
        "training_pairing_scope": "PREFIX_1_1000_ONLY",
        "training_tau_full_pair": False,
        "native_epsilon_same_index_matches": 2000,
        "stock_noise_same_index_matches": 2000,
        "tau_same_index_matches": 1005,
        "tau_mismatch_first_update": 1001,
        "tau_mismatch_count": 995,
        "C_full_stochastic_resume_equivalence": False,
        "original_pair_failure_preserved": True,
        "final2000_single_factor_causal_attribution_admitted": False,
    }
    assert all(deviation.get(key) == value for key, value in expected.items())


def check_strict_pair_integrity(integrity: dict, arm: str) -> None:
    if arm == "R":
        assert integrity.get("paired_noise_identity") == "PASS_ALL_2000_STEPS"


def validate_as_executed_binding(card: dict, card_sha: str, arm: str, step: int,
                                 integrity: dict, deviation: dict, receipt: dict,
                                 readback: dict, receipt_sha: str, readback_sha: str) -> None:
    validate_deviation_manifest(deviation)
    assert integrity.get("schema") == "PI0248_AS_EXECUTED_ARM_INTEGRITY_V1"
    assert integrity.get("status") == "PASS_AS_EXECUTED_ARM_INTEGRITY"
    assert integrity.get("authorization") == AS_EXECUTED_AUTH
    assert integrity.get("protocol_commit") == AS_EXECUTED_PROTOCOL
    assert integrity.get("canonical_issue_comment_id") == AS_EXECUTED_ISSUE_COMMENT
    assert integrity.get("arm") == arm == card.get("arm")
    assert integrity.get("family") == card.get("family")
    assert integrity.get("selected_checkpoint") == "NONE"
    training = integrity["training"]
    assert training.get("optimizer_updates") == 2000
    assert training.get("training_card_sha256") == card_sha
    assert training.get("execution_code_commit") == card.get("source", {}).get("execution_code_commit")
    assert training.get("schedule_rows") == 2000 and training.get("replayed_updates") == 0
    checks = integrity["integrity_checks"]
    assert checks.get("objective_formula_match_rows") == checks.get("objective_finite_rows") == 2000
    assert checks.get("full_s4_forward_and_backward_head_evaluations") == 8000
    assert checks.get("mode_restoration_rows") == 2000 and checks.get("mode_restoration_all_true") is True
    membership = checks["optimizer_membership"]
    assert membership == {
        "trainable_parameter_tensors": 471, "frozen_parameter_tensors": 559,
        "groups": {"diffusion": 456, "projector": 15}, "illegal_frozen": 0, "missing_trainable": 0,
    }
    assert checks.get("frozen_backbone_delta_l2") == 0 and checks.get("frozen_vlln_delta_l2") == 0
    assert checks.get("projector_full_tensor_reconciliation") is True
    if arm == "C":
        assert integrity.get("legacy_artifacts", {}).get("legacy_status") == "PASS_PI0248_CR_TRAINING_INTEGRITY"
    else:
        assert integrity.get("legacy_artifacts", {}).get("new_schema_only") is True
        assert integrity.get("legacy_artifacts", {}).get("old_manifest_status") == "NOT_WRITTEN_CROSS_ARM_PAIR_GATE"
    assert card.get("stage", {}).get("steps") == 2000
    assert card.get("stage", {}).get("checkpoint_steps") == [1000, 2000]
    binding = integrity["checkpoints"][str(step)]
    expected_revision, expected_tree = PINNED_CHECKPOINTS[(arm, step)]
    inventory = receipt["source_record"]["inventory"]
    repo_id = card["archive"]["repo_id"]
    assert receipt.get("durable_path", "").startswith(repo_id + "/")
    assert receipt.get("status") == "DURABLE_VERIFIED"
    assert receipt.get("family") == card["family"]
    assert receipt.get("checkpoint_step") == step and receipt.get("max_steps") == 2000
    assert receipt.get("hf_manifest_commit_id") == expected_revision
    assert inventory.get("tree_sha256") == expected_tree and len(inventory.get("entries", [])) == 9
    assert binding.get("revision") == expected_revision and binding.get("tree_sha256") == expected_tree
    assert binding.get("durable_receipt_sha256") == receipt_sha
    assert binding.get("readback_sha256") == readback_sha
    assert binding.get("full_trainer_files") == 9
    assert readback.get("status") == "PASS_PT_FULL_HF_READBACK"
    assert readback.get("revision") == expected_revision and readback.get("source_tree_sha256") == expected_tree
    assert readback.get("repo_id") == repo_id and len(readback.get("files", [])) == 9


def check_as_executed_evidence(card: dict, card_sha: str, metadata_root: Path, step: int,
                               receipt_path: Path, deviation_path: Path) -> tuple[dict, dict, dict, dict, list[dict], str]:
    durable_path = metadata_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
    readback_path = metadata_root / "hf_readbacks" / f"checkpoint-{step}.json"
    as_receipt = read(receipt_path)
    deviation = read(deviation_path)
    durable, readback = map(read, (durable_path, readback_path))
    validate_as_executed_binding(card, card_sha, card["arm"], step, as_receipt, deviation,
                                 durable, readback, sha(durable_path), sha(readback_path))
    inventory = durable["source_record"]["inventory"]
    files = [row for row in inventory["entries"]
             if row["path"] in ("config.json", "model.safetensors.index.json")
             or row["path"].endswith(".safetensors")]
    assert len(files) >= 4
    prefix = durable["durable_path"][len(card["archive"]["repo_id"]) + 1:]
    as_receipt["_integrity_file_sha256"] = sha(receipt_path)
    as_receipt["_deviation_manifest_sha256"] = sha(deviation_path)
    return {}, as_receipt, durable, readback, files, prefix


def check_scope_evidence(card: dict, card_sha: str, metadata_root: Path, step: int,
                         analysis_scope: str, as_executed_receipt_path: Path | None,
                         deviation_manifest_path: Path | None):
    if analysis_scope == "strict":
        manifest, integrity, receipt, readback, files, prefix = check_evidence(card, card_sha, metadata_root, step)
        integrity["_integrity_file_sha256"] = sha(metadata_root / "pi0248_training_integrity.json")
        integrity["_deviation_manifest_sha256"] = None
        return manifest, integrity, receipt, readback, files, prefix
    if analysis_scope == "as-executed":
        assert as_executed_receipt_path is not None and deviation_manifest_path is not None
        return check_as_executed_evidence(card, card_sha, metadata_root, step,
                                          as_executed_receipt_path, deviation_manifest_path)
    raise ValueError(f"unknown analysis scope: {analysis_scope}")


def materialize(card_path: Path, metadata_root: Path, arm: str, step: int, analysis_scope: str = "strict", as_executed_receipt_path: Path | None = None, deviation_manifest_path: Path | None = None) -> None:
    from huggingface_hub import snapshot_download

    card, card_sha = check_card(card_path, arm)
    _, integrity, receipt, readback, files, prefix = check_scope_evidence(card, card_sha, metadata_root, step, analysis_scope, as_executed_receipt_path, deviation_manifest_path)
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
        "schema": "PI0248_AS_EXECUTED_CR_INFERENCE_VIEW_V1" if analysis_scope == "as-executed" else "PI0248_CR_INFERENCE_VIEW_V1", "arm": arm, "checkpoint_step": step,
        "optimizer_restored": False, "run_card_sha256": card_sha,
        "prefix": prefix, "revision": receipt["hf_manifest_commit_id"],
        "checkpoint_tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
        "durable_receipt_sha256": sha(metadata_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"),
        "hf_readback_sha256": sha(metadata_root / "hf_readbacks" / f"checkpoint-{step}.json"),
        "training_integrity_sha256": integrity["_integrity_file_sha256"],
        "analysis_scope": analysis_scope,
        "deviation_manifest_sha256": integrity["_deviation_manifest_sha256"],
        "files": files,
    }
    write(view / "pi0245_inference_restore.json", restore)
    print(json.dumps({"status": "PASS_PI0248_PINNED_INFERENCE_VIEW", "arm": arm,
                      "checkpoint_step": step, "revision": restore["revision"],
                      "tree_sha256": restore["checkpoint_tree_sha256"]}))


def screen(card_path: Path, metadata_root: Path, arm: str, step: int, analysis_scope: str = "strict", as_executed_receipt_path: Path | None = None, deviation_manifest_path: Path | None = None) -> None:
    import torch
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from transformers import AutoModel

    card, card_sha = check_card(card_path, arm)
    _, integrity, receipt, readback, _, prefix = check_scope_evidence(card, card_sha, metadata_root, step, analysis_scope, as_executed_receipt_path, deviation_manifest_path)
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
        "schema": "PI0248_AS_EXECUTED_SCREEN_CARD_V1" if analysis_scope == "as-executed" else "PI0248_CR_SCREEN_CARD_V1", "stage": "M64_REPAIR", "arm": arm,
        "analysis_scope": analysis_scope,
        "evaluation_authorization": AS_EXECUTED_AUTH if analysis_scope == "as-executed" else AUTH,
        "training_card_authorization": AUTH,
        "checkpoint_step": step, "runtime_id": "S4", "selected_checkpoint": "NONE",
        "run_card_sha256": card_sha, "m64_parent_run_card_sha256": m64.CARD_SHA,
        "authorization": AS_EXECUTED_AUTH if analysis_scope == "as-executed" else AUTH, "input_manifest_sha256": sha(m64.M64 / "screen_inputs_manifest.json"),
        "training_manifest_sha256": integrity.get("training_manifest_sha256"),
        "training_integrity_sha256": integrity["_integrity_file_sha256"],
        "deviation_manifest_sha256": integrity["_deviation_manifest_sha256"],
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
                context=ctx, model_name=(f"PI0248_AS_EXECUTED_{arm}{step}" if analysis_scope == "as-executed" else f"PI0248_{arm}{step}"),
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
        "schema": "PI0248_AS_EXECUTED_CR_SCREEN_REPORT_V1" if analysis_scope == "as-executed" else "PI0248_CR_SCREEN_REPORT_V1",
        "analysis_scope": analysis_scope,
        "status": (("PASS_PI0248_AS_EXECUTED_FINAL2000" if passed else "SCIENCE_FAIL_PI0248_AS_EXECUTED_FINAL2000") if analysis_scope == "as-executed" else ("PASS_PI0248_CR_FINAL2000" if passed else "SCIENCE_FAIL_PI0248_CR_FINAL2000"))
        if step == 2000 else ("AS_EXECUTED_INTERMEDIATE_ONLY" if analysis_scope == "as-executed" else "OBSERVATION_ONLY"),
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


def cpu_replay(card_path: Path, metadata_root: Path, arm: str, step: int, analysis_scope: str = "strict", as_executed_receipt_path: Path | None = None, deviation_manifest_path: Path | None = None) -> None:
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import pi0247_d1_screen as d1
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    card, card_sha = check_card(card_path, arm)
    _, integrity, receipt, _, _, _ = check_scope_evidence(card, card_sha, metadata_root, step, analysis_scope, as_executed_receipt_path, deviation_manifest_path)
    _, fit, ctx = m64.inputs()
    root = SCREEN_ROOT / arm / f"screen_step{step}"
    report_path = root / "screen_report.json"
    report = read(report_path)
    score = report["score_card"]
    if step == 1000:
        assert report["status"] == ("AS_EXECUTED_INTERMEDIATE_ONLY" if analysis_scope == "as-executed" else "OBSERVATION_ONLY")
    else:
        allowed = ("PASS_PI0248_AS_EXECUTED_FINAL2000", "SCIENCE_FAIL_PI0248_AS_EXECUTED_FINAL2000") if analysis_scope == "as-executed" else ("PASS_PI0248_CR_FINAL2000", "SCIENCE_FAIL_PI0248_CR_FINAL2000")
        assert report["status"] in allowed
    assert report["arm"] == arm and report["checkpoint_step"] == step
    assert report["selected_checkpoint"] == "NONE" and report["run_card_sha256"] == card_sha
    assert score["run_card_sha256"] == card_sha and score["runtime_id"] == "S4"
    assert score.get("training_manifest_sha256") == integrity.get("training_manifest_sha256")
    assert score["training_integrity_sha256"] == integrity["_integrity_file_sha256"]
    assert score["deviation_manifest_sha256"] == integrity["_deviation_manifest_sha256"]
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
        expected_model_name = f"PI0248_AS_EXECUTED_{arm}{step}" if analysis_scope == "as-executed" else f"PI0248_{arm}{step}"
        assert record["model_name"] == expected_model_name and record["runtime"] == "S4" and record["steps"] == 4
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
        "schema": "PI0248_AS_EXECUTED_CR_INDEPENDENT_CPU_REPLAY_V1" if analysis_scope == "as-executed" else "PI0248_CR_INDEPENDENT_CPU_REPLAY_V1", "status": "PASS_INDEPENDENT_CPU_REPLAY",
        "analysis_scope": analysis_scope,
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
    parser.add_argument("--mode", choices=("prepare-inputs", "materialize", "screen", "cpu-replay", "self-check"), required=True)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--arm", choices=("C", "R"))
    parser.add_argument("--step", type=int, choices=STEPS)
    parser.add_argument("--analysis-scope", choices=("strict", "as-executed"), default="strict")
    parser.add_argument("--as-executed-receipt", type=Path)
    parser.add_argument("--deviation-manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "self-check":
        deviation = {
            "schema": "PI0248_AS_EXECUTED_PAIR_DEVIATION_V1", "protocol_commit": AS_EXECUTED_PROTOCOL,
            "canonical_issue_comment_id": AS_EXECUTED_ISSUE_COMMENT, "authorization": AS_EXECUTED_AUTH,
            "original_paired_execution_status": "STOP_UNPAIRED_NATIVE_TAU_IDENTITY_MISMATCH",
            "training_pairing_scope": "PREFIX_1_1000_ONLY", "training_tau_full_pair": False,
            "native_epsilon_same_index_matches": 2000, "stock_noise_same_index_matches": 2000,
            "tau_same_index_matches": 1005, "tau_mismatch_first_update": 1001, "tau_mismatch_count": 995,
            "C_full_stochastic_resume_equivalence": False, "original_pair_failure_preserved": True,
            "final2000_single_factor_causal_attribution_admitted": False,
        }
        card = {"arm": "R", "family": "PI0248_R_ROTATION_ENDPOINT_M64_2K_V1",
                "source": {"execution_code_commit": "0a37e542f3d653964c894bb550d428557282462f"},
                "stage": {"steps": 2000, "checkpoint_steps": [1000, 2000]},
                "archive": {"repo_id": "leoyuer/pi0246-gate-pass-campaign"}}
        integrity = {
            "schema": "PI0248_AS_EXECUTED_ARM_INTEGRITY_V1", "status": "PASS_AS_EXECUTED_ARM_INTEGRITY",
            "authorization": AS_EXECUTED_AUTH, "protocol_commit": AS_EXECUTED_PROTOCOL,
            "canonical_issue_comment_id": AS_EXECUTED_ISSUE_COMMENT, "arm": "R", "family": card["family"],
            "selected_checkpoint": "NONE",
            "training": {"optimizer_updates": 2000, "training_card_sha256": "card",
                         "execution_code_commit": card["source"]["execution_code_commit"], "schedule_rows": 2000, "replayed_updates": 0},
            "integrity_checks": {"objective_formula_match_rows": 2000, "objective_finite_rows": 2000,
                "full_s4_forward_and_backward_head_evaluations": 8000, "mode_restoration_rows": 2000,
                "mode_restoration_all_true": True,
                "optimizer_membership": {"trainable_parameter_tensors": 471, "frozen_parameter_tensors": 559,
                    "groups": {"diffusion": 456, "projector": 15}, "illegal_frozen": 0, "missing_trainable": 0},
                "frozen_backbone_delta_l2": 0, "frozen_vlln_delta_l2": 0, "projector_full_tensor_reconciliation": True},
            "legacy_artifacts": {"new_schema_only": True, "old_manifest_status": "NOT_WRITTEN_CROSS_ARM_PAIR_GATE"},
            "checkpoints": {"1000": {"revision": PINNED_CHECKPOINTS[("R", 1000)][0],
                "tree_sha256": PINNED_CHECKPOINTS[("R", 1000)][1], "durable_receipt_sha256": "receipt",
                "readback_sha256": "readback", "full_trainer_files": 9}}
        }
        checkpoint = {"status": "DURABLE_VERIFIED", "family": card["family"], "checkpoint_step": 1000,
            "max_steps": 2000, "hf_manifest_commit_id": PINNED_CHECKPOINTS[("R", 1000)][0],
            "source_record": {"inventory": {"tree_sha256": PINNED_CHECKPOINTS[("R", 1000)][1], "entries": [{}] * 9}}}
        readback = {"status": "PASS_PT_FULL_HF_READBACK", "revision": checkpoint["hf_manifest_commit_id"],
            "source_tree_sha256": integrity["checkpoints"]["1000"]["tree_sha256"],
            "repo_id": card["archive"]["repo_id"], "files": [{}] * 9}
        validate_as_executed_binding(card, "card", "R", 1000, integrity, deviation, checkpoint, readback, "receipt", "readback")
        for bad in (
            {**integrity, "authorization": "wrong"},
            {**integrity, "checkpoints": {"1000": {**integrity["checkpoints"]["1000"], "revision": "wrong"}}},
        ):
            try:
                validate_as_executed_binding(card, "card", "R", 1000, bad, deviation, checkpoint, readback, "receipt", "readback")
            except AssertionError:
                pass
            else:
                raise AssertionError("invalid as-executed integrity/checkpoint was admitted")
        for bad_deviation in (None, {**deviation, "authorization": "wrong"}):
            try:
                validate_as_executed_binding(card, "card", "R", 1000, integrity, bad_deviation, checkpoint, readback, "receipt", "readback")
            except (AssertionError, AttributeError):
                pass
            else:
                raise AssertionError("missing/unauthorized deviation was admitted")
        try:
            check_strict_pair_integrity({"paired_noise_identity": "STOP_UNPAIRED_NATIVE_TAU_IDENTITY_MISMATCH"}, "R")
        except AssertionError:
            pass
        else:
            raise AssertionError("strict scope accepted the original tau pairing failure")
        assert not m64.group([], 64, True)["quality_pass"]
        reject_rows = [{"status": "VALID", "gate_decision": "REJECT",
                        "metrics": {"xyz_rmse_m": .01, "so3_rmse_rad": .01}}] * 64
        assert m64.group(reject_rows, 64, True)["quality_pass"] is False
        print("PASS_AS_EXECUTED_AUTHORIZATION_AND_CHECKPOINT_BOUNDARY")
        print("PASS_STRICT_TAU_PAIR_REJECTION")
        print("PASS_EMPTY_AND_REJECT_QUALITY_GATES")
        return
    if args.mode == "prepare-inputs":
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        m64.inputs()
        print(json.dumps({"status": "PASS_PI0248_SHARED_M64_INPUTS",
                          "manifest_sha256": sha(m64.M64 / "screen_inputs_manifest.json")}))
        return
    if not all((args.card, args.metadata_root, args.arm, args.step)):
        parser.error("materialize/screen/cpu-replay require --card, --metadata-root, --arm, and --step")
    if args.analysis_scope == "as-executed":
        if not all((args.as_executed_receipt, args.deviation_manifest)):
            parser.error("as-executed scope requires --as-executed-receipt and --deviation-manifest")
    elif args.as_executed_receipt or args.deviation_manifest:
        parser.error("as-executed evidence may only be passed with --analysis-scope as-executed")
    if args.mode == "materialize":
        os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="1")
        materialize(args.card, args.metadata_root, args.arm, args.step, args.analysis_scope,
                    args.as_executed_receipt, args.deviation_manifest)
    elif args.mode == "screen":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        screen(args.card, args.metadata_root, args.arm, args.step, args.analysis_scope,
               args.as_executed_receipt, args.deviation_manifest)
    elif args.mode == "cpu-replay":
        cpu_replay(args.card, args.metadata_root, args.arm, args.step, args.analysis_scope,
                   args.as_executed_receipt, args.deviation_manifest)
    else:
        parser.error("unknown mode")


if __name__ == "__main__":
    main()
