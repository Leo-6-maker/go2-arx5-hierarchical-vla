#!/usr/bin/env python3
"""Seal the PI0252 A/G cards from the fixed PI0251 C2000 and R1 inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REVISION = "7017368c42800927a17be5561056ceff58585de7"
TREE = "dbc5c5ff07ac73cd4bf448b5e679d0641a51ea4b33695246db79d0a5767a8144"
AUTH = "PI0252_R1_SPEED_MARGIN_TO_FULL_FINETUNE_GO"
CANONICAL = 5862818190
POINTER = 5862824588
G0_SHA = "e184f76f8155a0445640b675ea7b559d99ceaed29cd92a722e54ab5f85bbae5d"
STEPS = 2000
CHECKPOINTS = (1000, 2000)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"refusing to replace existing sealed PI0252 preparation file: {path}")
        return
    path.write_text(data, encoding="utf-8")


def checkpoint_pair(parent: dict[str, Any], integrity: dict[str, Any]) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    output = Path(parent["paths"]["output_root"])
    matches = []
    for receipt_path in output.glob("recovery_attempts/*/durable_receipts/checkpoint-2000.durable_receipt.json"):
        readback_path = receipt_path.parents[1] / "hf_readbacks" / "checkpoint-2000.json"
        if not readback_path.is_file():
            continue
        receipt, readback = read(receipt_path), read(readback_path)
        inventory = receipt.get("source_record", {}).get("inventory", {})
        if (receipt.get("status") == "DURABLE_VERIFIED" and receipt.get("family") == parent["family"]
                and receipt.get("checkpoint_step") == 2000
                and receipt.get("hf_manifest_commit_id") == REVISION
                and inventory.get("tree_sha256") == TREE
                and readback.get("status") == "PASS_PT_FULL_HF_READBACK"
                and readback.get("revision") == REVISION
                and readback.get("source_tree_sha256") == TREE):
            matches.append((receipt_path, readback_path, receipt, readback))
    if len(matches) != 1:
        raise RuntimeError(f"expected one PI0251 C2000 durable/readback pair, found {len(matches)}")
    record = integrity.get("checkpoints", {}).get("2000", {})
    receipt_path, readback_path, receipt, readback = matches[0]
    if (record.get("tree_sha256") != TREE or record.get("hf_manifest_commit_id") != REVISION
            or record.get("durable_receipt_sha256") != sha(receipt_path)
            or record.get("full_hf_readback_sha256") != sha(readback_path)):
        raise RuntimeError("PI0251 C integrity record differs from the selected durable C2000 pair")
    entries = {row["path"]: row for row in receipt["source_record"]["inventory"]["entries"]}
    observed = {row["path"].split("/checkpoint-2000/", 1)[-1]: row for row in readback["files"]
                if "/checkpoint-2000/" in row["path"]}
    if set(entries) != set(observed) or any(
            int(observed[name]["bytes"]) != int(entry["bytes"])
            or observed[name]["sha256"] != entry["sha256"] for name, entry in entries.items()):
        raise RuntimeError("PI0251 C2000 full Trainer readback differs from its durable inventory")
    return receipt_path, readback_path, receipt, readback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-card", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--pbs-script", type=Path, required=True)
    parser.add_argument("--g0-receipt", type=Path, required=True)
    parser.add_argument("--g1-receipt", type=Path, required=True)
    parser.add_argument("--timestamp-path", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()

    parent_path = args.parent_card.resolve()
    parent = read(parent_path)
    parent_sha = sha(parent_path)
    if (parent.get("arm") != "C" or parent.get("schema") != "PI0251_CE_RUN_CARD_V1"
            or parent.get("selected_checkpoint") != "NONE"
            or read(parent_path.with_name("run_card.seal.json")).get("sha256") != parent_sha):
        raise RuntimeError("PI0252 requires the sealed PI0251 C run card as its common initializer")
    output_c = Path(parent["paths"]["output_root"])
    integrity_path = output_c / "pi0251_training_integrity.json"
    integrity = read(integrity_path)
    if (integrity.get("status") != "PASS_PI0251_C_TRAINING_INTEGRITY"
            or integrity.get("arm") != "C" or integrity.get("run_card_sha256") != parent_sha
            or integrity.get("optimizer_updates") != STEPS):
        raise RuntimeError("PI0251 C2000 lacks its read-only training-integrity PASS")
    receipt_path, readback_path, receipt, readback = checkpoint_pair(parent, integrity)
    g0_source, g1_source = args.g0_receipt.resolve(), args.g1_receipt.resolve()
    if sha(g0_source) != G0_SHA or read(g0_source).get("status") != "PI0252_GATE_CONFIGURATION_BINDING_CONFLICT":
        raise RuntimeError("the sealed PI0252 G0 attribution receipt changed")
    if (read(g1_source).get("status") != "PASS_PI0252_AG_CPU_FIXTURES"
            or read(g1_source).get("test_count") != 5):
        raise RuntimeError("PI0252 speed/identity CPU fixture receipt is incomplete")

    task_root, execution_root = args.task_root.resolve(), args.execution_root.resolve()
    if task_root.exists() and any(item.resolve() != execution_root for item in task_root.iterdir()):
        raise RuntimeError(f"PI0252 task root already has files; refusing to overwrite: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    (task_root / "logs").mkdir(parents=True, exist_ok=True)
    for arm in ("A", "G"):
        (task_root / arm / "logs").mkdir(parents=True, exist_ok=True)
    evidence_root = task_root / "evidence"
    evidence_root.mkdir(exist_ok=True)
    for source, name in ((g0_source, "PI0252_G0_READONLY_ATTRIBUTION_20260928.json"),
                         (g1_source, "PI0252_AG_CPU_SELF_CHECK.json")):
        target = evidence_root / name
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != sha(source):
            raise RuntimeError(f"PI0252 evidence copy changed: {name}")

    sys.path.insert(0, str(execution_root))
    spec = importlib.util.spec_from_file_location("pi0252_prepare_engine", execution_root / "pi0251_ce_train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("PI0252 shared training adapter is missing")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.REVISION, engine.TREE_SHA256 = REVISION, TREE
    schedule_ref = copy.deepcopy(parent["schedule"])
    schedule_rows = read(Path(schedule_ref["path"]))["rows"]
    if len(schedule_rows) != STEPS or sha(Path(schedule_ref["path"])) != schedule_ref["sha256"]:
        raise RuntimeError("PI0252 inherited exposure schedule is not the sealed 64/2000 table")
    timestamp_path = args.timestamp_path.resolve()
    timestamp_sha = sha(timestamp_path)
    dt_rows = engine._derive_actual_dt_rows(schedule_rows, timestamp_path)
    dt_path = task_root / "evidence" / "pi0252_actual_h40_dt_table.json"
    write_once(dt_path, {"schema": "PI0252_ACTUAL_H40_DT_TABLE_V1", "timestamp_sha256": timestamp_sha,
                         "schedule_sha256": schedule_ref["sha256"], "rows": dt_rows})

    rng_spec = importlib.util.spec_from_file_location("pi0252_prepare_step_rng", execution_root / "pi0252_step_rng.py")
    if rng_spec is None or rng_spec.loader is None:
        raise RuntimeError("PI0252 namespaced RNG module is missing")
    rng = importlib.util.module_from_spec(rng_spec)
    rng_spec.loader.exec_module(rng)
    key_path = task_root / "evidence" / "pi0252_step_key_table.json"
    write_once(key_path, {"schema": "PI0252_STEP_KEY_TABLE_V1", "seed": 22301,
                          "schedule_sha256": schedule_ref["sha256"],
                          "rows": rng.build_key_table(schedule_rows)})

    processor = parent["source"]["processor_bindings"]
    view_root = task_root / "initializer_C2000"
    inventory = receipt["source_record"]["inventory"]
    entries = {row["path"]: row for row in inventory["entries"]}
    observed = {str(row["path"]).split("/checkpoint-2000/", 1)[-1]: row for row in readback["files"]
                if "/checkpoint-2000/" in str(row["path"])}
    model_files = engine._materialize_x2000_view(
        view_root, parent["archive"]["repo_id"], parent["archive"]["path_prefix"], entries,
        {name: row["path"] for name, row in observed.items()}, Path(parent["paths"]["base_model_path"]),
        processor["view_files"],
    )
    view_path = task_root / "initializer_C2000_view_receipt.json"
    write_once(view_path, {"status": "PASS_PI0252_C2000_CT_VIEW", "revision": REVISION,
                           "checkpoint_tree_sha256": TREE, "model_files": model_files,
                           "source_durable_receipt_sha256": sha(receipt_path),
                           "source_hf_readback_sha256": sha(readback_path),
                           "processor_identity": {key: value for key, value in processor.items()
                                                  if key != "view_files"}})

    code_names = ("pi0251_ce_train.py", "pi0252_ag_train.py", "pi0252_step_rng.py",
                  "pi0252_speed_margin.py", "pi0252_ag_train_h200.pbs")
    code_paths = {name: (args.pbs_script.resolve() if name.endswith(".pbs") else execution_root / name)
                  for name in code_names}
    code_paths.update({"pi0248_cr_train.py": Path(parent["paths"]["legacy_execution_driver_root"]) / "pi0248_cr_train.py",
                       "pi0248_rotation_cpu.py": Path(parent["paths"]["legacy_execution_driver_root"]) / "pi0248_rotation_cpu.py"})
    if any(not path.is_file() for path in code_paths.values()):
        raise RuntimeError("PI0252 sealed A/G execution sources are incomplete")
    code_hashes = {name: sha(path) for name, path in code_paths.items()}
    source = {key: copy.deepcopy(parent["source"][key]) for key in (
        "execution_code_sha256", "legacy_execution_code_commit", "legacy_execution_driver_root",
        "processor_bindings", "rotation_check_path", "rotation_check_sha256", "runtime_bindings",
        "p0_effective_sha256", "modality_config_sha256", "groot_source_commit",
    )}
    source.update({
        "execution_code_commit": args.execution_commit,
        "custom_execution_code_sha256": code_hashes,
        "initializer_view_receipt_path": str(view_path),
        "initializer_view_receipt_sha256": sha(view_path),
        "processor_bindings": processor,
        "c2000_parent_run_card_path": str(parent_path), "c2000_parent_run_card_sha256": parent_sha,
        "c2000_integrity_path": str(integrity_path), "c2000_integrity_sha256": sha(integrity_path),
        "c2000_durable_receipt_path": str(receipt_path), "c2000_durable_receipt_sha256": sha(receipt_path),
        "c2000_hf_readback_path": str(readback_path), "c2000_hf_readback_sha256": sha(readback_path),
        "step_key_table_path": str(key_path), "step_key_table_sha256": sha(key_path),
        "dt_table_path": str(dt_path), "dt_table_sha256": sha(dt_path),
        "timestamp_path": str(timestamp_path), "timestamp_sha256": timestamp_sha,
        "step_rng_module": "pi0252_step_rng.py", "step_rng_module_name": "pi0252_step_rng_runtime",
        "audit_prefix": "pi0252", "paired_arms": ["A", "G"], "reference_arm": "A",
        "intervention_arm": "G", "peer_arm": "A", "peer_integrity_status": "PASS_PI0252_A_TRAINING_INTEGRITY",
        "objective_kind": "peak_linear_speed",
        "g0_receipt_path": str(evidence_root / "PI0252_G0_READONLY_ATTRIBUTION_20260928.json"),
        "g0_receipt_sha256": sha(evidence_root / "PI0252_G0_READONLY_ATTRIBUTION_20260928.json"),
        "g1_cpu_selfcheck_path": str(evidence_root / "PI0252_AG_CPU_SELF_CHECK.json"),
        "g1_cpu_selfcheck_sha256": sha(evidence_root / "PI0252_AG_CPU_SELF_CHECK.json"),
        "pair_recipe_binding_path": str(task_root / "pi0252_ag_pair_recipe.json"),
        "pair_run_binding_path": str(task_root / "pi0252_ag_pair_binding.json"),
        "authorization": AUTH, "canonical_issue_comment": CANONICAL, "pointer_pr_comment": POINTER,
    })
    recipe = {
        "schema": "PI0252_AG_PAIR_RECIPE_V1", "status": "SEALED", "selected_checkpoint": "NONE",
        "authorization": AUTH, "canonical_issue_comment": CANONICAL, "pointer_pr_comment": POINTER,
        "start_revision": REVISION, "start_tree_sha256": TREE,
        "schedule_sha256": schedule_ref["sha256"], "step_key_table_sha256": sha(key_path),
        "dt_table_sha256": sha(dt_path), "runtime_id": "S4", "optimizer_steps_each": STEPS,
        "learning_rate": 1e-5, "checkpoint_steps": list(CHECKPOINTS), "new_adamw_each_arm": True,
        "pairing": ["source", "key", "model_input_sha256", "native_tau_identity", "native_epsilon", "stock_initial_noise"],
        "pairing_excludes": ["stock_vl_features", "model_output", "gradient"],
        "losses": {
            "A": {"family": "PI0252R1_M64_A_V1",
                  "loss": "L_native + 0.4*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4) + 0.1*ell_rot(a_free_S4)",
                  "coefficients": {"ell_pos": 0.4, "ell_delta": 0.1, "ell_rot": 0.1, "ell_speed_margin": 0.0}},
            "G": {"family": "PI0252R1_M64_G_V1", "loss": "J_A + 0.1*ell_speed_margin(a_free_S4)",
                  "coefficients": {"ell_pos": 0.4, "ell_delta": 0.1, "ell_rot": 0.1, "ell_speed_margin": 0.1}},
        },
    }
    recipe_path = Path(source["pair_recipe_binding_path"])
    write_once(recipe_path, recipe)
    source["pair_recipe_binding_sha256"] = sha(recipe_path)

    paths_base = copy.deepcopy(parent["paths"])
    paths_base.update({"output_base": str(task_root), "base_model_path": str(view_root),
                       "execution_driver_root": str(execution_root),
                       "execution_driver": str(execution_root / "pi0252_ag_train.py"),
                       "pbs_script": str(args.pbs_script.resolve()),
                       "initializer_view_receipt_path": str(view_path)})
    initializer = copy.deepcopy(parent["initializer"])
    initializer.update({"path": str(view_root), "screen_revision": REVISION,
                        "screen_manifest_sha256": sha(readback_path), "full_checkpoint_tree_sha256": TREE,
                        "full_checkpoint_bytes": int(inventory["total_bytes"]),
                        "durable_receipt_path": str(receipt_path), "durable_receipt_sha256": sha(receipt_path),
                        "model_files": model_files, "source_family": receipt["family"]})
    for key in ("x2000_durable_receipt_path", "x2000_durable_receipt_sha256"):
        initializer.pop(key, None)

    card_paths = {}
    for arm in ("A", "G"):
        doc = copy.deepcopy(parent)
        paths = {**paths_base, "output_root": str(task_root / arm)}
        config = copy.deepcopy(parent["resolved_config"])
        config["start_from_checkpoint"] = str(view_root)
        expected_loss = recipe["losses"][arm]["loss"]
        doc.update({
            "schema": "PI0252_AG_RUN_CARD_V1", "protocol": "PI0252", "training_manifest_family": "AG",
            "integrity_schema": "PI0252_AG_TRAINING_INTEGRITY_V1", "status": "SEALED_READY_NOT_STARTED",
            "authorization": AUTH, "execution_authorization": AUTH,
            "canonical_issue_comment": CANONICAL, "pointer_pr_comment": POINTER,
            "execution_update_commit": args.execution_commit,
            "family": recipe["losses"][arm]["family"], "loss": expected_loss,
            "loss_coefficients": recipe["losses"][arm]["coefficients"], "seed": 22301,
            "selected_checkpoint": "NONE", "d1_runtime_id": "S4",
            "stage": {"name": "M64_AG", "steps": STEPS,
                      "checkpoint_steps": list(CHECKPOINTS), "source_revision": REVISION},
            "initializer": copy.deepcopy(initializer), "paths": paths,
            "archive": {**copy.deepcopy(parent["archive"]),
                        "path_prefix": f"arms/PI0252/r1-speed-margin-20260928/{arm}",
                        "checkpoint_steps": list(CHECKPOINTS), "full_trainer_checkpoints": True},
            "source": copy.deepcopy(source), "schedule": copy.deepcopy(schedule_ref),
            "resolved_config": config,
            "loss_contract": {
                "objective": expected_loss,
                "generation": "complete pure-noise label-free S4 rollout before physical losses",
                "ell_pos": "whole-window physical XYZ mean squared Euclidean error / (0.05 m)^2",
                "ell_delta": "whole-window physical XYZ residual-increment mean square using fixed 0.05 s scale",
                "ell_rot": "PI0248 C_T physical rotation chordal objective, theta_ref=0.05 rad",
                "ell_speed_margin": "relu(max_j(sum_xyz(delta_xyz^2)/(2*actual_dt_j)^2)-0.81)",
                "ell_speed_margin_lambda": 0.1 if arm == "G" else 0.0,
                "actual_dt_table_path": str(dt_path), "actual_dt_table_sha256": sha(dt_path),
                "horizon": 40, "physical_dt_s": 0.05, "euler_steps": 4,
                "backpropagate_through_all_euler_steps": True,
                "expert_prefix_input": False, "active_action_smoothing": False,
                "position_units": "m", "rotation_theta_ref_rad": 0.05, "S4_full_gradient": True,
            },
            "optimizer": {**copy.deepcopy(parent["optimizer"]),
                          "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS),
                          "learning_rate": 1e-5, "warmup_steps": 10,
                          "optimizer_state": "fresh AdamW; no C optimizer resume"},
            "paired_arm_root": str(task_root / ("G" if arm == "A" else "A")),
            "hf_preflight": {"status": "PENDING_PER_ARM_PUBLIC_PREFIX_PREFLIGHT"},
        })
        for stale in ("d1_report_sha256", "d2_report_sha256", "input_identity_audit_sha256", "r2_authority"):
            doc.pop(stale, None)
        if arm == "G":
            doc["source"]["peer_run_card_path"] = str(task_root / "A" / "run_card.json")
            doc["source"]["peer_run_card_sha256"] = sha(card_paths["A"])
        card_path = task_root / arm / "run_card.json"
        card_paths[arm] = card_path
        write_once(card_path, doc)
        write_once(card_path.with_name("run_card.seal.json"), {"sha256": sha(card_path)})
        write_once(card_path.parent / "preparation.json", {
            "status": "READY_NOT_STARTED", "selected_checkpoint": "NONE", "run_card_sha256": sha(card_path),
            "config": config, "schedule_sha256": schedule_ref["sha256"],
            "initializer_revision": REVISION, "initializer_tree_sha256": TREE,
        })
    write_once(Path(source["pair_run_binding_path"]), {
        "schema": "PI0252_AG_PAIR_RUN_BINDING_V1", "status": "SEALED",
        "pair_recipe_binding_sha256": sha(recipe_path), "cards": {
            arm: {"path": str(card_paths[arm]), "sha256": sha(card_paths[arm])} for arm in ("A", "G")},
        "selected_checkpoint": "NONE",
    })
    write_once(task_root / "preparation.json", {
        "status": "READY_NOT_STARTED", "authorization": AUTH, "canonical_issue_comment": CANONICAL,
        "pointer_pr_comment": POINTER, "c2000_revision": REVISION, "c2000_tree_sha256": TREE,
        "schedule_sha256": schedule_ref["sha256"], "dt_table_sha256": sha(dt_path),
        "step_key_table_sha256": sha(key_path),
        "cards": {arm: {"path": str(card_paths[arm]), "sha256": sha(card_paths[arm])} for arm in ("A", "G")},
        "pair_recipe_sha256": sha(recipe_path), "pair_binding_sha256": sha(Path(source["pair_run_binding_path"])),
    })
    print(json.dumps(read(task_root / "preparation.json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
