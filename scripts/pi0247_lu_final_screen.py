#!/usr/bin/env python3
"""Final-2000 L/U screen using the sealed D2 inputs and original D2 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

S4 = {"id": "S4", "steps": 4, "inactive": "stock"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wanted_checkpoint_file(name: str) -> bool:
    return (name in {"config.json", "embodiment_id.json", "processor_config.json", "statistics.json",
                     "model.safetensors.index.json"} or name.endswith(".safetensors")
            or name.startswith("experiment_cfg/"))


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to replace conflicting evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verified_checkpoint(
    arm: str, step: int, run_card_path: Path, manifest_path: Path, checkpoint_path: Path,
    *, execution_state_path: Path | None = None, durable_receipt_path: Path | None = None,
    hf_readback_path: Path | None = None, materialization_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if step not in (1000, 2000):
        raise RuntimeError("only the protocol checkpoint-1000 observation and checkpoint-2000 decision are allowed")
    run_card_path, manifest_path = run_card_path.resolve(strict=True), manifest_path.resolve(strict=True)
    run_card = read_json(run_card_path)
    run_card_sha = sha(run_card_path)
    seal = read_json(run_card_path.with_name("run_card.seal.json"))
    if seal.get("sha256") != run_card_sha:
        raise RuntimeError("L/U run card seal mismatch")

    materialized = materialization_receipt_path is not None
    output = Path(run_card["paths"]["output_root"])
    manifest = read_json(manifest_path)
    state = read_json(execution_state_path) if materialized and execution_state_path else read_json(
        output / "execution_state.json"
    )
    checkpoint = checkpoint_path.resolve(strict=True)
    record = manifest.get("checkpoints", {}).get(str(step), {})
    recorded_checkpoint = Path(record.get("checkpoint_path", ""))
    checkpoint_matches = (
        recorded_checkpoint == output / "training" / f"checkpoint-{step}"
        if materialized else recorded_checkpoint.resolve() == checkpoint
    )
    if (
        run_card.get("schema") != "PI0247_LU_RUN_CARD_V1"
        or run_card.get("arm") != arm
        or manifest.get("status") != "PASS_TRAINING_INTEGRITY"
        or manifest.get("arm") != arm
        or manifest.get("global_step") != 2000
        or manifest.get("selected_checkpoint") != "NONE"
        or manifest.get("run_card_sha256") != run_card_sha
        or state.get("status") != "TRAINING_COMPLETE"
        or state.get("training_manifest_sha256") != sha(manifest_path)
        or record.get("step") != step
        or record.get("family") != run_card.get("family")
        or not checkpoint_matches
    ):
        raise RuntimeError(f"checkpoint-{step} is not bound to a valid, unselected L/U training manifest")

    if materialized:
        if not all((execution_state_path, durable_receipt_path, hf_readback_path)):
            raise RuntimeError("HF materialization requires its closed training evidence package")
        persistent_receipt, hf_readback = durable_receipt_path, hf_readback_path
    else:
        archive_root = checkpoint.parent.parent
        persistent_receipt = archive_root / "persistent_receipts" / f"checkpoint-{step}.durable_receipt.json"
        hf_readback = archive_root / "hf_readbacks" / f"checkpoint-{step}.json"
    if (sha(persistent_receipt) != record.get("durable_receipt_sha256")
            or sha(hf_readback) != record.get("full_hf_readback_sha256")):
        raise RuntimeError(f"checkpoint-{step} durable receipt or HF readback changed after finalization")
    receipt = read_json(persistent_receipt)
    inventory = receipt.get("source_record", {}).get("inventory", {})
    if (receipt.get("status") != "DURABLE_VERIFIED"
            or receipt.get("family") != run_card["family"]
            or receipt.get("checkpoint_step") != step
            or inventory.get("tree_sha256") != record.get("tree_sha256")
            or inventory.get("file_count") != record.get("file_count")
            or inventory.get("total_bytes") != record.get("total_bytes")):
        raise RuntimeError(f"checkpoint-{step} persistent receipt does not match the training manifest")
    if materialized:
        readback = read_json(hf_readback)
        materialization = read_json(materialization_receipt_path)
        expected_files = [
            {key: row[key] for key in ("path", "bytes", "sha256")}
            for row in inventory.get("entries", []) if wanted_checkpoint_file(row["path"])
        ]
        if (
            readback.get("status") != "PASS_PT_FULL_HF_READBACK"
            or readback.get("pi0247_status") != "PASS_PI0247_LU_FULL_HF_READBACK"
            or readback.get("revision") != record.get("hf_manifest_commit_id")
            or readback.get("source_tree_sha256") != record.get("tree_sha256")
            or readback.get("repo_id") != run_card.get("archive", {}).get("repo_id")
            or readback.get("path_prefix") != run_card.get("archive", {}).get("path_prefix")
            or materialization.get("schema") != "PI0247_LU_HF_INFERENCE_MATERIALIZATION_V1"
            or materialization.get("arm") != arm
            or materialization.get("checkpoint_step") != step
            or materialization.get("repo_id") != readback.get("repo_id")
            or materialization.get("path_prefix") != readback.get("path_prefix")
            or materialization.get("revision") != readback.get("revision")
            or materialization.get("source_checkpoint_tree_sha256") != record.get("tree_sha256")
            or materialization.get("run_card_sha256") != run_card_sha
            or materialization.get("training_manifest_sha256") != sha(manifest_path)
            or materialization.get("durable_receipt_sha256") != sha(persistent_receipt)
            or materialization.get("hf_readback_sha256") != sha(hf_readback)
            or materialization.get("execution_state_sha256") != sha(execution_state_path)
            or materialization.get("files") != expected_files
            or sha(hf_readback) != record.get("full_hf_readback_sha256")
        ):
            raise RuntimeError(f"checkpoint-{step} HF materialization does not bind to the closed training receipt")
        if manifest.get("execution_code_commit") != run_card.get("source", {}).get("execution_code_commit"):
            raise RuntimeError("detached inference lineage disagrees with the sealed training source commit")
        for row in expected_files:
            path = checkpoint / row["path"]
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise RuntimeError(f"materialized HF inference file differs from the sealed full readback: {path}")
    return run_card, manifest, receipt


def inference_view(
    step: int, checkpoint: Path, manifest_path: Path, run_card_path: Path,
    receipt: dict[str, Any], model_root: Path, p1: Any,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    model_root = model_root.resolve()
    source_entries = receipt["source_record"]["inventory"]["entries"]
    entries = [row for row in source_entries if wanted_checkpoint_file(row["path"])]
    names = {row["path"] for row in entries}
    if not {"config.json", "model.safetensors.index.json"} <= names or sum(
        name.endswith(".safetensors") for name in names
    ) < 2:
        raise RuntimeError(f"checkpoint-{step} receipt lacks the complete inference weight set")

    if model_root.exists():
        restore = read_json(model_root / "pi0245_inference_restore.json")
        if (restore.get("checkpoint_step") != step
                or restore.get("checkpoint_tree_sha256") != receipt["source_record"]["inventory"]["tree_sha256"]
                or restore.get("source_training_manifest_sha256") != sha(manifest_path)
                or restore.get("source_run_card_sha256") != sha(run_card_path)):
            raise RuntimeError("existing inference view belongs to a different checkpoint or training lineage")
        p1.verify_model_files(model_root)
        return restore

    partial = model_root.with_name(model_root.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserving incomplete inference view for inspection: {partial}")
    partial.mkdir(parents=True)
    try:
        files = []
        for row in entries:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe checkpoint path: {relative}")
            source = (checkpoint / relative).resolve(strict=True)
            if not source.is_relative_to(checkpoint) or not source.is_file() or sha(source) != row["sha256"]:
                raise RuntimeError(f"checkpoint inference file changed: {relative}")
            target = partial / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            files.append({"path": relative.as_posix(), "bytes": row["bytes"], "sha256": row["sha256"]})

        run_card = read_json(run_card_path)
        restore = {
            "schema": "PI0247_LU_CHECKPOINT_INFERENCE_VIEW_V1",
            "checkpoint_step": step, "family": run_card["family"],
            "prefix": run_card["archive"]["path_prefix"] + f"/checkpoint-{step}",
            "revision": "local-tree-sha256:" + receipt["source_record"]["inventory"]["tree_sha256"],
            "source_training_manifest_sha256": sha(manifest_path),
            "source_run_card_sha256": sha(run_card_path),
            "checkpoint_tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"],
            "optimizer_restored": False, "files": files,
        }
        write_immutable(partial / "pi0245_inference_restore.json", restore)
        p1.verify_model_files(partial)
        os.replace(partial, model_root)
        return restore
    except Exception:
        # Keep any partial view as evidence; a rerun must not silently replace it.
        raise


def make_noise_identity(context: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    audit = context["d2_audit_rows"]
    noise = context["noise"]
    n1_sha = hashlib.sha256(noise["N1"].tobytes()).hexdigest()
    identity: dict[str, Any] = {
        "B0": {},
        "B1": {"shared_across_sources": True, "source_tensor_sha256": n1_sha,
               "file_sha256": context["facts"]["noise_files_sha256"]["N1"]},
    }
    for source in context["rows"]:
        source_id = source["source_episode_id"]
        item = audit[source_id]
        identity["B0"][source_id] = {
            "source_tensor_sha256": item["b0_source_tensor_sha256"],
            "noise_packet_sha256": item["s0_noise_packet_sha256"],
            "per_source_packet": True,
        }
    initials: dict[str, dict[str, str]] = {"B0": {}, "B1": {}}
    for row in rows:
        name = "invalid_output.npz" if row.get("metrics") is None else "tensors.npz"
        with np.load(Path(row["source_cell"]) / name, allow_pickle=False) as data:
            initials[row["noise_name"]][row["source_episode_id"]] = hashlib.sha256(
                data["noise_initial"].tobytes()
            ).hexdigest()
    for source, item in identity["B0"].items():
        item["initial_bfloat16_sha256"] = initials["B0"][source]
    b1 = set(initials["B1"].values())
    if len(b1) != 1 or len(initials["B1"]) != 16:
        raise RuntimeError("B1 final-screen initial noise is not the sealed shared tensor")
    identity["B1"]["initial_bfloat16_sha256"] = next(iter(b1))
    return identity


def describe_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize checkpoint-1000 observations without applying the final gate."""
    import numpy as np

    valid = [row for row in rows if row.get("metrics") and row.get("status") == "VALID"]
    xyz = np.asarray([row["metrics"]["xyz_rmse_m"] for row in valid], dtype=np.float64)
    so3 = np.asarray([row["metrics"]["so3_rmse_rad"] for row in valid], dtype=np.float64)
    return {
        "cell_count": len(rows), "valid_count": len(valid), "invalid_count": len(rows) - len(valid),
        "gate_counts": {decision: sum(row.get("gate_decision") == decision for row in rows)
                        for decision in ("PASS", "WARN", "REJECT", "UNSCORABLE")},
        "median_xyz_rmse_m": float(np.median(xyz)) if xyz.size else None,
        "p95_xyz_rmse_m": float(np.quantile(xyz, .95, method="linear")) if xyz.size else None,
        "median_so3_rmse_rad": float(np.median(so3)) if so3.size else None,
        "quality_gate_applied": False,
    }


def build_screen(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pi0247_d1_screen as d1
    import pi0247_d2_full16_screen as d2

    run_card, train_manifest, receipt = verified_checkpoint(
        args.arm, args.step, args.run_card, args.training_manifest, args.checkpoint,
        execution_state_path=args.execution_state, durable_receipt_path=args.durable_receipt,
        hf_readback_path=args.hf_readback, materialization_receipt_path=args.materialization_receipt,
    )
    d2_card, d2_card_sha = d2.read_card(args.d2_card, args.d2_card_sha)
    facts, context = d1.prepare(Path(d2_card["paths"]["stage_root"]), Path(d2_card["paths"]["code_root"]))
    context["facts"] = facts
    if d1.canonical_sha(facts) != d2_card["static_preflight_canonical_sha256"]:
        raise RuntimeError("D2 inputs differ from the sealed static preflight")
    d2.prepare_inputs(d2_card, context)
    if len(context["rows"]) != 16:
        raise RuntimeError("final L/U screen no longer has the original 16 source windows")

    d2_report_path = Path(d2_card["paths"]["output_root"]) / "pi0247_d2_full16_report.json"
    d2_report = read_json(d2_report_path)
    d2_report_sha = sha(d2_report_path)
    if (d2_report.get("status") != "D2_VALID_FAIL_LU_REQUIRED"
            or d2_report.get("r_star_frozen") != "S4"
            or run_card.get("d2_report_sha256") != d2_report_sha
            or run_card.get("input_identity_audit_sha256") != d2_card.get("input_identity_audit_sha256")
            or run_card.get("d1_report_sha256") != d2_card.get("d1_report_sha256")
            or run_card.get("initializer", {}).get("screen_manifest_sha256")
               != d2_card.get("weights", [{}])[0].get("manifest_sha256")):
        raise RuntimeError("D2 gate/report binding does not authorize this final L/U screen")

    if not args.materialization_receipt:
        lu_module_path = Path(run_card["paths"]["repo_code_root"]) / "scripts/pi0247_lu_train.py"
        sys.path.insert(0, str(lu_module_path.parent))
        import pi0247_lu_train as lu
        lu._verify_card(args.run_card, args.arm)
        lu._verify_code(run_card, Path(run_card["paths"]["repo_code_root"]),
                        Path(run_card["paths"]["code_root"]), Path(run_card["paths"]["training_runtime_root"]))
        lu._verify_runtime_bindings(run_card["paths"], run_card["source"]["runtime_bindings"])

    ct_paths = context["paths"]
    modality_sha = d1.sha(Path(ct_paths["modality_config"]))
    processor_config_sha = d1.sha(Path(ct_paths["processor_base_model"]) / "processor_config.json")
    base_statistics_sha = d1.sha(Path(ct_paths["processor_base_model"]) / "statistics.json")
    flat_stats_sha = d1.sha(Path(ct_paths["train_dataset"]) / "meta/stats.json")
    relative_stats_sha = d1.sha(Path(ct_paths["train_dataset"]) / "meta/relative_stats.json")
    ct_binding_path = Path(d2_card["paths"]["stage_root"]) / "p1_corelab/runtime_binding.json"
    ct_binding = read_json(ct_binding_path)
    t_parent = {row["path"]: row for row in context["cards"]["T"]["parent"]["files"]}
    # ponytail: stage21 overlays NEW_EMBODIMENT from C_T; fingerprint the base statistics without using them for this embodiment.
    if (modality_sha != run_card["source"]["modality_config_sha256"]
            or ct_binding.get("p0_effective_processor_xyz_sha256") != run_card["source"]["p0_effective_sha256"]
            or ct_binding.get("modality_config_sha256") != modality_sha
            or ct_binding.get("relative_stats_sha256") != relative_stats_sha
            or t_parent.get("processor_config.json", {}).get("sha256") != processor_config_sha
            or ct_binding.get("processor_class")
               != f"{type(context['processor']).__module__}.{type(context['processor']).__qualname__}"):
        raise RuntimeError("the frozen C_T processor or statistics differ from the L training binding")

    args.output_root.mkdir(parents=True, exist_ok=True)
    model_root = args.output_root / f"inference_view_step_{args.step}"
    restore = inference_view(args.step, args.checkpoint, args.training_manifest, args.run_card,
                             receipt, model_root, context["p1"])
    model_files = context["p1"].verify_model_files(model_root)
    from transformers import AutoModel
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    import torch

    score_card = {
        "schema": "PI0247_LU_FINAL_SCREEN_CARD_V1", "arm": args.arm,
        "checkpoint_step": args.step, "runtime": S4,
        "family": run_card["family"], "loss_contract": run_card["loss_contract"],
        "loss_coefficients": run_card["loss_coefficients"],
        "initializer": run_card["initializer"],
        "checkpoint_binding": train_manifest["checkpoints"][str(args.step)],
        "archive": {key: run_card["archive"][key] for key in ("repo_id", "path_prefix", "public")},
        "physical_contract": run_card["physical_contract"],
        "processor_identity": {
            "modality_config_sha256": modality_sha,
            "p0_effective_sha256": run_card["source"]["p0_effective_sha256"],
            "frozen_C_T": {
                "T_run_card_sha256": d1.sha(Path(d2_card["paths"]["stage_root"]) / "screen_inputs/T_run_card.json"),
                "p1_closure_sha256": facts["p1_closure_sha256"],
                "runtime_binding_sha256": d1.sha(ct_binding_path),
                "processor_config_sha256": processor_config_sha,
                "model_base_statistics_sha256": base_statistics_sha,
                "T_card_base_statistics_sha256": t_parent.get("statistics.json", {}).get("sha256"),
                "model_base_statistics_differs_from_T_parent": (
                    base_statistics_sha != t_parent.get("statistics.json", {}).get("sha256")
                ),
                "NEW_EMBODIMENT_statistics_source": "C_T train_dataset/meta/stats.json",
                "NEW_EMBODIMENT_stats_sha256": flat_stats_sha,
                "relative_stats_sha256": relative_stats_sha,
                "processor_class": ct_binding["processor_class"],
            },
        },
        "run_card_sha256": sha(args.run_card),
        "training_manifest_sha256": sha(args.training_manifest),
        "checkpoint_tree_sha256": restore["checkpoint_tree_sha256"],
        "hf_revision": train_manifest["checkpoints"][str(args.step)]["hf_manifest_commit_id"],
        "hf_materialization_receipt_sha256": sha(args.materialization_receipt)
            if args.materialization_receipt else None,
        "inference_restore_sha256": model_files["manifest_sha256"],
        "d2_run_card_sha256": d2_card_sha,
        "d1_run_card_sha256": d2_card["d1_run_card_sha256"],
        "d1_report_sha256": d2_card["d1_report_sha256"],
        "d2_report_sha256": d2_report_sha,
        "input_identity_audit_sha256": d2_card["input_identity_audit_sha256"],
        "source_episode_ids": [row["source_episode_id"] for row in context["rows"]],
        "noise_identity_source": d2_report["noise_identity"],
        "d1_runtime_id": "S4", "decoder_and_gate": "frozen PI0247 D1 runner",
        "resource": {"device": d2_card["resource"]["device"]},
        "output_root": str(args.output_root.resolve()),
        "runner_sha256": sha(Path(__file__).resolve()),
    }
    score_card_path = args.output_root / "score_card.json"
    write_immutable(score_card_path, score_card)
    card_sha = sha(score_card_path)
    write_immutable(args.output_root / "score_card.sha256.json", {"sha256": card_sha})

    model = AutoModel.from_pretrained(model_root, local_files_only=True, trust_remote_code=True)
    model.eval().to(device=d2_card["resource"]["device"], dtype=torch.bfloat16)
    if int(model.action_head.num_inference_timesteps) != 4:
        raise RuntimeError("L/U final screen model is not the frozen four-step S4 runtime")

    all_rows = []
    gate = TrajectorySafetyGateV1()
    for source_row in context["rows"]:
        source = source_row["source_episode_id"]
        for noise_name in ("B0", "B1"):
            if noise_name == "B0":
                noise_path = Path(context["d2_noise_packet_paths"][source])
                context["noise"][noise_name] = context["d2_noise"][source][noise_name]
            else:
                noise_path = Path(context["noise_root"]) / "N1.npz"
                context["noise"][noise_name] = context["d2_noise"][source][noise_name]
            context["facts"]["noise_files_sha256"][noise_name] = d1.sha(noise_path)
            cell = args.output_root / "cells" / f"{args.arm}{args.step}__S4__{source.split(':', 1)[0]}__{noise_name}"
            record = d1.run_one(
                args=args, card=score_card, card_sha=card_sha, context=context,
                model_name=f"{args.arm}{args.step}", runtime=S4, source=source, noise_name=noise_name,
                model=model, model_files=model_files, bridge=bridge, selector=selector,
                stage43=stage43, gate=gate,
            )
            tensor_name = "invalid_output.npz" if record.get("metrics") is None else "tensors.npz"
            with np.load(cell / tensor_name, allow_pickle=False) as data:
                clipped = int(np.count_nonzero(np.abs(data["raw_native"][..., :3]) > 1.0))
                head_evaluations = int(data["trace_x"].shape[0])
            all_rows.append({
                **record, "source_cell": str(cell), "noise_name": noise_name,
                "raw_xyz_clipped_scalars": clipped,
                "head_denoising_forwards": head_evaluations,
                "gate_issues": record.get("gate", {}).get("issues", []),
                "tail35_39_pooled_sse_fraction": record.get("metrics", {}).get(
                    "tail35_39_pooled_sse_fraction"),
            })

    if len(all_rows) != 32 or len({(row["source_episode_id"], row["noise_name"]) for row in all_rows}) != 32:
        raise RuntimeError("final-only screen does not close the original 16 × B0/B1 matrix")
    noise_identity = make_noise_identity(context, all_rows)
    groups = {name: describe_group([row for row in all_rows if row["noise_name"] == name])
              for name in ("B0", "B1")}
    if args.step == 2000:
        groups = {name: d2.evaluate_group([row for row in all_rows if row["noise_name"] == name])
                  for name in ("B0", "B1")}
        invalid = any(row.get("status") in d2.INVALID for row in all_rows)
        status = "INVALID_SCREEN" if invalid else (
            "PASS_BOTH_NOISE_GROUPS" if all(item["quality_pass"] for item in groups.values()) else "VALID_FAIL"
        )
    else:
        status = "OBSERVATION_ONLY"
    report = {
        "schema": "PI0247_LU_FINAL_SCREEN_REPORT_V1", "status": status,
        "arm": args.arm, "checkpoint_step": args.step, "runtime_id": "S4",
        "family": run_card["family"], "loss_contract": run_card["loss_contract"],
        "loss_coefficients": run_card["loss_coefficients"],
        "initializer": run_card["initializer"],
        "checkpoint_binding": score_card["checkpoint_binding"],
        "archive": score_card["archive"], "physical_contract": score_card["physical_contract"],
        "processor_identity": score_card["processor_identity"],
        "run_card_sha256": score_card["run_card_sha256"],
        "training_manifest_sha256": score_card["training_manifest_sha256"],
        "checkpoint_tree_sha256": score_card["checkpoint_tree_sha256"],
        "hf_revision": score_card["hf_revision"],
        "hf_materialization_receipt_sha256": score_card["hf_materialization_receipt_sha256"],
        "score_card_sha256": card_sha, "d2_report_sha256": d2_report_sha,
        "source_episode_ids": score_card["source_episode_ids"],
        "noise_identity": noise_identity, "thresholds_per_group": d2_report["thresholds_per_group"],
        "counts": {"logical_cells": 32, "new_queries": 32, "model_queries": 32,
                   "backbone_forwards": 32,
                   "head_denoising_evaluations": sum(int(row["head_denoising_forwards"]) for row in all_rows)},
        "group_results": groups, "cells": all_rows,
        "adjudication_checkpoint": args.step == 2000,
        "selected_checkpoint": "NONE",
    }
    write_immutable(args.output_root / "final_screen_report.json", report)
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"status": status, "arm": args.arm, "groups": groups,
                      "report": str(args.output_root / "final_screen_report.json")}, sort_keys=True), flush=True)
    return report


def choose_fit_recipe(results: dict[str, dict[str, Any]]) -> str | None:
    if any(results[arm]["status"] not in {"PASS_BOTH_NOISE_GROUPS", "VALID_FAIL"} for arm in ("L", "U")):
        return None
    eligible = [arm for arm in ("L", "U") if results[arm]["status"] == "PASS_BOTH_NOISE_GROUPS"]
    if not eligible:
        return None
    return min(eligible, key=lambda arm: (
        results[arm]["group_results"]["B0"]["p95_xyz_rmse_m"],
        results[arm]["group_results"]["B1"]["p95_xyz_rmse_m"],
        results[arm]["group_results"]["B0"]["median_xyz_rmse_m"],
        arm != "L",
    ))


def compare(args: argparse.Namespace) -> dict[str, Any]:
    d2_report = read_json(args.d2_report)
    reports = {
        ("L", 1000): read_json(args.l1000_report),
        ("L", 2000): read_json(args.l2000_report),
        ("U", 1000): read_json(args.u1000_report),
        ("U", 2000): read_json(args.u2000_report),
    }
    arm_reports = {arm: reports[(arm, 2000)] for arm in ("L", "U")}
    if d2_report.get("status") != "D2_VALID_FAIL_LU_REQUIRED" or d2_report.get("r_star_frozen") != "S4":
        raise RuntimeError("comparison lacks the frozen D2-valid-fail route")
    source_ids = d2_report["source_episode_ids"]
    baseline = {(row["source_episode_id"], row["noise_name"]): row for row in d2_report["cells"]
                if row.get("model_name") == "P2000" and row.get("runtime") == "S4"}
    indexed: dict[tuple[str, int], dict[tuple[str, str], dict[str, Any]]] = {}
    for (arm, step), report in reports.items():
        if (report.get("schema") != "PI0247_LU_FINAL_SCREEN_REPORT_V1"
                or report.get("arm") != arm or report.get("checkpoint_step") != step
                or report.get("runtime_id") != "S4" or report.get("source_episode_ids") != source_ids
                or report.get("d2_report_sha256") != sha(args.d2_report)
                or report.get("noise_identity") != d2_report.get("noise_identity")):
            raise RuntimeError(f"{arm} final screen differs from the sealed D2 source/noise identities")
        if (step == 1000 and (report.get("status") != "OBSERVATION_ONLY"
                              or report.get("adjudication_checkpoint") is not False
                              or any(row.get("quality_gate_applied") is not False
                                     for row in report.get("group_results", {}).values()))):
            raise RuntimeError("checkpoint-1000 observations were incorrectly used as a quality gate")
        if (step == 2000 and report.get("status") not in
                {"PASS_BOTH_NOISE_GROUPS", "VALID_FAIL", "INVALID_SCREEN"}):
            raise RuntimeError(f"{arm} final-2000 screen is incomplete")
        rows = {(row["source_episode_id"], row["noise_name"]): row for row in report["cells"]}
        if len(rows) != 32 or set(rows) != set(baseline):
            raise RuntimeError(f"{arm} final screen is not the exact 16 × two-noise D2 matrix")
        indexed[(arm, step)] = rows
    for arm in ("L", "U"):
        low, final = reports[(arm, 1000)], reports[(arm, 2000)]
        if (low.get("training_manifest_sha256") != final.get("training_manifest_sha256")
                or low.get("run_card_sha256") != final.get("run_card_sha256")):
            raise RuntimeError(f"{arm} checkpoint screens do not share one sealed training lineage")

    paired = []
    for source in source_ids:
        for noise in ("B0", "B1"):
            key = (source, noise)
            p, l, u = baseline[key], indexed[("L", 2000)][key], indexed[("U", 2000)][key]
            pm, lm, um = p.get("metrics"), l.get("metrics"), u.get("metrics")
            def delta(left: dict[str, Any] | None, right: dict[str, Any] | None, metric: str) -> float | None:
                return float(left[metric] - right[metric]) if left and right else None
            paired.append({
                "source_episode_id": source, "noise_name": noise,
                "P2000": {"metrics": pm, "gate_decision": p["gate_decision"]},
                "L2000": {"metrics": lm, "gate_decision": l["gate_decision"]},
                "U2000": {"metrics": um, "gate_decision": u["gate_decision"]},
                "delta_xyz_L_minus_P_m": delta(lm, pm, "xyz_rmse_m"),
                "delta_xyz_U_minus_P_m": delta(um, pm, "xyz_rmse_m"),
                "delta_xyz_U_minus_L_m": delta(um, lm, "xyz_rmse_m"),
                "delta_so3_L_minus_P_rad": delta(lm, pm, "so3_rmse_rad"),
                "delta_so3_U_minus_P_rad": delta(um, pm, "so3_rmse_rad"),
                "delta_so3_U_minus_L_rad": delta(um, lm, "so3_rmse_rad"),
            })
    if len(baseline) != 32:
        raise RuntimeError("D2 P-final baseline does not contain all 32 reusable cells")

    winner = choose_fit_recipe(arm_reports)
    both_valid_fail = all(report.get("status") == "VALID_FAIL" for report in arm_reports.values())
    if winner:
        route = "AUTO_M64"
        status = "PI0247_LU_FIT_RECIPE_QUALIFIED"
    elif both_valid_fail:
        route = "STOP_NO_M64"
        status = "PI0247_M16_HYPOTHESIS_SET_EXHAUSTED"
    else:
        route = "RESOLVE_TYPED_SCREEN_FAILURE"
        status = "PI0247_LU_SCREEN_INVALID_OR_INCOMPLETE"

    comparison = {
        "schema": "PI0247_LU_FINAL_COMPARISON_V1", "status": status,
        "route": route, "winner_arm": winner,
        "d2_report_sha256": sha(args.d2_report),
        "report_sha256": {f"{arm}{step}": sha(path) for (arm, step), path in {
            ("L", 1000): args.l1000_report, ("L", 2000): args.l2000_report,
            ("U", 1000): args.u1000_report, ("U", 2000): args.u2000_report,
        }.items()},
        "runtime_id": "S4", "source_episode_ids": source_ids,
        "counts": {"d2_reused_cells": 32, "new_lu_cells": 128, "total_logical_cells": 160,
                   "head_denoising_evaluations": sum(
                       int(report["counts"]["head_denoising_evaluations"]) for report in reports.values())},
        "checkpoint1000_observations": {arm: reports[(arm, 1000)]["group_results"] for arm in ("L", "U")},
        "final2000_group_results": {arm: report["group_results"] for arm, report in arm_reports.items()},
        "paired_source_noise_comparisons": paired,
        "selected_checkpoint": "NONE",
    }
    write_immutable(args.comparison_output, comparison)
    if winner:
        winning_report = arm_reports[winner]
        write_immutable(args.comparison_output.with_name("fit_recipe_manifest.json"), {
            "schema": "PI0247_FIT_RECIPE_MANIFEST_V1", "status": "PASS_LU_FIT_RECIPE",
            "winner_arm": winner, "family": winning_report["family"],
            "checkpoint_step": 2000, "checkpoint_tree_sha256": winning_report["checkpoint_tree_sha256"],
            "training_manifest_sha256": winning_report["training_manifest_sha256"],
            "checkpoint_binding": winning_report["checkpoint_binding"],
            "archive": winning_report["archive"], "physical_contract": winning_report["physical_contract"],
            "loss_contract": winning_report["loss_contract"],
            "loss_coefficients": winning_report["loss_coefficients"],
            "initializer": winning_report["initializer"],
            "processor_identity": winning_report["processor_identity"],
            "runtime_id": "S4", "noise_identity": winning_report["noise_identity"],
            "final_screen_report_sha256": sha(args.l2000_report if winner == "L" else args.u2000_report),
            "selected_checkpoint": "NONE",
        })
    print(json.dumps(comparison, sort_keys=True), flush=True)
    return comparison


def self_check() -> None:
    def passed(p95_b0: float, p95_b1: float, median_b0: float) -> dict[str, Any]:
        return {"status": "PASS_BOTH_NOISE_GROUPS", "group_results": {
            "B0": {"p95_xyz_rmse_m": p95_b0, "median_xyz_rmse_m": median_b0},
            "B1": {"p95_xyz_rmse_m": p95_b1},
        }}
    assert choose_fit_recipe({"L": passed(.030, .034, .018), "U": passed(.029, .034, .017)}) == "U"
    assert choose_fit_recipe({"L": passed(.030, .034, .018), "U": {"status": "VALID_FAIL"}}) == "L"
    assert choose_fit_recipe({"L": {"status": "VALID_FAIL"}, "U": {"status": "VALID_FAIL"}}) is None
    assert choose_fit_recipe({"L": passed(.030, .034, .018), "U": {"status": "INVALID_SCREEN"}}) is None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--arm", choices=("L", "U"))
    parser.add_argument("--step", type=int, choices=(1000, 2000))
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--execution-state", type=Path)
    parser.add_argument("--durable-receipt", type=Path)
    parser.add_argument("--hf-readback", type=Path)
    parser.add_argument("--materialization-receipt", type=Path)
    parser.add_argument("--d2-card", type=Path)
    parser.add_argument("--d2-card-sha", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--d2-report", type=Path)
    parser.add_argument("--l1000-report", type=Path)
    parser.add_argument("--l2000-report", type=Path)
    parser.add_argument("--u1000-report", type=Path)
    parser.add_argument("--u2000-report", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("PI0247 L/U final-only screen self-check: PASS")
    elif args.compare:
        assert all((args.d2_report, args.l1000_report, args.l2000_report,
                    args.u1000_report, args.u2000_report, args.comparison_output))
        compare(args)
    else:
        required = (args.arm, args.step, args.run_card, args.training_manifest, args.checkpoint,
                    args.d2_card, args.d2_card_sha, args.output_root)
        assert all(required), "L/U checkpoint screen arguments are required"
        build_screen(args)


if __name__ == "__main__":
    main()
