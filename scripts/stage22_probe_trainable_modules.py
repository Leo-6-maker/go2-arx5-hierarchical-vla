#!/usr/bin/env python3
"""Probe Stage22 GR00T trainability without training or inference.

The probe reconstructs the model through the frozen Stage20S pipeline and
records the live ``Parameter.requires_grad`` state.  It performs no forward
pass, backward pass, optimizer step, checkpoint write, WBC execution, or
trajectory evaluation.  The JSON output is an evidence artifact for the
direct-launch H2 auditor and must be hashed in the H2 provenance manifest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


FROZEN_LAUNCHER_GIT_SHA = "deb920a070141e6a32a88cd0da6719828647e1e6"
FROZEN_LAUNCHER_FILE_SHA256 = "f42a1990847cf8f5326c66db37043ffed6e43458dd907fdbd08da5dd905b00a2"
EXPECTED_MODULE_ROLES = ("projector", "state", "action", "diffusion", "visual", "llm", "vlln")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _same_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).resolve(strict=True) == expected.resolve(strict=True)
    except OSError:
        return False


def _validate_run_manifest(
    path: Path,
    *,
    recipe: str,
    base_model_path: Path,
    dataset_dir: Path,
    modality_config: Path,
) -> dict[str, Any]:
    """Validate the run identity before attaching the probe to that run."""

    payload = _load_json(path.resolve(strict=True))
    expected_diffusion = recipe == "R1"
    expected_index_sha256 = _sha256_file(base_model_path / "model.safetensors.index.json")
    checks = (
        payload.get("schema") == "STAGE20S_GR00T_TRAINING_RUN_V1",
        payload.get("status") == "PASS",
        payload.get("simulation_only") is True,
        payload.get("hardware_control") is False,
        payload.get("max_steps") == 50,
        payload.get("action_horizon") == 40,
        payload.get("action_sample_period_s") == 0.05,
        payload.get("action_sample_period_provenance") == "TEST_ONLY_SIM_VISUAL_REACH_V0",
        payload.get("tune_projector") is True,
        payload.get("tune_diffusion_model") is expected_diffusion,
        payload.get("tune_visual") is False,
        payload.get("tune_llm") is False,
        payload.get("tune_vlln") is False,
        _same_path(payload.get("base_model_path"), base_model_path),
        payload.get("base_model_index_sha256") == expected_index_sha256,
        _same_path(payload.get("dataset_dir"), dataset_dir),
        _same_path(payload.get("modality_config"), modality_config),
    )
    if not all(checks):
        raise ValueError(f"run manifest does not match frozen {recipe} probe contract: {path}")
    return payload


def _digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def _load_launcher(path: Path):
    spec = importlib.util.spec_from_file_location("stage20s_frozen_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen launcher module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _role_components(model: Any) -> dict[str, list[tuple[str, Any]]]:
    action_head = model.action_head
    backbone_model = getattr(model.backbone, "model", None)
    if backbone_model is None:
        raise RuntimeError("GR00T backbone does not expose the expected model module")
    return {
        "projector": [("action_head.position_embedding", getattr(action_head, "position_embedding", None))],
        "state": [("action_head.state_encoder", action_head.state_encoder)],
        "action": [
            ("action_head.action_encoder", action_head.action_encoder),
            ("action_head.action_decoder", action_head.action_decoder),
        ],
        "diffusion": [("action_head.model", action_head.model)],
        "visual": [("backbone.model.visual", backbone_model.visual)],
        "llm": [("backbone.model.language_model", backbone_model.language_model)],
        "vlln": [
            ("action_head.vlln", action_head.vlln),
            ("action_head.vl_self_attention", action_head.vl_self_attention),
        ],
    }


def _component_parameters(components: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for prefix, module in components:
        if module is None:
            continue
        for local_name, parameter in module.named_parameters(recurse=True):
            full_name = f"{prefix}.{local_name}" if local_name else prefix
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            result.append((full_name, parameter))
    return result


def _inventory(model: Any, *, recipe: str, base_model_path: Path, launcher_path: Path) -> dict[str, Any]:
    all_parameters = list(model.named_parameters())
    all_by_id = {id(parameter): (name, parameter) for name, parameter in all_parameters}
    components = _role_components(model)
    rows: list[dict[str, Any]] = []
    claimed: dict[int, str] = {}
    duplicate_claims: list[str] = []
    for role in EXPECTED_MODULE_ROLES:
        parameters = _component_parameters(components[role])
        overlap = [name for _, parameter in parameters if id(parameter) in claimed]
        duplicate_claims.extend(f"{role}:{name}" for name, parameter in parameters if id(parameter) in claimed)
        for name, parameter in parameters:
            claimed.setdefault(id(parameter), role)
        sorted_names = sorted(name for name, _ in parameters)
        parameter_lines = sorted(
            f"{name}\t{parameter.numel()}\t{int(bool(parameter.requires_grad))}"
            for name, parameter in parameters
        )
        parameter_count = sum(int(parameter.numel()) for _, parameter in parameters)
        requires_grad_count = sum(
            int(parameter.numel()) for _, parameter in parameters if parameter.requires_grad
        )
        rows.append(
            {
                "role": role,
                "name": "+".join(prefix for prefix, module in components[role] if module is not None),
                "parameter_count": parameter_count,
                "requires_grad_parameter_count": requires_grad_count,
                "trainable_parameter_count": requires_grad_count,
                "all_requires_grad": bool(parameters) and all(
                    bool(parameter.requires_grad) for _, parameter in parameters
                ),
                "parameter_name_count": len(sorted_names),
                "parameter_names_sha256": _digest_lines(sorted_names),
                "requires_grad_state_sha256": _digest_lines(parameter_lines),
                "overlap_parameter_names": overlap,
            }
        )
    unassigned = [name for identity, (name, _) in all_by_id.items() if identity not in claimed]
    global_lines = sorted(
        f"{name}\t{parameter.numel()}\t{int(bool(parameter.requires_grad))}"
        for name, parameter in all_parameters
    )
    total_trainable = sum(
        int(parameter.numel()) for _, parameter in all_parameters if parameter.requires_grad
    )
    return {
        "schema": "STAGE22_TRAINABLE_MODULE_INVENTORY_V1",
        "status": "PASS",
        "recipe": recipe,
        "collection_method": "live_model_named_parameters",
        "model_construction": "FROZEN_STAGE20S_PIPELINE_CREATE_MODEL",
        "frozen_launcher_git_sha": FROZEN_LAUNCHER_GIT_SHA,
        "base_model_path": str(base_model_path),
        "base_model_index_sha256": _sha256_file(base_model_path / "model.safetensors.index.json"),
        "config_flags": {
            "tune_projector": True,
            "tune_diffusion_model": recipe == "R1",
            "tune_visual": False,
            "tune_llm": False,
            "tune_vlln": False,
        },
        "modules": rows,
        "total_parameter_count": sum(int(parameter.numel()) for _, parameter in all_parameters),
        "total_trainable_parameters": total_trainable,
        "parameter_inventory_digest_sha256": _digest_lines(global_lines),
        "coverage": {
            "complete": not unassigned and not duplicate_claims,
            "unassigned_parameter_count": len(unassigned),
            "unassigned_parameter_names": sorted(unassigned),
            "overlapping_parameter_count": len(duplicate_claims),
            "overlapping_parameter_names": sorted(duplicate_claims),
        },
        "source": {
            "probe_script": "stage22_probe_trainable_modules.py",
            "probe_schema": "STAGE22_TRAINABLE_MODULE_INVENTORY_V1",
            "launcher_path": str(launcher_path),
            "launcher_file_sha256": _sha256_file(launcher_path),
            "no_forward": True,
            "no_backward": True,
            "no_optimizer_step": True,
            "no_inference": True,
            "no_training": True,
        },
    }


def probe_trainable_modules(
    *,
    stage10_root: Path,
    base_model_path: Path,
    dataset_dir: Path,
    modality_config: Path,
    launcher_path: Path,
    run_manifest: Path,
    recipe: str,
    output: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    if recipe not in {"R0", "R1"}:
        raise ValueError("recipe must be R0 or R1")
    if not local_files_only:
        raise ValueError("the inventory probe requires --local-files-only")
    if output.exists():
        raise ValueError(f"refusing to overwrite inventory artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage10_root = stage10_root.resolve(strict=True)
    base_model_path = base_model_path.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    modality_config = modality_config.resolve(strict=True)
    if not (base_model_path / "model.safetensors.index.json").is_file():
        raise ValueError(f"base model index is missing: {base_model_path}")
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise ValueError(f"materialized dataset is missing: {dataset_dir}")
    if not modality_config.is_file():
        raise ValueError(f"modality config is missing: {modality_config}")
    run_manifest = run_manifest.resolve(strict=True)
    run_payload = _validate_run_manifest(
        run_manifest,
        recipe=recipe,
        base_model_path=base_model_path,
        dataset_dir=dataset_dir,
        modality_config=modality_config,
    )
    launcher_path = launcher_path.resolve(strict=True)
    if launcher_path.name != "stage20s_train_gr00t.py":
        raise ValueError("inventory probe must use the frozen stage20s_train_gr00t.py launcher")
    launcher_sha256 = _sha256_file(launcher_path)
    if launcher_sha256 != FROZEN_LAUNCHER_FILE_SHA256:
        raise ValueError("launcher file does not match the frozen Stage20S source")
    launcher = _load_launcher(launcher_path)
    from argparse import Namespace

    with tempfile.TemporaryDirectory(prefix="stage22_inventory_", dir=str(output.parent)) as work:
        args = Namespace(
            stage10_root=stage10_root,
            base_model_path=base_model_path,
            dataset_dir=dataset_dir,
            modality_config=modality_config,
            output_dir=Path(work) / "unused_training_output",
            max_steps=50,
            save_steps=25,
            learning_rate=1e-4,
            seed=2222,
            tune_diffusion_model=recipe == "R1",
            local_files_only=True,
        )
        config = launcher._build_config(args)
        sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
        from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

        save_cfg_dir = Path(work) / "probe_config"
        save_cfg_dir.mkdir(parents=True, exist_ok=True)
        pipeline = Gr00tN1d7Pipeline(config, save_cfg_dir)
        model = pipeline._create_model()
        payload = _inventory(
            model,
            recipe=recipe,
            base_model_path=base_model_path,
            launcher_path=launcher_path,
        )
        payload["runtime"] = {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "cuda": __import__("torch").version.cuda,
        }
        payload["source_run_manifest_sha256"] = _sha256_file(run_manifest)
        payload["source_run_manifest_path"] = str(run_manifest)
        payload["run_manifest_identity"] = {
            "schema": run_payload.get("schema"),
            "status": run_payload.get("status"),
            "max_steps": run_payload.get("max_steps"),
            "action_horizon": run_payload.get("action_horizon"),
            "tune_diffusion_model": run_payload.get("tune_diffusion_model"),
        }
        del model
        del pipeline
        gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--recipe", choices=("R0", "R1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    payload = probe_trainable_modules(
        stage10_root=args.stage10_root.resolve(strict=True),
        base_model_path=args.base_model_path.resolve(strict=True),
        dataset_dir=args.dataset_dir.resolve(strict=True),
        modality_config=args.modality_config.resolve(strict=True),
        launcher_path=args.launcher,
        run_manifest=args.run_manifest,
        recipe=args.recipe,
        output=args.output,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
