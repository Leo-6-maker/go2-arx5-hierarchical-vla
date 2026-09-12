#!/usr/bin/env python3
"""Record one reproduced L1 runtime-weight observation without updating a model.

PI0130 permits at most one H200 reproduction because the historical 612326
canary did not serialize the action-loss dtype or the post-cast XYZ
coefficients.  This surface deliberately performs one model construction and
one single-batch forward only.  It never creates an optimizer, calls
backward, starts training, writes a checkpoint, evaluates TRAIN24, or reads a
locked split.

The resulting JSON is explicitly reproduced evidence.  It is not historical
612326 telemetry and does not authorize the later recovered-input evaluator.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45e_l1_runtime_admission as admission  # noqa: E402
import stage45e_l1_h200_train as train_impl  # noqa: E402


PI0130_COMMENT = 5642258588
HISTORICAL_TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
REPRODUCED_SCHEMA = "STAGE45E_L1_REPRODUCED_RUNTIME_COEFFICIENT_EVIDENCE_V1"
REPRODUCED_STATUS = "REPRODUCED_RUNTIME_COEFFICIENT_EVIDENCE"
SOURCE_SEMANTICS = "FROZEN_612326_L1_CONFIG_AND_LOSS"
FORBIDDEN_TOP_LEVEL_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}
FORBIDDEN_CALLS = {
    "backward",
    "step",
    "zero_grad",
    "save_pretrained",
    "save_model",
    "save_checkpoint",
    "gr00t_run",
    "qsub",
}


class RuntimeWeightReproductionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeWeightReproductionError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_durable(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"reproduction output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-500:]}")
    return completed.stdout.strip()


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    forbidden_imports = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    forbidden_calls = sorted(calls.intersection(FORBIDDEN_CALLS))
    markers = {
        "absolute_repo_resolution": str(REPO) == str(Path(__file__).resolve().parents[1]),
        "single_forward_contract": True,
        "runtime_evidence_helper": "capture_effective_weight_evidence" in source,
        "no_optimizer_surface": "step" not in forbidden_calls and "zero_grad" not in forbidden_calls,
        "no_backward_surface": "backward" not in forbidden_calls,
        "no_training_entrypoint": "gr00t_run" not in forbidden_calls and "qsub" not in forbidden_calls,
        "no_checkpoint_write": "save_checkpoint" not in forbidden_calls and "save_model" not in forbidden_calls,
        "reproduced_label": REPRODUCED_STATUS in source,
        "historical_boundary": HISTORICAL_TRAINING_HEAD in source and "not historical" in source,
    }
    status = "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL"
    return {
        "schema": "STAGE45E_L1_REPRODUCED_RUNTIME_COEFFICIENT_STATIC_GUARD_V1",
        "status": status,
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": forbidden_calls,
        "markers": markers,
        "model_construction": 0,
        "model_forward": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "training": 0,
        "pbs": 0,
        "gpu": 0,
        "evaluation": 0,
        "automatic_retry": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def _default_args() -> argparse.Namespace:
    values = {
        key: Path(value)
        for key, value in train_impl.DEFAULTS.items()
        if key != "python"
    }
    return argparse.Namespace(**values)


def _write_failure(output_root: Path, state: Mapping[str, Any], error: BaseException) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        failure_path = output_root / "stage45e_l1_reproduced_runtime_weight_failure.json"
        if failure_path.exists() or failure_path.is_symlink():
            return
        payload = {
            "schema": "STAGE45E_L1_REPRODUCED_RUNTIME_COEFFICIENT_FAILURE_V1",
            "status": "FAIL",
            "pi_gate": PI0130_COMMENT,
            "execution_head": state.get("execution_head"),
            "branch": state.get("branch"),
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "phase": state.get("phase"),
            "model_imported": state.get("model_imported"),
            "model_construction_started": state.get("model_construction_started"),
            "model_construction": state.get("model_construction"),
            "model_forward": state.get("model_forward"),
            "action_head_forwards": state.get("action_head_forwards"),
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "automatic_retry": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "timestamp_utc": now_utc(),
        }
        write_json_durable(failure_path, payload)
    except Exception:
        # The original exception is the authoritative failure; do not mask it
        # with a best-effort failure-seal write error.
        pass


def execute(expected_head: str, output_root: Path) -> int:
    state: dict[str, Any] = {
        "phase": "START",
        "model_imported": False,
        "model_construction_started": False,
        "model_construction": 0,
        "model_forward": 0,
        "action_head_forwards": 0,
        "execution_head": None,
        "branch": None,
    }
    torch = None
    try:
        require(os.environ.get("PBS_JOBID"), "--execute requires a PBS allocation")
        require(expected_head and len(expected_head) == 40, "full expected execution HEAD is required")
        require(Path(sys.executable).resolve() == Path(os.environ["STAGE45E_L1_PYTHON"]).resolve(), "reproduction Python mismatch")
        require(not admission.model_imports_present(), "model module was imported before reproduction admission")
        guard = static_guard()
        require(guard["status"] == "PASS", "reproduction static guard failed")

        state["phase"] = "PRE_MODEL_ADMISSION"
        checkout = train_impl.checkout_observation(expected_head)
        state["execution_head"] = checkout["observed_head"]
        state["branch"] = checkout["observed_branch"]
        gpu = train_impl.gpu_observation()
        runtime = train_impl.runtime_identity(gpu)
        paths = train_impl._paths(_default_args())
        require(str(paths["base_checkpoint"]) == train_impl.DEFAULTS["base_model"], "BASE path is not sealed")
        require(str(paths["full24_dataset"]) == train_impl.DEFAULTS["dataset"], "FULL24 path is not sealed")
        require(str(paths["groot_source"]) == train_impl.DEFAULTS["groot_source"], "GR00T source path is not sealed")
        require(str(paths["cosmos_root"]) == train_impl.DEFAULTS["cosmos_root"], "Cosmos path is not sealed")
        require(str(paths["full24_runtime_statistics"]) == train_impl.DEFAULTS["stats"], "FULL24 stats path is not sealed")
        code_contract = train_impl._validate_l1_code()
        input_audit = train_impl._validate_inputs(paths)

        # Use the evaluator's already-admitted, inference-only binding shape:
        # this reproducer has no training boundary and no evaluation budget.
        import stage45e_l1_train24_eval as eval_impl

        output_root = Path(output_root).expanduser().resolve()
        require(not output_root.exists() and not output_root.is_symlink(), f"reproduction root must be fresh: {output_root}")
        path_values = {key: str(value) for key, value in paths.items()}
        binding, observed = eval_impl._build_binding(checkout, gpu, runtime, path_values, output_root)
        admission_result = admission.validate_pre_model_binding(
            binding,
            observed,
            authority_token=admission.AUTHORITY_TOKEN,
            phase="EVAL",
        )
        output_root.mkdir(parents=False, exist_ok=False)
        write_json_durable(output_root / "stage45e_l1_reproduction_static_guard.json", guard)
        write_json_durable(output_root / "stage45e_l1_reproduction_runtime_binding.json", {"binding": binding, "observed": observed})
        write_json_durable(output_root / "stage45e_l1_reproduction_pre_model_admission.json", admission_result)

        state["phase"] = "MODEL_IMPORT"
        env_values = train_impl._set_runtime_environment(output_root, paths["cosmos_root"])
        if str(paths["groot_source"]) not in sys.path:
            sys.path.insert(0, str(paths["groot_source"]))
        train_impl._bind_cosmos_processor(paths["cosmos_root"] / "snapshots" / admission.COSMOS_REVISION)
        from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory

        install_structured_dataset_factory()
        from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
        from go2arx5_vla import stage45e_l1_loss as l1
        from go2arx5_vla.stage45e_l1_runtime_weight_evidence import capture_effective_weight_evidence

        l1_patch = l1.install_l1_loss_patch()
        state["model_imported"] = True
        state["phase"] = "MODEL_CONSTRUCTION"
        canary_root = output_root / "canary"
        canary_root.mkdir(parents=False, exist_ok=False)
        config = train_impl._load_config(paths["training_config"], canary_root, paths["base_checkpoint"], paths["full24_dataset"])
        canary_config_dir = canary_root / "experiment_cfg"
        canary_config_dir.mkdir(parents=False, exist_ok=False)
        state["model_construction_started"] = True
        pipeline = Gr00tN1d7Pipeline(config, canary_config_dir)
        pipeline.setup()
        state["model_construction"] = 1
        model = pipeline.return_model()
        model.to(device=admission.DEVICE)
        model.eval()

        sample = next(iter(pipeline.return_dataset()[0]))
        batch = pipeline.return_collator()([sample])
        inputs = batch["inputs"]
        l1.reset_runtime_counters()
        state["phase"] = "ONE_FORWARD"
        # Match the admitted BF16 forward representation while keeping the
        # reproducer strictly no-grad/no-update.
        with torch_module_context() as torch_module:
            torch = torch_module
            with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                with torch_module.no_grad():
                    outputs = model(inputs)
            state["model_forward"] = 1
            runtime_weight_evidence = capture_effective_weight_evidence(
                torch_module, outputs["action_loss"], inputs["action_mask"]
            )
            require(bool(torch_module.isfinite(outputs["loss"]).item()), "reproduced L1 loss is non-finite")

        state["action_head_forwards"] = l1.runtime_counters()["action_head_forwards"]
        require(state["action_head_forwards"] == 1, "reproduced action-head count drift")
        checkpoint_dirs = sorted(canary_root.glob("checkpoint-*"))
        require(not checkpoint_dirs, "reproduced canary wrote a checkpoint")
        del outputs, inputs, batch, sample, model, pipeline, config
        gc.collect()
        torch.cuda.empty_cache()

        state["phase"] = "SEAL"
        counters = {
            "model_construction": 1,
            "model_forward": 1,
            "action_head_forwards": 1,
            "backward": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "training": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        }
        result = {
            "schema": REPRODUCED_SCHEMA,
            "status": REPRODUCED_STATUS,
            "pi_gate": PI0130_COMMENT,
            "contract_pi_gate": admission.PI_GATE_COMMENT,
            "execution_head": checkout["observed_head"],
            "branch": checkout["observed_branch"],
            "pbs_jobid": os.environ["PBS_JOBID"],
            "host_gpu_runtime": gpu,
            "runtime_identity": runtime,
            "historical_training_head": HISTORICAL_TRAINING_HEAD,
            "historical_training_root": "/scratch/e1192548/stage45e_l1_h200_20260911_377b5012_612326.hopper-m-02",
            "source_semantics": SOURCE_SEMANTICS,
            "l1_contract": code_contract,
            "input_audit": input_audit,
            "paths": path_values,
            "runtime_environment": env_values,
            "l1_patch": l1_patch,
            "evidence": runtime_weight_evidence,
            "execution_counters": counters,
            "model_mode": "eval",
            "precision": {"bf16_autocast": True, "fp16": False},
            "evaluation_executed": False,
            "automatic_retry": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "claim_boundary": "Reproduced no-update H200 forward observation only; this is not historical 612326 telemetry, does not prove gradient correctness, and does not authorize TRAIN24 evaluation.",
            "timestamp_utc": now_utc(),
        }
        evidence_path = output_root / "stage45e_l1_reproduced_runtime_weight_evidence.json"
        write_json_durable(evidence_path, result)
        result["evidence_artifact"] = {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
        }
        # The artifact is written exactly once.  The hash-bearing companion is
        # intentionally separate so the evidence file remains immutable.
        write_json_durable(output_root / "stage45e_l1_reproduced_runtime_weight_pointer.json", result["evidence_artifact"])
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        if torch is not None:
            try:
                state["action_head_forwards"] = train_impl.l1.runtime_counters().get("action_head_forwards", state.get("action_head_forwards", 0))
            except Exception:
                pass
        _write_failure(Path(output_root).expanduser().resolve(), state, exc)
        print(f"STAGE45E_L1_RUNTIME_WEIGHT_REPRODUCTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


class torch_module_context:
    """Lazy Torch loader used only after pre-model admission."""

    def __enter__(self) -> Any:
        import torch

        return torch

    def __exit__(self, *_args: Any) -> None:
        return None


def plan() -> dict[str, Any]:
    guard = static_guard()
    require(guard["status"] == "PASS", "reproduction static guard failed")
    return {
        "schema": REPRODUCED_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "pi_gate": PI0130_COMMENT,
        "historical_training_head": HISTORICAL_TRAINING_HEAD,
        "source_semantics": SOURCE_SEMANTICS,
        "model_construction": 0,
        "model_forward": 0,
        "action_head_forwards": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "training": 0,
        "one_and_only_one_canary": True,
        "automatic_retry": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "static_guard": guard,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-head", default=os.environ.get("STAGE45E_L1_EXPECTED_HEAD"))
    parser.add_argument("--output-root", type=Path, default=os.environ.get("STAGE45E_L1_REPRODUCTION_OUTPUT_ROOT"))
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        require(args.expected_head is not None, "--expected-head is required")
        require(args.output_root is not None, "--output-root is required")
        return execute(args.expected_head, Path(args.output_root))
    except Exception as exc:
        print(f"STAGE45E_L1_RUNTIME_WEIGHT_REPRODUCTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
