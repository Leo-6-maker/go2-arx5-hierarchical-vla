"""CPU/static regressions for the manifest-driven inference engine."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.h200_admission import H200AdmissionError
from go2arx5_vla.evaluation.h200_engine import (
    AdapterInference,
    H200_RUNTIME_CONFIG_SCHEMA,
    H200EngineError,
    run_h200_inference_engine,
)
from go2arx5_vla.evaluation.reusable import (
    GENERIC_EVAL_STATUS,
    GENERIC_EVALUATION_MANIFEST_SCHEMA,
)
from go2arx5_vla.evaluation.structured_action_eval import DecodedStructuredPrediction
from go2arx5_vla.experiments.reusable import RUNTIME_LATE_BOUND_FIELDS, sha256_file, write_json


SCRIPT = ROOT / "scripts" / "run_h200_inference_backend.py"
GENERIC_ENGINE = ROOT / "src" / "go2arx5_vla" / "evaluation" / "h200_engine.py"
REGISTRY = ROOT / "src" / "go2arx5_vla" / "evaluation" / "h200_adapters.py"
VERSIONED_ADAPTER = ROOT / "src" / "go2arx5_vla" / "evaluation" / "groot_n1d7_adapter.py"
HISTORICAL_CANARY = ROOT / "scripts" / "reusable_h200_inference_canary.py"
HISTORICAL_ADAPTER = ROOT / "scripts" / "stage45c_h200_eval_runtime_adapter.py"


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "file"}


def _fixture(root: Path) -> tuple[Path, Path, dict, dict, tuple[str, ...], tuple[str, ...]]:
    inputs = root / "inputs"
    inputs.mkdir()
    noise = inputs / "noise.bin"
    noise.write_bytes(b"observed noise bank for CPU fixture")
    body = inputs / "body.rgb"
    body.write_bytes(b"body observation")
    wrist = inputs / "wrist.rgb"
    wrist.write_bytes(b"wrist observation")
    cohort = inputs / "cohort.json"
    cohort.write_bytes(b'{"cohort":"arbitrary-order"}\n')

    labels = ("checkpoint-zeta", "checkpoint-alpha")
    windows = ("target-B/window-7", "target-A/window-2", "target-C/window-9")
    checkpoints: list[dict] = []
    checkpoint_paths: dict[str, Path] = {}
    statistics_paths: dict[str, Path] = {}
    for index, label in enumerate(labels):
        checkpoint = inputs / f"{index}_{label}.checkpoint"
        checkpoint.write_bytes(f"checkpoint-{label}".encode())
        mapping = inputs / f"{index}_{label}.mapping"
        mapping.write_bytes(f"mapping-{label}".encode())
        statistics = inputs / f"{index}_{label}.stats"
        statistics.write_bytes(f"stats-{label}".encode())
        checkpoint_paths[label] = checkpoint
        statistics_paths[label] = statistics
        checkpoints.append(
            {
                "label": label,
                "identity": _artifact(checkpoint),
                "runtime_mapping": _artifact(mapping),
                "statistics": _artifact(statistics),
            }
        )

    window_values = []
    for index, window_id in enumerate(windows):
        window_values.append(
            {
                "id": window_id,
                "target_id": f"explicit-target-{index}",
                "anchor_world_m": [0.0, float(index), 0.0],
                "target_world_m": [1.0, float(index), 0.0],
                "source_timestamps": [float(step) * 0.05 for step in range(40)],
                "observation_timestamp": float(index) * 0.1,
                "adapter_input": {
                    "dataset_index": index,
                    "observations": {
                        "body_rgb": {
                            **_artifact(body),
                            "shape": [96, 96, 3],
                            "dtype": "uint8",
                        },
                        "wrist_rgb": {
                            **_artifact(wrist),
                            "shape": [96, 96, 3],
                            "dtype": "uint8",
                        },
                    },
                    "state_v1": [0.0] * 10,
                    "expert_physical_action": np.zeros((40, 10), dtype=np.float64).tolist(),
                },
            }
        )

    output = root / "evaluation-output"
    late_bound = {
        "repository_execution_head": "a" * 40,
        "python_runtime": "fixture-python-v1",
        "container_or_environment": "fixture-environment-v1",
        "groot_source": "fixture-groot-source-v1",
        "cosmos_cache_root": "fixture-cosmos-cache-v1",
        "checkpoint_paths": {
            label: _artifact(checkpoint_paths[label]) for label in labels
        },
        "dataset_path": _artifact(cohort),
        "statistics_path": {
            label: _artifact(statistics_paths[label]) for label in labels
        },
        "output_root": str(output.resolve()),
        "scheduler_project": "fixture-project-v1",
        "scheduler_queue": "fixture-queue-v1",
        "scheduler_resources": "fixture-resources-v1",
        "gpu_device": "fixture-device-v1",
        "host": "fixture-host-v1",
        "offline_mode": True,
        "runtime_admission_identity": "fixture-admission-v1",
    }
    manifest = {
        "schema": GENERIC_EVALUATION_MANIFEST_SCHEMA,
        "status": GENERIC_EVAL_STATUS,
        "experiment": {
            "id": "generic-engine-arbitrary-order",
            "stage": "future-generic",
            "execution_head": "a" * 40,
            "protocol_sha256": "b" * 64,
        },
        "timestamp_profile": "GENERIC_MONOTONIC_H_V1",
        "cohort": {
            "name": "ARBITRARY_COHORT",
            "split": "TRAIN",
            "identity": _artifact(cohort),
            "window_order": "explicit_manifest_order",
            "windows": window_values,
        },
        "checkpoints": checkpoints,
        "contract": {
            "raw_action_shape": [1, 40, 132],
            "decoded_action_shape": [40, 10],
            "state_v1_shape": [10],
            "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
            "target_xyz_in_model_input": False,
            "raw_before_decode": True,
            "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
            "inverse_normalization": "manifest_bound_stats",
            "geometry": "K16",
            "timing": "T16",
        },
        "sampler": {
            "profile": "GROOT_N1D7_SOURCE_MAPPED_EULER4_V1",
            "timesteps": [0, 250, 500, 750],
            "dt": 0.25,
            "expected_calls_per_window": 4,
            "same_noise_across_checkpoints": True,
            "noise_bank": str(noise.resolve()),
            "noise_bank_sha256": sha256_file(noise),
        },
        "runtime": {
            "adapter": "groot_n1d7_structured_action_v1",
            "offline": True,
            "no_model_construction": True,
            "no_training": True,
            "identity": "generic-engine-runtime-v1",
            "backend": "groot_h200_inference_backend_v1",
            "late_bound": late_bound,
        },
        "output": {
            "root": str(output.resolve()),
            "raw_predictions_manifest": "raw_predictions.jsonl",
            "fresh": True,
        },
        "locked_boundaries": {
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "simulator": "LOCKED",
            "hardware": "LOCKED",
            "new_data": "LOCKED",
        },
        "analysis": {
            "paired_comparisons": [
                {"name": "zeta_vs_alpha", "left": labels[0], "right": labels[1]}
            ]
        },
    }
    runtime_config = {
        "schema": H200_RUNTIME_CONFIG_SCHEMA,
        "repository_root": str(ROOT.resolve()),
        "expected_branch": "fixture-branch-v1",
        "expected_execution_head": "a" * 40,
        "require_clean_checkout": True,
        "adapter": "groot_n1d7_structured_action_v1",
        "observed_runtime": copy.deepcopy(late_bound),
        "observed_sampler": {
            "noise_bank": _artifact(noise),
            "source": "cpu-fixture-observation",
        },
    }
    manifest_path = root / "evaluation-manifest.json"
    config_path = root / "runtime-config.json"
    write_json(manifest_path, manifest)
    write_json(config_path, runtime_config)
    return manifest_path, config_path, manifest, runtime_config, labels, windows


def _probe(_repo_root: Path) -> dict[str, object]:
    return {
        "repository_root": str(ROOT.resolve()),
        "repository_execution_head": "a" * 40,
        "repository_branch": "fixture-branch-v1",
        "python_runtime": "fixture-python-v1",
        "host": "fixture-host-v1",
        "offline_mode": True,
        "gpu_device": "fixture-device-v1",
        "runtime_admission_identity": "fixture-admission-v1",
        "worktree_clean": True,
    }


class RecordingAdapter:
    def __init__(self, manifest: dict, *, fail_at: int | None = None) -> None:
        self.manifest = manifest
        self.fail_at = fail_at
        self.prepared = False
        self.prepare_calls = 0
        self.close_calls = 0
        self.infer_pairs: list[tuple[str, str]] = []
        self.runtime_flags = {
            "adapter_id": "cpu-fixture",
            "adapter_version": "test-v1",
            "model_imported": False,
            "model_construction": False,
            "model_forward": False,
        }

    def prepare(self, _context: object) -> None:
        self.prepared = True
        self.prepare_calls += 1

    def infer(
        self,
        *,
        checkpoint: dict,
        window: dict,
        noise_bank: Path,
        context: object,
    ) -> AdapterInference:
        del noise_bank, context
        self.assert_prepared()
        pair = (str(checkpoint["label"]), str(window["id"]))
        self.infer_pairs.append(pair)
        if self.fail_at is not None and len(self.infer_pairs) == self.fail_at:
            raise RuntimeError("induced late adapter failure")
        value = np.full((1, 40, 132), len(self.infer_pairs) / 100.0, dtype=np.float32)
        adapter_input = window["adapter_input"]
        return AdapterInference(
            raw=value,
            sampler_calls=4,
            observations=adapter_input["observations"],
            state_v1=adapter_input["state_v1"],
            expert_physical_action=adapter_input["expert_physical_action"],
            metadata={"fixture": True},
        )

    def assert_prepared(self) -> None:
        if not self.prepared:
            raise AssertionError("adapter was used before admission preparation")

    def close(self) -> None:
        self.close_calls += 1


class RecordingDecoder:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls = 0

    def __call__(
        self,
        raw: np.ndarray,
        state: np.ndarray,
        *,
        stats_path: Path,
        source_timestamps: object,
        observation_timestamp: float,
        chunk_id: str,
    ) -> DecodedStructuredPrediction:
        del state, stats_path, source_timestamps, observation_timestamp, chunk_id
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("induced late decoder failure")
        physical = np.zeros((40, 10), dtype=np.float64)
        physical[:, 0] = float(raw[0, 0, 0])
        return DecodedStructuredPrediction(
            physical_action=physical,
            decoder_details={"fixture_decoder": True},
            trajectory_chunk=None,
        )


class GeneralH200EngineTests(unittest.TestCase):
    def test_arbitrary_checkpoint_and_explicit_window_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, _config, labels, windows = _fixture(root)
            adapter = RecordingAdapter(manifest)
            summary = run_h200_inference_engine(
                manifest_path,
                config_path,
                repo_root=ROOT,
                adapter=adapter,
                runtime_probe=_probe,
                decoder=RecordingDecoder(),
            )

            self.assertEqual(adapter.prepare_calls, 1)
            self.assertEqual(
                adapter.infer_pairs,
                [(label, window) for label in labels for window in windows],
            )
            self.assertEqual(summary["checkpoint_order"], list(labels))
            self.assertEqual(summary["window_order"], list(windows))
            self.assertEqual(summary["checkpoint_count"], 2)
            self.assertEqual(summary["window_count"], 3)
            self.assertEqual(summary["raw_prediction_count"], 6)
            self.assertEqual(summary["sampler_calls"], 24)
            output = root / "evaluation-output"
            self.assertTrue((output / "engine_admission.json").is_file())
            self.assertTrue((output / "engine_raw_seal.json").is_file())
            self.assertTrue((output / "engine_run_manifest.json").is_file())
            self.assertTrue((output / "engine_artifact_inventory_sha256.txt").is_file())
            self.assertEqual(
                len((output / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()),
                6,
            )

    def test_wrong_head_and_runtime_fail_before_adapter_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, config, _labels, _windows = _fixture(root)
            adapter = RecordingAdapter(manifest)
            bad_manifest = copy.deepcopy(manifest)
            bad_manifest["experiment"]["execution_head"] = "c" * 40
            write_json(manifest_path, bad_manifest)
            with self.assertRaises((H200EngineError, H200AdmissionError)):
                run_h200_inference_engine(
                    manifest_path,
                    config_path,
                    repo_root=ROOT,
                    adapter=adapter,
                    runtime_probe=_probe,
                    decoder=RecordingDecoder(),
                )
            self.assertEqual(adapter.prepare_calls, 0)

    def test_tampered_bound_artifact_and_unknown_adapter_fail_before_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, _config, _labels, _windows = _fixture(root)
            checkpoint_path = Path(manifest["checkpoints"][0]["identity"]["path"])
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
            adapter = RecordingAdapter(manifest)
            with self.assertRaises((H200EngineError, H200AdmissionError)):
                run_h200_inference_engine(
                    manifest_path,
                    config_path,
                    repo_root=ROOT,
                    adapter=adapter,
                    runtime_probe=_probe,
                    decoder=RecordingDecoder(),
                )
            self.assertEqual(adapter.prepare_calls, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, config, _labels, _windows = _fixture(root)
            unknown_manifest = copy.deepcopy(manifest)
            unknown_manifest["runtime"]["adapter"] = "unlisted-adapter-v1"
            unknown_config = copy.deepcopy(config)
            unknown_config["adapter"] = "unlisted-adapter-v1"
            unknown_manifest_path = root / "unknown-manifest.json"
            unknown_config_path = root / "unknown-config.json"
            write_json(unknown_manifest_path, unknown_manifest)
            write_json(unknown_config_path, unknown_config)
            adapter = RecordingAdapter(unknown_manifest)
            with self.assertRaises((H200EngineError, H200AdmissionError, ValueError)):
                run_h200_inference_engine(
                    unknown_manifest_path,
                    unknown_config_path,
                    repo_root=ROOT,
                    adapter=adapter,
                    runtime_probe=_probe,
                    decoder=RecordingDecoder(),
                )
            self.assertEqual(adapter.prepare_calls, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, config, _labels, _windows = _fixture(root)
            bad_config = copy.deepcopy(config)
            bad_config["observed_runtime"]["host"] = "wrong-host"
            bad_config_path = root / "wrong-runtime-config.json"
            write_json(bad_config_path, bad_config)
            adapter = RecordingAdapter(manifest)
            with self.assertRaises((H200EngineError, H200AdmissionError)):
                run_h200_inference_engine(
                    manifest_path,
                    bad_config_path,
                    repo_root=ROOT,
                    adapter=adapter,
                    runtime_probe=_probe,
                    decoder=RecordingDecoder(),
                )
            self.assertEqual(adapter.prepare_calls, 0)

    def test_induced_adapter_failure_seals_partial_raw_without_derived_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, _config, _labels, _windows = _fixture(root)
            adapter = RecordingAdapter(manifest, fail_at=5)
            with self.assertRaises(RuntimeError):
                run_h200_inference_engine(
                    manifest_path,
                    config_path,
                    repo_root=ROOT,
                    adapter=adapter,
                    runtime_probe=_probe,
                    decoder=RecordingDecoder(),
                )
            output = root / "evaluation-output"
            failure = json.loads((output / "engine_failure_seal.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["raw_count"], 4)
            self.assertFalse(failure["derived_started"])
            staging = Path(failure["staging_root"])
            raw_seal = json.loads((staging / "raw_seal.json").read_text(encoding="utf-8"))
            self.assertEqual(raw_seal["status"], "FAILED_PARTIAL")
            self.assertEqual(raw_seal["raw_count"], 4)
            self.assertEqual(len(list((staging / "raw").rglob("*.npy"))), 4)
            self.assertFalse((output / "engine_run_manifest.json").exists())

    def test_late_decode_failure_sees_durable_raw_before_first_derived_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, config_path, manifest, _config, _labels, _windows = _fixture(root)
            decoder = RecordingDecoder(fail_at=3)
            with self.assertRaises(RuntimeError):
                run_h200_inference_engine(
                    manifest_path,
                    config_path,
                    repo_root=ROOT,
                    adapter=RecordingAdapter(manifest),
                    runtime_probe=_probe,
                    decoder=decoder,
                )
            output = root / "evaluation-output"
            self.assertEqual(
                len((output / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()),
                3,
            )
            self.assertEqual(len(list((output / "raw").rglob("*.npy"))), 3)
            failure = json.loads((output / "engine_failure_seal.json").read_text(encoding="utf-8"))
            self.assertTrue(failure["raw_sealed"])
            self.assertTrue(failure["derived_started"])
            self.assertTrue((output / "failure_seal.json").is_file())

    def test_registry_is_exact_and_model_imports_are_deferred(self) -> None:
        from go2arx5_vla.evaluation.h200_adapters import (
            adapter_spec,
            registered_adapter_ids,
        )

        self.assertEqual(registered_adapter_ids(), ("groot_n1d7_structured_action_v1",))
        self.assertEqual(adapter_spec("groot_n1d7_structured_action_v1").version, "v1")
        with self.assertRaises(ValueError):
            adapter_spec("not-allowlisted")
        self.assertNotIn("go2arx5_vla.evaluation.groot_n1d7_adapter", sys.modules)
        from go2arx5_vla.evaluation.h200_adapters import resolve_adapter

        resolved = resolve_adapter("groot_n1d7_structured_action_v1", object())
        self.assertEqual(resolved.adapter_id, "groot_n1d7_structured_action_v1")
        self.assertEqual(resolved.adapter_version, "v1")

        for path in (GENERIC_ENGINE, REGISTRY, VERSIONED_ADAPTER, SCRIPT):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            forbidden = ("torch", "gr00t", "transformers", "safetensors")
            self.assertEqual(
                [
                    name
                    for name in imports
                    if name in forbidden or name.startswith(tuple(item + "." for item in forbidden))
                ],
                [],
                str(path),
            )

    def test_generic_surface_has_no_instance_specific_tokens_and_admission_precedes_adapter(self) -> None:
        forbidden = ("stage45", "full24", "train24", "stage22")
        for path in (GENERIC_ENGINE, SCRIPT, REGISTRY):
            source = path.read_text(encoding="utf-8").lower()
            self.assertFalse(any(token in source for token in forbidden), str(path))
        source = GENERIC_ENGINE.read_text(encoding="utf-8")
        self.assertLess(
            source.index("validate_h200_inference_admission("),
            source.index("resolve_adapter"),
        )
        self.assertIn("raw_seal", source)
        self.assertIn("derived_started", source)

    def test_absolute_foreign_cwd_empty_pythonpath_self_check_and_wrong_root(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = ""
        environment["PYTHONNOUSERSITE"] = "1"
        with tempfile.TemporaryDirectory() as temporary:
            foreign = Path(temporary) / "foreign-cwd"
            wrong = Path(temporary) / "wrong-root"
            foreign.mkdir()
            wrong.mkdir()
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(ROOT), "--self-check"],
                cwd=foreign,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["repository_root"], str(ROOT.resolve()))
            self.assertEqual(payload["src_root"], str((ROOT / "src").resolve()))
            self.assertTrue(payload["scripts_namespace"])
            self.assertEqual(payload["pythonpath"], "")
            self.assertEqual(payload["model_modules_present"], [])

            blocked = subprocess.run(
                [sys.executable, str(SCRIPT), "--repo-root", str(wrong), "--self-check"],
                cwd=foreign,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(blocked.returncode, 2)
            self.assertEqual(blocked.stdout, "")
            blocked_payload = json.loads(blocked.stderr)
            self.assertEqual(blocked_payload["status"], "BLOCKED")
            self.assertNotIn("registered_adapters", blocked_payload)

    def test_versioned_adapter_keeps_source_mapped_action_head_symbols(self) -> None:
        current = ast.parse(VERSIONED_ADAPTER.read_text(encoding="utf-8"))
        historical = ast.parse(HISTORICAL_ADAPTER.read_text(encoding="utf-8"))
        current_functions = {
            node.name: node
            for node in current.body
            if isinstance(node, ast.FunctionDef)
        }
        historical_functions = {
            node.name: node
            for node in historical.body
            if isinstance(node, ast.FunctionDef)
        }
        current_source = ast.get_source_segment(
            VERSIONED_ADAPTER.read_text(encoding="utf-8"), current_functions["_load_frameworks"]
        )
        self.assertIn("gr00t.model.gr00t_n1d7", current_source)
        adapter_source = VERSIONED_ADAPTER.read_text(encoding="utf-8")
        historical_source = HISTORICAL_ADAPTER.read_text(encoding="utf-8")
        for symbol in (
            "action_encoder",
            "position_embedding",
            "torch.cat",
            "action_decoder",
            "use_alternate_vl_dit",
        ):
            self.assertIn(symbol, adapter_source)
            self.assertIn(symbol, historical_source)
        self.assertIn("_offline_step", historical_functions)
        self.assertIn("raw = current.detach()", adapter_source)

    def test_historical_canary_remains_a_separate_committed_surface(self) -> None:
        self.assertTrue(HISTORICAL_CANARY.is_file())
        generic_source = GENERIC_ENGINE.read_text(encoding="utf-8")
        self.assertNotIn("reusable_h200_inference_canary", generic_source)
        self.assertNotIn("run_h200_inference_backend", HISTORICAL_CANARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
