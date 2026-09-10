"""CPU/synthetic regression tests for the reusable train/eval boundary."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.analyzer import (
    CheckpointAnalysisError,
    analyze_rows,
    load_rows,
    write_analysis,
)
from go2arx5_vla.evaluation.reusable import (
    GENERIC_EVAL_STATUS,
    GENERIC_EVALUATION_MANIFEST_SCHEMA,
    GENERIC_PREDICTION_RECORD_SCHEMA,
    ReusableEvaluationError,
    build_evaluation_plan,
    run_evaluation,
    validate_evaluation_manifest,
)
from go2arx5_vla.evaluation.h200_admission import (
    H200_INFERENCE_ADMISSION_SCHEMA,
    H200_INFERENCE_ADMISSION_STATUS,
    H200AdmissionError,
    validate_h200_inference_admission,
)
from go2arx5_vla.evaluation.h200_binding import ADAPTER_RUNTIME_BINDING_SCHEMA
from go2arx5_vla.evaluation.structured_action_eval import (
    DecodedStructuredPrediction,
)
from go2arx5_vla.experiments.profiles import (
    SAMPLER_PROFILES,
    sampler_profile,
)
from go2arx5_vla.experiments.reusable import (
    RUNTIME_LATE_BOUND_FIELDS,
    sha256_file,
    sha256_tree,
    tree_deployment_metadata,
    write_json,
)
from go2arx5_vla.training.reusable import (
    ReusableTrainingError,
    build_training_plan,
    validate_training_execution_admission,
    validate_training_manifest,
    verify_training_artifacts,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path), "kind": "file"}


class ReusableInfrastructureTests(unittest.TestCase):
    def _evaluation_fixture(self, root: Path) -> tuple[Path, dict, list[dict]]:
        input_dir = root / "inputs"
        input_dir.mkdir()
        noise_path = input_dir / "noise.bin"
        noise_path.write_bytes(b"same noise bank")
        body_path = input_dir / "body.rgb"
        body_path.write_bytes(b"body modality fixture")
        wrist_path = input_dir / "wrist.rgb"
        wrist_path.write_bytes(b"wrist modality fixture")
        cohort_path = input_dir / "cohort.json"
        cohort_path.write_bytes(b'{"cohort":"fixture"}\n')
        groot_source_path = input_dir / "groot-source.txt"
        groot_source_path.write_bytes(b"groot source fixture\n")
        cosmos_path = input_dir / "cosmos-cache.txt"
        cosmos_path.write_bytes(b"cosmos cache fixture\n")
        modality_path = input_dir / "modality-config.py"
        modality_path.write_bytes(b"MODALITY = 'fixture'\n")
        processor_path = input_dir / "processor-checkpoint.bin"
        processor_path.write_bytes(b"processor checkpoint fixture\n")
        base_path = input_dir / "base-checkpoint.bin"
        base_path.write_bytes(b"base checkpoint fixture\n")

        checkpoint_files: dict[str, Path] = {}
        mapping_files: dict[str, Path] = {}
        stats_files: dict[str, Path] = {}
        checkpoints = []
        for label, byte in (("CP0", b"checkpoint zero"), ("CP1", b"checkpoint one")):
            checkpoint_path = input_dir / f"{label}.bin"
            checkpoint_path.write_bytes(byte)
            stats_path = input_dir / f"{label}.stats"
            stats_path.write_bytes(b"statistics fixture")
            mapping_path = input_dir / f"{label}.mapping"
            mapping_path.write_bytes(b"runtime mapping fixture")
            checkpoint_files[label] = checkpoint_path
            mapping_files[label] = mapping_path
            stats_files[label] = stats_path
            checkpoints.append(
                {
                    "label": label,
                    "identity": _artifact(checkpoint_path),
                    "runtime_mapping": _artifact(mapping_path),
                    "statistics": _artifact(stats_path),
                }
            )

        windows = []
        for index in range(2):
            windows.append(
                {
                    "id": f"target_{index}:window-00",
                    "target_id": f"target_{index}",
                    "anchor_world_m": [0.0, 0.0, 0.0],
                    "target_world_m": [1.0, 0.0, 0.0],
                    "source_timestamps": [float(step) * 0.05 for step in range(40)],
                    "observation_timestamp": 0.0,
                }
            )
        manifest = {
            "schema": GENERIC_EVALUATION_MANIFEST_SCHEMA,
            "status": GENERIC_EVAL_STATUS,
            "experiment": {
                "id": "synthetic-reusable-eval",
                "stage": "future",
                "execution_head": "a" * 40,
                "protocol_sha256": "b" * 64,
            },
            "timestamp_profile": "GENERIC_MONOTONIC_H_V1",
            "cohort": {
                "name": "SYNTHETIC_TWO_WINDOW",
                "split": "TRAIN",
                "identity": _artifact(cohort_path),
                "window_order": "explicit_manifest_order",
                "windows": windows,
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
                "language": "move the end effector to the fixture target",
            },
            "sampler": {
                "profile": "GROOT_N1D7_SOURCE_MAPPED_EULER4_V1",
                "timesteps": [0, 250, 500, 750],
                "dt": 0.25,
                "expected_calls_per_window": 4,
                "same_noise_across_checkpoints": True,
                "noise_bank": str(noise_path),
                "noise_bank_sha256": sha256_file(noise_path),
            },
            "runtime": {
                "adapter": "synthetic-stage-adapter",
                "offline": True,
                "no_model_construction": True,
                "no_training": True,
                "identity": "runtime-fixture-v1",
                "backend": "groot_h200_inference_backend_v1",
                "late_bound": {
                    "repository_execution_head": "a" * 40,
                    "python_runtime": "fixture-python-v1",
                    "container_or_environment": "fixture-environment-v1",
                    "groot_source": _artifact(groot_source_path),
                    "cosmos_cache_root": _artifact(cosmos_path),
                    "checkpoint_paths": {
                        label: _artifact(path)
                        for label, path in checkpoint_files.items()
                    },
                    "dataset_path": _artifact(cohort_path),
                    "statistics_path": {
                        label: _artifact(path) for label, path in stats_files.items()
                    },
                    "output_root": str(root / "evaluation-output"),
                    "scheduler_project": "fixture-project-v1",
                    "scheduler_queue": "fixture-queue-v1",
                    "scheduler_resources": "fixture-resources-v1",
                    "gpu_device": "fixture-gpu-device-v1",
                    "host": "fixture-host-v1",
                    "offline_mode": True,
                    "runtime_admission_identity": "fixture-admission-v1",
                },
                "adapter_input_identities": {
                    "modality_config": _artifact(modality_path),
                    "processor_checkpoint": _artifact(processor_path),
                    "base_checkpoint": _artifact(base_path),
                },
            },
            "output": {
                "root": str(root / "evaluation-output"),
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
                    {"name": "cp0_vs_cp1", "left": "CP0", "right": "CP1"}
                ]
            },
        }
        records = []
        for label, fill in (("CP0", 0.1), ("CP1", 0.2)):
            for index, window in enumerate(windows):
                raw_path = input_dir / f"{label}_{index}.npy"
                raw = np.full((1, 40, 132), fill, dtype=np.float32)
                np.save(raw_path, raw)
                records.append(
                    {
                        "schema": GENERIC_PREDICTION_RECORD_SCHEMA,
                        "checkpoint_label": label,
                        "checkpoint_sha256": next(
                            item["identity"]["sha256"]
                            for item in checkpoints
                            if item["label"] == label
                        ),
                        "window_id": window["id"],
                        "raw_path": str(raw_path),
                        "raw_sha256": sha256_file(raw_path),
                        "raw_array_sha256": _sha256_bytes(
                            np.ascontiguousarray(raw).tobytes()
                        ),
                        "noise_sha256": _sha256_bytes(
                            b"window-noise-" + str(index).encode()
                        ),
                        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
                        "target_xyz_in_model_input": False,
                        "observations": {
                            "body_rgb": {
                                **_artifact(body_path),
                                "shape": [96, 96, 3],
                                "dtype": "uint8",
                            },
                            "wrist_rgb": {
                                **_artifact(wrist_path),
                                "shape": [96, 96, 3],
                                "dtype": "uint8",
                            },
                        },
                        "state_v1": [0.0] * 10,
                        "expert_physical_action": np.zeros(
                            (40, 10), dtype=np.float64
                        ).tolist(),
                    }
                )
        manifest_path = root / "evaluation-manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path, manifest, records

    def _h200_admission(self, manifest: dict) -> dict:
        return {
            "schema": H200_INFERENCE_ADMISSION_SCHEMA,
            "backend": manifest["runtime"]["backend"],
            "runtime_identity": manifest["runtime"]["identity"],
            "timestamp_profile": manifest["timestamp_profile"],
            "runtime_late_bound": json.loads(
                json.dumps(manifest["runtime"]["late_bound"])
            ),
            "adapter_runtime_binding": {
                "schema": ADAPTER_RUNTIME_BINDING_SCHEMA,
                "adapter_id": manifest["runtime"]["adapter"],
                "adapter_version": "v1",
                "immutable": True,
                "dataset": manifest["runtime"]["late_bound"]["dataset_path"],
                "groot_source": manifest["runtime"]["late_bound"]["groot_source"],
                "cosmos_cache": manifest["runtime"]["late_bound"]["cosmos_cache_root"],
                "modality_config": manifest["runtime"]["adapter_input_identities"]["modality_config"],
                "processor_checkpoint": manifest["runtime"]["adapter_input_identities"]["processor_checkpoint"],
                "base_checkpoint": manifest["runtime"]["adapter_input_identities"]["base_checkpoint"],
                "cuda_device": manifest["runtime"]["late_bound"]["gpu_device"],
                "checkpoint": manifest["runtime"]["late_bound"]["checkpoint_paths"],
                "statistics": manifest["runtime"]["late_bound"]["statistics_path"],
                "noise_bank": _artifact(Path(manifest["sampler"]["noise_bank"])),
                "language": manifest["contract"]["language"],
            },
            "cohort": {
                "name": manifest["cohort"]["name"],
                "split": manifest["cohort"]["split"],
                "window_ids": [
                    item["id"] for item in manifest["cohort"]["windows"]
                ],
                "target_ids": [
                    item["target_id"] for item in manifest["cohort"]["windows"]
                ],
            },
            "sampler": {
                "profile": manifest["sampler"]["profile"],
                "timesteps": list(manifest["sampler"]["timesteps"]),
                "dt": manifest["sampler"]["dt"],
                "expected_calls_per_window": manifest["sampler"][
                    "expected_calls_per_window"
                ],
                "same_noise_across_checkpoints": True,
                "noise_bank": _artifact(Path(manifest["sampler"]["noise_bank"])),
            },
            "output_fresh": True,
            "locked_boundaries": dict(manifest["locked_boundaries"]),
        }

    def test_plan_validates_explicit_cohort_and_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _manifest_path, manifest, _ = self._evaluation_fixture(Path(temporary))
            validate_evaluation_manifest(manifest)
            plan = build_evaluation_plan(manifest)
            self.assertEqual(plan["raw_prediction_count"], 4)
            self.assertEqual(plan["sampler_calls"], 16)
            self.assertFalse(plan["model_construction"])

    def test_sampler_profile_requires_strict_json_numeric_types(self) -> None:
        invalid_values = {
            "timesteps_fractional": lambda candidate: candidate["sampler"].update(
                {"timesteps": [0.9, 250, 500, 750]}
            ),
            "timesteps_boolean": lambda candidate: candidate["sampler"].update(
                {"timesteps": [False, 250, 500, 750]}
            ),
            "timesteps_numeric_string": lambda candidate: candidate["sampler"].update(
                {"timesteps": ["0", 250, 500, 750]}
            ),
            "calls_fractional": lambda candidate: candidate["sampler"].update(
                {"expected_calls_per_window": 4.5}
            ),
            "calls_boolean": lambda candidate: candidate["sampler"].update(
                {"expected_calls_per_window": True}
            ),
            "calls_numeric_string": lambda candidate: candidate["sampler"].update(
                {"expected_calls_per_window": "4"}
            ),
            "dt_boolean": lambda candidate: candidate["sampler"].update(
                {"dt": True}
            ),
            "dt_numeric_string": lambda candidate: candidate["sampler"].update(
                {"dt": "0.25"}
            ),
            "dt_nonfinite": lambda candidate: candidate["sampler"].update(
                {"dt": float("nan")}
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            _manifest_path, manifest, _records = self._evaluation_fixture(Path(temporary))
            for case, mutate in invalid_values.items():
                with self.subTest(case=case):
                    candidate = json.loads(json.dumps(manifest))
                    mutate(candidate)
                    with self.assertRaises(ReusableEvaluationError):
                        validate_evaluation_manifest(candidate)

    def test_raw_is_sealed_before_decode_and_analyzer_is_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest, records = self._evaluation_fixture(root)
            raw_manifest_path = root / "evaluation-output" / "raw_predictions.jsonl"
            decode_calls = 0

            def fake_decoder(
                raw,
                state,
                *,
                stats_path,
                source_timestamps,
                observation_timestamp,
                chunk_id,
            ):
                nonlocal decode_calls
                persisted_lines = raw_manifest_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertEqual(len(persisted_lines), decode_calls + 1)
                decode_calls += 1
                physical = np.zeros((40, 10), dtype=np.float64)
                physical[:, 0] = float(raw[0, 0, 0])
                return DecodedStructuredPrediction(
                    physical_action=physical,
                    decoder_details={"synthetic_decoder": True},
                    trajectory_chunk=None,
                )

            summary = run_evaluation(
                manifest,
                records,
                output_dir=root / "evaluation-output",
                manifest_path=manifest_path,
                records_base_dir=root / "inputs",
                decoder=fake_decoder,
            )
            self.assertEqual(decode_calls, 4)
            self.assertEqual(summary["raw_prediction_count"], 4)
            self.assertEqual(summary["sampler_calls"], 16)
            self.assertNotIn("skipped", summary["verified_artifacts"])
            self.assertTrue(summary["verified_artifacts"]["cohort"]["verified"])
            raw_lines = raw_manifest_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 4)
            self.assertTrue(
                all(
                    json.loads(line)["raw_persisted_before_decode"]
                    for line in raw_lines
                )
            )
            decoded_paths = sorted((root / "evaluation-output" / "decoded").rglob("*.npy"))
            self.assertEqual(len(decoded_paths), 4)
            self.assertTrue(
                all(np.load(path, allow_pickle=False).shape == (40, 10) for path in decoded_paths)
            )
            self.assertTrue((root / "evaluation-output" / "per_window.csv").is_file())
            self.assertTrue((root / "evaluation-output" / "checkpoint_aggregates.json").is_file())
            self.assertTrue((root / "evaluation-output" / "artifact_inventory_sha256.txt").is_file())
            self.assertTrue((root / "evaluation-output" / "failure_seal.json").is_file())

            rows, source = load_rows(root / "evaluation-output")
            analysis = analyze_rows(
                rows,
                paired_comparisons=[
                    {"name": "cp0_vs_cp1", "left": "CP0", "right": "CP1"}
                ],
            )
            self.assertEqual(analysis["window_count"], 2)
            self.assertEqual(analysis["paired_comparisons"][0]["window_count"], 2)
            self.assertGreater(
                analysis["paired_comparisons"][0]["metric_distributions"][
                    "position_rmse_m"
                ]["mean"],
                0.0,
            )
            analysis_manifest = write_analysis(
                analysis,
                output_dir=root / "analysis-output",
                source_path=source,
            )
            self.assertEqual(analysis_manifest["threshold_used"], None)
            self.assertTrue((root / "analysis-output" / "report.md").is_file())

    def test_artifact_verification_is_mandatory_before_decode(self) -> None:
        tamper_cases = (
            ("checkpoint", "identity"),
            ("runtime_mapping", "runtime_mapping"),
            ("statistics", "statistics"),
        )
        for case_name, artifact_key in tamper_cases:
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest, records = self._evaluation_fixture(root)
                artifact_path = Path(manifest["checkpoints"][0][artifact_key]["path"])
                artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
                decoder_calls = 0

                def decoder(*args, **kwargs):
                    nonlocal decoder_calls
                    decoder_calls += 1
                    raise AssertionError("decoder must not be called")

                with self.assertRaises(ReusableEvaluationError):
                    run_evaluation(
                        manifest,
                        records,
                        output_dir=root / "evaluation-output",
                        manifest_path=manifest_path,
                        records_base_dir=root / "inputs",
                        decoder=decoder,
                    )
                self.assertEqual(decoder_calls, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest, records = self._evaluation_fixture(root)
            noise_path = Path(manifest["sampler"]["noise_bank"])
            noise_path.write_bytes(noise_path.read_bytes() + b"tampered")
            decoder_calls = 0

            def decoder(*args, **kwargs):
                nonlocal decoder_calls
                decoder_calls += 1
                raise AssertionError("decoder must not be called")

            with self.assertRaises(ReusableEvaluationError):
                run_evaluation(
                    manifest,
                    records,
                    output_dir=root / "evaluation-output",
                    manifest_path=manifest_path,
                    records_base_dir=root / "inputs",
                    decoder=decoder,
                )
            self.assertEqual(decoder_calls, 0)

    def test_h200_admission_cross_binds_all_runtime_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest, _records = self._evaluation_fixture(root)
            admission = self._h200_admission(manifest)
            result = validate_h200_inference_admission(
                manifest, admission, manifest_path=manifest_path
            )
            self.assertEqual(result["status"], H200_INFERENCE_ADMISSION_STATUS)
            self.assertFalse(result["model_imported"])
            self.assertFalse(result["model_construction"])
            self.assertFalse(result["model_forward"])
            self.assertFalse(result["pbs_submission"])

            mutations = {
                "head": lambda item: item["runtime_late_bound"].update(
                    {"repository_execution_head": "c" * 40}
                ),
                "checkpoint": lambda item: item["runtime_late_bound"][
                    "checkpoint_paths"
                ]["CP0"].update({"sha256": "c" * 64}),
                "statistics": lambda item: item["runtime_late_bound"][
                    "statistics_path"
                ]["CP0"].update({"sha256": "c" * 64}),
                "dataset": lambda item: item["runtime_late_bound"][
                    "dataset_path"
                ].update({"sha256": "c" * 64}),
                "cohort": lambda item: item["cohort"].update(
                    {"window_ids": ["wrong-window"]}
                ),
                "output": lambda item: item["runtime_late_bound"].update(
                    {"output_root": str(root / "other-output")}
                ),
                "offline": lambda item: item["runtime_late_bound"].update(
                    {"offline_mode": False}
                ),
                "runtime_identity": lambda item: item["runtime_late_bound"].update(
                    {"runtime_admission_identity": "wrong-admission"}
                ),
                "sampler": lambda item: item["sampler"].update(
                    {"profile": "wrong-profile"}
                ),
                "sampler_timesteps_fractional": lambda item: item["sampler"].update(
                    {"timesteps": [0.9, 250, 500, 750]}
                ),
                "sampler_timesteps_boolean": lambda item: item["sampler"].update(
                    {"timesteps": [False, 250, 500, 750]}
                ),
                "sampler_timesteps_numeric_string": lambda item: item["sampler"].update(
                    {"timesteps": ["0", 250, 500, 750]}
                ),
                "sampler_calls_fractional": lambda item: item["sampler"].update(
                    {"expected_calls_per_window": 4.5}
                ),
                "sampler_calls_boolean": lambda item: item["sampler"].update(
                    {"expected_calls_per_window": True}
                ),
                "sampler_calls_numeric_string": lambda item: item["sampler"].update(
                    {"expected_calls_per_window": "4"}
                ),
                "sampler_dt_boolean": lambda item: item["sampler"].update(
                    {"dt": True}
                ),
                "sampler_dt_numeric_string": lambda item: item["sampler"].update(
                    {"dt": "0.25"}
                ),
                "noise": lambda item: item["sampler"]["noise_bank"].update(
                    {"sha256": "c" * 64}
                ),
                "locked_boundary": lambda item: item["locked_boundaries"].update(
                    {"Holdout-F": "OPEN"}
                ),
            }
            for case_name, mutate in mutations.items():
                with self.subTest(case_name=case_name):
                    candidate = json.loads(json.dumps(admission))
                    mutate(candidate)
                    with self.assertRaises(H200AdmissionError):
                        validate_h200_inference_admission(
                            manifest, candidate, manifest_path=manifest_path
                        )
            output_root = Path(manifest["output"]["root"])
            output_root.mkdir()
            (output_root / "stale.json").write_text("stale\n", encoding="utf-8")
            with self.assertRaises(H200AdmissionError):
                validate_h200_inference_admission(
                    manifest, admission, manifest_path=manifest_path
                )

    def test_generic_analyzer_requires_explicit_target_id(self) -> None:
        rows = [
            {
                "checkpoint": {"label": "arbitrary", "sha256": "a" * 64},
                "window_id": "stage22_expert_legacy:window-1",
                "metrics": {"position_rmse_m": 1.0},
            }
        ]
        with self.assertRaises(CheckpointAnalysisError):
            analyze_rows(rows)
        analyzer_source = (
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "analyzer.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("stage22", analyzer_source.lower())

    def test_analysis_output_must_be_fresh_and_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            result = analyze_rows(
                [
                    {
                        "checkpoint": {"label": "CP", "sha256": "a" * 64},
                        "window_id": "window-1",
                        "target_id": "target-1",
                        "metrics": {"position_rmse_m": 1.0},
                    }
                ]
            )
            output = root / "analysis-output"
            output.mkdir()
            (output / "stale.json").write_text("stale\n", encoding="utf-8")
            with self.assertRaises(CheckpointAnalysisError):
                write_analysis(result, output_dir=output, source_path=source)

    def test_manifest_rejects_target_xyz_and_noise_or_order_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest, records = self._evaluation_fixture(Path(temporary))
            bad_manifest = json.loads(json.dumps(manifest))
            bad_manifest["contract"]["target_xyz_in_model_input"] = True
            with self.assertRaises(ReusableEvaluationError):
                validate_evaluation_manifest(bad_manifest)
            bad_records = list(records)
            bad_records[1] = dict(bad_records[1])
            bad_records[1]["noise_sha256"] = "c" * 64
            with self.assertRaises(ReusableEvaluationError):
                run_evaluation(
                    manifest,
                    bad_records,
                    output_dir=Path(temporary) / "evaluation-output",
                    manifest_path=manifest_path,
                    records_base_dir=Path(temporary) / "inputs",
                    decoder=lambda *args, **kwargs: None,
                )

    def test_timestamp_profiles_allow_portable_monotonic_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _manifest_path, manifest, _records = self._evaluation_fixture(Path(temporary))
            for timestamps in (
                [float(step) / 20.0 for step in range(40)],
                [float(step) / 25.0 for step in range(40)],
                [float(step) * 0.037 + (0.001 if step > 10 else 0.0) for step in range(40)],
            ):
                candidate = json.loads(json.dumps(manifest))
                candidate["cohort"]["windows"][0]["source_timestamps"] = timestamps
                validate_evaluation_manifest(candidate)

            nonmonotonic = [float(step) / 20.0 for step in range(40)]
            nonmonotonic[20], nonmonotonic[21] = 2.0, 1.0
            nonfinite = [float(step) / 20.0 for step in range(40)]
            nonfinite[7] = float("nan")
            for timestamps in (nonmonotonic, nonfinite):
                candidate = json.loads(json.dumps(manifest))
                candidate["cohort"]["windows"][0]["source_timestamps"] = timestamps
                with self.assertRaises(ReusableEvaluationError):
                    validate_evaluation_manifest(candidate)

            fixed = json.loads(json.dumps(manifest))
            fixed["timestamp_profile"] = "STAGE40_T16_20HZ_V1"
            fixed["cohort"]["windows"][0]["source_timestamps"] = [
                float(step) / 25.0 for step in range(40)
            ]
            with self.assertRaises(ReusableEvaluationError):
                validate_evaluation_manifest(fixed)

    def test_sampler_profile_and_descriptive_only_analysis(self) -> None:
        self.assertIn("GROOT_N1D7_SOURCE_MAPPED_EULER4_V1", SAMPLER_PROFILES)
        self.assertEqual(
            sampler_profile("GROOT_N1D7_SOURCE_MAPPED_EULER4_V1").calls_per_window,
            4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            _manifest_path, manifest, _records = self._evaluation_fixture(Path(temporary))
            bad = json.loads(json.dumps(manifest))
            bad["sampler"]["profile"] = "unknown-sampler-v9"
            with self.assertRaises(ReusableEvaluationError):
                validate_evaluation_manifest(bad)

        rows = [
            {
                "checkpoint": {"label": "arbitrary-a", "sha256": "a" * 64},
                "window_id": "window-1",
                "target_id": "target-1",
                "metrics": {"position_rmse_m": 1.0},
            },
            {
                "checkpoint": {"label": "arbitrary-b", "sha256": "b" * 64},
                "window_id": "window-1",
                "target_id": "target-1",
                "metrics": {"position_rmse_m": 2.0},
            },
        ]
        result = analyze_rows(rows)
        self.assertEqual(result["paired_comparisons"], [])

    def test_late_failure_leaves_incremental_decoded_and_rows_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _manifest_path, manifest, records = self._evaluation_fixture(root)
            calls = 0

            def failing_decoder(
                raw,
                state,
                *,
                stats_path,
                source_timestamps,
                observation_timestamp,
                chunk_id,
            ):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("induced late decode failure")
                return DecodedStructuredPrediction(
                    physical_action=np.zeros((40, 10), dtype=np.float64),
                    decoder_details={"synthetic_decoder": True},
                    trajectory_chunk=None,
                )

            output = root / "evaluation-output"
            with self.assertRaises(RuntimeError):
                run_evaluation(
                    manifest,
                    records,
                    output_dir=output,
                    manifest_path=root / "evaluation-manifest.json",
                    records_base_dir=root / "inputs",
                    decoder=failing_decoder,
                )
            rows_path = output / "per_window.jsonl"
            self.assertEqual(len(rows_path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len(list((output / "decoded").rglob("*.npy"))), 2)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["rows_completed"], 2)
            seal = json.loads((output / "failure_seal.json").read_text(encoding="utf-8"))
            self.assertEqual(seal["status"], "FAILED")

    def test_future_tree_identity_ignores_safe_mode_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            copied = root / "copied"
            source.mkdir()
            (source / "nested").mkdir()
            (source / "nested" / "payload.bin").write_bytes(b"stable bytes")
            (source / "manifest.json").write_text("{\"v\":1}\n", encoding="utf-8")
            shutil.copytree(source, copied)
            os.chmod(source / "nested" / "payload.bin", 0o600)
            os.chmod(copied / "nested" / "payload.bin", 0o644)
            self.assertEqual(sha256_tree(source), sha256_tree(copied))
            self.assertTrue(tree_deployment_metadata(source)["safe"])
            self.assertTrue(tree_deployment_metadata(copied)["safe"])

    def test_templates_are_non_executable_and_use_sentinels(self) -> None:
        eval_template = json.loads(
            (ROOT / "manifests" / "REUSABLE_CHECKPOINT_EVAL_MANIFEST_TEMPLATE_V1.json").read_text(
                encoding="utf-8"
            )
        )
        train_template = json.loads(
            (ROOT / "manifests" / "REUSABLE_ADAPTATION_TRAINING_MANIFEST_TEMPLATE_V1.json").read_text(
                encoding="utf-8"
            )
        )
        validate_evaluation_manifest(eval_template)
        validate_training_manifest(train_template)
        with self.assertRaises(ReusableEvaluationError):
            build_evaluation_plan(eval_template)
        with self.assertRaises(ReusableTrainingError):
            build_training_plan(train_template)
        self.assertTrue(
            all(value == "__REQUIRED__" for value in train_template["recipe"].values())
        )
        serialized_eval = json.dumps(eval_template, sort_keys=True)
        self.assertNotIn("0.05", serialized_eval)
        self.assertNotIn("1.95", serialized_eval)

    def test_backend_boundaries_are_explicit_in_plans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _eval_manifest_path, eval_manifest, _records = self._evaluation_fixture(root)
            self.assertEqual(
                build_evaluation_plan(eval_manifest)["backend_boundary"],
                "groot_h200_inference_backend_v1",
            )
            _train_manifest_path, train_manifest = self._training_fixture(root)
            plan = build_training_plan(train_manifest)
            self.assertEqual(
                plan["backend_boundary"], "groot_h200_training_backend_v1"
            )
            self.assertEqual(
                set(plan["late_bound_runtime_fields"]), set(RUNTIME_LATE_BOUND_FIELDS)
            )
            bad = json.loads(json.dumps(train_manifest))
            del bad["runtime"]["late_bound"][RUNTIME_LATE_BOUND_FIELDS[0]]
            with self.assertRaises(ReusableTrainingError):
                validate_training_manifest(bad)

    def _training_fixture(self, root: Path) -> tuple[Path, dict]:
        base = root / "base.bin"
        dataset = root / "dataset.json"
        stats = root / "stats.json"
        for path, payload in (
            (base, b"base checkpoint"),
            (dataset, b'{"dataset":"fixture"}\n'),
            (stats, b'{"stats":"fixture"}\n'),
        ):
            path.write_bytes(payload)
        identity = lambda path: {
            "path": str(path),
            "sha256": sha256_file(path),
            "kind": "file",
        }
        manifest = {
            "schema": "GO2ARX5_REUSABLE_ADAPTATION_TRAINING_MANIFEST_V1",
            "status": "READY_NOT_EXECUTED",
            "experiment": {
                "id": "synthetic-training",
                "stage": "future",
                "execution_head": "d" * 40,
                "protocol_sha256": "e" * 64,
            },
            "dataset": {
                "name": "FIXTURE",
                "split": "TRAIN",
                "manifest": identity(dataset),
                "statistics": identity(stats),
                "provenance": {"identity": "dataset-provenance-v1", "sha256": "f" * 64},
            },
            "base_checkpoint": {"label": "BASE", "identity": identity(base)},
            "recipe": {
                "max_steps": 20,
                "warmup_steps": 1,
                "scheduler": "cosine",
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "seed": 2222,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "bf16": True,
                "fp16": False,
                "tune_projector": True,
                "tune_diffusion": True,
                "tune_visual": False,
                "tune_llm": False,
                "tune_vlln": False,
                "resume": False,
                "fresh_start": True,
                "save_only_model": False,
                "save_steps": [10, 20],
            },
            "runtime": {
                "identity": "training-runtime-fixture-v1",
                "offline": True,
                "preflight_required": True,
                "backend": "groot_h200_training_backend_v1",
                "late_bound": {
                    field: f"fixture-{field}" for field in RUNTIME_LATE_BOUND_FIELDS
                },
            },
            "output": {
                "root": str(root / "train-output"),
                "fresh": True,
                "failure_seal": True,
            },
            "locked_boundaries": {
                "Validation-E": "LOCKED",
                "Holdout-F": "LOCKED",
                "WBC": "LOCKED",
                "simulator": "LOCKED",
                "hardware": "LOCKED",
                "new_data": "LOCKED",
            },
        }
        manifest_path = root / "training-manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path, manifest

    def test_training_plan_requires_explicit_no_resume_and_full_state_saves(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest = self._training_fixture(Path(temporary))
            validate_training_manifest(manifest)
            plan = build_training_plan(manifest)
            self.assertFalse(plan["training_started"])
            self.assertTrue(plan["fresh_start"])
            self.assertEqual(plan["recipe"]["save_steps"], [10, 20])
            verified = verify_training_artifacts(manifest, manifest_path=manifest_path)
            self.assertEqual(verified["base_checkpoint"]["verified"], True)
            plan_with_verified = dict(plan)
            plan_with_verified["verified_artifacts"] = verified
            self.assertEqual(
                validate_training_execution_admission(plan_with_verified), verified
            )
            skipped_plan = dict(plan)
            skipped_plan["verified_artifacts"] = {"skipped": True}
            with self.assertRaises(ReusableTrainingError):
                validate_training_execution_admission(skipped_plan)
            bad = json.loads(json.dumps(manifest))
            del bad["recipe"]["resume"]
            with self.assertRaises(ReusableTrainingError):
                validate_training_manifest(bad)

    def test_training_controls_require_strict_json_integers(self) -> None:
        fractional_values = {
            "max_steps": 20.5,
            "warmup_steps": 1.5,
            "seed": 2222.5,
            "batch_size": 1.5,
            "gradient_accumulation_steps": 1.5,
            "save_steps": [10, 20.5],
        }
        with tempfile.TemporaryDirectory() as temporary:
            _manifest_path, manifest = self._training_fixture(Path(temporary))
            for field, value in fractional_values.items():
                with self.subTest(field=field):
                    candidate = json.loads(json.dumps(manifest))
                    candidate["recipe"][field] = value
                    with self.assertRaises(ReusableTrainingError):
                        validate_training_manifest(candidate)
            boolean_steps = json.loads(json.dumps(manifest))
            boolean_steps["recipe"]["max_steps"] = True
            with self.assertRaises(ReusableTrainingError):
                validate_training_manifest(boolean_steps)

    def test_training_learning_rates_require_strict_json_numbers(self) -> None:
        invalid_values = {
            "learning_rate_boolean": ("learning_rate", True),
            "learning_rate_numeric_string": ("learning_rate", "1e-4"),
            "learning_rate_nonfinite": ("learning_rate", float("nan")),
            "weight_decay_boolean": ("weight_decay", True),
            "weight_decay_numeric_string": ("weight_decay", "1e-5"),
            "weight_decay_nonfinite": ("weight_decay", float("inf")),
        }
        with tempfile.TemporaryDirectory() as temporary:
            _manifest_path, manifest = self._training_fixture(Path(temporary))
            for case, (field, value) in invalid_values.items():
                with self.subTest(case=case):
                    candidate = json.loads(json.dumps(manifest))
                    candidate["recipe"][field] = value
                    with self.assertRaises(ReusableTrainingError):
                        validate_training_manifest(candidate)

    def test_generic_sources_have_no_runtime_imports_or_process_launch(self) -> None:
        paths = [
            ROOT / "scripts" / "eval_checkpoint_cohort.py",
            ROOT / "scripts" / "analyze_checkpoint_eval.py",
            ROOT / "scripts" / "train_adaptation.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "reusable.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "analyzer.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "h200_admission.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "h200_binding.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "stage45c_compatibility.py",
            ROOT / "src" / "go2arx5_vla" / "training" / "reusable.py",
            ROOT / "scripts" / "validate_h200_inference_admission.py",
        ]
        forbidden = {"torch", "gr00t", "cuda", "subprocess"}
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(
                imported.isdisjoint(forbidden), (path, imported & forbidden)
            )
            self.assertNotIn("os.system", path.read_text(encoding="utf-8"))
        self.assertNotIn(
            "verify_artifacts",
            (ROOT / "scripts" / "eval_checkpoint_cohort.py").read_text(
                encoding="utf-8"
            ),
        )

    def test_future_generic_layer_has_no_machine_or_stage_literals(self) -> None:
        paths = [
            ROOT / "scripts" / "eval_checkpoint_cohort.py",
            ROOT / "scripts" / "analyze_checkpoint_eval.py",
            ROOT / "scripts" / "train_adaptation.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "reusable.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "analyzer.py",
            ROOT / "src" / "go2arx5_vla" / "evaluation" / "h200_admission.py",
            ROOT / "src" / "go2arx5_vla" / "training" / "reusable.py",
            ROOT / "src" / "go2arx5_vla" / "experiments" / "reusable.py",
            ROOT / "scripts" / "validate_h200_inference_admission.py",
        ]
        forbidden = (
            "Stage45",
            "FULL24",
            "STRUCTURED@1000",
            "e1192548",
            "/nfs/home/svu",
            "/scratch/e1192548",
            "CFP03-CF-004",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertTrue(
                all(token not in source for token in forbidden),
                (path, [token for token in forbidden if token in source]),
            )


if __name__ == "__main__":
    unittest.main()
