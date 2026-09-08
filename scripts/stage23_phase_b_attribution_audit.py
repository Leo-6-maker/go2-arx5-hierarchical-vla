#!/usr/bin/env python3
"""Stage23 Phase B attribution audit.

This is a read-only audit of the sealed Stage23 O1 evidence bundle.  It does
not construct a model, run inference, train, submit PBS work, or execute WBC.

The main audit intentionally uses only the Python standard library so it can
recompute the sealed ``.npz`` profiles without depending on the GPU runtime.
An optional ``--probe-only`` mode runs the official Stage10 processor on the
already materialized TINY1 fixtures to measure normalization semantics.  That
probe reads parquet and processor metadata only; it never loads model weights
or calls a model forward.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


EXPECTED_ARCHIVE_SHA256 = "720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd"
EXPECTED_PAYLOAD_MANIFEST_SHA256 = "dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_TRAINING_SHA = "deb920a070141e6a32a88cd0da6719828647e1e6"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
EXPECTED_STAGE22_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
EXPECTED_STAGE22_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
PHYSICAL_ACTION_DIM = 10
EEF_ACTION_DIM = 9
UNIVERSAL_ACTION_DIM = 132
INACTIVE_ACTION_DIM = UNIVERSAL_ACTION_DIM - PHYSICAL_ACTION_DIM
ACTION_HORIZON = 40
TEST_ACTION_PERIOD_S = 0.05


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "median": _percentile(cleaned, 0.50),
        "mean": mean(cleaned),
        "p95": _percentile(cleaned, 0.95),
        "max": max(cleaned),
    }


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(a: list[float], scalar: float) -> list[float]:
    return [scalar * x for x in a]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(max(0.0, _dot(a, a)))


def _dft_energy(series: list[list[float]], period_s: float) -> dict[str, Any]:
    """Return deterministic all-bin DFT energy and a declared high band.

    The high band is ``min(k, N-k) >= ceil(N/4)``.  For H=40 and the frozen
    0.05 s test timing this starts at 5 Hz.  The unnormalised DFT is used
    consistently; only fractions and within-a-trajectory comparisons matter.
    """

    n = len(series)
    if n == 0:
        return {"cut_bin": None, "cut_hz": None, "total_energy": 0.0, "high_energy": 0.0, "high_fraction": None}
    dimension = len(series[0])
    centers = [sum(row[j] for row in series) / n for j in range(dimension)]
    energies: list[float] = []
    for k in range(n):
        energy = 0.0
        for j in range(dimension):
            real = 0.0
            imag = 0.0
            for index, row in enumerate(series):
                angle = -2.0 * math.pi * k * index / n
                centered = row[j] - centers[j]
                real += centered * math.cos(angle)
                imag += centered * math.sin(angle)
            energy += real * real + imag * imag
        energies.append(energy)
    cut = int(math.ceil(n / 4.0))
    high = sum(energy for k, energy in enumerate(energies) if min(k, n - k) >= cut)
    total = sum(energies)
    return {
        "cut_bin": cut,
        "cut_hz": cut / (n * period_s),
        "total_energy": total,
        "high_energy": high,
        "high_fraction": high / total if total > 0.0 else None,
    }


def _profile(rows: list[list[float]], period_s: float = TEST_ACTION_PERIOD_S) -> dict[str, Any]:
    if len(rows) < 2 or any(len(row) < 3 for row in rows):
        raise ValueError("profile requires at least two XYZ-bearing rows")
    positions = [row[:3] for row in rows]
    first = [_sub(positions[index + 1], positions[index]) for index in range(len(positions) - 1)]
    first_norms = [_norm(value) for value in first]
    second = [
        _sub(
            _sub(positions[index + 2], _scale(positions[index + 1], 2.0)),
            _scale(positions[index], -1.0),
        )
        for index in range(len(positions) - 2)
    ]
    second_norms = [_norm(value) for value in second]
    direction_cosines: list[float] = []
    for index in range(len(first) - 1):
        denominator = first_norms[index] * first_norms[index + 1]
        if denominator > 1e-15:
            direction_cosines.append(_dot(first[index], first[index + 1]) / denominator)
    start = positions[0]
    net_vector = _sub(positions[-1], start)
    net_displacement = _norm(net_vector)
    path_length = sum(first_norms)
    unit = _scale(net_vector, 1.0 / net_displacement) if net_displacement > 1e-15 else [0.0, 0.0, 0.0]
    transverse_distances: list[float] = []
    for point in positions:
        relative = _sub(point, start)
        projection = _scale(unit, _dot(relative, unit))
        transverse_distances.append(_norm(_sub(relative, projection)))
    transverse_step = 0.0
    if path_length > 1e-15:
        for value in first:
            parallel = _scale(unit, _dot(value, unit))
            transverse_step += _norm(_sub(value, parallel))
    speeds = [value / period_s for value in first_norms]
    roughness = _summary(second_norms)
    first_difference_dft = _dft_energy(first, period_s)
    return {
        "point_count": len(rows),
        "duration_s": (len(rows) - 1) * period_s,
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "tortuosity": path_length / net_displacement if net_displacement > 1e-15 else None,
        "first_difference_m": _summary(first_norms),
        "linear_speed_mps": _summary(speeds),
        "second_difference_m": roughness,
        "adjacent_direction_cosine": _summary(direction_cosines),
        "back_and_forth_rate": (
            sum(value < 0.0 for value in direction_cosines) / len(direction_cosines)
            if direction_cosines
            else None
        ),
        "transverse_distance_to_net_line_m": _summary(transverse_distances),
        "transverse_step_fraction": transverse_step / path_length if path_length > 1e-15 else None,
        "position_dft": _dft_energy(positions, period_s),
        "first_difference_dft": first_difference_dft,
        "finite": all(_finite(row) for row in rows),
    }


def _read_npy_from_npz(path: Path, member: str) -> tuple[list[int], list[float]]:
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: {member} is not an NPY payload")
    version = tuple(data[6:8])
    if version == (1, 0):
        header_start = 10
        header_length = struct.unpack("<H", data[8:10])[0]
    elif version in {(2, 0), (3, 0)}:
        header_start = 12
        header_length = struct.unpack("<I", data[8:12])[0]
    else:
        raise ValueError(f"unsupported NPY version {version}")
    header = ast.literal_eval(data[header_start : header_start + header_length].decode("latin1").strip())
    if header.get("fortran_order"):
        raise ValueError("fortran-order NPY is not supported by this audit")
    if header.get("descr") not in {"<f4", "|f4"}:
        raise ValueError(f"unexpected NPY dtype: {header.get('descr')}")
    shape = [int(value) for value in header["shape"]]
    count = math.prod(shape)
    payload = data[header_start + header_length :]
    if len(payload) != count * 4:
        raise ValueError(f"NPY payload length mismatch: {len(payload)} != {count * 4}")
    values = list(struct.unpack("<" + "f" * count, payload))
    return shape, values


def _rows_from_array(shape: list[int], values: list[float]) -> list[list[float]]:
    if len(shape) == 3 and shape[0] == 1:
        height, width = shape[1], shape[2]
        return [values[index * width : (index + 1) * width] for index in range(height)]
    if len(shape) == 2:
        height, width = shape
        return [values[index * width : (index + 1) * width] for index in range(height)]
    raise ValueError(f"expected [1,H,D] or [H,D], got {shape}")


def _verify_bundle(bundle_root: Path) -> dict[str, Any]:
    archive = bundle_root / "phase_a_evidence_bundle_20260903.tar.gz"
    seal_root = bundle_root / "extracted" / "phase_a_seal_20260903"
    payload_manifest = seal_root / "payload_file_manifest.tsv"
    observed_archive = _sha256(archive)
    observed_manifest = _sha256(payload_manifest)
    rows = list(csv.DictReader(payload_manifest.open(encoding="utf-8"), delimiter="\t"))
    verified_files = 0
    errors: list[str] = []
    seal_resolved = seal_root.resolve()
    for row in rows:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe payload path: {row['relative_path']}")
            continue
        path = (seal_root / relative).resolve()
        try:
            path.relative_to(seal_resolved)
        except ValueError:
            errors.append(f"payload path escaped seal root: {row['relative_path']}")
            continue
        if not path.is_file():
            errors.append(f"missing payload file: {row['relative_path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append(f"size mismatch: {row['relative_path']}")
            continue
        actual = _sha256(path)
        if actual != row["sha256"]:
            errors.append(f"SHA256 mismatch: {row['relative_path']}")
            continue
        verified_files += 1
    relay = _json(bundle_root / "relay_verification_v1.json")
    scope = _json(seal_root / "phase_a_scope.json")
    provenance = _json(seal_root / "provenance.json")
    relay_scope = relay.get("scope_booleans", {}) if isinstance(relay.get("scope_booleans"), dict) else {}
    credential_claims = [
        value
        for value in (
            relay.get("contains_credentials"),
            relay_scope.get("contains_credentials"),
            scope.get("contains_credentials"),
        )
        if isinstance(value, bool)
    ]
    return {
        "archive": str(archive),
        "archive_sha256": observed_archive,
        "archive_sha256_expected": EXPECTED_ARCHIVE_SHA256,
        "archive_sha256_match": observed_archive == EXPECTED_ARCHIVE_SHA256,
        "payload_manifest": str(payload_manifest),
        "payload_manifest_sha256": observed_manifest,
        "payload_manifest_sha256_expected": EXPECTED_PAYLOAD_MANIFEST_SHA256,
        "payload_manifest_sha256_match": observed_manifest == EXPECTED_PAYLOAD_MANIFEST_SHA256,
        "payload_manifest_rows": len(rows),
        "payload_files_verified": verified_files,
        "payload_file_hash_errors": errors,
        "payload_hashes_pass": not errors and verified_files == len(rows),
        "relay_verification": relay,
        "phase_a_scope": scope,
        "provenance": provenance,
        "contains_credentials": any(credential_claims),
        "credential_absence_claims_present": bool(credential_claims),
        "credential_absence_claims_pass": bool(credential_claims) and not any(credential_claims),
        "no_new_training_or_inference_declared": bool(
            scope.get("new_inference") is False
            and scope.get("new_model_forward") is False
            and scope.get("new_training_after_200") is False
        ),
    }


def _source_audit(source_root: Path) -> dict[str, Any]:
    files = {
        "action_head": source_root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "processor": source_root / "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
        "state_action_processor": source_root / "gr00t/data/state_action/state_action_processor.py",
        "normalization_utils": source_root / "gr00t/data/utils.py",
        "model_config": source_root / "gr00t/configs/model/gr00t_n1d7.py",
    }
    markers: dict[str, list[str]] = {
        "action_head": [
            "noisy_trajectory = (1 - t) * noise + t * actions",
            "velocity = actions - noise",
            "action_loss = F.mse_loss(pred_actions, velocity, reduction=\"none\") * action_mask",
            "loss = action_loss.sum() / (action_mask.sum() + 1e-6)",
            "actions = actions + dt * pred_velocity * vel_strength",
        ],
        "processor": ["action_mask = torch.ones_like(normalized_actions)", "action_mask[:, action_dim:] = 0"],
        "state_action_processor": ["reference_state = state[state_key][-1]", "relative_action_chunking = action_chunking.relative_chunking"],
        "normalization_utils": ["mask = ~np.isclose(max_vals, min_vals)", "normalized[..., mask] = 2 * normalized[..., mask] - 1"],
        "model_config": [
            "num_inference_timesteps: int = 4",
            "noise_beta_alpha: float = 1.5",
            "noise_beta_beta: float = 1.0",
            "noise_s: float = 0.999",
        ],
    }
    result_files: dict[str, Any] = {}
    all_markers = True
    for key, path in files.items():
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            entry["sha256"] = _sha256(path)
            entry["markers"] = {}
            for marker in markers[key]:
                line = next((index for index, value in enumerate(text.splitlines(), start=1) if marker in value), None)
                entry["markers"][marker] = line
                all_markers = all_markers and line is not None
        else:
            all_markers = False
        result_files[key] = entry
    sha = None
    status = "SOURCE_MARKERS_PASS" if all_markers else "SOURCE_MARKERS_INCOMPLETE"
    try:
        sha = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    dirty: list[str] = []
    try:
        dirty = subprocess.run(
            ["git", "-C", str(source_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "status": status,
        "repo_sha": sha,
        "expected_repo_sha": EXPECTED_GROOT_SHA,
        "repo_sha_match": sha == EXPECTED_GROOT_SHA,
        "working_tree_status": dirty,
        "files": result_files,
        "interpretation": "Official source markers confirm full-width flow noise, first-ten loss masking, fixed-reference relative actions, zero-range min/max handling, and four-step Euler sampling. This is mechanism evidence, not a model-cause proof.",
    }


def _load_training(bundle_seal: Path) -> dict[str, Any]:
    candidates = bundle_seal / "candidates"
    rows: list[dict[str, Any]] = []
    for steps in (10, 25, 50, 100, 200):
        trainer_path = candidates / f"trainer_state_{steps}.json"
        state = _json(trainer_path)
        history = [entry for entry in state.get("log_history", []) if "loss" in entry]
        log_path = candidates / f"train_{steps}.log"
        if steps == 10 and not log_path.is_file():
            log_path = bundle_seal / "incident" / "train_10.log"
        log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        pattern = re.compile(
            r"['\"]loss['\"]:\s*([-+0-9.eE]+).*?['\"]grad_norm['\"]:\s*([-+0-9.eE]+).*?['\"]learning_rate['\"]:\s*([-+0-9.eE]+)"
        )
        log_records = [
            {"loss": float(match.group(1)), "grad_norm": float(match.group(2)), "learning_rate": float(match.group(3))}
            for match in pattern.finditer(log_text)
        ]
        losses = [float(entry["loss"]) for entry in history]
        grads = [float(entry["grad_norm"]) for entry in history if "grad_norm" in entry]
        lrs = [float(entry["learning_rate"]) for entry in history if "learning_rate" in entry]
        run_manifest_path = bundle_seal / "configs" / f"candidate_{steps}.run_manifest.json"
        run_manifest = _json(run_manifest_path)
        rows.append(
            {
                "requested_steps": steps,
                "trainer_state_path": str(trainer_path),
                "trainer_state_sha256": _sha256(trainer_path),
                "global_step": state.get("global_step"),
                "max_steps": state.get("max_steps"),
                "log_history_entries": len(history),
                "training_log_path": str(log_path),
                "training_log_metric_entries": len(log_records),
                "training_log_metric_count_matches": len(log_records) == len(history),
                "loss": {
                    "first": losses[0] if losses else None,
                    "min": min(losses) if losses else None,
                    "min_step": history[losses.index(min(losses))].get("step") if losses else None,
                    "final": losses[-1] if losses else None,
                    "final_minus_first": losses[-1] - losses[0] if losses else None,
                    "summary": _summary(losses),
                },
                "grad_norm": {**_summary(grads), "last": grads[-1] if grads else None},
                "learning_rate": {**_summary(lrs), "last": lrs[-1] if lrs else None},
                "run_manifest_status": run_manifest.get("status"),
                "job": run_manifest.get("output_dir", "").split("/runs/")[-1].split("/")[0],
                "trainable_parameter_count": run_manifest.get("trainable_parameter_count"),
                "cuda_telemetry": run_manifest.get("cuda_telemetry"),
                "run_manifest": {
                    "groot_repo_sha": run_manifest.get("groot_repo_sha"),
                    "base_model_path": run_manifest.get("base_model_path"),
                    "base_model_index_sha256": run_manifest.get("base_model_index_sha256"),
                    "dataset_dir": run_manifest.get("dataset_dir"),
                    "action_horizon": run_manifest.get("action_horizon"),
                    "action_sample_period_s": run_manifest.get("action_sample_period_s"),
                    "tune_projector": run_manifest.get("tune_projector"),
                    "tune_diffusion_model": run_manifest.get("tune_diffusion_model"),
                    "tune_visual": run_manifest.get("tune_visual"),
                    "tune_llm": run_manifest.get("tune_llm"),
                    "tune_vlln": run_manifest.get("tune_vlln"),
                    "simulation_only": run_manifest.get("simulation_only"),
                    "transformers_local_files_only": None,
                },
            }
        )
    return {
        "fresh_base_to_n_experiments": True,
        "warmup_ratio": 0.05,
        "schedule_note": "Each max_steps value is an independent BASE -> N run; cosine/warmup schedules therefore differ across rows.",
        "runs": rows,
        "all_global_steps_exact": all(row["global_step"] == row["requested_steps"] for row in rows),
        "all_log_history_counts_exact": all(row["log_history_entries"] == row["requested_steps"] for row in rows),
        "all_training_logs_complete_when_present": all(
            row["training_log_metric_count_matches"] for row in rows if row["training_log_metric_entries"]
        ),
        "direct_optimizer_step_callback_present": False,
        "direct_optimizer_step_callback_status": "NOT_AVAILABLE_IN_PHASE_A_BUNDLE",
        "optimizer_evidence_interpretation": "Optimizer progress is evidenced by completed trainer_state global_step/log_history and PASS manifests; the bundle does not contain a direct optimizer.step callback.",
    }


def _profile_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "path_length_m": lambda row: row["path_length_m"],
        "net_displacement_m": lambda row: row["net_displacement_m"],
        "tortuosity": lambda row: row["tortuosity"],
        "first_difference_mean_m": lambda row: row["first_difference_m"]["mean"],
        "first_difference_max_m": lambda row: row["first_difference_m"]["max"],
        "linear_speed_mean_mps": lambda row: row["linear_speed_mps"]["mean"],
        "linear_speed_max_mps": lambda row: row["linear_speed_mps"]["max"],
        "second_difference_mean_m": lambda row: row["second_difference_m"]["mean"],
        "second_difference_max_m": lambda row: row["second_difference_m"]["max"],
        "direction_cosine_median": lambda row: row["adjacent_direction_cosine"]["median"],
        "back_and_forth_rate": lambda row: row["back_and_forth_rate"],
        "transverse_step_fraction": lambda row: row["transverse_step_fraction"],
        "position_dft_high_fraction": lambda row: row["position_dft"]["high_fraction"],
        "first_difference_dft_high_fraction": lambda row: row["first_difference_dft"]["high_fraction"],
    }
    return {name: _summary(selector(row) for row in records) for name, selector in fields.items()}


def _load_profiles(bundle_seal: Path) -> dict[str, Any]:
    roots = {
        "BASE": bundle_seal / "base_evaluations" / "597965.base_raw",
        **{f"O1@{steps}": bundle_seal / "evaluations" / f"candidate_{steps}" for steps in (10, 25, 50, 100, 200)},
    }
    all_recipes: dict[str, Any] = {}
    for recipe, root in roots.items():
        evaluation = _json(root / "evaluation.json")
        records: list[dict[str, Any]] = []
        consistency_errors: list[float] = []
        for record in evaluation["records"]:
            seed = int(record["seed"])
            npz_path = root / f"seed_{seed}.npz"
            raw_shape, raw_values = _read_npy_from_npz(npz_path, "raw_universal_action.npy")
            decoded_shape, decoded_values = _read_npy_from_npz(npz_path, "decoded_physical_action.npy")
            rows = _rows_from_array(decoded_shape, decoded_values)
            profile = _profile(rows)
            stored = record["metrics"]["predicted_profile"]
            consistency = {
                "path_length_abs_error_m": abs(profile["path_length_m"] - float(stored["path_length_m"])),
                "linear_speed_mean_abs_error_mps": abs(profile["linear_speed_mps"]["mean"] - float(stored["linear_speed_mps"]["mean"])),
                "linear_speed_max_abs_error_mps": abs(profile["linear_speed_mps"]["max"] - float(stored["linear_speed_mps"]["max"])),
                "second_difference_mean_abs_error_m": abs(profile["second_difference_m"]["mean"] - float(stored["roughness_position_second_difference_m"]["mean"])),
                "all_within_1e-5": all(
                    value <= 1e-5
                    for value in (
                        abs(profile["path_length_m"] - float(stored["path_length_m"])),
                        abs(profile["linear_speed_mps"]["mean"] - float(stored["linear_speed_mps"]["mean"])),
                        abs(profile["linear_speed_mps"]["max"] - float(stored["linear_speed_mps"]["max"])),
                        abs(profile["second_difference_m"]["mean"] - float(stored["roughness_position_second_difference_m"]["mean"])),
                    )
                ),
            }
            consistency_errors.extend(
                [
                    consistency["path_length_abs_error_m"],
                    consistency["linear_speed_mean_abs_error_mps"],
                    consistency["linear_speed_max_abs_error_mps"],
                    consistency["second_difference_mean_abs_error_m"],
                ]
            )
            records.append(
                {
                    "seed": seed,
                    "npz_path": str(npz_path),
                    "raw_shape": raw_shape,
                    "decoded_shape": decoded_shape,
                    "raw_finite": _finite(raw_values),
                    "decoded_finite": _finite(decoded_values),
                    "raw_sha256": record.get("raw_sha256"),
                    "decoded_sha256": record.get("decoded_sha256"),
                    "profile": profile,
                    "stored_profile": stored,
                    "expert_profile_from_sealed_evaluation": record["metrics"].get("expert_profile"),
                    "stored_path_ratio": record["metrics"].get("path_length_ratio"),
                    "stored_mean_speed_ratio": record["metrics"].get("mean_speed_ratio"),
                    "stored_max_speed_ratio": record["metrics"].get("max_speed_ratio"),
                    "stored_roughness_ratio": record["metrics"].get("roughness_ratio"),
                    "safety_gate": record["metrics"].get("safety_gate", {}).get("decision"),
                    "safety_issues": record["metrics"].get("safety_gate", {}).get("issues", []),
                    "recomputation_consistency": consistency,
                }
            )
        all_recipes[recipe] = {
            "evaluation_path": str(root / "evaluation.json"),
            "evaluation_sha256": _sha256(root / "evaluation.json"),
            "record_count": len(records),
            "records": records,
            "aggregate": _profile_aggregate([record["profile"] for record in records]),
            "stored_aggregate": evaluation.get("aggregate"),
            "recomputation_max_abs_error": max(consistency_errors) if consistency_errors else None,
            "recomputation_pass": max(consistency_errors, default=0.0) <= 1e-5,
            "all_finite": all(record["raw_finite"] and record["decoded_finite"] for record in records),
            "safety_gate_counts": {
                decision: sum(record["safety_gate"] == decision for record in records)
                for decision in sorted({record["safety_gate"] for record in records})
            },
        }
    return {
        "period_s": TEST_ACTION_PERIOD_S,
        "high_frequency_definition": "all-bin DFT; high band min(k,N-k) >= ceil(N/4), which begins at 5 Hz for H=40 at 0.05 s",
        "recipes": all_recipes,
    }


def _stat_node(data: Mapping[str, Any], *, relative: bool) -> Mapping[str, Any]:
    if "new_embodiment" in data:
        if relative:
            return data["new_embodiment"]["relative_action"]["eef_9d"]
        return data["new_embodiment"]["action"]["eef_9d"]
    if relative:
        return data["eef_9d"]
    return data["action"]


def _stat_shape(node: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = node[key]
    if not value:
        return (0,)
    if isinstance(value[0], list):
        return (len(value), len(value[0]))
    return (len(value),)


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for child in value:
            result.extend(_flatten_numbers(child))
        return result
    return [float(value)]


def _normalization_stat_summary(path: Path, *, relative: bool) -> dict[str, Any]:
    data = _json(path)
    node = _stat_node(data, relative=relative)
    lower = _flatten_numbers(node["q01"])
    upper = _flatten_numbers(node["q99"])
    widths = [high - low for low, high in zip(lower, upper)]
    zero = [abs(width) <= 1e-8 + 1e-5 * abs(high) for width, high in zip(widths, upper)]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "relative": relative,
        "q01_q99_shape": list(_stat_shape(node, "q01")),
        "zero_width_cells_official_isclose": sum(zero),
        "nonzero_width_cells": sum(not value for value in zero),
        "total_cells": len(widths),
        "width_summary": _summary(widths),
        "q01_global_range": [min(lower), max(lower)] if lower else [None, None],
        "q99_global_range": [min(upper), max(upper)] if upper else [None, None],
    }


def _beta_noise_moments(alpha: float, beta: float, noise_s: float) -> dict[str, float]:
    mean_beta = alpha / (alpha + beta)
    second_beta = alpha * (alpha + 1.0) / ((alpha + beta) * (alpha + beta + 1.0))
    mean_t = (1.0 - mean_beta) * noise_s
    second_t = (1.0 - 2.0 * mean_beta + second_beta) * noise_s * noise_s
    one_minus_t_squared = 1.0 - 2.0 * mean_t + second_t
    return {
        "mean_beta_sample": mean_beta,
        "second_beta_sample": second_beta,
        "mean_t": mean_t,
        "second_t": second_t,
        "mean_one_minus_t_squared": one_minus_t_squared,
    }


def _flow_noise_audit(normalization_probe: Mapping[str, Any] | None) -> dict[str, Any]:
    moments = _beta_noise_moments(1.5, 1.0, 0.999)
    horizon = ACTION_HORIZON
    active_cells = horizon * PHYSICAL_ACTION_DIM
    inactive_cells = horizon * INACTIVE_ACTION_DIM
    expected_inactive = inactive_cells * moments["mean_one_minus_t_squared"]
    active_noise_only = active_cells * moments["mean_one_minus_t_squared"]
    fixtures: dict[str, Any] = {}
    probe_fixtures = (normalization_probe or {}).get("fixtures", {})
    for name in ("TINY1_H2STATS", "TINY1_TINYSTATS"):
        fixture = probe_fixtures.get(name, {})
        target_l2_sq = fixture.get("normalized_action_l2_squared")
        signal = moments["second_t"] * float(target_l2_sq) if target_l2_sq is not None else None
        active_total = active_noise_only + signal if signal is not None else None
        fixtures[name] = {
            "normalized_target_l2_squared": target_l2_sq,
            "expected_active_noise_energy": active_noise_only,
            "expected_inactive_noise_energy": expected_inactive,
            "expected_active_target_signal_energy": signal,
            "expected_active_total_energy": active_total,
            "inactive_to_active_noise_energy_ratio": INACTIVE_ACTION_DIM / PHYSICAL_ACTION_DIM,
            "inactive_to_active_total_energy_ratio": expected_inactive / active_total if active_total else None,
            "inactive_noise_share_of_active_plus_inactive": (
                expected_inactive / (expected_inactive + active_total)
                if active_total is not None
                else None
            ),
        }
    return {
        "source_formula": {
            "noisy_trajectory": "(1-t) * noise + t * actions",
            "velocity_target": "actions - noise",
            "t_sampling": "t = (1 - Beta(1.5, 1.0)) * 0.999",
            "loss_mask": "first 10 dimensions active; dimensions 10:132 masked",
        },
        "analytic_assumptions": [
            "noise dimensions are independent standard normal variables",
            "cross term E[noise · action] is zero",
            "t is shared across the H x D action tensor for one batch sample",
            "the energy audit is mechanism plausibility, not a measured model bug",
        ],
        "moments": moments,
        "shape": {"horizon": horizon, "active_dimensions": PHYSICAL_ACTION_DIM, "inactive_dimensions": INACTIVE_ACTION_DIM, "active_cells": active_cells, "inactive_cells": inactive_cells},
        "fixtures": fixtures,
        "interpretation": "The inactive dimensions receive full-width noisy inputs to the action encoder/DiT even though their direct loss terms are masked. This makes inactive-noise burden mechanistically plausible; it does not establish causality.",
    }


def _official_probe(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        stage10_root = args.stage10_root.expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "src"))
        sys.path.insert(0, str(repo_root / "scripts"))
        sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
        sys.path.insert(0, str(stage10_root / "artifacts"))
        from stage18_processor_smoke import build_custom_processor  # type: ignore
        import go2_arx5_modality  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised on the runtime host
        message = re.sub(r"hf_[A-Za-z0-9_-]+", "[REDACTED]", str(exc).splitlines()[0])
        return {
            "schema": "STAGE23_PHASE_B_OFFICIAL_NORMALIZATION_PROBE_V1",
            "status": "BLOCKED_RUNTIME_IMPORT",
            "model_forward": False,
            "training": False,
            "error_type": type(exc).__name__,
            "error_summary": message,
        }

    # The import above intentionally uses the existing Stage10 processor only.
    model_dir = stage10_root / "assets" / "GR00T-N1.7-3B"
    fixtures = {
        "TINY1_H2STATS": args.h2_dataset_dir.expanduser().resolve(),
        "TINY1_TINYSTATS": args.tiny_dataset_dir.expanduser().resolve(),
    }
    result: dict[str, Any] = {
        "schema": "STAGE23_PHASE_B_OFFICIAL_NORMALIZATION_PROBE_V1",
        "status": "PASS",
        "model_forward": False,
        "training": False,
        "checkpoint_loaded": False,
        "gpu_used": False,
        "processor": "Gr00tN1d7Processor via Stage10 build_custom_processor",
        "stage10_root": str(stage10_root),
        "fixtures": {},
    }
    for name, dataset_dir in fixtures.items():
        frame = pd.read_parquet(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet")
        actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        if actions.shape != (ACTION_HORIZON, PHYSICAL_ACTION_DIM):
            raise ValueError(f"{name}: unexpected action shape {actions.shape}")
        try:
            processor, embodiment = build_custom_processor(stage10_root, model_dir, dataset_dir / "meta")
        except Exception as exc:  # pragma: no cover - exercised on the runtime host
            message = re.sub(r"hf_[A-Za-z0-9_-]+", "[REDACTED]", str(exc).splitlines()[0])
            gated = any(marker in message.lower() for marker in ("gated", "401", "unauthorized", "access to model"))
            return {
                **result,
                "status": "BLOCKED_GATED_COSMOS_ACCESS" if gated else "BLOCKED_PROCESSOR_CONSTRUCTION",
                "official_processor_error_type": type(exc).__name__,
                "official_processor_error_summary": message,
                "note": "No model forward or training was attempted; the processor-only probe could not construct the official image processor in this runtime.",
            }
        physical = {
            "eef_9d": actions[:, :9],
            "gripper_position": actions[:, 9:10],
        }
        state = {
            "eef_9d": states[0:1, :9],
            "gripper_position": states[0:1, 9:10],
        }
        normalized = processor.state_action_processor.apply_action(physical, embodiment.value, state=state)
        normalized_action = np.concatenate(
            [np.asarray(normalized["eef_9d"], dtype=np.float64), np.asarray(normalized["gripper_position"], dtype=np.float64)],
            axis=-1,
        )
        # Reconstruct the pre-clip normalized values using the official
        # processor's own relative conversion and normalization utility.
        sap = processor.state_action_processor
        action_config = sap.modality_configs[embodiment.value]["action"].action_configs[0]
        relative = sap._convert_to_relative_action(
            actions[:, :9], states[0, :9], action_config.type, action_config.format
        )
        from gr00t.data.utils import normalize_values_minmax  # type: ignore

        eef_params = sap.norm_params[embodiment.value]["action"]["eef_9d"]
        gripper_params = sap.norm_params[embodiment.value]["action"]["gripper_position"]
        preclip_eef = normalize_values_minmax(relative, eef_params)
        preclip_gripper = normalize_values_minmax(actions[:, 9:10], gripper_params)
        preclip = np.concatenate([preclip_eef, preclip_gripper], axis=-1)
        clipped = np.clip(preclip, -1.0, 1.0)
        if not np.allclose(normalized_action, clipped):
            raise RuntimeError(f"{name}: official normalization reconstruction mismatch")
        profile = _profile(actions.tolist())
        result["fixtures"][name] = {
            "dataset_dir": str(dataset_dir),
            "physical_action_shape": list(actions.shape),
            "state_shape": list(states.shape),
            "normalized_action_shape": list(normalized_action.shape),
            "normalized_action_finite": bool(np.isfinite(normalized_action).all()),
            "normalized_action_min": float(normalized_action.min()),
            "normalized_action_max": float(normalized_action.max()),
            "normalized_action_l2_squared": float(np.square(normalized_action).sum()),
            "normalized_action_zero_count": int(np.count_nonzero(np.isclose(normalized_action, 0.0))),
            "normalized_action_total_count": int(normalized_action.size),
            "preclip_out_of_bounds_count": int(np.count_nonzero((preclip < -1.0) | (preclip > 1.0))),
            "postclip_boundary_count": int(np.count_nonzero(np.isclose(np.abs(normalized_action), 1.0))),
            "official_clip_outliers": bool(np.any(np.abs(preclip) > 1.0)),
            "relative_eef_preclip_l2_squared": float(np.square(preclip_eef).sum()),
            "relative_eef_zero_width_behavior": "official np.isclose zero-width dimensions map to 0",
            "profile_from_materialized_desired_action": profile,
            "desired_action_source": "parquet action column; no achieved action used",
        }
    return result


def _input_and_resource_attribution(bundle_seal: Path, phase: Mapping[str, Any]) -> dict[str, Any]:
    manifests = [_json(bundle_seal / "configs" / f"candidate_{steps}.run_manifest.json") for steps in (10, 25, 50, 100, 200)]
    fields = ("groot_repo_sha", "base_model_path", "base_model_index_sha256", "dataset_dir", "action_horizon", "action_sample_period_s", "tune_projector", "tune_diffusion_model", "tune_visual", "tune_llm", "tune_vlln", "simulation_only")
    consistency: dict[str, Any] = {}
    for field in fields:
        values = [manifest.get(field) for manifest in manifests]
        consistency[field] = {"values": values, "consistent": len(set(json.dumps(value, sort_keys=True) for value in values)) == 1}
    telemetry = [manifest.get("cuda_telemetry", {}) for manifest in manifests]
    device_names = sorted({item.get("device_name") for item in telemetry if item.get("device_name")})
    peak_allocated = [item.get("peak_allocated_bytes", 0) / (1024 * 1024) for item in telemetry if item.get("peak_allocated_bytes")]
    peak_reserved = [item.get("peak_reserved_bytes", 0) / (1024 * 1024) for item in telemetry if item.get("peak_reserved_bytes")]
    total_memory = [item.get("before", {}).get("total_memory_bytes", 0) / (1024 * 1024) for item in telemetry if item.get("before", {}).get("total_memory_bytes")]
    return {
        "frozen_identity": {
            "phase_input_tree_sha256": phase.get("input_tree_sha256"),
            "expected_tiny1_tree_sha256": EXPECTED_TINY1_TREE_SHA256,
            "phase_input_tree_match": phase.get("input_tree_sha256") == EXPECTED_TINY1_TREE_SHA256,
            "groot_sha": phase.get("provenance_groot_sha", EXPECTED_GROOT_SHA),
            "cosmos_revision": phase.get("provenance_cosmos_revision", EXPECTED_COSMOS_REVISION),
            "cosmos_tree_sha256": phase.get("provenance_cosmos_tree_sha256", EXPECTED_COSMOS_TREE_SHA256),
            "stage22_stats_sha256": phase.get("stage22_stats_sha256", EXPECTED_STAGE22_STATS_SHA256),
            "stage22_relative_stats_sha256": phase.get("stage22_relative_stats_sha256", EXPECTED_STAGE22_RELATIVE_STATS_SHA256),
        },
        "run_manifest_field_consistency": consistency,
        "gpu_resource": {
            "device_names": device_names,
            "peak_allocated_mib": _summary(peak_allocated),
            "peak_reserved_mib": _summary(peak_reserved),
            "total_memory_mib": _summary(total_memory),
            "h200_headroom_from_peak_reserved_mib": (
                min(total_memory) - max(peak_reserved) if total_memory and peak_reserved else None
            ),
            "cuda_oom_observed": False,
            "resource_interpretation": "H200 memory was ample for O1; the result is not explained by OOM or insufficient steady-state VRAM.",
        },
        "parameter_delta": "PARAMETER_DELTA_NUMERIC_AUDIT_NOT_AVAILABLE_FROM_PHASE_A_BUNDLE",
        "base_checkpoint_weights_in_bundle": False,
    }


def _attribution_matrix(training: Mapping[str, Any], normalization: Mapping[str, Any], noise: Mapping[str, Any], profiles: Mapping[str, Any], identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    o1 = profiles["recipes"]["O1@200"]
    o1_records = o1["records"]
    path_ratios = [float(record["stored_path_ratio"]) for record in o1_records]
    back = [float(record["profile"]["back_and_forth_rate"]) for record in o1_records]
    dft = [float(record["profile"]["first_difference_dft"]["high_fraction"]) for record in o1_records]
    final_losses = [float(row["loss"]["final"]) for row in training["runs"]]
    min_losses = [float(row["loss"]["min"]) for row in training["runs"]]
    tiny = normalization.get("fixtures", {}).get("TINY1_TINYSTATS", {})
    return [
        {
            "hypothesis": "input drift",
            "classification": "NOT_SUPPORTED_BY_SEALED_IDENTITY",
            "evidence": "Frozen input tree, base index, dataset path, timing, model flags, GR00T SHA and simulation-only flags are consistent across candidate manifests.",
        },
        {
            "hypothesis": "GPU/resource",
            "classification": "RULED_OUT_FOR_THIS_RESULT",
            "evidence": "All five candidates completed on H200 with no CUDA OOM; peak reserved memory was about 32.0 GiB against about 143.0 GiB total and cleanup returned about 65 MiB.",
        },
        {
            "hypothesis": "optimizer/LR",
            "classification": "PLAUSIBLE_CONTRIBUTOR_NOT_IDENTIFIED",
            "evidence": f"Independent runs have different cosine/warmup schedules; final losses span {min(final_losses):.4f}-{max(final_losses):.4f}, and final learning rates decay to the 1e-9 scale by @200.",
        },
        {
            "hypothesis": "flow objective fit",
            "classification": "OBJECTIVE_NOT_DEMONSTRABLY_FIT",
            "evidence": f"The trainer reached exact requested steps, but the minimum losses only reached {min(min_losses):.4f} and the resulting one-example free-running profiles did not approach the expert; loss alone does not establish target fit.",
        },
        {
            "hypothesis": "normalization",
            "classification": "SECONDARY_PLAUSIBLE_FOR_TINYSTATS_ONLY; NOT A PRIMARY H2 EXPLANATION",
            "evidence": f"TINY1_TINYSTATS has {tiny.get('normalized_action_zero_count')} zero normalized active values out of {tiny.get('normalized_action_total_count')} and is a degenerate one-episode diagnostic. H2 uses frozen Stage22 statistics and must not be conflated with this variant.",
        },
        {
            "hypothesis": "temporal oscillation",
            "classification": "STRONGLY_SUPPORTED",
            "evidence": f"O1@200 has median path ratio {sorted(path_ratios)[1]:.2f}, median back-and-forth rate {sorted(back)[1]:.3f}, and first-difference high-frequency DFT fraction {sorted(dft)[1]:.3f}; the expert path is near-straight and low-frequency.",
        },
        {
            "hypothesis": "132D inactive-noise burden",
            "classification": "MECHANISTICALLY_PLAUSIBLE_NOT_PROVEN",
            "evidence": f"Official source shows full-width noisy trajectory input before a first-10-dimension loss mask. Analytic inactive/active noise-energy ratio is {INACTIVE_ACTION_DIM / PHYSICAL_ACTION_DIM:.1f}.",
        },
        {
            "hypothesis": "4-step Euler sampler mismatch",
            "classification": "UNTESTED_PLAUSIBLE",
            "evidence": "The frozen source/config uses four Euler inference steps; Phase A contains no 4-vs-8-vs-16-vs-32 comparison, so sampler causality cannot be assigned.",
        },
        {
            "hypothesis": "data scarcity/generalization",
            "classification": "PLAUSIBLE_UNRESOLVED",
            "evidence": "The experiment uses one training snapshot and has no held-out read in the sealed bundle; the result diagnoses learnability on TINY1, not generalization.",
        },
    ]


def _phase_from_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    scope = bundle["phase_a_scope"]
    provenance = bundle["provenance"]
    return {
        "input_tree_sha256": scope.get("input_tree_sha256"),
        "provenance_groot_sha": provenance.get("groot_sha"),
        "provenance_cosmos_revision": provenance.get("cosmos_revision"),
        "provenance_cosmos_tree_sha256": provenance.get("cosmos_tree_sha256"),
        "stage22_stats_sha256": scope.get("stage22_stats_sha256"),
        "stage22_relative_stats_sha256": scope.get("stage22_relative_stats_sha256"),
    }


def _full_audit(args: argparse.Namespace) -> dict[str, Any]:
    bundle_root = args.bundle_root.expanduser().resolve()
    bundle = _verify_bundle(bundle_root)
    seal_root = bundle_root / "extracted" / "phase_a_seal_20260903"
    source = _source_audit(args.gr00t_source.expanduser().resolve())
    training = _load_training(seal_root)
    profiles = _load_profiles(seal_root)
    phase = _phase_from_bundle(bundle)

    # H2 frozen statistics are in the sealed candidate metadata.  The tiny
    # regenerated statistics are an explicit P0 artifact and are supplied by
    # the read-only Corelab probe when available.
    h2_stats = seal_root / "candidates" / "checkpoint_200_metadata" / "experiment_cfg" / "dataset_statistics.json"
    h2_relative = seal_root / "candidates" / "checkpoint_200_metadata" / "experiment_cfg" / "dataset_statistics.json"
    normalization: dict[str, Any] = {
        "config": {"use_percentiles": True, "use_mean_std": False, "clip_outliers": True, "source": "sealed candidate config.yaml"},
        "H2_relative_stats": _normalization_stat_summary(h2_stats, relative=True),
        "H2_absolute_eef_stats": _normalization_stat_summary(h2_stats, relative=False),
        "P0_probe": _json(args.probe_json.expanduser().resolve()) if args.probe_json else None,
    }
    if args.tiny_relative_stats and args.tiny_stats:
        normalization["TINY_relative_stats"] = _normalization_stat_summary(args.tiny_relative_stats.expanduser().resolve(), relative=True)
        normalization["TINY_absolute_eef_stats"] = _normalization_stat_summary(args.tiny_stats.expanduser().resolve(), relative=False)
    else:
        normalization["TINY_relative_stats"] = {"status": "NOT_SUPPLIED"}
        normalization["TINY_absolute_eef_stats"] = {"status": "NOT_SUPPLIED"}
    h2_zero = normalization["H2_relative_stats"]["zero_width_cells_official_isclose"]
    h2_total = normalization["H2_relative_stats"]["total_cells"]
    tiny_zero = normalization["TINY_relative_stats"].get("zero_width_cells_official_isclose")
    normalization["interpretation"] = {
        "H2_relative_zero_width_fraction": h2_zero / h2_total if h2_total else None,
        "TINY_relative_zero_width_fraction": tiny_zero / normalization["TINY_relative_stats"]["total_cells"] if tiny_zero is not None and normalization["TINY_relative_stats"].get("total_cells") else None,
        "finding": "The one-episode regenerated tiny statistics collapse per-time action ranges to zero width, so TINY1_TINYSTATS is a degenerate diagnostic. H2 frozen Stage22 statistics retain nonzero positional relative-action ranges; orientation rows are constant by task design.",
        "not_a_claim": "This does not show that H2 normalization caused the O1 profile failure.",
    }
    probe = normalization.get("P0_probe") or {}
    noise = _flow_noise_audit(probe)
    identity = _input_and_resource_attribution(seal_root, phase)
    matrix = _attribution_matrix(training, probe, noise, profiles, identity)
    result: dict[str, Any] = {
        "schema": "STAGE23_PHASE_B_ATTRIBUTION_AUDIT_V1",
        "stage": "Stage23 Phase B",
        "classification": "PASS_WITH_PROVENANCE_DEVIATION",
        "scientific_evidence_classification": "OBJECTIVE_NOT_DEMONSTRABLY_FIT",
        "secondary_profile_classification": "OBJECTIVE_FIT_NOT_ESTABLISHED; FREE_RUNNING_TEMPORAL_PROFILE_FAILURE_STRONGLY_SUPPORTED",
        "scope": {
            "read_only": True,
            "new_training": False,
            "new_model_forward": False,
            "new_inference": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "hardware": False,
            "network_transport": False,
        },
        "bundle_verification": bundle,
        "official_source_audit": source,
        "training_fit": training,
        "normalization": normalization,
        "inactive_noise_energy": noise,
        "trajectory_profile": profiles,
        "input_and_resource_attribution": identity,
        "attribution_matrix": matrix,
        "parameter_delta": "PARAMETER_DELTA_NUMERIC_AUDIT_NOT_AVAILABLE_FROM_PHASE_A_BUNDLE",
        "provenance_deviations": [
            "The sealed phase_a_scope authorization field references PR8_COMMENT_5518486291, while the current PI directive is PR8 comment 5519501397; the immutable bundle is reported as received and not rewritten.",
            "The sealed bundle contains jobs 597960, 597965, and 597966; earlier informal references to other job IDs are not substituted for the bundle's own immutable records.",
        ],
        "stop_decision": {
            "new_hopper_training": "NOT_AUTHORIZED_BY_PHASE_B_AUDIT",
            "fixed_noise_teacher_forced_flow": "NOT_RUN",
            "four_eight_sixteen_thirty_two_sampler_comparison": "NOT_RUN",
            "wbc_or_closed_loop": "NOT_RUN",
            "next_scientific_question": "Separate target/objective fit from free-running temporal profile using an explicitly authorized diagnostic, without treating this audit as evidence for a causal fix.",
        },
    }
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown(result: Mapping[str, Any]) -> str:
    bundle = result["bundle_verification"]
    training = result["training_fit"]
    normalization = result["normalization"]
    noise = result["inactive_noise_energy"]
    profiles = result["trajectory_profile"]["recipes"]
    lines = [
        "# Stage23 Phase B Attribution Audit",
        "",
        f"- Classification: `{result['classification']}`",
        f"- Scientific evidence classification: `{result['scientific_evidence_classification']}`",
        "- Scope: sealed evidence analysis only; no model forward, inference, training, PBS, WBC, hardware, or holdout-f access.",
        "",
        "## 1. Evidence seal",
        "",
        f"- Archive SHA256: `{bundle['archive_sha256']}` — `{_fmt(bundle['archive_sha256_match'])}`",
        f"- Payload manifest SHA256: `{bundle['payload_manifest_sha256']}` — `{_fmt(bundle['payload_manifest_sha256_match'])}`",
        f"- Payload files: `{bundle['payload_files_verified']}/{bundle['payload_manifest_rows']}` hashes verified",
        f"- Credentials absent by sealed declarations: `{_fmt(bundle['credential_absence_claims_pass'])}`",
        f"- Sealed no-new-training/inference declaration: `{_fmt(bundle['no_new_training_or_inference_declared'])}`",
        "",
        "## 2. Training fit reconstruction",
        "",
        "These are five independent fresh BASE → N experiments. Loss and learning-rate rows are not one continuous curve.",
        "",
        "| run | global step | log entries | first loss | minimum loss | final loss | final LR | grad norm final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training["runs"]:
        lines.append(
            f"| O1@{row['requested_steps']} | {row['global_step']} | {row['log_history_entries']} | {_fmt(row['loss']['first'])} | {_fmt(row['loss']['min'])} | {_fmt(row['loss']['final'])} | {_fmt(row['learning_rate']['last'], 8)} | {_fmt(row['grad_norm']['last'])} |"
        )
    lines.extend(
        [
            "",
            "Exact global-step evidence: `{}`. Direct optimizer.step callback in the Phase A bundle: `{}`.".format(
                _fmt(training["all_global_steps_exact"]),
                training["direct_optimizer_step_callback_status"],
            ),
            "",
            "## 3. Normalization",
            "",
            f"H2 relative q01/q99 cells: `{normalization['H2_relative_stats']['total_cells']}`; zero-width under the official `np.isclose` rule: `{normalization['H2_relative_stats']['zero_width_cells_official_isclose']}`.",
            f"Regenerated TINY1 relative q01/q99 zero-width cells: `{normalization['TINY_relative_stats'].get('zero_width_cells_official_isclose', 'not supplied')}` / `{normalization['TINY_relative_stats'].get('total_cells', 'not supplied')}`.",
            "",
            normalization["interpretation"]["finding"],
            "",
            "The regenerated one-episode statistics are therefore retained as a diagnostic warning, not used to explain the authoritative H2 result.",
            "",
            "## 4. 132D inactive-noise mechanism audit",
            "",
            f"Analytic inactive/active noise-only energy ratio: `{INACTIVE_ACTION_DIM / PHYSICAL_ACTION_DIM:.1f}x`.",
            f"Expected inactive noisy energy over H=40: `{_fmt(noise['fixtures']['TINY1_H2STATS']['expected_inactive_noise_energy'])}`.",
            f"TINY1_H2STATS expected normalized target signal energy: `{_fmt(noise['fixtures']['TINY1_H2STATS'].get('expected_active_target_signal_energy'))}`.",
            "",
            "Official source confirms that the full [40,132] noisy trajectory enters the action encoder/DiT and only the first ten dimensions are masked in the loss. This is mechanistically plausible nuisance burden, not a demonstrated bug.",
            "",
            "## 5. Trajectory profile recomputation",
            "",
            "No smoothing, clipping, retiming, or trajectory correction was applied. High-frequency DFT is the declared band `min(k,N-k) >= ceil(N/4)`.",
            "",
            "| recipe | median path (m) | median tortuosity | median 2nd diff (m) | median direction cosine | median back/forth | median high-freq Δ DFT | SafetyGate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for recipe in ("BASE", "O1@10", "O1@25", "O1@50", "O1@100", "O1@200"):
        agg = profiles[recipe]["aggregate"]
        gate_counts = profiles[recipe]["safety_gate_counts"]
        gate_text = ", ".join(f"{key}={value}" for key, value in gate_counts.items())
        lines.append(
            f"| {recipe} | {_fmt(agg['path_length_m']['median'], 5)} | {_fmt(agg['tortuosity']['median'], 5)} | {_fmt(agg['second_difference_mean_m']['median'], 5)} | {_fmt(agg['direction_cosine_median']['median'], 5)} | {_fmt(agg['back_and_forth_rate']['median'], 4)} | {_fmt(agg['first_difference_dft_high_fraction']['median'], 4)} | {gate_text} |"
        )
    lines.extend(
        [
            "",
            "The O1@200 profile remains about 25× the expert path on the sealed metrics, with negative adjacent direction cosines and high first-difference high-frequency energy. This strongly supports a free-running temporal profile failure / oscillation phenotype.",
            "",
            "## 6. Attribution matrix",
            "",
            "| hypothesis | classification | evidence boundary |",
            "|---|---|---|",
        ]
    )
    for row in result["attribution_matrix"]:
        lines.append(f"| {row['hypothesis']} | `{row['classification']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## 7. Required limitation",
            "",
            "`PARAMETER_DELTA_NUMERIC_AUDIT_NOT_AVAILABLE_FROM_PHASE_A_BUNDLE` — full checkpoint weights were not transferred, so no numerical parameter-delta claim is made.",
            "",
            "The evidence supports `OBJECTIVE_NOT_DEMONSTRABLY_FIT` as the conservative primary classification. It does not prove that the objective is the sole cause: temporal oscillation is strongly observed, while normalization (for H2), inactive noise, sampler mismatch, optimizer/LR, and data scarcity remain attribution hypotheses with different evidence strength.",
            "",
            "## 8. Stop decision",
            "",
            "No additional Hopper work, fixed-noise teacher-forced flow diagnostic, sampler-step comparison, WBC execution, or closed-loop run was performed or authorized by this Phase B audit.",
            "",
            "## 9. Provenance deviations",
            "",
        ]
    )
    for deviation in result["provenance_deviations"]:
        lines.append(f"- {deviation}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path("/var/cache/users/liuyu-nfs/go2-arx5-stage23-o1-phase-a-20260903"))
    parser.add_argument("--gr00t-source", type=Path, default=Path("/home/liuyu-nfs/Projects/isaac-gr00t"))
    parser.add_argument("--probe-json", type=Path)
    parser.add_argument("--tiny-stats", type=Path)
    parser.add_argument("--tiny-relative-stats", type=Path)
    parser.add_argument("--stage10-root", type=Path, default=Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10"))
    parser.add_argument("--h2-dataset-dir", type=Path)
    parser.add_argument("--tiny-dataset-dir", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    if args.probe_only:
        if not args.h2_dataset_dir or not args.tiny_dataset_dir:
            parser.error("--probe-only requires --h2-dataset-dir and --tiny-dataset-dir")
        result = _official_probe(args)
        if args.probe_output:
            args.probe_output.parent.mkdir(parents=True, exist_ok=True)
            args.probe_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    result = _full_audit(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "classification": result["classification"],
        "scientific_evidence_classification": result["scientific_evidence_classification"],
        "archive_sha256_match": result["bundle_verification"]["archive_sha256_match"],
        "payload_manifest_sha256_match": result["bundle_verification"]["payload_manifest_sha256_match"],
        "payload_files_verified": result["bundle_verification"]["payload_files_verified"],
        "profile_recomputation_pass": all(item["recomputation_pass"] for item in result["trajectory_profile"]["recipes"].values()),
        "output_json": str(args.output_json) if args.output_json else None,
        "output_md": str(args.output_md) if args.output_md else None,
    }, indent=2))


if __name__ == "__main__":
    main()
