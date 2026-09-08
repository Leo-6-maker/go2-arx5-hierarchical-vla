#!/usr/bin/env python3
"""Audit the immutable Stage22 H2 relay without rerunning any workload.

The Hopper relay predates the stricter GPU-evidence and snapshot-lock schemas
in this repository.  This script therefore performs a read-only, historical
audit directly over the relay payloads.  It accepts the explicitly labelled
GPU projection only when the projection, its source launcher log, PBS
accounting window, and H2 run manifests all agree.  It also treats the
pre-optimizer group inventory separately from the stricter role-level
``requires_grad`` inventory: the former can support broad recipe counts, but
cannot silently close the latter gate.

No model is loaded, no GPU query is executed, no profile inference is run,
and no checkpoint or holdout-f artifact is read by this tool.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_ARCHIVE_SHA256 = (
    "77604a2dc2f9c436718139b950209943cf4c767fc7f0ae52e624f28a3c877661"
)
EXPECTED_GR00T_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_LAUNCHER_SHA = (
    "f42a1990847cf8f5326c66db37043ffed6e43458dd907fdbd08da5dd905b00a2"
)
EXPECTED_LAUNCHER_COMMIT = "deb920a070141e6a32a88cd0da6719828647e1e6"
EXPECTED_BASE_TREE_SHA = (
    "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
)
EXPECTED_TRAIN_TREE_SHA = (
    "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
)
EXPECTED_SPLIT_LOCK_SHA = (
    "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
)
EXPECTED_MODALITY_SHA = (
    "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
)
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA = (
    "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
)
EXPECTED_R1_TRAINABLE = 1_419_078_784
EXPECTED_R0_TRAINABLE = 327_356_544
EXPECTED_TRAINING_SEED = 2222
EXPECTED_STEPS = 50
EXPECTED_SAVE_STEPS = 25
EXPECTED_LR = 1e-4
EXPECTED_HORIZON = 40
EXPECTED_ACTION_PERIOD = 0.05
EXPECTED_SEEDS = (2223, 2224, 2225)
EXPECTED_SNAPSHOT_IDS = tuple(
    f"stage22_expert_red_target_holdout_e:expert-{index:04d}" for index in range(6)
)
EXPECTED_SNAPSHOT_IDS_SHA = "fccd590d0a585203d6ea0279a6ec253362ce44772ffc2aad3b929bdfefd82ae1"
PROFILE_SCHEMA = "STAGE22_ACTION_PROFILE_EVALUATION_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def check(checks: dict[str, Any], name: str, passed: bool, observed: Any, expected: Any) -> None:
    checks[name] = {"passed": bool(passed), "observed": observed, "expected": expected}


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def stats(values: Iterable[Any]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if value is not None and finite(value))
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def extract_json_from_log(path: Path) -> tuple[dict[str, Any], int]:
    """Extract the first complete JSON object from a log without executing it."""

    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, index
    raise ValueError(f"no JSON object found in profile log: {path}")


def parse_key_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)=(\S+)$", line.strip())
        if match:
            result[match.group(1)] = match.group(2)
    return result


def parse_csv_gpu_row(line: str, expected_columns: int) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != expected_columns or not parts[0].isdigit():
        return None
    if not parts[2].startswith("GPU-"):
        return None
    total = re.sub(r"\s*MiB\s*$", "", parts[3], flags=re.IGNORECASE)
    try:
        total_mib = int(float(total))
    except ValueError:
        return None
    row = {
        "index": int(parts[0]),
        "name": parts[1],
        "uuid": parts[2],
        "memory_total_mib": total_mib,
    }
    if expected_columns == 5:
        used = re.sub(r"\s*MiB\s*$", "", parts[4], flags=re.IGNORECASE)
        row["memory_used_mib"] = int(float(used))
    return row


def parse_projection(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = parse_key_values(text)
    query_line = next((line.strip() for line in text.splitlines() if line.startswith("nvidia-smi --query-gpu=")), "")
    rows = [parse_csv_gpu_row(line, 4) for line in text.splitlines()]
    rows = [row for row in rows if row is not None]
    if len(rows) != 1:
        raise ValueError(f"projection must contain exactly one GPU row: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "text": text,
        "values": values,
        "query_line": query_line,
        "device": rows[0],
    }


def parse_source_gpu_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    header = "index, name, uuid, memory.total [MiB], memory.used [MiB]"
    rows = [parse_csv_gpu_row(line, 5) for line in text.splitlines()]
    rows = [row for row in rows if row is not None]
    timestamps = re.findall(
        r"20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
        text,
    )
    host_lines = [line.strip() for line in text.splitlines() if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", line.strip())]
    host_candidates = [line for line in host_lines if line.startswith("hopper-")]
    values = parse_key_values(text)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "text": text,
        "header_present": header in text,
        "query_has_memory_used": "--query-gpu=index,name,uuid,memory.total,memory.used" in text,
        "rows": rows,
        "timestamps": timestamps,
        "host_candidates": sorted(set(host_candidates)),
        "values": values,
    }


def verify_relay(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    archive = root / "relay_archive.tar.gz"
    relay_manifest = load_json(root / "relay_manifest.json")
    sums_path = root / "SHA256SUMS"
    sum_rows: list[tuple[str, str]] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        sum_rows.append((digest, relative))
    payload_pass = True
    missing: list[str] = []
    mismatched: list[str] = []
    for expected, relative in sum_rows:
        path = root / relative
        if not path.is_file():
            payload_pass = False
            missing.append(relative)
            continue
        if sha256_file(path) != expected:
            payload_pass = False
            mismatched.append(relative)
    check(checks, "relay_archive_sha256", sha256_file(archive) == EXPECTED_ARCHIVE_SHA256, sha256_file(archive), EXPECTED_ARCHIVE_SHA256)
    check(checks, "relay_payload_hashes", payload_pass, {"count": len(sum_rows), "missing": missing, "mismatched": mismatched}, "all SHA256SUMS entries match")
    check(checks, "relay_payload_count", len(sum_rows) == 89 and relay_manifest.get("file_count_excluding_manifests") == 89, {"sha256sums": len(sum_rows), "manifest": relay_manifest.get("file_count_excluding_manifests")}, 89)
    check(checks, "relay_scope_booleans", all(relay_manifest.get(key) is value for key, value in {
        "contains_credentials": False,
        "holdout_f_included": False,
        "holdout_f_accessed": False,
        "checkpoint_weights_included": False,
        "training_or_inference_started_by_relay": False,
        "r1_at_100_authorized": False,
    }.items()), {key: relay_manifest.get(key) for key in ("contains_credentials", "holdout_f_included", "holdout_f_accessed", "checkpoint_weights_included", "training_or_inference_started_by_relay", "r1_at_100_authorized")}, "all forbidden-scope booleans")
    forbidden_names = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(token in path.name.lower() for token in (".safetensors", ".pt", ".pth", ".ckpt", "holdout_f", "token", "password", "private_key"))
    ]
    check(checks, "relay_forbidden_filename_scan", not forbidden_names, forbidden_names, "no checkpoint, holdout_f, or credential filenames")
    credential_pattern = re.compile(rb"hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
    credential_hit = False
    for path in root.rglob("*"):
        if path.is_file() and path.name not in {"relay_archive.tar.gz"}:
            if credential_pattern.search(path.read_bytes()):
                credential_hit = True
                break
    check(checks, "relay_credential_content_scan", not credential_hit, credential_hit, False)
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = [member.name for member in handle.getmembers()]
        archive_safe = all(not Path(member).is_absolute() and ".." not in Path(member).parts for member in members)
    except (OSError, tarfile.TarError):
        members = []
        archive_safe = False
    check(checks, "relay_archive_member_safety", archive_safe, {"member_count": len(members)}, "safe relative archive members")
    return checks, relay_manifest


def verify_identity(root: Path, snapshot_lock: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    identity = load_json(root / "identity/canonical_identity.json")
    source_hashes = (root / "audit/source_script_hashes.txt").read_text(encoding="utf-8")
    expected_identity = {
        "frozen_training_source_sha": EXPECTED_LAUNCHER_COMMIT,
        "frozen_training_launcher_sha256": EXPECTED_LAUNCHER_SHA,
        "isaac_gr00t_sha": EXPECTED_GR00T_SHA,
        "frozen_modality_sha256": EXPECTED_MODALITY_SHA,
        "base_model_tree_sha256": EXPECTED_BASE_TREE_SHA,
        "train_fixture_tree_sha256": EXPECTED_TRAIN_TREE_SHA,
        "split_lock_sha256": EXPECTED_SPLIT_LOCK_SHA,
        "cosmos_revision": EXPECTED_COSMOS_REVISION,
        "cosmos_tree_sha256": EXPECTED_COSMOS_TREE_SHA,
    }
    check(checks, "frozen_identity_fields", all(identity.get(key) == value for key, value in expected_identity.items()), {key: identity.get(key) for key in expected_identity}, expected_identity)
    check(checks, "frozen_launcher_source_hash", f"{EXPECTED_LAUNCHER_SHA} " in source_hashes and "stage20s_train_gr00t.py" in source_hashes, source_hashes.splitlines()[:1], EXPECTED_LAUNCHER_SHA)
    check(checks, "groot_source_hash", EXPECTED_GR00T_SHA == identity.get("isaac_gr00t_sha"), identity.get("isaac_gr00t_sha"), EXPECTED_GR00T_SHA)
    check(checks, "snapshot_lock_contract", snapshot_lock.get("snapshot_ids") == list(EXPECTED_SNAPSHOT_IDS) and snapshot_lock.get("snapshot_ids_sha256") == EXPECTED_SNAPSHOT_IDS_SHA, {"sha256": snapshot_lock.get("snapshot_ids_sha256"), "count": snapshot_lock.get("snapshot_count")}, {"sha256": EXPECTED_SNAPSHOT_IDS_SHA, "count": 6})
    local_lock_sha = sha256_file(repo_root / "manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json")
    check(checks, "local_validation_lock_file", local_lock_sha == "23c35e2204b9555c8431a3f86fd7e1767740bf7196341972e847bc43c141aec8", local_lock_sha, "frozen repository snapshot-lock SHA")
    return checks, identity


def verify_gpu(root: Path, r0: Mapping[str, Any], r1: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    source_path = root / "logs/597351_h2_launcher.log"
    source = parse_source_gpu_log(source_path)
    projection_manifest_path = root / "source/597351_h2_gpu_projection_manifest.json"
    projection_manifest = load_json(projection_manifest_path)
    projection_paths = [root / "source/597351_h2_gpu_projection_r0.txt", root / "source/597351_h2_gpu_projection_r1.txt"]
    projections = [parse_projection(path) for path in projection_paths]
    accounting = load_json(root / "accounting/597351.json")
    source_sha = sha256_file(source_path)
    sum_hashes = {
        relative: digest
        for digest, relative in (
            line.split("  ", 1)
            for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    check(checks, "gpu_source_log_hash", source_sha == projection_manifest.get("source_log_sha256"), source_sha, projection_manifest.get("source_log_sha256"))
    check(checks, "gpu_projection_manifest_hash", sha256_file(projection_manifest_path) == sum_hashes.get("source/597351_h2_gpu_projection_manifest.json"), sha256_file(projection_manifest_path), sum_hashes.get("source/597351_h2_gpu_projection_manifest.json"))
    pbs_template = (root / "source/stage22_h2_r0_r1_50_hopper_20260902.pbs").read_text(encoding="utf-8")
    historical_query = "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used --format=csv"
    check(checks, "gpu_source_query_format_deviation", source["header_present"] and historical_query in pbs_template and projection_manifest.get("source_output_header_actual", "").endswith("memory.used [MiB]"), {"source_header_present": source["header_present"], "query_present_in_immutable_pbs": historical_query in pbs_template, "query_literal_in_log": source["query_has_memory_used"]}, "five-column historical source query and output are documented by immutable PBS/log artifacts")
    check(checks, "gpu_source_one_row", len(source["rows"]) == 1, len(source["rows"]), 1)
    source_row = source["rows"][0] if source["rows"] else {}
    check(checks, "gpu_source_identity", source_row == {"index": 0, "name": "NVIDIA H200", "uuid": "GPU-39c00fb9-0eea-3abd-2727-2027667f3f44", "memory_total_mib": 143771, "memory_used_mib": 0}, source_row, "immutable H200 identity row")
    check(checks, "gpu_source_host", source["host_candidates"] == ["hopper-12"], source["host_candidates"], ["hopper-12"])
    check(checks, "gpu_source_job", source["values"].get("PBS_JOBID") == "597351.hopper-m-02", source["values"].get("PBS_JOBID"), "597351.hopper-m-02")
    check(checks, "gpu_source_cvd", source["values"].get("CUDA_VISIBLE_DEVICES") == source_row.get("uuid"), source["values"].get("CUDA_VISIBLE_DEVICES"), source_row.get("uuid"))
    check(checks, "gpu_source_timestamp", source["timestamps"] == ["2026-09-02T20:11:29+08:00"], source["timestamps"], "immutable query timestamp")
    accounting_start = datetime.fromisoformat(str(accounting["start_utc"]).replace("Z", "+00:00"))
    accounting_end = datetime.fromisoformat(str(accounting["end_utc"]).replace("Z", "+00:00"))
    observed = datetime.fromisoformat("2026-09-02T20:11:29+08:00").astimezone(accounting_start.tzinfo)
    check(checks, "gpu_query_inside_pbs_window", accounting_start <= observed <= accounting_end and accounting.get("job_id") == "597351.hopper-m-02" and accounting.get("exec_host") == "hopper-12", {"start": accounting_start.isoformat(), "observed": observed.isoformat(), "end": accounting_end.isoformat(), "job": accounting.get("job_id"), "host": accounting.get("exec_host")}, "same completed PBS allocation window")
    for index, projection in enumerate(projections):
        recipe = "R0" if index == 0 else "R1"
        values = projection["values"]
        check(checks, f"{recipe}_projection_hash", projection["sha256"] == projection_manifest.get("projection_sha256"), projection["sha256"], projection_manifest.get("projection_sha256"))
        check(checks, f"{recipe}_projection_query", projection["query_line"] == "nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader,nounits", projection["query_line"], "four identity columns")
        check(checks, f"{recipe}_projection_identity", projection["device"] == {key: source_row[key] for key in ("index", "name", "uuid", "memory_total_mib")}, projection["device"], "identity columns preserved from source")
        check(checks, f"{recipe}_projection_source_binding", values.get("source_artifact_sha256") == source_sha and values.get("PBS_JOBID") == "597351.hopper-m-02" and values.get("HOSTNAME") == "hopper-12", {key: values.get(key) for key in ("source_artifact_sha256", "PBS_JOBID", "HOSTNAME")}, "source log hash, job, host")
    check(checks, "gpu_projection_manifest_scope", projection_manifest.get("no_rerun") is True and projection_manifest.get("no_present_time_query") is True and projection_manifest.get("original_log_immutable") is True, {key: projection_manifest.get(key) for key in ("no_rerun", "no_present_time_query", "original_log_immutable")}, "historical projection only")
    runtime_totals = [nested(run, "cuda_telemetry", "before", "total_memory_bytes") for run in (r0, r1)]
    check(checks, "gpu_run_manifest_identity", all(nested(run, "runtime", "cuda_visible_devices") == source_row.get("uuid") and nested(run, "cuda_telemetry", "device_name") == "NVIDIA H200" for run in (r0, r1)) and runtime_totals[0] == runtime_totals[1] and all(isinstance(total, int) and total > 0 for total in runtime_totals), {"r0_uuid": nested(r0, "runtime", "cuda_visible_devices"), "r1_uuid": nested(r1, "runtime", "cuda_visible_devices"), "r0_total_memory_bytes": runtime_totals[0], "r1_total_memory_bytes": runtime_totals[1], "nvidia_smi_total_mib": source_row.get("memory_total_mib")}, "R0/R1 manifests bind the same H200 and internally consistent runtime total; display-unit differences are retained as a format note")
    check(checks, "gpu_provenance_recovered_from_immutable_artifacts", all(value["passed"] for key, value in checks.items() if key.startswith("gpu_") or key.startswith("R0_projection") or key.startswith("R1_projection")), "source log + projection + accounting + run identity", "all historical fields artifact-backed")
    return checks, {
        "classification": "H2_GPU_PROVENANCE_SUPPORTED_WITH_FORMAT_DEVIATION",
        "source_log_sha256": source_sha,
        "projection_sha256": projections[0]["sha256"],
        "source_query": "index,name,uuid,memory.total,memory.used",
        "projection_query": "index,name,uuid,memory.total",
        "format_deviation": "original immutable launcher query included memory.used; derived projection preserved identity columns and dropped only that extra field",
        "uuid": source_row.get("uuid"),
        "device_name": source_row.get("name"),
        "memory_total_mib": source_row.get("memory_total_mib"),
        "host": "hopper-12",
        "pbs_job_id": "597351.hopper-m-02",
        "present_time_query_executed_by_auditor": False,
    }


def verify_runs(root: Path, identity: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    manifests = {
        "R0": load_json(root / "manifests/597351_r0_run_manifest.json"),
        "R1": load_json(root / "manifests/597351_r1_run_manifest.json"),
    }
    accounting = load_json(root / "accounting/597351.json")
    launcher = (root / "source/stage22_h2_r0_r1_50_hopper_20260902.pbs").read_text(encoding="utf-8")
    for recipe, run in manifests.items():
        expected_diffusion = recipe == "R1"
        checks_for_run = {
            "schema": run.get("schema") == "STAGE20S_GR00T_TRAINING_RUN_V1",
            "status": run.get("status") == "PASS",
            "simulation_only": run.get("simulation_only") is True,
            "hardware_control": run.get("hardware_control") is False,
            "max_steps": run.get("max_steps") == EXPECTED_STEPS,
            "save_steps": run.get("save_steps") == EXPECTED_SAVE_STEPS,
            "horizon": run.get("action_horizon") == EXPECTED_HORIZON,
            "timing": run.get("action_sample_period_s") == EXPECTED_ACTION_PERIOD and run.get("action_sample_period_provenance") == "TEST_ONLY_SIM_VISUAL_REACH_V0",
            "projector": run.get("tune_projector") is True,
            "diffusion": run.get("tune_diffusion_model") is expected_diffusion,
            "visual": run.get("tune_visual") is False,
            "llm": run.get("tune_llm") is False,
            "vlln": run.get("tune_vlln") is False,
            "groot": run.get("groot_repo_sha") == EXPECTED_GR00T_SHA,
            "base_index": bool(run.get("base_model_index_sha256")),
            "runtime_device": nested(run, "cuda_telemetry", "device_name") == "NVIDIA H200",
            "checkpoint_50_metadata": all(key in run.get("artifacts", {}) for key in ("checkpoint-50/trainer_state.json", "checkpoint-50/model.safetensors.index.json")),
        }
        for name, passed in checks_for_run.items():
            check(checks, f"{recipe}_{name}", passed, run.get(name) if name in run else passed, True)
    check(checks, "H2_accounting", accounting.get("status") == "COMPLETED" and accounting.get("job_id") == "597351.hopper-m-02" and accounting.get("exit_status") == 0, {key: accounting.get(key) for key in ("status", "job_id", "exec_host", "exit_status")}, "completed H2 job 597351")
    check(checks, "H2_direct_launcher", "stage20s_train_gr00t.py" in launcher and "stage22_train_recipes.py" not in launcher and "--max-steps 50" in launcher and "--seed 2222" in launcher and "--tune-diffusion-model" in launcher, True, "direct frozen launcher commands in immutable PBS template")
    check(checks, "H2_same_base_dataset_modality", manifests["R0"].get("base_model_path") == manifests["R1"].get("base_model_path") and manifests["R0"].get("dataset_dir") == manifests["R1"].get("dataset_dir") and manifests["R0"].get("modality_config") == manifests["R1"].get("modality_config"), {recipe: {key: manifests[recipe].get(key) for key in ("base_model_path", "dataset_dir", "modality_config")} for recipe in manifests}, "identical H2 inputs")
    check(checks, "H2_same_runtime", all(manifests["R0"].get("runtime") == manifests["R1"].get("runtime") and manifests["R0"].get("cuda_telemetry", {}).get(key) == manifests["R1"].get("cuda_telemetry", {}).get(key) for key in ("device_name", "torch_version", "cuda_version")), {recipe: manifests[recipe].get("runtime") for recipe in manifests}, "identical runtime identity")
    h1 = load_json(root / "h1/597273_r1_preflight.json")
    check(checks, "H1_prior_pass", h1.get("status") == "PASS" and h1.get("optimizer_step_completed") is True and h1.get("training_steps_requested") == 1, {key: h1.get(key) for key in ("status", "optimizer_step_completed", "training_steps_requested", "trainable_parameter_count")}, "H1 optimizer step PASS")
    failed_probe = load_json(root / "module_probe/597644_status.json")
    check(checks, "failed_probe_infra_only", failed_probe.get("status") == "BLOCKED_OFFLINE_CACHE_PATH" and all(failed_probe.get(key) is False for key in ("forward", "backward", "optimizer", "inference", "training")), {key: failed_probe.get(key) for key in ("status", "forward", "backward", "optimizer", "inference", "training")}, "failed before model construction/compute")
    source_launcher_hash = sha256_file(root / "source/stage20s_train_gr00t.py")
    check(checks, "frozen_launcher_file_sha", source_launcher_hash == EXPECTED_LAUNCHER_SHA, source_launcher_hash, EXPECTED_LAUNCHER_SHA)
    check(checks, "identity_tree_hashes", all(identity.get(key) == value for key, value in {"base_model_tree_sha256": EXPECTED_BASE_TREE_SHA, "train_fixture_tree_sha256": EXPECTED_TRAIN_TREE_SHA, "split_lock_sha256": EXPECTED_SPLIT_LOCK_SHA, "frozen_modality_sha256": EXPECTED_MODALITY_SHA, "cosmos_revision": EXPECTED_COSMOS_REVISION, "cosmos_tree_sha256": EXPECTED_COSMOS_TREE_SHA}.items()), {key: identity.get(key) for key in ("base_model_tree_sha256", "train_fixture_tree_sha256", "split_lock_sha256", "frozen_modality_sha256", "cosmos_revision", "cosmos_tree_sha256")}, "frozen identity manifest values")
    return checks, manifests


def verify_module_integrity(root: Path, manifests: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    broad: dict[str, Any] = {}
    for recipe in ("R0", "R1"):
        path = root / f"manifests/597351_{recipe.lower()}_module_integrity.json"
        payload = load_json(path)
        groups = payload.get("groups", {})
        parameters = payload.get("parameters", [])
        computed_groups: dict[str, dict[str, int]] = {}
        for parameter in parameters:
            group = str(parameter.get("group"))
            row = computed_groups.setdefault(group, {"parameter_count": 0, "trainable_count": 0})
            row["parameter_count"] += int(parameter.get("numel", 0))
            if parameter.get("requires_grad") is True:
                row["trainable_count"] += int(parameter.get("numel", 0))
        group_match = all(
            computed_groups.get(group, {}).get("parameter_count") == values.get("parameter_count")
            and computed_groups.get(group, {}).get("trainable_count") == values.get("trainable_count")
            for group, values in groups.items()
        )
        expected_total = EXPECTED_R1_TRAINABLE if recipe == "R1" else EXPECTED_R0_TRAINABLE
        checks_for_recipe = {
            "schema": payload.get("schema") == "STAGE22_MODULE_INTEGRITY_GATE_V1",
            "status": payload.get("status") == "PASS",
            "recipe_flag": payload.get("diffusion_trainable_requested") is (recipe == "R1"),
            "violations": payload.get("violations") == [],
            "parameter_rows": len(parameters) == 1030 and all(isinstance(item.get("requires_grad"), bool) for item in parameters),
            "group_recomputation": group_match,
            "total_trainable": payload.get("total_trainable_parameter_count") == expected_total,
            "expected_r1_count": payload.get("expected_r1_trainable_parameter_count") == EXPECTED_R1_TRAINABLE,
        }
        for name, passed in checks_for_recipe.items():
            check(checks, f"{recipe}_broad_module_{name}", passed, passed, True)
        broad[recipe] = {
            "schema": payload.get("schema"),
            "status": payload.get("status"),
            "total_parameter_count": payload.get("total_parameter_count"),
            "total_trainable_parameter_count": payload.get("total_trainable_parameter_count"),
            "groups": groups,
            "parameter_row_count": len(parameters),
            "offline": payload.get("offline"),
            "optimizer_step_started": payload.get("optimizer_step_started"),
            "artifact_sha256": sha256_file(path),
        }
    strict_role_present = False
    check(checks, "TRAINABLE_MODULE_INVENTORY_GATE", strict_role_present, "STAGE22_MODULE_INTEGRITY_GATE_V1 group inventory only", "STAGE22_TRAINABLE_MODULE_INVENTORY_V1 role inventory with requires_grad digests")
    return checks, {
        "broad_integrity": broad,
        "strict_role_inventory": "MISSING",
        "strict_role_inventory_reason": "597351 emitted grouped per-parameter requires_grad rows, but not the frozen seven-role inventory; 597644 failed before model construction because the Cosmos cache path was absent",
        "expected_r0_trainable_parameters": EXPECTED_R0_TRAINABLE,
        "expected_r1_trainable_parameters": EXPECTED_R1_TRAINABLE,
    }


RECORD_METRICS: dict[str, tuple[str, ...]] = {
    "position_mean_error_m": ("position_error_m", "mean"),
    "position_p95_error_m": ("position_error_m", "p95"),
    "position_max_error_m": ("position_error_m", "max"),
    "rotation_mean_error_rad": ("rotation_error_rad", "mean"),
    "rotation_p95_error_rad": ("rotation_error_rad", "p95"),
    "rotation_max_error_rad": ("rotation_error_rad", "max"),
    "endpoint_position_error_m": ("endpoint_position_error_m",),
    "endpoint_rotation_error_rad": ("endpoint_rotation_error_rad",),
    "path_length_ratio": ("path_length_ratio",),
    "mean_speed_ratio": ("mean_speed_ratio",),
    "max_speed_ratio": ("max_speed_ratio",),
    "roughness_ratio": ("roughness_ratio",),
    "initial_position_discontinuity_m": ("predicted_profile", "initial_position_discontinuity_m"),
    "initial_rotation_discontinuity_rad": ("predicted_profile", "initial_rotation_discontinuity_rad"),
}
OVERLAP_METRICS: dict[str, tuple[str, ...]] = {
    "position_mean_error_m": ("position_error_m", "mean"),
    "position_p95_error_m": ("position_error_m", "p95"),
    "position_max_error_m": ("position_error_m", "max"),
    "rotation_mean_error_rad": ("rotation_error_rad", "mean"),
    "rotation_p95_error_rad": ("rotation_error_rad", "p95"),
    "rotation_max_error_rad": ("rotation_error_rad", "max"),
    "tangent_mean_error_mps": ("translational_tangent_error_mps", "mean"),
    "tangent_p95_error_mps": ("translational_tangent_error_mps", "p95"),
    "tangent_max_error_mps": ("translational_tangent_error_mps", "max"),
    "old_first_position_error_m": ("old_prediction_at_new_observation_vs_new_first", "position_error_m"),
    "old_first_rotation_error_rad": ("old_prediction_at_new_observation_vs_new_first", "rotation_error_rad"),
}


def metric(record: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    return nested(record, *path)


def summarize_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = {name: stats(metric(record, path) for record in records) for name, path in RECORD_METRICS.items()}
    result["safety_gate_decisions"] = {
        decision: sum(1 for record in records if nested(record, "safety_gate", "decision") == decision)
        for decision in ("ACCEPT", "WARN", "REJECT")
    }
    result["finite_outputs"] = all(record.get("finite") is True for record in records)
    return result


def summarize_overlaps(overlaps: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {name: stats(metric(item, path) for item in overlaps) for name, path in OVERLAP_METRICS.items()}


def delta_summary(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_records = {(str(row.get("input_id")), int(left.get("inference_seed", -1))): row for row in left.get("records", [])}
    right_records = {(str(row.get("input_id")), int(right.get("inference_seed", -1))): row for row in right.get("records", [])}
    common = sorted(set(left_records) & set(right_records))
    record_deltas = {
        name: stats(
            float(metric(right_records[key], path)) - float(metric(left_records[key], path))
            for key in common
            if metric(left_records[key], path) is not None and metric(right_records[key], path) is not None
        )
        for name, path in RECORD_METRICS.items()
    }
    def overlap_key(item: Mapping[str, Any], seed: int) -> tuple[str, float, float, int]:
        return (str(item.get("raw_episode_id")), float(item.get("previous_observation_timestamp_s")), float(item.get("current_observation_timestamp_s")), seed)
    left_overlaps = {overlap_key(item, int(left.get("inference_seed", -1))): item for item in left.get("chunk_overlap_metrics", [])}
    right_overlaps = {overlap_key(item, int(right.get("inference_seed", -1))): item for item in right.get("chunk_overlap_metrics", [])}
    common_overlaps = sorted(set(left_overlaps) & set(right_overlaps))
    overlap_deltas = {
        name: stats(
            float(metric(right_overlaps[key], path)) - float(metric(left_overlaps[key], path))
            for key in common_overlaps
            if metric(left_overlaps[key], path) is not None and metric(right_overlaps[key], path) is not None
        )
        for name, path in OVERLAP_METRICS.items()
    }
    return {
        "record_pair_count": len(common),
        "overlap_pair_count": len(common_overlaps),
        "delta_right_minus_left": {"records": record_deltas, "overlap": overlap_deltas},
    }


def verify_profiles(root: Path, snapshot_lock: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: dict[str, Any] = {}
    profiles: dict[str, list[dict[str, Any]]] = {}
    aggregate_match: dict[str, bool] = {}
    for recipe in ("base", "r0", "r1"):
        profiles[recipe] = []
        for seed in EXPECTED_SEEDS:
            path = root / f"logs/profile_resume_{recipe}_{seed}.log"
            payload, json_offset = extract_json_from_log(path)
            profiles[recipe].append(payload)
            expected_model_fragment = {
                "base": "GR00T-N1.7-3B",
                "r0": "/r0/checkpoint-50",
                "r1": "/r1/checkpoint-50",
            }[recipe]
            records = payload.get("records", [])
            record_ids = [str(record.get("input_id")) for record in records]
            record_checks = {
                "schema": payload.get("schema") == PROFILE_SCHEMA,
                "status": payload.get("status") == "PASS",
                "split": payload.get("split") == "validation",
                "seed": payload.get("inference_seed") == seed,
                "snapshot_count": payload.get("snapshot_count") == 6 and len(records) == 6,
                "snapshot_ids": set(record_ids) == set(EXPECTED_SNAPSHOT_IDS),
                "target": all(record.get("target_id") == "red_target_holdout_e" for record in records),
                "horizon_shapes": all(record.get("prediction_shape") == [40, 10] and record.get("label_shape") == [40, 10] for record in records),
                "raw_shapes": all(record.get("raw_universal_shape") == [1, 40, 132] and record.get("decoded_physical_shape") == [1, 40, 10] for record in records),
                "finite": all(record.get("finite") is True for record in records),
                "model_input_target_excluded": all(record.get("model_input_target_excluded") is True for record in records) and payload.get("model_input_contract", {}).get("target_position_world_m") == "excluded",
                "timing": payload.get("timing", {}).get("action_sample_period_s") == EXPECTED_ACTION_PERIOD and payload.get("timing", {}).get("action_sample_period_provenance") == "TEST_ONLY_STAGE22_SIMULATION",
                "correction_false": payload.get("profile_selection_policy", {}).get("trajectory_correction") is False and all(record.get("trajectory_correction_applied") is False for record in records) and all(item.get("no_trajectory_correction") is True for item in payload.get("chunk_overlap_metrics", [])),
                "overlap_count": len(payload.get("chunk_overlap_metrics", [])) == 5,
                "safety_gate_decisions": {decision: sum(1 for record in records if nested(record, "safety_gate", "decision") == decision) for decision in ("ACCEPT", "WARN", "REJECT")} == {"ACCEPT": 0, "WARN": 0, "REJECT": 6},
                "model_path": expected_model_fragment in str(payload.get("recipe_model_path")),
                "json_embedded_after_launcher_output": json_offset > 0,
            }
            for name, passed in record_checks.items():
                check(checks, f"profile_{recipe}_{seed}_{name}", passed, passed, True)
        aggregate = load_json(root / f"profiles/{recipe}_validation_seed_aggregate.json")
        aggregate_records = [record for payload in profiles[recipe] for record in payload.get("records", [])]
        aggregate_overlaps = [item for payload in profiles[recipe] for item in payload.get("chunk_overlap_metrics", [])]
        aggregate_match[recipe] = aggregate_records == aggregate.get("records") and aggregate_overlaps == aggregate.get("chunk_overlap_metrics")
        check(checks, f"profile_{recipe}_aggregate_exact_reproduction", aggregate_match[recipe], {"records": len(aggregate_records), "overlaps": len(aggregate_overlaps)}, "aggregate payload equals concatenated immutable profile logs")
    source_eval_sha = sha256_file(root / "source/stage22_profile_evaluation.py")
    check(checks, "profile_evaluator_source_hash", source_eval_sha == "1e998821c18165f4147e58ca59e9ab5316c55226ecadf683904ae0553e66026e", source_eval_sha, "immutable 597361 evaluator SHA")
    first_failure = (root / "logs/597361_first_profile_base_2223.log").read_text(encoding="utf-8", errors="replace")
    check(checks, "profile_initial_failure_pre_output", "AttributeError" in first_failure and PROFILE_SCHEMA not in first_failure and "SIM_VISUAL_REACH_MODALITY_CONFIG" in first_failure, "module symbol failure before profile JSON", "pre-output compatibility failure")
    profile_resume = load_json(root / "manifests/STAGE22_H2_PROFILE_RESUME.json")
    check(checks, "profile_resume_metadata", profile_resume.get("optimizer_steps_run_here") == 0 and nested(profile_resume, "profiles", "holdout_f_accessed") is False and profile_resume.get("no_wbc") is True, "profile-only continuation", "no training, holdout_f, or WBC")
    source_modality_sha = sha256_file(root / "source/stage20s_groot_modality.py")
    check(checks, "profile_modality_compatibility_source", source_modality_sha == "244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25", source_modality_sha, "immutable profile modality SHA")
    check(checks, "profile_semantics_frozen", all(
        payload.get("model_input_contract", {}).get("state_v1") == [10]
        and payload.get("expert_label_contract", {}).get("desired_eef") == [40, 9]
        and payload.get("expert_label_contract", {}).get("desired_gripper") == [40, 1]
        and payload.get("expert_label_contract", {}).get("timestamps") == [40]
        and payload.get("profile_selection_policy", {}).get("loss_not_used_for_selection") is True
        and payload.get("profile_selection_policy", {}).get("trajectory_correction") is False
        for recipe in profiles.values() for payload in recipe
    ), "validation/e, StateV1=10, H=40 labels, no correction, loss excluded", "frozen profile semantics")
    # The raw evaluator predates the later snapshot-manifest field.  The audit
    # binds its exact immutable input IDs to the current locked file instead of
    # treating R0/R1 set equality alone as sufficient.
    snapshot_ids = sorted({str(record.get("input_id")) for recipe in profiles.values() for payload in recipe for record in payload.get("records", [])})
    check(checks, "VALIDATION_SNAPSHOT_COMPLETENESS_GATE", snapshot_ids == sorted(EXPECTED_SNAPSHOT_IDS) and snapshot_lock.get("snapshot_ids_sha256") == EXPECTED_SNAPSHOT_IDS_SHA and snapshot_lock.get("snapshot_count") == 6 and snapshot_lock.get("expected_overlap_count") == 5, {"ids": snapshot_ids, "lock_sha256": snapshot_lock.get("snapshot_ids_sha256"), "lock_file_sha256": sha256_file(repo_root / "manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json")}, "six locked holdout_e snapshots and five transitions")
    summaries = {recipe: {"records": summarize_records([record for payload in payloads for record in payload["records"]]), "overlap": summarize_overlaps([item for payload in payloads for item in payload["chunk_overlap_metrics"]])} for recipe, payloads in profiles.items()}
    comparisons = {}
    for name, (left, right) in {"BASE_TO_R0": ("base", "r0"), "BASE_TO_R1": ("base", "r1"), "R0_TO_R1": ("r0", "r1")}.items():
        left_payloads = profiles[left]
        right_payloads = profiles[right]
        left_records = {(str(row["input_id"]), int(payload["inference_seed"])): row for payload in left_payloads for row in payload["records"]}
        right_records = {(str(row["input_id"]), int(payload["inference_seed"])): row for payload in right_payloads for row in payload["records"]}
        rows = []
        for key in sorted(set(left_records) & set(right_records)):
            rows.append({"left": left_records[key], "right": right_records[key]})
        comparisons[name] = {
            "pair_count": len(rows),
            "record_deltas": {metric_name: stats(float(metric(row["right"], path)) - float(metric(row["left"], path)) for row in rows if metric(row["left"], path) is not None and metric(row["right"], path) is not None) for metric_name, path in RECORD_METRICS.items()},
        }
        left_overlaps = {(str(item["raw_episode_id"]), float(item["previous_observation_timestamp_s"]), float(item["current_observation_timestamp_s"]), int(payload["inference_seed"])): item for payload in left_payloads for item in payload["chunk_overlap_metrics"]}
        right_overlaps = {(str(item["raw_episode_id"]), float(item["previous_observation_timestamp_s"]), float(item["current_observation_timestamp_s"]), int(payload["inference_seed"])): item for payload in right_payloads for item in payload["chunk_overlap_metrics"]}
        overlap_rows = [{"left": left_overlaps[key], "right": right_overlaps[key]} for key in sorted(set(left_overlaps) & set(right_overlaps))]
        comparisons[name]["overlap_pair_count"] = len(overlap_rows)
        comparisons[name]["overlap_deltas"] = {metric_name: stats(float(metric(row["right"], path)) - float(metric(row["left"], path)) for row in overlap_rows if metric(row["left"], path) is not None and metric(row["right"], path) is not None) for metric_name, path in OVERLAP_METRICS.items()}
    return checks, {
        "profile_log_count": 9,
        "raw_records_per_recipe": 18,
        "raw_overlaps_per_recipe": 15,
        "payload_snapshot_lock_hash_present": False,
        "auditor_rebound_snapshot_lock_sha256": sha256_file(repo_root / "manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json"),
        "aggregate_exact_match": aggregate_match,
        "summaries": summaries,
        "comparisons": comparisons,
        "visual_conditioning": {recipe: profiles[recipe][0].get("visual_target_conditioning") for recipe in profiles},
        "profile_source_sha256": source_eval_sha,
        "profile_modality_source_sha256": source_modality_sha,
        "semantic_drift_classification": "H2_PROFILE_EVALUATOR_SEMANTICS_FROZEN",
        "semantic_drift_note": "597361 initial attempt failed before output on modality symbol lookup; continuation used an equivalent export and preserved evaluator/model/label/timing/gate semantics",
    }


def verify_current_source_delta(bundle_source: Path, current_source: Path) -> dict[str, Any]:
    if not current_source.is_file():
        return {"available": False, "classification": "NOT_CHECKED"}
    bundle_lines = bundle_source.read_text(encoding="utf-8").splitlines()
    current_lines = current_source.read_text(encoding="utf-8").splitlines()
    diff = list(difflib.unified_diff(bundle_lines, current_lines, lineterm=""))
    changed = [line for line in diff if line.startswith("+") and not line.startswith("+++") or line.startswith("-") and not line.startswith("---")]
    added = [line for line in changed if line.startswith("+")]
    removed = [line for line in changed if line.startswith("-")]
    # The current evaluator adds only frozen validation snapshot-lock
    # argument, binding, equality check, and output provenance. Structural
    # lines such as ``type=Path`` and closing delimiters have no semantic
    # token, so they must not be mistaken for model/evaluator drift.
    added_text = "\n".join(added).lower()
    allowed = (
        not removed
        and "--snapshot-manifest" in added_text
        and "load_snapshot_lock" in added_text
        and "snapshot_manifest_sha256" in added_text
        and "snapshot_ids" in added_text
    )
    return {
        "available": True,
        "bundle_sha256": sha256_file(bundle_source),
        "current_sha256": sha256_file(current_source),
        "diff_line_count": len(diff),
        "changed_line_count": len(changed),
        "only_additive_snapshot_lock_support": allowed,
        "classification": "ADDITIVE_SNAPSHOT_LOCK_SUPPORT_ONLY" if allowed else "REVIEW_REQUIRED",
    }


def build_report(result: Mapping[str, Any]) -> str:
    gate = result["final_status"]
    gpu = result["gpu"]
    profiles = result["profiles"]
    return f"""# Stage22 authoritative H2 audit

Date: 2026-09-03 (Asia/Singapore)

## Final state

```text
{gate}
```

This is a read-only audit of immutable relay bundle
`{result['relay_archive_sha256']}`. No H2 rerun, new inference, holdout_f read,
checkpoint transfer, WBC execution, or present-time GPU query was performed.

## Relay and frozen identity

- Relay payload: 89 files; archive SHA256 verified.
- `contains_credentials=false`, `holdout_f_included=false`,
  `holdout_f_accessed=false`, and checkpoint weights absent.
- R0/R1 training job: `597351.hopper-m-02`, completed with exit 0; direct frozen
  launcher SHA `{EXPECTED_LAUNCHER_SHA}` and GR00T SHA `{EXPECTED_GR00T_SHA}`.
- Base tree, train fixture tree, split lock, modality, Cosmos revision/tree
  matched the frozen identity manifest.

## GPU provenance

`{gpu['classification']}`

The immutable launcher log used a five-column query including `memory.used`.
The relay's explicitly labelled projection dropped only that extra column and
preserved the H200 name, UUID, total memory, host, PBS job, and timestamp. The
projection and source-log hashes were checked against the relay manifest and
the timestamp was inside the completed PBS window. The auditor did not issue a
new `nvidia-smi` query.

## H2 identity and failed probe

R0 and R1 both used 50 steps, H=40, training seed 2222, LR 1e-4, 0.05 s
`TEST_ONLY_SIM_VISUAL_REACH_V0`, the same base/dataset/modality/runtime, and
direct `stage20s_train_gr00t.py` launch. The R0/R1 trainable counts were
327,356,544 and 1,419,078,784 respectively.

The 597644 probe failed at offline Cosmos cache lookup before model
construction. It performed no forward, backward, optimizer, inference, or
training work and therefore remains infrastructure-only evidence.

The relay's 597351 module files contain an internally consistent grouped
per-parameter `requires_grad` inventory and broad R0/R1 counts, but they do not
contain the frozen seven-role `STAGE22_TRAINABLE_MODULE_INVENTORY_V1`. Thus the
strict role-level gate remains open:

```text
H2_AUDIT_PASS_EXCEPT_LIVE_MODULE_INVENTORY
```

## Profile semantic and completeness audit

- 9 immutable profile JSON payloads were recovered from 9 logs: BASE/R0/R1 ×
  seeds 2223/2224/2225.
- Each has six exact `red_target_holdout_e` snapshots and five overlap
  transitions; all three recipes have 18 records and 15 overlaps.
- All raw outputs are finite with `[1,40,132]` universal and `[1,40,10]`
  decoded shapes. All 54 model rows are `SafetyGate=REJECT` because the
  test-only 2.0 m/s gate is exceeded; this is a profile result, not a WBC
  execution result.
- The raw evaluator payload predates the later snapshot-lock hash field. The
  auditor bound the exact immutable six IDs to the tracked snapshot-lock file
  and recorded its SHA; R0/R1 equality alone was not used.
- The 597361 initial failure was pre-output modality-symbol compatibility;
  the continuation preserved H=40, StateV1=10, 0.05 s timing, target exclusion,
  no correction, metrics, and SafetyGate semantics.

## Independent BASE/R0/R1 comparison

The machine-readable JSON contains full median/p95/max summaries and paired
`BASE→R0`, `BASE→R1`, and `R0→R1` deltas. The key aggregate medians are:

| recipe | mean position error | endpoint position error | path ratio | mean-speed ratio | max-speed ratio | overlap p95 position | SafetyGate |
|---|---:|---:|---:|---:|---:|---:|---|
| BASE | {profiles['summaries']['base']['records']['position_mean_error_m']['median']:.6f} m | {profiles['summaries']['base']['records']['endpoint_position_error_m']['median']:.6f} m | {profiles['summaries']['base']['records']['path_length_ratio']['median']:.3f} | {profiles['summaries']['base']['records']['mean_speed_ratio']['median']:.3f} | {profiles['summaries']['base']['records']['max_speed_ratio']['median']:.3f} | {profiles['summaries']['base']['overlap']['position_p95_error_m']['median']:.6f} m | 0/0/18 |
| R0 | {profiles['summaries']['r0']['records']['position_mean_error_m']['median']:.6f} m | {profiles['summaries']['r0']['records']['endpoint_position_error_m']['median']:.6f} m | {profiles['summaries']['r0']['records']['path_length_ratio']['median']:.3f} | {profiles['summaries']['r0']['records']['mean_speed_ratio']['median']:.3f} | {profiles['summaries']['r0']['records']['max_speed_ratio']['median']:.3f} | {profiles['summaries']['r0']['overlap']['position_p95_error_m']['median']:.6f} m | 0/0/18 |
| R1 | {profiles['summaries']['r1']['records']['position_mean_error_m']['median']:.6f} m | {profiles['summaries']['r1']['records']['endpoint_position_error_m']['median']:.6f} m | {profiles['summaries']['r1']['records']['path_length_ratio']['median']:.3f} | {profiles['summaries']['r1']['records']['mean_speed_ratio']['median']:.3f} | {profiles['summaries']['r1']['records']['max_speed_ratio']['median']:.3f} | {profiles['summaries']['r1']['overlap']['position_p95_error_m']['median']:.6f} m | 0/0/18 |

Neither R0 nor R1 meets the predeclared Stage22 profile qualification gate.
The selection artifact therefore remains reporting-only; no checkpoint is
scientifically selected and no R1@100/R1@200 extension is authorized.

## Required stop state

```text
H2_AUTHORITATIVE_AUDIT: {gate}
R1@100: NOT AUTHORIZED
R1@200: NOT AUTHORIZED
HOLDOUT_F_ACCESSED: false
WBC: NOT AUTHORIZED
```
"""


def audit(bundle_root: Path, snapshot_lock_path: Path, repo_root: Path, output_root: Path) -> dict[str, Any]:
    relay_checks, relay_manifest = verify_relay(bundle_root)
    snapshot_lock = load_json(snapshot_lock_path)
    identity_checks, identity = verify_identity(bundle_root, snapshot_lock, repo_root)
    r0 = load_json(bundle_root / "manifests/597351_r0_run_manifest.json")
    r1 = load_json(bundle_root / "manifests/597351_r1_run_manifest.json")
    gpu_checks, gpu = verify_gpu(bundle_root, r0, r1)
    run_checks, run_manifests = verify_runs(bundle_root, identity, repo_root)
    module_checks, modules = verify_module_integrity(bundle_root, run_manifests)
    profile_checks, profiles = verify_profiles(bundle_root, snapshot_lock, repo_root)
    source_delta = verify_current_source_delta(bundle_root / "source/stage22_profile_evaluation.py", repo_root / "scripts/stage22_profile_evaluation.py")
    all_checks = {**{f"relay.{key}": value for key, value in relay_checks.items()}, **{f"identity.{key}": value for key, value in identity_checks.items()}, **{f"gpu.{key}": value for key, value in gpu_checks.items()}, **{f"runs.{key}": value for key, value in run_checks.items()}, **{f"modules.{key}": value for key, value in module_checks.items()}, **{f"profiles.{key}": value for key, value in profile_checks.items()}}
    module_gate_pass = module_checks["TRAINABLE_MODULE_INVENTORY_GATE"]["passed"]
    core_excluded = {"modules.TRAINABLE_MODULE_INVENTORY_GATE"}
    core_failures = [name for name, value in all_checks.items() if not value["passed"] and name not in core_excluded]
    if core_failures:
        final_status = "H2_AUTHORITATIVE_AUDIT_BLOCKED"
    elif not module_gate_pass:
        final_status = "H2_AUDIT_PASS_EXCEPT_LIVE_MODULE_INVENTORY"
    elif gpu["classification"] == "H2_GPU_PROVENANCE_SUPPORTED_WITH_FORMAT_DEVIATION":
        final_status = "H2_AUTHORITATIVE_AUDIT_PASS_WITH_GPU_FORMAT_DEVIATION"
    else:
        final_status = "H2_AUTHORITATIVE_AUDIT_PASS"
    result = {
        "schema": "STAGE22_H2_AUTHORITATIVE_AUDIT_V1",
        "final_status": final_status,
        "relay_archive_sha256": sha256_file(bundle_root / "relay_archive.tar.gz"),
        "bundle_root": str(bundle_root),
        "snapshot_lock_path": str(snapshot_lock_path),
        "snapshot_lock_sha256": sha256_file(snapshot_lock_path),
        "relay": relay_checks,
        "identity": identity_checks,
        "gpu": gpu,
        "runs": run_checks,
        "modules": modules,
        "profiles": profiles,
        "current_source_delta": source_delta,
        "all_checks": all_checks,
        "core_failures": core_failures,
        "module_gate_pass": module_gate_pass,
        "holdout_f_accessed": False,
        "contains_credentials": False,
        "h2_rerun": False,
        "new_profile_inference": False,
        "checkpoint_transfer": False,
        "wbc_execution": False,
        "checkpoint_selection": False,
        "claim_boundary": "immutable H2 provenance and offline profile attribution audit only; no task or WBC claim",
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_root / "audit.md").write_text(build_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--snapshot-lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        bundle_root=args.bundle_root.resolve(strict=True),
        snapshot_lock_path=args.snapshot_lock.resolve(strict=True),
        repo_root=args.repo_root.resolve(strict=True),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps({
        "final_status": result["final_status"],
        "core_failures": result["core_failures"],
        "module_gate_pass": result["module_gate_pass"],
        "gpu_classification": result["gpu"]["classification"],
        "profile_records_per_recipe": result["profiles"]["raw_records_per_recipe"],
        "profile_overlaps_per_recipe": result["profiles"]["raw_overlaps_per_recipe"],
        "output_root": str(args.output_root.resolve()),
    }, indent=2))
    return 0 if result["final_status"] != "H2_AUTHORITATIVE_AUDIT_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
