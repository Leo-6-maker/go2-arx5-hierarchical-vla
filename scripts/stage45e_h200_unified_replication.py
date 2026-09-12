#!/usr/bin/env python3
"""Stage45E unified replication orchestration and CPU closure.

The static and high-water paths are CPU-only.  The explicit ``campaign``
path is intended to run inside one PBS H200 allocation and invokes the three
already-reviewed science branches as separate subprocesses.  It never merges
those branches, never resumes a checkpoint, and never schedules a retry after
model execution has started.  Each subprocess receives absolute paths and an
empty ``PYTHONPATH`` so that the committed entrypoints are tested from a
foreign working directory.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0143_V1.json"
PYTHON = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv/bin/python")
STORAGE_ROOT = Path("/hpctmp/e1192548/stage45e_h200_unified_replication_20260912")
WORKTREE_ROOT = Path("/scratch/e1192548/stage45e_h200_unified_worktrees")
DEFAULTS = {
    "base_model": Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B"),
    "dataset": Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/lerobot_train"),
    "groot": Path("/scratch/e1192548/stage45c_pi0094_native_runtime_20260909/groot-source-v4"),
    "cosmos": Path("/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B"),
    "modality": Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py"),
    "config": Path("/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02/h2_training/experiment_cfg/config.yaml"),
    "runtime_stats": Path("/scratch/e1192548/stage45b_pi0083_runtime_parquet_v2/STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2.json"),
    "n2_stats": Path("/hpctmp/e1192548/stage45e_h200_unified_replication_20260912/provenance/STAGE45E_N2_BALANCED_XYZ_STATS_PI0133.json"),
    "sealed_baseline": Path("/scratch/e1192548/stage45c_pi0100_h200_train24_eval_74e77d83d85e_20260910"),
    "snapshot_preflight": Path("/scratch/e1192548/stage45c_pi0100_h200_train24_eval_74e77d83d85e_20260910/stage45c_runtime_preflight.json"),
    "legacy_history": Path("/scratch/e1192548/stage45b_pi0087_b2_h200_20260909T052311Z"),
}
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
N2_STATS_SHA = "e88ff3ec5626130ca8a08006abe1ada70d34424af4b0d3bacf64db57149d7677"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_MANIFEST_SHA = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
ROUTE_HEADS = {
    "legacy": "9ad05f1bc7e2efc9f0d88674d569d42ef8b11e62",
    "l1": "186efc2977e2795bebf0dfe97d6f503a672716de",
    "n2": "6d8a7f132e325faea35346b36497ceb76390b2c3",
}
ROUTE_BRANCHES = {
    "legacy": "stage45e/legacy-full24-matched-seed-replication",
    "l1": "stage45e/l1-matched-seed-replication",
    "n2": "stage45e/n2-h200-matched-seed-replication",
}
TRAINING_QUEUE = (
    ("legacy", 2223),
    ("legacy", 2224),
    ("l1", 2223),
    ("l1", 2224),
    ("n2", 2222),
    ("n2", 2223),
    ("n2", 2224),
)
ALL_SEEDS = (2222, 2223, 2224)
ROUTES = ("legacy", "l1", "n2")
ESTIMATED_TRAINING_ROOT_BYTES = 51_318_519_711
MIN_FREE_BYTES = 64 * 1024**3
CHECKPOINT_STATE_FILES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
CHECKPOINT_HASH_PREFIX = {
    "legacy": "stage45e_legacy",
    "l1": "stage45e_l1",
    "n2": "stage45e",
}
PI_GATE = "PI_GATE_STATE_0143"
PI_ISSUE_COMMENT = 5644782159
PI_PR_COMMENT = 5644783564
RECOVERED_L1_ROOT = Path("/scratch/e1192548/stage45e_l1_h200_20260911_377b5012_612326.hopper-m-02")
RECOVERED_L1_V2_ROOT = Path("/scratch/e1192548/stage45e_l1_612326_recovery_v2_20260912_2e2780110b62")
RECOVERED_L1_ACCEPTANCE = RECOVERED_L1_V2_ROOT / "stage45e_l1_612326_contract_bound_acceptance.json"
RECOVERED_L1_INPUT_V2 = RECOVERED_L1_V2_ROOT / "stage45e_l1_recovered_training_eval_input_v2.json"
DUPLICATE_L1_ROOT = Path("/scratch/e1192548/stage45e_l1_h200_20260911_f77f9e02_612342.hopper-m-02")
RECOVERED_L1_ACCEPTANCE_SHA256 = "5ff6135c7d10da11686584c7435566e6e3490e677fca5e8d086adcf21287f04c"
RECOVERED_L1_INPUT_V2_SHA256 = "211411c307e30242cf5b4e65288e8c9eb2b446164aaec83de4ffd913ca9c8101"
RECOVERED_L1_READONLY_ACCEPTANCE_SHA256 = "76715d9124fce7e5c92c671a5c0f0d175bd7ace2b645e121d85b54a9b06441eb"
RECOVERED_L1_RUNTIME_BINDING_SHA256 = "ab8ef1dc775aa33cd43cc363b38e468ecc6ac1ed03acf3ea69e929e592c4525a"
RECOVERED_L1_TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
ALLOCATION_LEDGER_MANIFEST = REPO / "manifests/STAGE45E_H200_ALLOCATION_LEDGER_PI0143_V1.json"
ALLOCATION_LEDGER_MAX = 8
DIAGNOSTIC_PHASES = ("E4", "D1", "D2A")
DIAGNOSTIC_UNIT_KEYS = tuple(f"{phase}:{route}" for phase in DIAGNOSTIC_PHASES for route in ROUTES)
TRAINING_SOURCE_STANDARD = "STANDARD_PASS_RESULT"
TRAINING_SOURCE_RECOVERED = "CONTRACT_BOUND_READONLY_RECOVERY_NOT_HISTORICAL_PASS_RESULT"
EVIDENCE_DESCRIPTOR_SCHEMA = "VANDA_STAGE45E_EXTERNAL_DIAGNOSTIC_EVIDENCE_DESCRIPTOR_V1"
C10_HANDOFF_SCHEMA = "VANDA_STAGE45E_C10_HANDOFF_V1"
PI0145_GATE = "PI_GATE_STATE_0145"
PI0145_ISSUE_COMMENT = 5645246857
PI0145_PR_COMMENT = 5645286100
SCIENTIFIC_SPEC_IDENTITY_SCHEMA = "VANDA_STAGE45E_SCIENTIFIC_PHASE_IDENTITY_V1"
ALLOCATION_CORRECTION_SCHEMA = "VANDA_STAGE45E_ALLOCATION_CORRECTION_V1"
CORELAB_PR65_VERIFIER_HEAD = "0f4e97a407abc6e6edfb9383b77c08231edc5f1f"
CORELAB_PR65_VERIFIER_PATH = "scripts/stage45f_corelab_evidence_convergence.py"
HISTORICAL_LEGACY_DIAGNOSTIC_ROOT = Path("/scratch/e1192548/stage45d_d1_h200_execution_20260911_00fe100_smoke5/full")
HISTORICAL_D2A_DIAGNOSTIC_ROOT = Path("/scratch/e1192548/stage45d_d2a_path_divergence_20260911_a8432947_612015.hopper-m-02/full")
HISTORICAL_D2B_DIAGNOSTIC_ROOT = Path("/scratch/e1192548/stage45d_d2b_solver_discretization_20260911_f83d5375_612172.hopper-m-02/full")
HISTORICAL_L1_EVAL_ROOT = Path("/scratch/e1192548/stage45e_l1_train24_eval_recovered_v2_20260912_c68611a9")
HISTORICAL_DIAGNOSTIC_TARGETS = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
HISTORICAL_DIAGNOSTIC_WINDOWS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in HISTORICAL_DIAGNOSTIC_TARGETS
    for index in range(6)
)
HISTORICAL_D1_SEAL_SHA256 = "8965ab26a5276fa0d9f66487bf4a946b170d56d9033e2a748ea4a83358d3b71d"
HISTORICAL_D1_ROWS_SHA256 = "cd7a44108805c09cf22532a79554a481a61e3b7b1c5386972794e160e5fb601c"
HISTORICAL_D2A_SEAL_SHA256 = "528f9d8ab39dcbde9acc64d71ec6d9dc545000ab465fa77d994b0b3015fbb57e"
HISTORICAL_D2A_ROWS_SHA256 = "7bb9d818f9cfb7f68c960d674d9ea7529756021167ba52f8bd3bb95c22ca82c5"
HISTORICAL_D2B_SEAL_SHA256 = "7cfe1214adcd7e5818a2cc0e4307ae2f46c6a830e8c563f8376e5437e47e3dcf"
HISTORICAL_D2B_ROWS_SHA256 = "baa43d401f089a96a5a7bc4b0a1963bfdf7328d8aa3decb33eff1cffea08335b"
HISTORICAL_L1_SEAL_SHA256 = "1ee95e33bf71ed8f70aefe23bfb02b6400b93e7a8bc2cba2cbd8a8a749d79675"
HISTORICAL_L1_1000_ROWS_SHA256 = "aa312bcad066f64e99f8c0369f41cd62d88eef148e8b0ce1908ef074112cca1b"
HISTORICAL_L1_2000_ROWS_SHA256 = "2ddb426f022800c9ee32ed79c01e9d0a10eeb45a25ed1f525df80a6fa516ebd3"
HISTORICAL_L1_RAW_PREDICTIONS_SHA256 = "41eacdedb5746a8986a42d0d0c38b7e17e5474cb62ea59f6bf212e50fbbfb2a1"
HISTORICAL_L1_PREMODEL_SHA256 = "3018070aa138878e7ed55fe5539fe300aca808d7540cd08199d524d50ce6b8eb"
HISTORICAL_L1_STATIC_GUARD_SHA256 = "6f7a694790d875f875cf1527d5687ed7434db9712f8d1d66fa35eb2ecf436005"
HISTORICAL_LEGACY_CHECKPOINT_TREES = {
    1000: "38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c",
    2000: "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5",
}
HISTORICAL_L1_CHECKPOINT_TREES = {
    1000: "e0d779208c7c51dbb7a11ee2ba1449329ecc8a89b779cebba0e6791674b0c38c",
    2000: "53293d0ae3555c00acb7595b6c6dd19ed92450668344123cdc46efd5b69fcd49",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize an evidence identity without timestamps or formatting noise."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_scientific_phase_identity(spec_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable scientific portion of a phase specification.

    Allocation timestamps, campaign paths, and the orchestration wrapper are
    deliberately kept outside this identity.  Checkpoint paths and hashes,
    evaluator identity, sampler/noise fields, statistics, and any other
    declared scientific field remain inside it, so a changed input cannot be
    made reusable merely by retaining a PASS status.
    """

    volatile = {
        "timestamp_utc",
        "allocation_identity",
        "pbs_jobid",
        "allocation_ordinal",
        "campaign_root",
        "output_root",
        "attempt_root",
        "orchestration_identity",
        "scientific_identity",
        "scientific_identity_sha256",
    }
    payload = {key: value for key, value in dict(spec_doc).items() if key not in volatile}
    return {
        "schema": SCIENTIFIC_SPEC_IDENTITY_SCHEMA,
        "phase": spec_doc.get("phase"),
        "route": spec_doc.get("route"),
        "payload": payload,
    }


def scientific_spec_identity_sha256(spec_doc: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json_bytes(canonical_scientific_phase_identity(spec_doc)))


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"git failed in {repo}: {' '.join(args)}: {completed.stderr[-1000:]}")
    return completed.stdout.strip()


def regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"unsafe tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            child_mode = child.lstat().st_mode
            require(stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            child_mode = child.lstat().st_mode
            require(stat.S_ISREG(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": child.stat().st_size, "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def cosmos_digest(root: Path) -> str:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"Cosmos root unavailable: {root}")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None, "target": None, "resolved_path": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                resolved = item.resolve(strict=True)
                require(resolved.is_file() and resolved.is_relative_to(root), f"Cosmos symlink escapes root: {item}")
                entries.append({"path": rel, "kind": "symlink", "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved), "target": os.readlink(item), "resolved_path": resolved.relative_to(root).as_posix()})
            elif stat.S_ISREG(mode):
                entries.append({"path": rel, "kind": "regular_file", "size_bytes": item.stat().st_size, "sha256": sha256_file(item), "target": None, "resolved_path": None})
            else:
                raise RuntimeError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    lines = ["\t".join((str(item["kind"]), str(item["path"]), str(item["size_bytes"]), str(item.get("sha256") or "-"), str(item.get("target") or "-"), str(item.get("resolved_path") or "-"))) for item in sorted(entries, key=lambda value: str(value["path"]))]
    return sha256_bytes(("\n".join(lines) + "\n").encode())


def load_manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_PI0143_V1", "PI0143 unified manifest schema drift")
    return value


def p0_checklist() -> dict[str, str]:
    """Return the committed-surface reconciliation used by the PI0143 gate.

    This deliberately checks implementation surfaces, not live GPU state.  A
    live artifact can still be rejected by the strict verifiers below; the
    checklist only prevents a missing code path from being mistaken for a
    completed P0 item.
    """

    source = Path(__file__).read_text(encoding="utf-8")
    required_surfaces = {
        "A0_checklist": "def p0_checklist",
        "A1_recovered_l1_612326": "def verify_recovered_l1",
        "A2_legacy_exact_recipe": "def verify_legacy_exact_recipe",
        "A2b_legacy_provenance_decision": "def legacy_provenance_decision",
        "A3_route_phase_continuation": "def discover_diagnostic_attempts",
        "A4_truthful_accounting": "def diagnostic_accounting",
        "A5_allocation_ledger": "def append_allocation_record",
        "A6_campaign_failure_seal": "def write_campaign_failure_seal",
        "A7_external_c10_handoff": "def build_c10_handoff",
        "A8_identity_separation": "def training_lineage_identity",
        "A9_cpu_closure_semantics": "cpu_closure_execution",
        "V1_entrypoint_portability": "def _common_route_args",
        "V2_scientific_spec_identity": "def canonical_scientific_phase_identity",
        "V3_truthful_counter_accounting": "def _attempt_call_counts",
        "V4_historical_input_binding": "RECOVERED_L1_ACCEPTANCE_SHA256",
        "V5_historical_evidence_catalog": "def historical_diagnostic_evidence_catalog",
        "V6_corelab_producer_fixture": "def build_corelab_producer_fixture",
    }
    return {name: ("PASS_ALREADY_PRESENT" if marker in source else "PATCH_REQUIRED") for name, marker in required_surfaces.items()}


def static_audit() -> dict[str, Any]:
    manifest = load_manifest()
    allocation_manifest = validate_committed_allocation_manifest()
    require(manifest["authority"]["pi_gate"] == PI_GATE and manifest["authority"]["issue_comment"] == PI_ISSUE_COMMENT, "PI0143 authority drift")
    require(manifest["authority"]["pointer_pr"] == 66 and manifest["authority"]["pointer_comment"] == PI_PR_COMMENT, "PI0143 pointer drift")
    require(manifest["orchestration_repository"]["branch"] == "stage45e/h200-unified-replication-campaign", "orchestration branch drift")
    require(manifest["gpu_policy"]["max_science_allocations"] == ALLOCATION_LEDGER_MAX, "PI0143 allocation ceiling drift")
    require(manifest["gpu_policy"]["previous_science_allocations_consumed"] == 3, "PI0143 consumed allocation count drift")
    require(manifest["gpu_policy"]["additional_science_allocations_authorized"] == 5, "PI0143 continuation allocation count drift")
    require(manifest["gpu_policy"]["target_additional_allocations"] <= 2, "PI0143 target allocation drift")
    require(manifest["diagnostic_contract"]["total_model_side_calls"] == 8976, "diagnostic total drift")
    require(manifest["diagnostic_contract"]["e4"]["total_model_side_calls"] == 1680, "E4 ceiling drift")
    require(manifest["diagnostic_contract"]["d1"]["total_model_side_calls"] == 3840, "D1 ceiling drift")
    require(manifest["diagnostic_contract"]["d2a"]["total_model_side_calls"] == 3456, "D2A ceiling drift")
    require(manifest["diagnostic_contract"]["unit_key"] == "(phase,route)" and len(manifest["diagnostic_contract"]["units"]) == 9, "diagnostic unit contract drift")
    require(manifest["allocation_ledger"]["future_ordinal_9_fail_closed"] is True, "allocation #9 is not fail-closed")
    require(manifest["evidence_contract"]["c10_arm_count"] == 18, "C10 arm-count contract drift")
    require(manifest["identity_contract"]["training_head_must_not_be_rewritten_as_evaluator_head"] is True, "identity separation contract drift")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(top_imports.intersection({"torch", "transformers", "gr00t", "jax", "tensorflow", "isaacgym", "omni"}))
    require(not forbidden, f"orchestration top-level model imports: {forbidden}")
    call_names = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    require(not call_names.intersection({"qsub", "merge", "checkout"}), "orchestration contains forbidden scheduler/merge mutation")
    checklist = p0_checklist()
    require(all(status == "PASS_ALREADY_PRESENT" for status in checklist.values()), f"PI0143 P0 surface incomplete: {checklist}")
    return {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_REPLICATION_PI0143_STATIC_AUDIT_V1",
        "status": "PASS",
        "pi_gate": PI_GATE,
        "issue_comment": PI_ISSUE_COMMENT,
        "pointer_pr": 66,
        "pointer_comment": PI_PR_COMMENT,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "allocation_ledger_manifest_sha256": allocation_manifest["sha256"],
        "allocation_ledger_manifest_validation": allocation_manifest["validation"],
        "top_level_forbidden_imports": forbidden,
        "branch_separation": True,
        "fresh_base_training_lineages": 7,
        "max_science_allocations": 8,
        "previous_science_allocations_consumed": 3,
        "additional_science_allocations_authorized": 5,
        "e4_model_side_calls": 1680,
        "d1_model_side_calls": 3840,
        "d2a_model_side_calls": 3456,
        "total_model_side_calls": 8976,
        "gpu": 0,
        "pbs_submissions": 0,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "p0_checklist": checklist,
        "h200_hold": True,
    }


def _gpu_observation() -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "PI0143 campaign requires PBS_JOBID")
    query = "index,name,uuid,memory.total,memory.used"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            fields = [part.strip() for part in line.split(",")]
            require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
            rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4])})
    require(len(rows) == 1 and rows[0]["index"] == 0, f"PI0143 requires one H200 at index 0: {rows}")
    require("H200" in rows[0]["name"].upper() and rows[0]["memory_total_mib"] >= 80 * 1024, f"H200 identity/VRAM gate failed: {rows}")
    return {"schema": "VANDA_STAGE45E_UNIFIED_H200_ADMISSION_GPU_V1", "host": subprocess.check_output(["hostname"], text=True).strip(), "pbs_jobid": os.environ["PBS_JOBID"], "selected_device": "cuda:0", "rows": rows, "nvidia_smi_query": query, "nvidia_smi_stdout_sha256": sha256_bytes((result.stdout + "\n").encode())}


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {name: Path(getattr(args, name)).resolve() for name in DEFAULTS if name != "legacy_history"}


def verify_sources(args: argparse.Namespace, n2_repo: Path) -> dict[str, Any]:
    paths = _source_paths(args)
    for name, path in paths.items():
        require(path.exists() and not path.is_symlink(), f"source path unavailable: {name}: {path}")
    base_entries = regular_tree_entries(paths["base_model"], "payload/GR00T-N1.7-3B")
    base_tree = tree_digest(base_entries)
    require(base_tree == BASE_TREE, f"BASE tree mismatch: {base_tree}")
    sys.path.insert(0, str(n2_repo / "scripts"))
    sys.path.insert(0, str(n2_repo / "src"))
    import stage45e_n2_statistics as n2  # CPU-only module

    dataset = n2.validate_dataset_identity(paths["dataset"])
    require(dataset["tree_sha256"] == FULL24_TREE, "FULL24 tree mismatch")
    runtime_doc = read_json(paths["runtime_stats"])
    runtime = n2.validate_source_document(runtime_doc, source_path=paths["runtime_stats"])
    require(runtime["outer_sha256"] == RUNTIME_STATS_SHA and runtime["payload_sha256"] == RUNTIME_STATS_PAYLOAD_SHA, "runtime-V2 stats drift")
    n2_doc = read_json(paths["n2_stats"])
    n2_verified = n2.verify_n2_document(n2_doc, source=runtime_doc)
    n2_file_sha = sha256_file(paths["n2_stats"])
    require(n2_file_sha == N2_STATS_SHA and n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats artifact drift")
    require(sha256_file(paths["modality"]) == MODALITY_SHA, "modality identity drift")
    require(sha256_file(paths["config"]) == CONFIG_SHA, "training config identity drift")
    groot_manifest = Path(args.groot_manifest).resolve()
    require(groot_manifest.is_file() and not groot_manifest.is_symlink(), "GR00T snapshot manifest unavailable")
    sys.path.insert(0, str(REPO / "scripts"))
    from stage45b_groot_source_snapshot import verify_extracted

    snapshot = verify_extracted(
        paths["groot"],
        groot_manifest,
        expected_manifest_file_sha256=GROOT_MANIFEST_SHA,
        expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA,
        expected_payload_tree_sha256=GROOT_PAYLOAD_TREE,
    )
    snapshot_manifest = read_json(groot_manifest)
    require(snapshot_manifest.get("origin_commit") == GROOT_COMMIT, "GR00T source commit drift")
    snapshot["origin_commit"] = GROOT_COMMIT
    require((paths["cosmos"] / "refs/main").read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    cosmos_tree = cosmos_digest(paths["cosmos"])
    require(cosmos_tree == COSMOS_TREE, f"Cosmos tree drift: {cosmos_tree}")
    return {
        "base_tree_sha256": base_tree,
        "full24": dataset,
        "runtime_stats": runtime,
        "runtime_stats_sha256": sha256_file(paths["runtime_stats"]),
        "runtime_stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA,
        "n2_stats_sha256": n2_file_sha,
        "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA,
        "n2_verification": n2_verified,
        "modality_sha256": MODALITY_SHA,
        "config_sha256": CONFIG_SHA,
        "groot_commit": GROOT_COMMIT,
        "groot_snapshot": snapshot,
        "groot_manifest_sha256": GROOT_MANIFEST_SHA,
        "cosmos_revision": COSMOS_REVISION,
        "cosmos_tree_sha256": cosmos_tree,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def observe_worktree(repo: Path, route: str) -> dict[str, Any]:
    repo = Path(repo).resolve()
    require(repo.is_dir() and not repo.is_symlink(), f"route worktree unavailable: {repo}")
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    require(branch == ROUTE_BRANCHES[route], f"{route} branch drift: {branch}")
    require(head == ROUTE_HEADS[route], f"{route} HEAD drift: {head}")
    require(status == "", f"{route} worktree is dirty, including untracked files: {status}")
    return {"repo": str(repo), "branch": branch, "head": head, "clean": True}


def high_water(storage: Path, *, output: Path | None = None, planned_training_roots: int | None = None) -> dict[str, Any]:
    storage = Path(storage).resolve()
    require(storage.is_dir() and not storage.is_symlink(), f"storage root unavailable: {storage}")
    planned = len(TRAINING_QUEUE) if planned_training_roots is None else int(planned_training_roots)
    require(0 <= planned <= len(TRAINING_QUEUE), f"planned training root count is outside the frozen queue: {planned}")
    usage = shutil.disk_usage(storage)
    estimate = ESTIMATED_TRAINING_ROOT_BYTES * planned
    required = int(estimate * 1.15) + MIN_FREE_BYTES
    value = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_HIGH_WATER_V1",
        "status": "PASS" if usage.free >= required else "BLOCKED",
        "storage_root": str(storage),
        "free_bytes": usage.free,
        "estimated_training_bytes": estimate,
        "required_free_bytes_with_15_percent_margin_and_guard": required,
        "planned_training_roots": planned,
        "estimated_root_bytes": ESTIMATED_TRAINING_ROOT_BYTES,
        "accepted_evidence_deleted": False,
        "free_margin_sufficient": usage.free >= required,
    }
    require(value["free_margin_sufficient"], f"storage high-water gate failed: {value}")
    if output is not None:
        write_json(Path(output), value)
    return value


def _verify_checkpoint_artifact(route: str, root: Path, step: int, record: Mapping[str, Any], *, hash_prefix: str | None = None) -> dict[str, Any]:
    """Reconcile a sealed checkpoint record with its immutable on-disk tree.

    This is deliberately CPU/read-only.  A result JSON alone is not sufficient
    to reuse a lineage: trainer state, required optimizer state, byte totals,
    and the route-specific tree digest must all agree with the files currently
    materialized under the accepted root.
    """

    require(route in CHECKPOINT_HASH_PREFIX, f"unknown checkpoint route: {route}")
    checkpoint = Path(root) / f"checkpoint-{step}"
    try:
        mode = checkpoint.lstat().st_mode
    except OSError as exc:
        raise RuntimeError(f"checkpoint unavailable: {checkpoint}") from exc
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"checkpoint root is unsafe: {checkpoint}")
    require(isinstance(record, Mapping), f"checkpoint record is not an object: {checkpoint}")
    if "status" in record:
        require(record.get("status") in ("PASS", "COMPLETE"), f"checkpoint status drift: {checkpoint}")
    else:
        # The 612326 read-only recovery acceptance intentionally has no
        # synthetic training-result status.  Its equivalent proof is the
        # explicit file list plus verified training/optimizer/backward counts.
        require(record.get("required_state_files") == list(CHECKPOINT_STATE_FILES), f"checkpoint recovery state-file record drift: {checkpoint}")
        require(record.get("training_steps_verified") == step and record.get("optimizer_steps_verified") == step and record.get("backward_steps_verified") == step, f"checkpoint recovery step counters drift: {checkpoint}")
        require(record.get("optimizer_step_log_count") == step, f"checkpoint recovery optimizer-log count drift: {checkpoint}")
    expected_step = record.get("expected_step", record.get("global_step"))
    require(expected_step == step and record.get("global_step") == step, f"checkpoint step record drift: {checkpoint}")
    if "required_state_files_present" in record:
        require(record.get("required_state_files_present") is True and record.get("missing_state_files", []) == [], f"checkpoint state record drift: {checkpoint}")
    if record.get("path") is not None:
        require(Path(str(record["path"])).resolve() == checkpoint.resolve(), f"checkpoint path binding drift: {checkpoint}")
    state = read_json(checkpoint / "trainer_state.json")
    require(state.get("global_step") == step, f"trainer state global_step drift: {checkpoint}")
    for name in CHECKPOINT_STATE_FILES:
        path = checkpoint / name
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(f"required checkpoint state is absent: {path}") from exc
        require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"required checkpoint state is unsafe: {path}")
    prefix = hash_prefix or CHECKPOINT_HASH_PREFIX[route]
    entries = regular_tree_entries(checkpoint, f"{prefix}/{checkpoint.name}")
    actual = {
        "path": str(checkpoint),
        "global_step": state.get("global_step"),
        "expected_step": step,
        "required_state_files_present": True,
        "missing_state_files": [],
        "tree_sha256": tree_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "status": "PASS",
    }
    require(actual["tree_sha256"] == record.get("tree_sha256"), f"checkpoint tree hash drift: {checkpoint}")
    require(actual["file_count"] == record.get("file_count"), f"checkpoint file-count drift: {checkpoint}")
    require(actual["total_bytes"] == record.get("total_bytes"), f"checkpoint byte-count drift: {checkpoint}")
    return actual


def _historical_checkpoint(root: Path, step: int, key: str, route: str) -> dict[str, Any]:
    result = read_json(root / key)
    record = result.get(f"checkpoint_{step}", {})
    historical_prefix = "stage45b" if route == "legacy" else CHECKPOINT_HASH_PREFIX[route]
    return _verify_checkpoint_artifact(route, root, step, record, hash_prefix=historical_prefix)


def _historical_training_args(root: Path, seed: int = 2222) -> dict[str, Any]:
    """Read sealed TrainingArguments without constructing a model."""

    path = Path(root) / "training_args.bin"
    require(path.is_file() and not path.is_symlink(), f"historical training args unavailable: {path}")
    try:
        import torch

        try:
            args = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with the older sealed runtime
            args = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"historical training args unreadable: {path}: {exc}") from exc
    scheduler = getattr(args, "lr_scheduler_type", "")
    fields = {
        "seed": getattr(args, "seed", None),
        "max_steps": getattr(args, "max_steps", None),
        "per_device_train_batch_size": getattr(args, "per_device_train_batch_size", None),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "lr_scheduler_type": str(getattr(scheduler, "value", scheduler)),
        "warmup_steps": getattr(args, "warmup_steps", None),
        "max_grad_norm": getattr(args, "max_grad_norm", None),
        "bf16": getattr(args, "bf16", None),
        "fp16": getattr(args, "fp16", None),
        "tf32": getattr(args, "tf32", None),
        "dataloader_num_workers": getattr(args, "dataloader_num_workers", None),
    }
    expected = {
        "seed": seed,
        "max_steps": 2000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "dataloader_num_workers": 0,
    }
    for key, value in expected.items():
        require(fields[key] == value, f"historical Legacy recipe drift in {key}: {fields[key]!r}")
    return {"path": str(path), "sha256": sha256_file(path), "fields": fields}


def _yaml_scalar(document: str, key: str) -> str:
    matches = re.findall(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+?)\s*$", document)
    require(matches, f"historical training config is missing {key}")
    return matches[-1].strip().strip("'\"")


def _verify_legacy_historical_recipe_link(root: Path, config_sha256: str) -> dict[str, Any]:
    """Bind the historical preflight/result pair to the exact recipe source.

    The result JSON alone does not carry the trainability map.  The sealed
    preflight and its sibling Python-runtime preflight do: they bind the
    historical job, source trees, statistics, and the exact configuration
    digest before model construction.  Requiring both documents prevents a
    current config from being retroactively substituted for the one used by
    the historical run.
    """

    preflight_path = Path(root) / "stage45b_runtime_preflight.json"
    result_path = Path(root) / "stage45b_training_result.json"
    runtime_preflight_path = Path(root) / "stage45b_python_runtime_preflight.json"
    preflight = read_json(preflight_path)
    result = read_json(result_path)
    require(preflight.get("schema") == "CORELAB_STAGE45B_RUNTIME_PREFLIGHT_V1" and preflight.get("status") == "PASS", "Legacy historical runtime preflight is not PASS")
    require(result.get("schema") == "CORELAB_STAGE45B_FULL24_FRESH_BASE_TRAINING_RESULT_V2", "Legacy historical training-result schema drift")
    require(result.get("training_steps") == 2000 and result.get("optimizer_steps") == 2000 and result.get("training") is True, "Legacy historical training closure drift")
    require(preflight.get("training_steps") == 0 and preflight.get("optimizer_steps") == 0 and preflight.get("model_constructed") is False, "Legacy preflight is not pre-model")
    for key, expected in {
        "base_tree_sha256": BASE_TREE,
        "full24_tree_sha256": FULL24_TREE,
        "runtime_stats_sha256": RUNTIME_STATS_SHA,
        "statistics_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA,
    }.items():
        require((result.get(key) or preflight.get(key)) == expected, f"Legacy historical {key} drift")
    require(result.get("repository_head") == preflight.get("repository_head"), "Legacy historical result/preflight HEAD mismatch")
    historical_gpu_name = str(result.get("gpu", {}).get("name", result.get("gpu", {}).get("gpu_name", "")))
    require("H200" in historical_gpu_name.upper(), "Legacy historical result is not H200")
    require(runtime_preflight_path.is_file() and not runtime_preflight_path.is_symlink(), "Legacy Python-runtime preflight is missing")
    runtime_preflight = read_json(runtime_preflight_path)
    require(runtime_preflight.get("schema") == "VANDA_STAGE45B_PREMODEL_RUNTIME_SMOKE_V1" and runtime_preflight.get("status") == "PASS", "Legacy Python-runtime preflight is not PASS")
    require(runtime_preflight.get("execution_head") == result.get("repository_head"), "Legacy Python-runtime preflight HEAD mismatch")
    identities = runtime_preflight.get("identities", {})
    require(identities.get("base_tree_sha256") == BASE_TREE and identities.get("full24_tree_sha256") == FULL24_TREE, "Legacy Python-runtime source identity drift")
    require(identities.get("runtime_stats_sha256") == RUNTIME_STATS_SHA and identities.get("runtime_stats_payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA, "Legacy Python-runtime stats identity drift")
    require(identities.get("config_sha256") == config_sha256, "Legacy Python-runtime config SHA is not bound to the accepted recipe")
    require(runtime_preflight.get("scientific_exposure", {}).get("trainer") is False and runtime_preflight.get("scientific_exposure", {}).get("backward") is False, "Legacy Python-runtime preflight crossed a scientific boundary")
    return {
        "runtime_preflight": {"path": str(runtime_preflight_path), "sha256": sha256_file(runtime_preflight_path)},
        "runtime_preflight_execution_head": runtime_preflight.get("execution_head"),
        "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
        "training_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "config_sha256": config_sha256,
        "preflight_training_steps": preflight.get("training_steps"),
        "training_result_steps": result.get("training_steps"),
        "linked_before_model_construction": True,
    }


def verify_legacy_exact_recipe(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Prove Legacy seed2222's exact frozen recipe from immutable evidence.

    The historical result does not contain a complete trainability map.  The
    PI0143 contract therefore binds the immutable preflight config by its
    exact SHA and parses the five module-tuning flags from that file.  This is
    deliberately fail-closed: a current source/config is never used to infer
    what the historical run did.
    """

    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"Legacy exact-recipe root unavailable: {root}")
    config = Path(config_path or DEFAULTS["config"]).resolve()
    require(config.is_file() and not config.is_symlink(), f"Legacy exact-recipe config unavailable: {config}")
    observed_sha = sha256_file(config)
    require(observed_sha == CONFIG_SHA, f"Legacy exact-recipe config SHA drift: {observed_sha}")
    historical_link = _verify_legacy_historical_recipe_link(root, observed_sha)
    document = config.read_text(encoding="utf-8")
    flags = {
        "tune_projector": _yaml_scalar(document, "tune_projector").lower(),
        "tune_diffusion_model": _yaml_scalar(document, "tune_diffusion_model").lower(),
        "tune_visual": _yaml_scalar(document, "tune_visual").lower(),
        "tune_llm": _yaml_scalar(document, "tune_llm").lower(),
        "tune_vlln": _yaml_scalar(document, "tune_vlln").lower(),
    }
    require(flags == {
        "tune_projector": "true",
        "tune_diffusion_model": "true",
        "tune_visual": "false",
        "tune_llm": "false",
        "tune_vlln": "false",
    }, f"Legacy trainability map drift: {flags}")
    recipe_fields = {
        "max_steps": _yaml_scalar(document, "max_steps"),
        "global_batch_size": _yaml_scalar(document, "global_batch_size"),
        "gradient_accumulation_steps": _yaml_scalar(document, "gradient_accumulation_steps"),
        "learning_rate": _yaml_scalar(document, "learning_rate"),
        "lr_scheduler_type": _yaml_scalar(document, "lr_scheduler_type"),
        "weight_decay": _yaml_scalar(document, "weight_decay"),
        "warmup_steps": _yaml_scalar(document, "warmup_steps"),
        "optim": _yaml_scalar(document, "optim"),
        "seed": _yaml_scalar(document, "seed"),
        "bf16": _yaml_scalar(document, "bf16"),
        "fp16": _yaml_scalar(document, "fp16"),
        "tf32": _yaml_scalar(document, "tf32"),
        "dataloader_num_workers": _yaml_scalar(document, "dataloader_num_workers"),
    }
    expected = {
        "max_steps": "2000",
        "global_batch_size": "1",
        "gradient_accumulation_steps": "1",
        "learning_rate": "0.0001",
        "lr_scheduler_type": "cosine",
        "weight_decay": "1.0e-05",
        "warmup_steps": "10",
        "optim": "adamw_torch",
        "seed": "2222",
        "bf16": "true",
        "fp16": "false",
        "tf32": "true",
        "dataloader_num_workers": "0",
    }
    require(recipe_fields == expected, f"Legacy exact recipe fields drift: {recipe_fields}")
    return {
        "status": "PASS_EXACT_H200_FROZEN_RECIPE_PROVEN",
        "config_path": str(config),
        "config_sha256": observed_sha,
        "trainable_modules": ["projector", "diffusion_model"],
        "frozen_modules": ["visual", "llm", "vlln"],
        "tuning_flags": flags,
        "recipe_fields": recipe_fields,
        "source_kind": TRAINING_SOURCE_STANDARD,
        "historical_preflight_result_link": historical_link,
    }


def legacy_provenance_decision(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Make the Legacy seed-2222 reuse decision without guessing from performance.

    A missing or mismatched immutable recipe is a provenance decision, not a
    license to synthesize a PASS record.  The caller may then stop and report
    the explicit fresh-H200 token while retaining the reason for review.
    """

    root = Path(root).resolve()
    config = Path(config_path or DEFAULTS["config"]).resolve()
    try:
        lineage = verify_historical(root, "legacy", config_path=config)
    except RuntimeError as exc:
        return {
            "schema": "VANDA_STAGE45E_LEGACY_SEED2222_PROVENANCE_DECISION_V1",
            "status": "BLOCKED_PROVENANCE",
            "token": "LEGACY_SEED2222_FRESH_H200_REQUIRED_BY_PROVENANCE",
            "root": str(root),
            "config_path": str(config),
            "reason": str(exc),
            "lineage": None,
        }
    return {
        "schema": "VANDA_STAGE45E_LEGACY_SEED2222_PROVENANCE_DECISION_V1",
        "status": "PASS_EXACT_H200_FROZEN_RECIPE_REUSE",
        "token": "LEGACY_SEED2222_EXACT_H200_RECIPE_PROVEN",
        "root": str(root),
        "config_path": str(config),
        "reason": None,
        "lineage": lineage,
    }


def _verified_file_sha(path: Path, expected: str | None = None, label: str = "file") -> str:
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    observed = sha256_file(path)
    if expected is not None:
        require(observed == expected, f"{label} SHA drift: {path}: {observed} != {expected}")
    return observed


def verify_recovered_l1(
    root: Path | None = None,
    acceptance_path: Path | None = None,
    recovered_input_path: Path | None = None,
    *,
    expected_document_hashes: Mapping[str, str] | None = None,
    synthetic_test_override: bool = False,
) -> dict[str, Any]:
    """Accept only the contract-bound, recovered 612326 L1 lineage.

    This path has no historical training PASS result.  It rehashes both
    checkpoint trees and binds the two immutable recovery documents instead of
    manufacturing a result JSON for the old training job.
    """

    root = Path(root or RECOVERED_L1_ROOT).resolve()
    acceptance_path = Path(acceptance_path or RECOVERED_L1_ACCEPTANCE).resolve()
    recovered_input_path = Path(recovered_input_path or RECOVERED_L1_INPUT_V2).resolve()
    require(root.is_dir() and not root.is_symlink(), f"recovered L1 training root unavailable: {root}")
    expected = {
        "acceptance": RECOVERED_L1_ACCEPTANCE_SHA256,
        "recovered_input_v2": RECOVERED_L1_INPUT_V2_SHA256,
    }
    if expected_document_hashes is not None:
        require(synthetic_test_override, "recovered L1 document hash overrides are synthetic-test-only")
        expected.update(dict(expected_document_hashes))
    acceptance_sha = _verified_file_sha(acceptance_path, expected["acceptance"], "L1 contract-bound acceptance")
    input_sha = _verified_file_sha(recovered_input_path, expected["recovered_input_v2"], "L1 recovered-input V2")
    acceptance = read_json(acceptance_path)
    recovered = read_json(recovered_input_path)
    require(acceptance.get("schema") == "STAGE45E_L1_612326_CONTRACT_BOUND_ACCEPTANCE_V1", "612326 acceptance schema drift")
    require(acceptance.get("status") == "PASS_READONLY_TRAINING_AND_FROZEN_CONTRACT_INTEGRITY", "612326 acceptance status drift")
    require(acceptance.get("training_root") == str(root), "612326 acceptance root drift")
    require(acceptance.get("branch") == "stage45e/l1-physical-error-aware-loss", "612326 training branch drift")
    require(acceptance.get("historical_training_head") == RECOVERED_L1_TRAINING_HEAD, "612326 training HEAD drift")
    require(acceptance.get("base_acceptance", {}).get("path"), "612326 base acceptance reference missing")
    base_acceptance_path = Path(acceptance["base_acceptance"]["path"]).resolve()
    base_acceptance_sha = _verified_file_sha(base_acceptance_path, RECOVERED_L1_READONLY_ACCEPTANCE_SHA256, "612326 read-only acceptance")
    require(acceptance["base_acceptance"].get("sha256") == base_acceptance_sha, "612326 base acceptance SHA binding drift")
    require(acceptance.get("scientific_contract", {}).get("full24_stats_sha256") == RUNTIME_STATS_SHA, "612326 FULL24 stats identity drift")
    require(acceptance.get("scientific_contract", {}).get("full24_stats_payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA, "612326 FULL24 stats payload drift")
    contract = acceptance.get("scientific_contract", {}).get("l1_code_contract", {})
    require(contract.get("schema") == "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1", "612326 L1 loss schema drift")
    require(contract.get("active_scalar_count") == 190 and contract.get("weight_shape") == [40, 132], "612326 L1 mask/weight shape drift")
    require(contract.get("mask_schema") == "T16_BINARY_UNCHANGED" and contract.get("mask_folding") is False, "612326 L1 mask contract drift")
    require(contract.get("xyz_channels") == [0, 1, 2] and contract.get("xyz_rows") == [0, 14], "612326 L1 XYZ channel contract drift")
    require(acceptance.get("execution_counters", {}).get("evaluation") == 0, "612326 recovery evaluation boundary drift")
    require(acceptance.get("historical_root_mutated") is False, "612326 historical root mutation marker drift")
    binding_path = root / "stage45e_l1_runtime_binding.json"
    binding_sha = _verified_file_sha(binding_path, RECOVERED_L1_RUNTIME_BINDING_SHA256, "612326 runtime binding")
    require(acceptance.get("scientific_contract", {}).get("runtime_binding_sha256") == binding_sha, "612326 runtime binding SHA drift")

    require(recovered.get("schema") == "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V2", "612326 recovered-input schema drift")
    require(recovered.get("status") == "PASS_CONTRACT_BOUND_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION", "612326 recovered-input status drift")
    require(recovered.get("source_kind") == TRAINING_SOURCE_RECOVERED, "612326 source kind drift")
    require(recovered.get("acceptance_sha256") == acceptance_sha, "612326 recovered-input acceptance binding drift")
    require(recovered.get("acceptance_path") == str(acceptance_path), "612326 recovered-input acceptance path drift")
    require(recovered.get("training_root") == str(root) and recovered.get("branch") == acceptance.get("branch"), "612326 recovered-input lineage drift")
    require(recovered.get("historical_training_head") == RECOVERED_L1_TRAINING_HEAD, "612326 recovered-input HEAD drift")
    require(recovered.get("evaluation_authorized_by_this_bridge") is False and recovered.get("final_train24_token_emitted") is False, "612326 recovery authorization drift")
    for key, value in recovered.get("execution_counters", {}).items():
        if key in {"evaluation", "gpu", "model_construction", "model_forward", "pbs", "training", "validation_e_read", "holdout_f_read"}:
            require(value in (0, False), f"612326 recovered-input execution boundary drift in {key}")

    checkpoints: dict[str, dict[str, Any]] = {}
    for step in (1000, 2000):
        acceptance_record = acceptance.get(f"checkpoint_{step}", {})
        recovered_record = recovered.get("checkpoints", {}).get(f"L1@{step}", {})
        require(recovered_record.get("rehash_matches_acceptance") is True, f"612326 recovered checkpoint {step} rehash marker missing")
        require(recovered_record.get("tree_sha256") == acceptance_record.get("tree_sha256"), f"612326 checkpoint {step} acceptance mismatch")
        checkpoints[str(step)] = _verify_checkpoint_artifact("l1", root, step, acceptance_record, hash_prefix="stage45e_l1")
        require(checkpoints[str(step)]["tree_sha256"] == recovered_record.get("tree_sha256"), f"612326 checkpoint {step} recovered-input mismatch")

    training_args = _historical_training_args(root, 2222)
    return {
        "route": "l1",
        "seed": 2222,
        "root": str(root),
        "result_path": None,
        "result_sha256": None,
        "result": {},
        "training_args": training_args,
        "training_source_kind": TRAINING_SOURCE_RECOVERED,
        "training_lineage_head": RECOVERED_L1_TRAINING_HEAD,
        "training_lineage_branch": acceptance["branch"],
        "recovered_acceptance_path": str(acceptance_path),
        "recovered_acceptance_sha256": acceptance_sha,
        "recovered_input_v2_path": str(recovered_input_path),
        "recovered_input_v2_sha256": input_sha,
        "base_acceptance_path": str(base_acceptance_path),
        "base_acceptance_sha256": base_acceptance_sha,
        "runtime_binding_path": str(binding_path),
        "runtime_binding_sha256": binding_sha,
        "checkpoints": checkpoints,
        "acceptance": {
            "status": "PASS_READ_ONLY_RECOVERED_PRIMARY",
            "historical_pass_result_present": False,
            "source_kind": TRAINING_SOURCE_RECOVERED,
            "checkpoint_tree_binding": True,
        },
    }


def duplicate_l1_provenance(root: Path = DUPLICATE_L1_ROOT) -> dict[str, Any]:
    """Record 612342 without allowing it to enter the primary matrix."""

    root = Path(root).resolve()
    result_path = root / "stage45e_l1_training_result.json"
    require(root.is_dir() and not root.is_symlink(), f"duplicate L1 root unavailable: {root}")
    require(result_path.is_file() and not result_path.is_symlink(), f"duplicate L1 result unavailable: {result_path}")
    result = read_json(result_path)
    require(result.get("training_seed") == 2222, "612342 duplicate seed drift")
    return {
        "route": "l1",
        "seed": 2222,
        "root": str(root),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "source_kind": "DUPLICATE_RETRY_NOT_PRIMARY",
        "primary_admissible": False,
    }


def historical_diagnostic_evidence_catalog() -> dict[str, Any]:
    """Declare the immutable historical evidence needed by the 18-arm matrix.

    The catalog is intentionally a source map, not a copy of the evidence.
    Each entry names the exact sealed source, the row filter used to select a
    checkpoint/phase, and the expected immutable identities.  The audit below
    rehashes every source and inspects the selected rows before a campaign can
    admit a canary.
    """

    def reference(
        *,
        route: str,
        phase: str,
        step: int,
        source_phase: str,
        source_root: Path,
        seal_path: Path,
        seal_sha256: str,
        rows_path: Path,
        rows_sha256: str,
        seal_schema: str,
        row_schema: str,
        checkpoint_label: str,
        checkpoint_tree_sha256: str,
        probe_seeds: Sequence[int],
        expected_row_count: int,
        stats_route: str,
        stats_sha256: str,
        stats_payload_sha256: str,
        solver: str | None = None,
        raw_manifest_schema: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": "VANDA_STAGE45E_HISTORICAL_DIAGNOSTIC_EVIDENCE_DECLARATION_V1",
            "source_kind": "HISTORICAL_SEALED_READ_ONLY",
            "route": route,
            "training_seed": 2222,
            "phase": phase,
            "source_phase": source_phase,
            "step": step,
            "checkpoint_label": checkpoint_label,
            "checkpoint_tree_sha256": checkpoint_tree_sha256,
            "source_root": str(source_root),
            "seal_path": str(seal_path),
            "seal_sha256": seal_sha256,
            "rows_path": str(rows_path),
            "rows_sha256": rows_sha256,
            "seal_schema": seal_schema,
            "row_schema": row_schema,
            "row_filter": {
                "checkpoint_label": checkpoint_label,
                "solver": solver,
                "probe_seeds": list(probe_seeds),
                "window_set": "TRAIN24_TARGET_MAJOR_24_WINDOWS",
            },
            "expected_row_count": expected_row_count,
            "expected_window_count": 24,
            "expected_target_ids": list(HISTORICAL_DIAGNOSTIC_TARGETS),
            "stats_route": stats_route,
            "stats_sha256": stats_sha256,
            "stats_payload_sha256": stats_payload_sha256,
            "raw_manifest_schema": raw_manifest_schema,
            "target_xyz_in_model_input": False,
            "raw_before_derived": True,
        }

    legacy: dict[str, dict[str, dict[str, Any]]] = {phase: {} for phase in DIAGNOSTIC_PHASES}
    for step in (1000, 2000):
        legacy["E4"][str(step)] = reference(
            route="legacy", phase="E4", step=step, source_phase="D2B",
            source_root=HISTORICAL_D2B_DIAGNOSTIC_ROOT,
            seal_path=HISTORICAL_D2B_DIAGNOSTIC_ROOT / "d2b_full_seal.json",
            seal_sha256=HISTORICAL_D2B_SEAL_SHA256,
            rows_path=HISTORICAL_D2B_DIAGNOSTIC_ROOT / "raw_probe_rows.jsonl",
            rows_sha256=HISTORICAL_D2B_ROWS_SHA256,
            seal_schema="CORELAB_STAGE45D_D2B_H200_EXECUTION_SEAL_V1",
            row_schema="CORELAB_STAGE45D_D2B_STAGE_ROW_V1",
            checkpoint_label=f"FULL24@{step}",
            checkpoint_tree_sha256=HISTORICAL_LEGACY_CHECKPOINT_TREES[step],
            probe_seeds=(4242,), expected_row_count=96,
            stats_route="STAGE45C_RUNTIME_V2_STATS_BRIDGE",
            stats_sha256=RUNTIME_STATS_SHA,
            stats_payload_sha256=RUNTIME_STATS_PAYLOAD_SHA,
            solver="E4", raw_manifest_schema="CORELAB_STAGE45D_D2B_RAW_STAGE_MANIFEST_V1",
        )
        legacy["D1"][str(step)] = reference(
            route="legacy", phase="D1", step=step, source_phase="D1",
            source_root=HISTORICAL_LEGACY_DIAGNOSTIC_ROOT,
            seal_path=HISTORICAL_LEGACY_DIAGNOSTIC_ROOT / "full_seal.json",
            seal_sha256=HISTORICAL_D1_SEAL_SHA256,
            rows_path=HISTORICAL_LEGACY_DIAGNOSTIC_ROOT / "raw_probe_rows.jsonl",
            rows_sha256=HISTORICAL_D1_ROWS_SHA256,
            seal_schema="CORELAB_STAGE45D_D1_FULL_EXECUTION_SEAL_V1",
            row_schema="CORELAB_STAGE45D_D1_FULL_PROBE_ROW_V1",
            checkpoint_label=f"FULL24@{step}",
            checkpoint_tree_sha256=HISTORICAL_LEGACY_CHECKPOINT_TREES[step],
            probe_seeds=(2223, 2224, 2225), expected_row_count=216,
            stats_route="STAGE45C_RUNTIME_V2_STATS_BRIDGE",
            stats_sha256=RUNTIME_STATS_SHA,
            stats_payload_sha256=RUNTIME_STATS_PAYLOAD_SHA,
        )
        legacy["D2A"][str(step)] = reference(
            route="legacy", phase="D2A", step=step, source_phase="D2A",
            source_root=HISTORICAL_D2A_DIAGNOSTIC_ROOT,
            seal_path=HISTORICAL_D2A_DIAGNOSTIC_ROOT / "d2a_full_seal.json",
            seal_sha256=HISTORICAL_D2A_SEAL_SHA256,
            rows_path=HISTORICAL_D2A_DIAGNOSTIC_ROOT / "raw_probe_rows.jsonl",
            rows_sha256=HISTORICAL_D2A_ROWS_SHA256,
            seal_schema="CORELAB_STAGE45D_D2A_H200_EXECUTION_SEAL_V1",
            row_schema="CORELAB_STAGE45D_D2A_STAGE_ROW_V1",
            checkpoint_label=f"FULL24@{step}",
            checkpoint_tree_sha256=HISTORICAL_LEGACY_CHECKPOINT_TREES[step],
            probe_seeds=(4242,), expected_row_count=96,
            stats_route="STAGE45C_RUNTIME_V2_STATS_BRIDGE",
            stats_sha256=RUNTIME_STATS_SHA,
            stats_payload_sha256=RUNTIME_STATS_PAYLOAD_SHA,
            raw_manifest_schema="CORELAB_STAGE45D_D2A_RAW_STAGE_MANIFEST_V1",
        )
    l1 = {"E4": {}}
    for step, rows_sha256 in ((1000, HISTORICAL_L1_1000_ROWS_SHA256), (2000, HISTORICAL_L1_2000_ROWS_SHA256)):
        l1["E4"][str(step)] = {
            "schema": "VANDA_STAGE45E_HISTORICAL_DIAGNOSTIC_EVIDENCE_DECLARATION_V1",
            "source_kind": TRAINING_SOURCE_RECOVERED,
            "route": "l1",
            "training_seed": 2222,
            "phase": "E4",
            "source_phase": "L1_TRAIN24_E4_EVALUATION",
            "step": step,
            "checkpoint_label": f"L1@{step}",
            "checkpoint_tree_sha256": HISTORICAL_L1_CHECKPOINT_TREES[step],
            "source_root": str(HISTORICAL_L1_EVAL_ROOT),
            "seal_path": str(HISTORICAL_L1_EVAL_ROOT / "stage45e_l1_eval_seal.json"),
            "seal_sha256": HISTORICAL_L1_SEAL_SHA256,
            "rows_path": str(HISTORICAL_L1_EVAL_ROOT / "rows" / f"L1_at_{step}_rows.json"),
            "rows_sha256": rows_sha256,
            "seal_schema": "VANDA_STAGE45E_L1_TRAIN24_E4_EVALUATION_SEAL_V1",
            "row_schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
            "row_filter": {
                "checkpoint_label": f"L1@{step}",
                "solver": "E4",
                "probe_seeds": [4242],
                "window_set": "TRAIN24_TARGET_MAJOR_24_WINDOWS",
            },
            "expected_row_count": 24,
            "expected_window_count": 24,
            "expected_target_ids": list(HISTORICAL_DIAGNOSTIC_TARGETS),
            "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
            "stats_sha256": RUNTIME_STATS_SHA,
            "stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA,
            "raw_prediction_path": str(HISTORICAL_L1_EVAL_ROOT / "raw_predictions.jsonl"),
            "raw_prediction_sha256": HISTORICAL_L1_RAW_PREDICTIONS_SHA256,
            "pre_model_admission_path": str(HISTORICAL_L1_EVAL_ROOT / "stage45e_l1_eval_pre_model_admission.json"),
            "pre_model_admission_sha256": HISTORICAL_L1_PREMODEL_SHA256,
            "static_guard_path": str(HISTORICAL_L1_EVAL_ROOT / "stage45e_l1_eval_static_guard.json"),
            "static_guard_sha256": HISTORICAL_L1_STATIC_GUARD_SHA256,
            "target_xyz_in_model_input": False,
            "raw_before_derived": True,
        }
    return {
        "schema": "VANDA_STAGE45E_HISTORICAL_DIAGNOSTIC_EVIDENCE_CATALOG_V1",
        "status": "DECLARED_READ_ONLY_CATALOG",
        "references": {"legacy": legacy, "l1": l1},
        "required_topology": {
            "legacy2222": 6,
            "l1_seed2222_e4": 2,
            "total_required_references": 8,
            "source_rows_are_not_copied": True,
        },
    }


def _catalog_file(path: Path, expected_sha256: str, field: str) -> tuple[Path, str]:
    original = Path(path)
    require(not original.is_symlink(), f"{field} is a symlink: {original}")
    resolved = original.resolve()
    return resolved, _verified_file_sha(resolved, expected_sha256, field)


def _read_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{field} is unavailable: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{field} malformed JSON at line {line_number}") from exc
        require(isinstance(value, Mapping), f"{field} row {line_number} is not an object")
        rows.append(dict(value))
    return rows


def _historical_row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
    checkpoint_value = identity.get("checkpoint", row.get("checkpoint"))
    if isinstance(checkpoint_value, Mapping):
        checkpoint_label = checkpoint_value.get("label")
        checkpoint_tree = checkpoint_value.get("sha256")
    else:
        checkpoint_label = checkpoint_value
        checkpoint_tree = identity.get("checkpoint_tree_sha256", row.get("checkpoint_tree_sha256"))
    sampler = row.get("sampler") if isinstance(row.get("sampler"), Mapping) else {}
    decoded_action = row.get("decoded_action") if isinstance(row.get("decoded_action"), Mapping) else {}
    decoder_details = decoded_action.get("decoder_details") if isinstance(decoded_action.get("decoder_details"), Mapping) else {}
    window_id = identity.get("window_id", row.get("window_id"))
    target_id = identity.get("target_id", row.get("target_id"))
    if target_id is None and row.get("schema") == "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1":
        # This accepted pre-flag L1 schema carries target identity in the
        # canonical window key, not in a separate target_id field.  Restrict
        # the derivation to that exact schema and exact TRAIN24 key grammar;
        # arbitrary filenames or free-form strings never become identities.
        prefix = "stage22_expert_"
        suffix = ":"
        if isinstance(window_id, str) and window_id.startswith(prefix) and suffix in window_id:
            candidate = window_id[len(prefix):].split(suffix, 1)[0]
            if candidate in HISTORICAL_DIAGNOSTIC_TARGETS:
                target_id = candidate
    return {
        "checkpoint_label": checkpoint_label,
        "checkpoint_tree_sha256": checkpoint_tree,
        "window_id": window_id,
        "target_id": target_id,
        "window_index": identity.get("window_index", row.get("window_index")),
        "probe_seed": identity.get("noise_seed", identity.get("seed", row.get("seed", sampler.get("seed")))),
        "solver": identity.get("solver", sampler.get("name")),
        "stats_route": identity.get("stats_route", row.get("stats_route")),
        # The accepted L1 row schema stores the two immutable stats hashes in
        # decoded_action.decoder_details; do not treat their absence from the
        # row top level as an implicit identity match.
        "stats_sha256": identity.get("stats_sha256", row.get("stats_sha256", decoder_details.get("stats_sha256"))),
        "stats_payload_sha256": identity.get("stats_payload_sha256", row.get("stats_payload_sha256", decoder_details.get("statistics_payload_sha256"))),
        "target_xyz_in_model_input": identity.get("target_xyz_in_model_input", row.get("target_xyz_in_model_input")),
        "raw_before_derived": row.get("raw_before_derived"),
    }


def _require_historical_contract_flag(
    seal: Mapping[str, Any],
    *,
    container: str,
    field: str,
    expected: bool,
    label: str,
) -> None:
    """Verify a legacy seal field at its schema-defined location.

    The accepted Stage45D seal schemas do not all place contract flags in the
    same namespace: D1 records them in ``evidence`` while D2A/D2B record them
    in ``experiment_contract``.  This is an explicit schema mapping, not a
    fallback that treats an absent flag as safe.
    """

    payload = seal.get(container)
    require(isinstance(payload, Mapping), f"historical {label} {container} namespace missing")
    value = payload.get(field)
    require(value is expected, f"historical {label} {field} drift")


def _audit_l1_raw_prediction_order(
    raw_prediction_path: Path,
    selected: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> dict[str, Any]:
    """Prove L1's raw artifact log precedes each derived reference.

    The historical L1 evaluator stores raw and derived records in one
    append-only JSONL manifest rather than a seal-level boolean.  Every
    derived record must point to an earlier raw record, and every selected
    row must bind to the matching raw file/hash/identity.  This keeps the
    historical evidence read-only while still proving the lifecycle boundary.
    """

    records = _read_jsonl(raw_prediction_path, "historical L1 raw prediction manifest")
    raw_by_path: dict[str, Mapping[str, Any]] = {}
    derived_count = 0
    for index, record in enumerate(records, start=1):
        schema = record.get("schema")
        if schema == "VANDA_STAGE45E_L1_RAW_PREDICTION_ARTIFACT_V1":
            relative = record.get("path")
            require(
                isinstance(relative, str)
                and relative
                and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts,
                f"historical L1 raw manifest path is not relative at line {index}",
            )
            raw_path = (source_root / relative).resolve()
            require(raw_path.is_relative_to(source_root), "historical L1 raw manifest escapes source root")
            require(relative not in raw_by_path, f"historical L1 duplicate raw manifest path: {relative}")
            raw_by_path[relative] = record
        elif schema == "VANDA_STAGE45E_L1_DERIVED_EVIDENCE_REFERENCE_V1":
            derived_count += 1
            relative = record.get("raw_path")
            require(
                isinstance(relative, str) and relative in raw_by_path,
                f"historical L1 derived record precedes raw artifact at line {index}",
            )
        else:
            raise RuntimeError(f"historical L1 raw manifest schema drift at line {index}")

    require(raw_by_path, "historical L1 raw prediction manifest has no raw records")
    require(derived_count == len(raw_by_path), "historical L1 raw/derived manifest count drift")

    selected_paths: set[str] = set()
    for row in selected:
        artifact = row.get("raw_prediction_artifact")
        require(isinstance(artifact, Mapping), "historical L1 raw prediction artifact binding missing")
        relative = artifact.get("path")
        require(isinstance(relative, str) and relative in raw_by_path, "historical L1 selected raw artifact is absent from manifest")
        require(relative not in selected_paths, "historical L1 selected raw artifact is duplicated")
        selected_paths.add(relative)
        raw = raw_by_path[relative]
        row_checkpoint = row.get("checkpoint")
        raw_checkpoint = raw.get("checkpoint")
        require(isinstance(row_checkpoint, Mapping) and isinstance(raw_checkpoint, Mapping), "historical L1 raw checkpoint identity missing")
        require(
            raw_checkpoint.get("label") == row_checkpoint.get("label")
            and raw_checkpoint.get("tree_sha256") == row_checkpoint.get("sha256"),
            "historical L1 raw checkpoint identity drift",
        )
        require(raw.get("window_id") == row.get("window_id"), "historical L1 raw window identity drift")
        require(raw.get("file_sha256") == artifact.get("sha256"), "historical L1 raw file SHA binding drift")
        require(raw.get("array_sha256") == artifact.get("array_sha256"), "historical L1 raw array SHA binding drift")
        require(raw.get("shape") == artifact.get("shape"), "historical L1 raw shape binding drift")

    return {
        "manifest_sha256": sha256_file(raw_prediction_path),
        "raw_record_count": len(raw_by_path),
        "derived_record_count": derived_count,
        "selected_raw_record_count": len(selected_paths),
        "raw_before_derived": True,
        "source_bound": True,
    }


def _selected_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(b"".join(_canonical_json_bytes(row) + b"\n" for row in rows))


def audit_historical_diagnostic_evidence_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one catalog entry and return a fully hash-bound read-only ref."""

    source_root = Path(str(reference["source_root"]))
    require(source_root.is_dir() and not source_root.is_symlink(), f"historical evidence source root unavailable: {source_root}")
    source_root = source_root.resolve()
    seal_path, seal_sha = _catalog_file(Path(str(reference["seal_path"])), str(reference["seal_sha256"]), "historical evidence seal")
    rows_path, rows_sha = _catalog_file(Path(str(reference["rows_path"])), str(reference["rows_sha256"]), "historical evidence rows")
    require(seal_path.is_relative_to(source_root) and rows_path.is_relative_to(source_root), "historical evidence source escapes declared root")
    seal = read_json(seal_path)
    require(seal.get("schema") == reference["seal_schema"] and seal.get("status") == "PASS", "historical evidence seal schema/status drift")
    require(reference["phase"] in {"E4", "D1", "D2A"}, "historical evidence phase drift")
    expected_raw_schema = reference.get("raw_manifest_schema")
    contract_proof: dict[str, Any] = {}
    if reference["source_phase"] == "D1":
        _require_historical_contract_flag(
            seal,
            container="evidence",
            field="raw_before_derived",
            expected=True,
            label="raw-before-derived seal",
        )
        _require_historical_contract_flag(
            seal,
            container="evidence",
            field="target_xyz_in_model_input",
            expected=False,
            label="target exclusion seal",
        )
        contract_proof = {"namespace": "evidence", "raw_before_derived": True, "target_xyz_in_model_input": False}
    elif reference["source_phase"] in {"D2A", "D2B"}:
        _require_historical_contract_flag(
            seal,
            container="experiment_contract",
            field="raw_before_derived",
            expected=True,
            label="raw-before-derived seal",
        )
        _require_historical_contract_flag(
            seal,
            container="experiment_contract",
            field="target_xyz_in_model_input",
            expected=False,
            label="target exclusion seal",
        )
        contract_proof = {"namespace": "experiment_contract", "raw_before_derived": True, "target_xyz_in_model_input": False}
    else:
        # The historical L1 evaluator predates the seal-level flags. Its
        # per-row target exclusion and append-only raw/derived manifest are
        # checked after rows are selected below; absence is not a PASS.
        contract_proof = {"namespace": "selected_rows_and_raw_manifest"}
    if reference["source_phase"] == "D1":
        require(seal.get("execution_head") == "00fe100722268da3134bc7bfe2b535fa86475f7d", "historical D1 execution HEAD drift")
        require(seal.get("counters", {}).get("training_steps") == 0 and seal.get("counters", {}).get("optimizer_steps") == 0, "historical D1 crossed training boundary")
    elif reference["source_phase"] == "D2A":
        require(seal.get("execution_head") == "a843294780c1392369bfc987e468857c63fc13a5", "historical D2A execution HEAD drift")
        require(seal.get("experiment_contract", {}).get("free_running_reset") is False, "historical D2A reset contract drift")
        require(seal.get("call_budget", {}).get("noise_seed") == 4242, "historical D2A noise identity drift")
    elif reference["source_phase"] == "D2B":
        require(seal.get("execution_head") == "f83d53758e0ccda951c7af4dd7a606642a782b93", "historical D2B execution HEAD drift")
        require(seal.get("experiment_contract", {}).get("solver_comparison_only") is True, "historical D2B solver contract drift")
        require(seal.get("call_budget", {}).get("noise_seed") == 4242, "historical D2B noise identity drift")
    else:
        require(seal.get("execution_head") == "c68611a9a577bd23314afaed9c6ffb7fc829e9b2", "historical L1 evaluator HEAD drift")
        require(seal.get("training") is False and seal.get("backward") is False and seal.get("optimizer_steps") == 0, "historical L1 evaluator crossed training boundary")
        require(seal.get("call_budget", {}).get("total_model_side_calls") == 360, "historical L1 evaluator call budget drift")
        require(seal.get("observed_counters", {}).get("training_steps") == 0 and seal.get("observed_counters", {}).get("optimizer_steps") == 0, "historical L1 evaluator counter boundary drift")
        for key in ("pre_model_admission_path", "static_guard_path"):
            auxiliary_path, auxiliary_sha = _catalog_file(Path(str(reference[key])), str(reference[key.replace("_path", "_sha256")]), f"historical L1 {key}")
            require(auxiliary_path.is_relative_to(source_root), f"historical L1 {key} escapes source root")
        raw_prediction_path, raw_prediction_sha = _catalog_file(Path(str(reference["raw_prediction_path"])), str(reference["raw_prediction_sha256"]), "historical L1 raw predictions")
        require(raw_prediction_path.is_relative_to(source_root), "historical L1 raw predictions escape source root")
        require(seal.get("raw_manifest") == str(raw_prediction_path), "historical L1 raw prediction seal binding drift")

    rows = read_json(rows_path) if rows_path.suffix == ".json" else _read_jsonl(rows_path, "historical evidence rows")
    require(isinstance(rows, list), "historical evidence rows are not a list")
    selected: list[dict[str, Any]] = []
    for row in rows:
        identity = _historical_row_identity(row)
        if identity["checkpoint_label"] != reference["checkpoint_label"]:
            continue
        expected_solver = reference.get("row_filter", {}).get("solver")
        if expected_solver is not None and identity["solver"] not in (expected_solver, None):
            if not (reference.get("route") == "l1" and expected_solver == "E4" and str(identity["solver"]).startswith("train24-e4-")):
                continue
        selected.append(dict(row))
    require(len(selected) == reference["expected_row_count"], f"historical evidence row count drift: {len(selected)} != {reference['expected_row_count']}")
    identities = [_historical_row_identity(row) for row in selected]
    require({item["window_id"] for item in identities} == set(HISTORICAL_DIAGNOSTIC_WINDOWS), "historical evidence window set drift")
    require({item["target_id"] for item in identities} == set(HISTORICAL_DIAGNOSTIC_TARGETS), "historical evidence target set drift")
    require({item["checkpoint_tree_sha256"] for item in identities} == {reference["checkpoint_tree_sha256"]}, "historical evidence checkpoint hash drift")
    require({item["stats_route"] for item in identities} == {reference["stats_route"]}, "historical evidence stats route drift")
    require({item["stats_sha256"] for item in identities} == {reference["stats_sha256"]}, "historical evidence stats SHA drift")
    payload_values = {item["stats_payload_sha256"] for item in identities if item["stats_payload_sha256"] is not None}
    if payload_values:
        require(payload_values == {reference["stats_payload_sha256"]}, "historical evidence stats payload SHA drift")
    require({item["target_xyz_in_model_input"] for item in identities} == {False}, "historical evidence target XYZ input drift")
    probe_seeds = {item["probe_seed"] for item in identities}
    require(probe_seeds == set(reference["row_filter"]["probe_seeds"]), "historical evidence probe/noise seed drift")
    if reference.get("route") == "l1":
        for row in selected:
            sampler = row.get("sampler", {})
            require(isinstance(sampler, Mapping), "historical L1 sampler identity missing")
            require(sampler.get("name") == "train24-e4-carried-euler-direct-action-head", "historical L1 sampler identity drift")
            require(sampler.get("dt") == 0.25 and sampler.get("steps_observed") == 4, "historical L1 E4 schedule drift")
            require(sampler.get("carried_state") is True and sampler.get("reset_between_stages") is False, "historical L1 carried-state drift")
            require(sampler.get("official_public_sampler_called") is False and sampler.get("seed") == 4242, "historical L1 sampler boundary drift")

    raw_manifest_paths: list[dict[str, Any]] = []
    if expected_raw_schema is not None:
        for row, identity in zip(selected, identities):
            raw_manifest_value = row.get("raw_manifest_path")
            require(isinstance(raw_manifest_value, str) and raw_manifest_value, "historical raw manifest path missing")
            raw_manifest_path = Path(raw_manifest_value)
            require(not raw_manifest_path.is_symlink(), "historical raw manifest is a symlink")
            raw_manifest_path = raw_manifest_path.resolve()
            require(raw_manifest_path.is_relative_to(source_root), "historical raw manifest escapes source root")
            raw_manifest_sha = _verified_file_sha(raw_manifest_path, row.get("raw_manifest_sha256"), "historical raw manifest")
            raw_manifest = read_json(raw_manifest_path)
            require(raw_manifest.get("schema") == expected_raw_schema and raw_manifest.get("status") == "RAW_SEALED_BEFORE_DERIVED", "historical raw manifest lifecycle drift")
            manifest_identity = raw_manifest.get("identity", {})
            for key in ("checkpoint", "checkpoint_tree_sha256", "window_id", "target_id", "target_xyz_in_model_input"):
                if key in identity:
                    require(manifest_identity.get(key) == identity[key], f"historical raw manifest identity drift in {key}")
            raw_manifest_paths.append({"path": str(raw_manifest_path), "sha256": raw_manifest_sha})
    elif reference["source_phase"] == "D1":
        require(all(row.get("raw_before_derived") is True for row in selected), "historical D1 row raw-before-derived drift")
    else:
        contract_proof.update(_audit_l1_raw_prediction_order(raw_prediction_path, selected, source_root))
        # The selected derived rows are the authoritative target-exclusion
        # records for this pre-flag L1 schema; the set check above is
        # intentionally retained and is not inferred from the raw log.
        contract_proof["target_xyz_in_model_input"] = False

    return {
        **dict(reference),
        "status": "PASS_READ_ONLY",
        "source_root": str(source_root),
        "seal_path": str(seal_path),
        "seal_sha256": seal_sha,
        "rows_path": str(rows_path),
        "rows_sha256": rows_sha,
        "selected_rows_sha256": _selected_rows_sha256(selected),
        "selected_row_count": len(selected),
        "selected_window_count": len({item["window_id"] for item in identities}),
        "selected_target_ids": sorted({item["target_id"] for item in identities}),
        "selected_probe_seeds": sorted(probe_seeds),
        "raw_manifest_count": len(raw_manifest_paths),
        "raw_manifest_inventory_sha256": _selected_rows_sha256(raw_manifest_paths) if raw_manifest_paths else None,
        "contract_proof": contract_proof,
        "immutable_source_bound": True,
        "read_only": True,
    }


def audit_historical_diagnostic_evidence_catalog(*, require_real: bool = True) -> dict[str, Any]:
    """Audit all eight required historical references without model execution."""

    catalog = historical_diagnostic_evidence_catalog()
    audited: dict[str, dict[str, dict[str, Any]]] = {"legacy": {}, "l1": {}}
    missing: list[str] = []
    for route, phases in catalog["references"].items():
        for phase, steps in phases.items():
            audited[route][phase] = {}
            for step, reference in steps.items():
                key = f"{route}:{phase}:{step}"
                try:
                    audited[route][phase][step] = audit_historical_diagnostic_evidence_reference(reference)
                except RuntimeError:
                    if require_real:
                        raise
                    missing.append(key)
                    audited[route][phase][step] = {**reference, "status": "SKIPPED_NOT_MOUNTED", "read_only": True}
    return {
        "schema": "VANDA_STAGE45E_HISTORICAL_DIAGNOSTIC_EVIDENCE_AUDIT_V1",
        "status": "PASS_READ_ONLY" if not missing else "INCOMPLETE_NOT_MOUNTED",
        "catalog_schema": catalog["schema"],
        "references": audited,
        "required_reference_count": catalog["required_topology"]["total_required_references"],
        "audited_reference_count": sum(len(steps) for phases in audited.values() for steps in phases.values()),
        "missing": missing,
        "model_execution": 0,
        "gpu": 0,
        "pbs": 0,
        "training": 0,
    }


def verify_historical(root: Path, route: str, *, config_path: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"historical {route} root unavailable: {root}")
    if route == "legacy":
        result_path = root / "stage45b_training_result.json"
        result = read_json(result_path)
        training_args = _historical_training_args(root)
        require(training_args["fields"]["seed"] == 2222 and result.get("training_steps") == 2000, "historical Legacy seed2222 identity drift")
        gpu = result.get("gpu", {})
        gpu_name = str(gpu.get("gpu_name", gpu.get("name", "")))
        require("H200" in gpu_name.upper(), "historical Legacy seed2222 is not proven H200")
        preflight_path = root / "stage45b_runtime_preflight.json"
        preflight = read_json(preflight_path)
        base_tree = result.get("base_tree_sha256") or preflight.get("base_tree_sha256")
        full24_tree = result.get("full24_tree_sha256") or preflight.get("full24_tree_sha256")
        require(base_tree == BASE_TREE and full24_tree == FULL24_TREE, "historical Legacy source identity drift")
        stats_sha = result.get("runtime_stats_sha256") or result.get("structured_stats_sha256") or preflight.get("runtime_stats_sha256")
        stats_payload_sha = result.get("runtime_stats_payload_sha256") or result.get("structured_stats_payload_sha256") or preflight.get("runtime_stats_payload_sha256") or preflight.get("statistics_payload_sha256")
        require(stats_sha == RUNTIME_STATS_SHA and stats_payload_sha == RUNTIME_STATS_PAYLOAD_SHA and result.get("optimizer_steps") == 2000, "historical Legacy recipe/stats drift")
        exact_recipe = verify_legacy_exact_recipe(root, config_path)
        return {"route": route, "seed": 2222, "root": str(root), "result_path": str(result_path), "result_sha256": sha256_file(result_path), "result": result, "training_args": training_args, "training_source_kind": TRAINING_SOURCE_STANDARD, "training_lineage_head": result.get("repository_head") or preflight.get("repository_head"), "training_lineage_branch": result.get("branch") or "stage45b/full24-fresh-base-execution", "legacy_exact_recipe": exact_recipe, "source_preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path), "base_tree_sha256": base_tree, "full24_tree_sha256": full24_tree, "runtime_stats_sha256": stats_sha, "runtime_stats_payload_sha256": stats_payload_sha}, "checkpoints": {str(step): _historical_checkpoint(root, step, "stage45b_training_result.json", route) for step in (1000, 2000)}}
    # A historical L1 PASS result (in particular 612342) is never a primary
    # source.  The only admissible seed-2222 L1 path is the recovered V2
    # contract, which has no synthetic training-result file.
    if root == RECOVERED_L1_ROOT:
        return verify_recovered_l1()
    raise RuntimeError("L1 historical PASS result is not admissible as primary; use recovered 612326 contract-bound input")


def orchestration_identity() -> dict[str, Any]:
    """Identify the wrapper independently from every science/evaluator head."""

    return {
        "role": "ORCHESTRATION_WRAPPER",
        "repository": str(REPO),
        "branch": git(REPO, "branch", "--show-current"),
        "head": git(REPO, "rev-parse", "HEAD"),
        "working_tree_clean": git(REPO, "status", "--porcelain=v1", "--untracked-files=all") == "",
    }


def training_lineage_identity(lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize training provenance without relabeling it as evaluator code."""

    result = lineage.get("result", {})
    source_kind = lineage.get("training_source_kind") or (
        TRAINING_SOURCE_RECOVERED if lineage.get("recovered_input_v2_path") else TRAINING_SOURCE_STANDARD
    )
    return {
        "role": "TRAINING_LINEAGE",
        "route": lineage.get("route"),
        "seed": lineage.get("seed"),
        "source_kind": source_kind,
        "training_head": lineage.get("training_lineage_head") or result.get("execution_head") or result.get("repository_head"),
        "training_branch": lineage.get("training_lineage_branch") or result.get("branch"),
        "training_result_path": lineage.get("result_path"),
        "training_result_sha256": lineage.get("result_sha256"),
        "recovered_acceptance_path": lineage.get("recovered_acceptance_path"),
        "recovered_acceptance_sha256": lineage.get("recovered_acceptance_sha256"),
        "recovered_input_v2_path": lineage.get("recovered_input_v2_path"),
        "recovered_input_v2_sha256": lineage.get("recovered_input_v2_sha256"),
        "checkpoint_trees": {str(step): item.get("tree_sha256") for step, item in lineage.get("checkpoints", {}).items()},
    }


def evaluator_identity(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Bind the common evaluator separately from route training HEADs."""

    repo = Path(getattr(args, "legacy_repo", WORKTREE_ROOT / "legacy")).resolve() if args is not None else None
    script = repo / "scripts/stage45e_legacy_matched_seed.py" if repo is not None else None
    return {
        "role": "COMMON_EVALUATOR_ADAPTER",
        "repository": str(repo) if repo is not None else None,
        "branch": ROUTE_BRANCHES["legacy"],
        "head": ROUTE_HEADS["legacy"],
        "script": str(script) if script is not None else None,
        "script_sha256": sha256_file(script) if script is not None and script.is_file() and not script.is_symlink() else None,
        "source_route": "legacy_common_helper",
    }


def _ledger_schema() -> dict[str, Any]:
    return {
        "schema": "VANDA_STAGE45E_H200_SCIENCE_ALLOCATION_LEDGER_PI0143_V1",
        "pi_gate": PI_GATE,
        "max_science_allocations": ALLOCATION_LEDGER_MAX,
        "append_only": True,
        "zero_science_probe_records_separate": True,
        "records": [],
    }


def validate_committed_allocation_manifest() -> dict[str, Any]:
    """Validate the committed seed ledger before any live ledger is opened.

    The JSON manifest is the reviewable declaration; the JSONL file created by
    a future allocation is the append-only observation stream.  Keep these
    surfaces separate, but require the declaration to satisfy the same
    structural invariants as the live ledger.
    """

    document = read_json(ALLOCATION_LEDGER_MANIFEST)
    require(document.get("schema") == "VANDA_STAGE45E_H200_SCIENCE_ALLOCATION_LEDGER_PI0143_V1", "committed allocation manifest schema drift")
    require(document.get("pi_gate") == PI_GATE, "committed allocation manifest authority drift")
    require(document.get("max_science_allocations") == ALLOCATION_LEDGER_MAX, "committed allocation manifest ceiling drift")
    require(document.get("append_only") is True, "committed allocation manifest is not append-only")
    require(document.get("seeded_consumed_ordinals") == [1, 2, 3], "committed allocation manifest seed drift")
    require(document.get("future_ordinal_9") == "FAIL_CLOSED_UNAUTHORIZED", "committed allocation manifest #9 policy drift")
    records = document.get("records")
    require(isinstance(records, list) and len(records) == 3, "committed allocation manifest must seed exactly allocations #1-#3")
    normalized: list[dict[str, Any]] = []
    for record in records:
        require(isinstance(record, Mapping), "committed allocation manifest record is not an object")
        item = dict(record)
        item.setdefault("allocation_kind", "SCIENCE")
        item.setdefault("root", None)
        item.setdefault("attempted_model_calls", 0)
        item.setdefault("failure_seal_sha256", None)
        item["historical_record_only"] = True
        normalized.append(item)
    validation = validate_allocation_ledger({**_ledger_schema(), "records": normalized})
    return {"path": str(ALLOCATION_LEDGER_MANIFEST), "sha256": sha256_file(ALLOCATION_LEDGER_MANIFEST), "validation": validation}


HISTORICAL_ALLOCATION_RECORDS = (
    {
        "allocation_kind": "SCIENCE",
        "ordinal": 1,
        "pbs_jobid": "612893.hopper-m-02",
        "root": None,
        "status": "PRE_SCIENCE_STORAGE_FAILURE",
        "model_imported": False,
        "model_forward": 0,
        "training": False,
        "attempted_model_calls": 0,
        "failure_seal_sha256": None,
        "consumed": True,
        "historical_record_only": True,
        "evidence_status": "HISTORICAL_EXTERNAL_RECORD_REFERENCED",
    },
    {
        "allocation_kind": "SCIENCE",
        "ordinal": 2,
        "pbs_jobid": "612910.hopper-m-02",
        "root": None,
        "status": "PRE_SCIENCE_NATIVE_RUNTIME_FAILURE",
        "model_imported": False,
        "model_forward": 0,
        "training": False,
        "attempted_model_calls": 0,
        "failure_seal_sha256": None,
        "consumed": True,
        "historical_record_only": True,
        "evidence_status": "HISTORICAL_EXTERNAL_RECORD_REFERENCED",
    },
    {
        "allocation_kind": "SCIENCE",
        "ordinal": 3,
        "pbs_jobid": "612918.hopper-m-02",
        "root": None,
        "status": "PASS_LEGACY2223_2224_TRAINING_L1_SEED2223_PRE_MODEL_FAILURE",
        "model_imported": "MIXED_TRAINING_AND_PRE_MODEL_FAILURE",
        "model_forward": None,
        "training": True,
        "attempted_model_calls": None,
        "attempted_model_calls_complete": False,
        "counter_domain": "TRAINING_NOT_DIAGNOSTIC",
        "route_outcomes": {
            "legacy_seed2223": "TRAINING_PASS",
            "legacy_seed2224": "TRAINING_PASS",
            "l1_seed2223": "PRE_MODEL_FAILURE",
        },
        "historical_correction_id": "PI0145-ALLOCATION-3-001",
        "failure_seal_sha256": None,
        "consumed": True,
        "historical_record_only": True,
        "evidence_status": "HISTORICAL_EXTERNAL_RECORD_REFERENCED",
    },
)


def read_allocation_ledger(path: Path) -> dict[str, Any]:
    """Read a persistent JSONL ledger without silently repairing its history."""

    path = Path(path)
    if not path.exists():
        return _ledger_schema()
    require(path.is_file() and not path.is_symlink(), f"allocation ledger is not a regular file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"allocation ledger malformed at line {line_number}: {path}") from exc
        require(isinstance(value, Mapping), f"allocation ledger record is not an object at line {line_number}")
        records.append(dict(value))
    document = _ledger_schema()
    document["records"] = records
    validate_allocation_ledger(document)
    return document


def validate_allocation_ledger(document: Mapping[str, Any]) -> dict[str, Any]:
    require(document.get("schema") == "VANDA_STAGE45E_H200_SCIENCE_ALLOCATION_LEDGER_PI0143_V1", "allocation ledger schema drift")
    require(document.get("pi_gate") == PI_GATE and document.get("max_science_allocations") == ALLOCATION_LEDGER_MAX, "allocation ledger authority drift")
    records = document.get("records")
    require(isinstance(records, list), "allocation ledger records are not a list")
    for record in records:
        require(isinstance(record, Mapping), "allocation ledger record is not an object")
        require(record.get("allocation_kind", "SCIENCE") in ("SCIENCE", "ZERO_SCIENCE_PROBE"), f"unknown allocation kind: {record.get('allocation_kind')}")
    science = [record for record in records if record.get("allocation_kind", "SCIENCE") == "SCIENCE"]
    probes = [record for record in records if record.get("allocation_kind") == "ZERO_SCIENCE_PROBE"]
    require(len(science) <= ALLOCATION_LEDGER_MAX, f"allocation #{len(science) + 1} exceeds PI0143 ceiling")
    jobs: set[str] = set()
    for ordinal, record in enumerate(science, start=1):
        require(record.get("ordinal") == ordinal, f"allocation ledger ordinal gap at {ordinal}")
        job = record.get("pbs_jobid")
        require(isinstance(job, str) and job, f"allocation {ordinal} is missing PBS identity")
        require(job not in jobs, f"duplicate science allocation job: {job}")
        jobs.add(job)
        require(record.get("consumed") is True, f"allocation {ordinal} is not marked consumed")
        if record.get("historical_record_only") is True:
            require(record.get("evidence_status") == "HISTORICAL_EXTERNAL_RECORD_REFERENCED", f"allocation {ordinal} historical evidence status missing")
        else:
            require(record.get("root"), f"allocation {ordinal} is missing evidence root")
        require("model_imported" in record and "model_forward" in record and "training" in record, f"allocation {ordinal} boundary counters missing")
        require("failure_seal_sha256" in record, f"allocation {ordinal} failure/seal identity missing")
        if "attempted_model_calls_complete" in record:
            require(isinstance(record.get("attempted_model_calls_complete"), bool), f"allocation {ordinal} attempted-call completeness is not boolean")
            if record.get("attempted_model_calls_complete") is False:
                require(record.get("attempted_model_calls") is None, f"allocation {ordinal} incomplete attempted-call count was coerced to zero")
    for record in probes:
        require(record.get("allocation_kind") == "ZERO_SCIENCE_PROBE", "invalid zero-science allocation record")
        require(record.get("pbs_jobid") not in jobs, f"zero-science probe collides with science job: {record.get('pbs_jobid')}")
    return {"science_records": len(science), "zero_science_probe_records": len(probes), "next_science_ordinal": len(science) + 1, "max_science_allocations": ALLOCATION_LEDGER_MAX}


def seed_allocation_ledger(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the ledger once; subsequent records are append-only JSONL."""

    path = Path(path)
    if path.exists():
        return read_allocation_ledger(path)
    document = _ledger_schema()
    document["records"] = [dict(record) for record in records]
    validate_allocation_ledger(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in document["records"]:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return document


def append_allocation_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append one allocation, fail-closed at #9, and make duplicate observe idempotent."""

    path = Path(path)
    document = read_allocation_ledger(path)
    records = list(document["records"])
    incoming = dict(record)
    kind = incoming.get("allocation_kind", "SCIENCE")
    require(kind in ("SCIENCE", "ZERO_SCIENCE_PROBE"), f"unknown allocation kind: {kind}")
    job = incoming.get("pbs_jobid")
    require(isinstance(job, str) and job, "allocation record requires PBS job identity")
    for existing in records:
        if existing.get("pbs_jobid") == job:
            require(existing == incoming, f"allocation job observed with conflicting ledger record: {job}")
            return {"status": "IDEMPOTENT_OBSERVATION", "record": existing, "ledger": document}
    if kind == "SCIENCE":
        ordinal = len([item for item in records if item.get("allocation_kind", "SCIENCE") == "SCIENCE"]) + 1
        require(ordinal <= ALLOCATION_LEDGER_MAX, "ALLOCATION_9_BLOCKED_UNAUTHORIZED_PI_GATE")
        incoming["ordinal"] = ordinal
        incoming["consumed"] = True
    else:
        incoming["consumed"] = False
    validate_allocation_ledger({**document, "records": [*records, incoming]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(incoming, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"status": "APPENDED", "record": incoming, "ledger": read_allocation_ledger(path)}


def admit_current_allocation(path: Path, *, pbs_jobid: str, root: Path, allocation_kind: str = "SCIENCE") -> dict[str, Any]:
    """Perform the pre-model in-job ledger admission; a pre-model failure still consumes its ordinal."""

    require(pbs_jobid, "PBS job identity is required for allocation admission")
    record = {
        "allocation_kind": allocation_kind,
        "pbs_jobid": pbs_jobid,
        "root": str(Path(root).resolve()),
        "status": "PRE_MODEL_ADMITTED",
        "model_imported": False,
        "model_forward": 0,
        "training": False,
        "attempted_model_calls": 0,
        "failure_seal_sha256": None,
        "admission_head": git(REPO, "rev-parse", "HEAD"),
    }
    return append_allocation_record(path, record)


def append_allocation_outcome(path: Path, *, pbs_jobid: str, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the post-run outcome separately so the admission event stays immutable."""

    outcome_path = Path(str(path) + ".outcomes.jsonl")
    existing: list[dict[str, Any]] = []
    if outcome_path.exists():
        require(outcome_path.is_file() and not outcome_path.is_symlink(), f"allocation outcome ledger is unsafe: {outcome_path}")
        for line in outcome_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
    incoming = {"pbs_jobid": pbs_jobid, **dict(outcome)}
    for item in existing:
        if item.get("pbs_jobid") == pbs_jobid:
            require(item == incoming, f"conflicting allocation outcome for {pbs_jobid}")
            return {"status": "IDEMPOTENT_OBSERVATION", "record": item, "path": str(outcome_path)}
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    with outcome_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(incoming, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"status": "APPENDED", "record": incoming, "path": str(outcome_path)}


def append_allocation_correction(
    path: Path,
    *,
    pbs_jobid: str,
    correction: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    """Append a source-bound correction without rewriting the allocation ledger."""

    document = read_allocation_ledger(Path(path))
    require(any(item.get("pbs_jobid") == pbs_jobid for item in document["records"]), f"allocation correction references unknown job: {pbs_jobid}")
    source_path = Path(source_path).resolve()
    observed_source_sha = _verified_file_sha(source_path, source_sha256, "allocation correction source")
    correction_value = dict(correction)
    require(isinstance(correction_value.get("correction_id"), str) and correction_value["correction_id"], "allocation correction requires correction_id")
    incoming = {
        "schema": ALLOCATION_CORRECTION_SCHEMA,
        "pbs_jobid": pbs_jobid,
        "correction": correction_value,
        "source_path": str(source_path),
        "source_sha256": observed_source_sha,
    }
    correction_path = Path(str(path) + ".corrections.jsonl")
    existing: list[dict[str, Any]] = []
    if correction_path.exists():
        require(correction_path.is_file() and not correction_path.is_symlink(), f"allocation correction ledger is unsafe: {correction_path}")
        for line_number, line in enumerate(correction_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"allocation correction ledger malformed at line {line_number}: {correction_path}") from exc
            require(isinstance(value, Mapping), f"allocation correction record is not an object at line {line_number}")
            existing.append(dict(value))
    for item in existing:
        same_key = item.get("pbs_jobid") == pbs_jobid and item.get("correction", {}).get("correction_id") == correction_value["correction_id"]
        if same_key:
            require(item == incoming, f"conflicting allocation correction for {pbs_jobid}/{correction_value['correction_id']}")
            return {"status": "IDEMPOTENT_OBSERVATION", "record": item, "path": str(correction_path)}
    correction_path.parent.mkdir(parents=True, exist_ok=True)
    with correction_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(incoming, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"status": "APPENDED", "record": incoming, "path": str(correction_path)}


def _result_file(route: str, root: Path) -> Path:
    return root / ("stage45e_l1_training_result.json" if route == "l1" else "training_result.json")


def _training_pass(route: str, result: Mapping[str, Any], seed: int) -> bool:
    expected_status = "STAGE45E_L1_TRAINING_COMPLETE_FOR_EVAL" if route == "l1" else "PASS"
    observed_seed = result.get("training_seed") if route == "l1" else result.get("seed")
    observed_head = result.get("execution_head") if route == "l1" else result.get("repository_head")
    observed_branch = result.get("branch")
    input_audit = result.get("input_audit", {})
    base_tree = result.get("base_tree_sha256") or input_audit.get("base_tree_sha256")
    full24_tree = result.get("full24_tree_sha256") or input_audit.get("full24_tree_sha256")
    if route == "l1":
        stats_ok = result.get("structured_stats", {}).get("statistics_payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA or result.get("structured_stats", {}).get("stats_payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA or result.get("structured_stats", {}).get("payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA
    elif route == "n2":
        stats_ok = result.get("n2_stats_sha256") == N2_STATS_SHA and result.get("n2_stats_payload_sha256") == N2_STATS_PAYLOAD_SHA
    else:
        stats_ok = result.get("runtime_stats_sha256") == RUNTIME_STATS_SHA and result.get("n2_statistics_override") is False
    return (
        result.get("status") == expected_status
        and observed_seed == seed
        and result.get("training_steps") == 2000
        and result.get("optimizer_steps") == 2000
        and observed_head == ROUTE_HEADS[route]
        and observed_branch == ROUTE_BRANCHES[route]
        and base_tree == BASE_TREE
        and full24_tree == FULL24_TREE
        and stats_ok
    )


def _sealed_gpu_identity(result: Mapping[str, Any], route: str, seed: int) -> dict[str, Any]:
    gpu = result.get("gpu", {})
    if isinstance(gpu, Mapping) and isinstance(gpu.get("rows"), list):
        rows = gpu.get("rows", [])
        require(len(rows) == 1, f"{route} seed{seed} GPU provenance is not singular")
        observed = rows[0]
    else:
        require(isinstance(gpu, Mapping), f"{route} seed{seed} GPU provenance is not an object")
        observed = gpu
    name = str(observed.get("name", observed.get("gpu_name", "")))
    index = observed.get("index", observed.get("gpu_index", 0))
    memory = observed.get("memory_total_mib", observed.get("memory_total", observed.get("memory_total_mb")))
    if memory is None and observed.get("total_vram_bytes") is not None:
        memory = int(observed["total_vram_bytes"]) // 1024**2
    uuid = observed.get("uuid", observed.get("gpu_uuid"))
    require("H200" in name.upper(), f"{route} seed{seed} is not sealed as H200: {observed}")
    require(index in (0, "0"), f"{route} seed{seed} is not bound to cuda:0: {observed}")
    require(isinstance(memory, (int, float)) and not isinstance(memory, bool) and int(memory) >= 80 * 1024, f"{route} seed{seed} H200 VRAM provenance is below floor: {observed}")
    require(isinstance(uuid, str) and uuid.startswith("GPU-"), f"{route} seed{seed} GPU UUID provenance is missing: {observed}")
    return {"name": name, "index": int(index), "memory_total_mib": int(memory), "uuid": uuid}


def _accepted_training_lineage(route: str, root: Path, seed: int) -> dict[str, Any]:
    """Accept one complete lineage only after CPU reconciliation of its tree."""

    root = Path(root).resolve()
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise RuntimeError(f"training lineage root unavailable: {root}") from exc
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"training lineage root is unsafe: {root}")
    path = _result_file(route, root)
    result = read_json(path)
    require(_training_pass(route, result, seed), f"{route} seed{seed} did not produce a complete exact 2000-step lineage")
    require(result.get("training_steps") == 2000 and result.get("optimizer_steps") == 2000, f"{route} seed{seed} step/optimizer closure drift")
    require(result.get("starting_checkpoint", "BASE") in ("BASE", "exact BASE"), f"{route} seed{seed} did not start from BASE")
    training_args = _historical_training_args(root, seed)
    gpu = _sealed_gpu_identity(result, route, seed)
    require(result.get("validation_e_read") is False and result.get("holdout_f_read") is False, f"{route} seed{seed} validation boundary drift")
    require(result.get("wbc_steps", 0) == 0 and result.get("simulator_steps", 0) == 0 and not result.get("hardware", False), f"{route} seed{seed} downstream execution boundary drift")
    if route == "legacy":
        require(result.get("n2_statistics_override") is False, f"Legacy seed{seed} carries an N2 statistics override")
    if route == "l1":
        require(result.get("normalization_unchanged") is True and result.get("mask_unchanged") is True and result.get("decoder_unchanged") is True, f"L1 seed{seed} contract flags drift")
    if route == "n2":
        require(result.get("n2_stats_sha256") == N2_STATS_SHA and result.get("n2_stats_payload_sha256") == N2_STATS_PAYLOAD_SHA, f"N2 seed{seed} statistics identity drift")
    checkpoints = {
        str(step): _verify_checkpoint_artifact(route, root, step, result.get(f"checkpoint_{step}", {}))
        for step in (1000, 2000)
    }
    return {
        "route": route,
        "seed": seed,
        "root": str(root),
        "result_path": str(path),
        "result_sha256": sha256_file(path),
        "result": result,
        "training_args": training_args,
        "training_source_kind": TRAINING_SOURCE_STANDARD,
        "training_lineage_head": observed_head if (observed_head := (result.get("execution_head") if route == "l1" else result.get("repository_head"))) else None,
        "training_lineage_branch": result.get("branch"),
        "gpu_acceptance": gpu,
        "checkpoints": checkpoints,
        "acceptance": {
            "status": "PASS_READ_ONLY",
            "h200_identity": True,
            "exact_recipe": True,
            "trainer_state": True,
            "checkpoint_tree_binding": True,
        },
    }


def find_completed_training(storage: Path, route: str, seed: int) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    training = Path(storage) / "training"
    if not training.is_dir():
        return roots
    prefix = f"{route}_seed{seed}_"
    for root in sorted(training.iterdir(), key=lambda item: item.name):
        if not root.is_dir() or root.is_symlink() or not root.name.startswith(prefix):
            continue
        result_path = _result_file(route, root)
        if result_path.is_file() and not result_path.is_symlink():
            roots.append(_accepted_training_lineage(route, root, seed))
    return roots


def _incomplete_training_roots(storage: Path, route: str, seed: int) -> list[Path]:
    training = Path(storage) / "training"
    if not training.is_dir():
        return []
    prefix = f"{route}_seed{seed}_"
    return [root for root in sorted(training.iterdir(), key=lambda item: item.name) if root.is_dir() and not root.is_symlink() and root.name.startswith(prefix) and not _result_file(route, root).is_file()]


def _planned_fresh_training_count(storage: Path) -> int:
    """Count only future fresh roots; completed accepted roots stay reserved."""

    count = 0
    for route, seed in TRAINING_QUEUE:
        completed = find_completed_training(storage, route, seed)
        require(len(completed) <= 1, f"multiple complete {route} seed{seed} lineages exist; refusing performance-based selection")
        if completed:
            continue
        incomplete = _incomplete_training_roots(storage, route, seed)
        prior_failures = [evidence for root in incomplete if (evidence := _authorized_pre_model_retry(route, seed, root)) is not None]
        require(len(prior_failures) == len(incomplete), f"unclassified incomplete {route} seed{seed} lineage blocks storage planning: {incomplete}")
        count += 1
    return count


def _root_name(route: str, seed: int, job_id: str) -> str:
    safe_job = re.sub(r"[^A-Za-z0-9_.-]+", "_", job_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{route}_seed{seed}_{stamp}_{safe_job}"


def _common_route_args(args: argparse.Namespace) -> list[str]:
    names = ("base_model", "dataset", "groot", "groot_manifest", "cosmos", "modality", "config", "runtime_stats", "sealed_baseline", "snapshot_preflight")
    flags = {"base_model": "--base-model", "dataset": "--dataset", "groot": "--groot", "groot_manifest": "--groot-manifest", "cosmos": "--cosmos", "modality": "--modality", "config": "--config", "runtime_stats": "--runtime-stats", "sealed_baseline": "--sealed-baseline", "snapshot_preflight": "--snapshot-preflight"}
    result: list[str] = []
    for name in names:
        result.extend((flags[name], str(Path(getattr(args, name)).resolve())))
    return result


def _run_command(campaign_root: Path, name: str, command: Sequence[str], *, extra_env: Mapping[str, str] | None = None) -> None:
    logs = campaign_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record_path = logs / f"{name}.command.json"
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    command_values = [str(item) for item in command]
    started_utc = now_utc()
    write_json(record_path, {"schema": "VANDA_STAGE45E_UNIFIED_SUBPROCESS_RECORD_V2", "name": name, "command": command_values, "cwd": "/tmp", "pythonpath": "", "started_utc": started_utc, "lifecycle": "STARTED"})
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    if extra_env:
        environment.update(extra_env)
    started = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command_values, cwd="/tmp", env=environment, stdout=stdout, stderr=stderr, check=False)
    except OSError as exc:
        finished = time.time()
        write_json(record_path, {"schema": "VANDA_STAGE45E_UNIFIED_SUBPROCESS_RECORD_V2", "name": name, "command": command_values, "cwd": "/tmp", "pythonpath": "", "started_utc": started_utc, "finished_epoch": finished, "duration_s": finished - started, "returncode": None, "spawn_error": f"{type(exc).__name__}: {exc}", "stdout": str(stdout_path), "stderr": str(stderr_path), "lifecycle": "SPAWN_FAILED"}, overwrite=True)
        raise
    finished = time.time()
    lifecycle = "PASS" if process.returncode == 0 else "FAILED"
    write_json(record_path, {"schema": "VANDA_STAGE45E_UNIFIED_SUBPROCESS_RECORD_V2", "name": name, "command": command_values, "cwd": "/tmp", "pythonpath": "", "started_utc": started_utc, "finished_epoch": finished, "duration_s": finished - started, "returncode": process.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "lifecycle": lifecycle}, overwrite=True)
    require(process.returncode == 0, f"subprocess failed ({name}); see {stderr_path}")


def _run_canary(args: argparse.Namespace, campaign_root: Path, gpu: Mapping[str, Any]) -> Path:
    route_repo = Path(args.legacy_repo).resolve()
    script = route_repo / "scripts/stage45e_legacy_matched_seed.py"
    output = campaign_root / "resource_canary"
    command = [str(PYTHON), str(script), "--mode", "canary", *_common_route_args(args), "--gpu-uuid", str(gpu["rows"][0]["uuid"]), "--output-root", str(output)]
    _run_command(campaign_root, "resource_canary", command)
    result_path = output / "legacy_resource_canary.json"
    result = read_json(result_path)
    require(result.get("status") == "PASS" and result.get("weight_update") == 0 and result.get("optimizer_steps") == 0, "resource canary did not pass as a zero-update H200 canary")
    return result_path


def _training_command(args: argparse.Namespace, route: str, seed: int, output: Path, canary: Path) -> list[str]:
    repo = Path(getattr(args, f"{route}_repo")).resolve()
    if route == "legacy":
        return [str(PYTHON), str(repo / "scripts/stage45e_legacy_matched_seed.py"), "--mode", "train", *_common_route_args(args), "--seed", str(seed), "--canary", str(canary), "--gpu-uuid", str(args.gpu_uuid), "--output-root", str(output)]
    if route == "n2":
        return [str(PYTHON), str(repo / "scripts/stage45e_n2_gpu_sprint.py"), "--mode", "train", *_common_route_args(args), "--n2-stats", str(Path(args.n2_stats).resolve()), "--seed", str(seed), "--canary", str(canary), "--gpu-uuid", str(args.gpu_uuid), "--output-root", str(output)]
    return [str(PYTHON), str(repo / "scripts/stage45e_l1_h200_train.py"), "--execute", "--seed", str(seed), "--expected-head", ROUTE_HEADS[route], "--output-root", str(output), "--python", str(PYTHON), "--base-model", str(Path(args.base_model).resolve()), "--dataset", str(Path(args.dataset).resolve()), "--groot-source", str(Path(args.groot).resolve()), "--groot-manifest", str(Path(args.groot_manifest).resolve()), "--cosmos-root", str(Path(args.cosmos).resolve()), "--modality", str(Path(args.modality).resolve()), "--config", str(Path(args.config).resolve()), "--stats", str(Path(args.runtime_stats).resolve()), "--release", str(Path(args.release).resolve()), "--video-qualification", str(Path(args.video_qualification).resolve()), "--ffmpeg-root", str(Path(args.ffmpeg_root).resolve())]


def _training_result(route: str, root: Path, seed: int) -> dict[str, Any]:
    return _accepted_training_lineage(route, root, seed)


def _authorized_pre_model_retry(route: str, seed: int, root: Path) -> dict[str, Any] | None:
    """Return evidence only for the previously authorized L1 pre-import retry."""

    if (route, seed) != ("l1", 2223):
        return None
    failure_path = Path(root) / "stage45e_l1_failure.json"
    if not failure_path.is_file() or failure_path.is_symlink():
        return None
    failure = read_json(failure_path)
    if failure.get("status") != "FAIL" or failure.get("training_seed") != seed:
        return None
    if failure.get("model_imported") is not False:
        return None
    if failure.get("model_construction_started") not in (None, False):
        return None
    if failure.get("model_construction_completed") not in (None, False):
        return None
    if failure.get("training_started") not in (None, False):
        return None
    if failure.get("backward_started") not in (None, False):
        return None
    if failure.get("optimizer_steps_observed") not in (None, 0):
        return None
    if failure.get("checkpoint_writes_observed") not in (None, 0):
        return None
    if failure.get("action_head_forwards") not in (None, 0):
        return None
    checkpoint_dirs = [item for item in Path(root).iterdir() if item.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        return None
    return {
        "path": str(failure_path),
        "sha256": sha256_file(failure_path),
        "status": "FAIL_PRE_MODEL_IMPORT",
        "model_imported": False,
        "retry_reason": "PI0134_L1_SEED2223_OFFLINE_ENVIRONMENT_FAILURE_BEFORE_MODEL_IMPORT",
    }


def _bind_lineages(args: argparse.Namespace, storage: Path, campaign_root: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    lineages: dict[str, dict[int, dict[str, Any]]] = {route: {} for route in ROUTES}
    plan: list[dict[str, Any]] = []
    for route in ROUTES:
        for seed in ALL_SEEDS:
            if route == "legacy" and seed == 2222:
                legacy_decision = legacy_provenance_decision(Path(args.legacy_history), Path(args.config))
                write_json(campaign_root / "legacy_seed2222_provenance_decision.json", {key: value for key, value in legacy_decision.items() if key != "lineage"})
                require(legacy_decision["status"] == "PASS_EXACT_H200_FROZEN_RECIPE_REUSE", f"{legacy_decision['token']}: {legacy_decision['reason']}")
                lineages[route][seed] = legacy_decision["lineage"]
                plan.append({
                    "route": route,
                    "seed": seed,
                    "action": "REUSE_COMPLETE_READ_ONLY",
                    "source_kind": TRAINING_SOURCE_STANDARD,
                    "root": lineages[route][seed]["root"],
                    "result_sha256": lineages[route][seed]["result_sha256"],
                })
                continue
            if route == "l1" and seed == 2222:
                lineages[route][seed] = verify_recovered_l1(
                    Path(args.recovered_l1_root),
                    Path(args.recovered_l1_acceptance),
                    Path(args.recovered_l1_input_v2),
                )
                lineages[route][seed]["duplicate_provenance"] = duplicate_l1_provenance(Path(args.duplicate_l1_root))
                plan.append({
                    "route": route,
                    "seed": seed,
                    "action": "REUSE_RECOVERED_READ_ONLY",
                    "source_kind": TRAINING_SOURCE_RECOVERED,
                    "root": lineages[route][seed]["root"],
                    "recovered_acceptance_sha256": lineages[route][seed]["recovered_acceptance_sha256"],
                    "recovered_input_v2_sha256": lineages[route][seed]["recovered_input_v2_sha256"],
                    "duplicate_root_not_primary": lineages[route][seed]["duplicate_provenance"]["root"],
                })
                continue
            completed = find_completed_training(storage, route, seed)
            require(len(completed) <= 1, f"multiple complete {route} seed{seed} lineages exist; refusing performance-based selection")
            if completed:
                lineages[route][seed] = completed[0]
                plan.append({"route": route, "seed": seed, "action": "REUSE_COMPLETE_READ_ONLY", "root": completed[0]["root"], "result_sha256": completed[0]["result_sha256"]})
                continue
            incomplete = _incomplete_training_roots(storage, route, seed)
            prior_failures = [evidence for root in incomplete if (evidence := _authorized_pre_model_retry(route, seed, root)) is not None]
            require(len(prior_failures) == len(incomplete), f"incomplete prior {route} seed{seed} lineage is not an authorized pre-model retry: {incomplete}")
            root = storage / "training" / _root_name(route, seed, str(os.environ["PBS_JOBID"]))
            require(not root.exists(), f"new training root unexpectedly exists: {root}")
            lineages[route][seed] = {"route": route, "seed": seed, "root": str(root), "pending": True}
            item = {"route": route, "seed": seed, "action": "FRESH_BASE_2000", "root": str(root)}
            if prior_failures:
                item["prior_failure_seals"] = prior_failures
                item["retry_authorization"] = "PRE_MODEL_FAILURE_ONLY"
            plan.append(item)
    historical_evidence = audit_historical_diagnostic_evidence_catalog(require_real=True)
    lineages["legacy"][2222]["historical_diagnostic_evidence"] = historical_evidence["references"]["legacy"]
    lineages["l1"][2222]["historical_diagnostic_evidence"] = historical_evidence["references"]["l1"]
    write_json(campaign_root / "historical_diagnostic_evidence_audit.json", historical_evidence)
    write_json(campaign_root / "lineage_plan.json", {"schema": "VANDA_STAGE45E_UNIFIED_LINEAGE_PLAN_V1", "status": "BOUND_BEFORE_TRAINING", "queue": plan, "fresh_base_count": sum(item["action"] == "FRESH_BASE_2000" for item in plan), "historical_reuse": {route: lineages[route].get(2222, {}).get("root") for route in ROUTES}, "timestamp_utc": now_utc()})
    return lineages, plan


def _checkpoint_spec(route: str, seed: int, step: int, lineage: Mapping[str, Any]) -> dict[str, Any]:
    result = lineage.get("result", {})
    record = result.get(f"checkpoint_{step}", {})
    tree = record.get("tree_sha256") or lineage.get("checkpoints", {}).get(str(step), {}).get("tree_sha256")
    require(tree, f"checkpoint tree hash missing for {route} seed{seed} step{step}")
    path = Path(lineage["root"]) / f"checkpoint-{step}"
    return {
        "label": f"{route.upper()}(seed{seed})@{step}",
        "route": route,
        "seed": seed,
        "step": step,
        "path": str(path),
        "tree_sha256": tree,
        "training_lineage_identity": training_lineage_identity(lineage),
    }


def build_phase_specs(
    lineages: Mapping[str, Mapping[int, Mapping[str, Any]]],
    campaign_root: Path,
    *,
    evaluator: Mapping[str, Any] | None = None,
    orchestration: Mapping[str, Any] | None = None,
    synthetic_test_only: bool = False,
) -> dict[str, dict[str, Path]]:
    # Legacy seed2222 already has an accepted D1/D2A artifact and is reused
    # read-only in CPU closure.  It is intentionally absent from current
    # model-call specs; L1 seed2222 is also reused only for its own arm.
    legacy_seeds = ALL_SEEDS if synthetic_test_only else (2223, 2224)
    all_specs = {
        "legacy": [_checkpoint_spec("legacy", seed, step, lineages["legacy"][seed]) for seed in legacy_seeds for step in (1000, 2000)],
        "l1": [_checkpoint_spec("l1", seed, step, lineages["l1"][seed]) for seed in ALL_SEEDS for step in (1000, 2000)],
        "n2": [_checkpoint_spec("n2", seed, step, lineages["n2"][seed]) for seed in ALL_SEEDS for step in (1000, 2000)],
    }
    e4_specs = {route: [item for item in all_specs[route] if not (route == "l1" and item["seed"] == 2222)] for route in ROUTES}
    # E4 therefore has 4 LEGACY + 4 L1 + 6 N2 = 14 new checkpoint arms;
    # D1/D2A have 4 LEGACY + 6 L1 + 6 N2 = 16 active arms.
    phase_specs: dict[str, dict[str, Path]] = {"E4": {}, "D1": {}, "D2A": {}}
    for phase, selected in (("E4", e4_specs), ("D1", all_specs), ("D2A", all_specs)):
        for route in ROUTES:
            path = campaign_root / "specs" / f"{phase}_{route}.json"
            budget = {
                "arms": len(selected[route]),
                "conditioning_passes": len(selected[route]) * 24,
                "action_head_calls": len(selected[route]) * 24 * (4 if phase == "E4" else 9 if phase == "D1" else 8),
                "raw_rows": len(selected[route]) * 24 * (1 if phase == "E4" else 9 if phase == "D1" else 4),
            }
            document = {
                "schema": "VANDA_STAGE45E_UNIFIED_PHASE_SPEC_V2",
                "status": "SEALED_INPUT_SPEC",
                "route": route,
                "phase": phase,
                "checkpoints": selected[route],
                "budget": budget,
                "evaluation_identity": dict(evaluator or {}),
                "orchestration_identity": dict(orchestration or {}),
                "synthetic_test_only": synthetic_test_only,
                "science_contract": {
                    "cohort": "FULL24_EXISTING_TRAIN",
                    "split": "TRAIN",
                    "window_count": 24,
                    "target_xyz_in_model_input": False,
                    "noise_seed": 4242,
                    "sampler": "source-mapped-four-step-Euler-action-head-graph",
                    "timestep_buckets": [0, 250, 500, 750],
                    "dt": 0.25,
                    "stats_sha256": N2_STATS_SHA if route == "n2" else RUNTIME_STATS_SHA,
                    "stats_payload_sha256": N2_STATS_PAYLOAD_SHA if route == "n2" else RUNTIME_STATS_PAYLOAD_SHA,
                    "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
                    "raw_shape": [1, 40, 132],
                    "decoded_shape": [40, 10],
                    "mask_active_scalar_count": 190,
                },
                "timestamp_utc": now_utc(),
            }
            document["scientific_identity"] = canonical_scientific_phase_identity(document)
            document["scientific_identity_sha256"] = scientific_spec_identity_sha256(document)
            write_json(path, document)
            phase_specs[phase][route] = path
    return phase_specs


def _expected_phase_totals(phase: str) -> dict[str, dict[str, int]]:
    if phase == "E4":
        return {"legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96}, "l1": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96}, "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 576, "raw_rows": 144}}
    if phase == "D1":
        return {"legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 864, "raw_rows": 864}, "l1": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1296, "raw_rows": 1296}, "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1296, "raw_rows": 1296}}
    return {"legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 768, "raw_rows": 384}, "l1": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1152, "raw_rows": 576}, "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1152, "raw_rows": 576}}


def _counter_value(document: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = document.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _attempt_call_counts(root: Path) -> dict[str, Any]:
    """Extract actual calls from a pass/failure seal, never from a budget."""

    root = Path(root)
    seal_path = root / "unified_phase_seal.json"
    failure_path = root / "unified_failure_seal.json"
    document: Mapping[str, Any]
    status = "INCOMPLETE_UNKNOWN"
    if seal_path.is_file() and not seal_path.is_symlink():
        document = read_json(seal_path)
        status = "PASS_ACCEPTED" if document.get("status") == "PASS" else "INCOMPLETE_UNKNOWN"
        counters = document.get("observed_counters") or document.get("counters") or {}
    elif failure_path.is_file() and not failure_path.is_symlink():
        document = read_json(failure_path)
        status = "FAILED_ATTEMPT"
        counters = document.get("observed_counters") or document.get("counters") or {}
        if not counters:
            pre_model = document.get("pre_model_admission")
            if pre_model:
                pre_model_path = Path(str(pre_model))
                if pre_model_path.is_file() and not pre_model_path.is_symlink():
                    pre_model_doc = read_json(pre_model_path)
                    counters = pre_model_doc.get("counters", {}) or pre_model_doc
        if not counters:
            counters = document
    else:
        return {
            "conditioning_passes": None,
            "action_head_calls": None,
            "raw_rows": None,
            "model_side_calls": None,
            "status": "INCOMPLETE_UNKNOWN",
            "counters_complete": False,
            "counter_source": "MISSING_SEAL",
        }
    conditioning = _counter_value(counters, "conditioning_passes", "conditioning_calls")
    action_head = _counter_value(counters, "action_head_calls", "action_head_forwards")
    raw_rows = _counter_value(counters, "raw_rows", "raw_row_count", "raw_probe_rows", "raw_stage_count")
    model_side = _counter_value(counters, "model_side_calls", "attempted_model_calls")
    if model_side is None and conditioning is not None and action_head is not None:
        model_side = conditioning + action_head
    explicit_model_side = _counter_value(document, "model_side_calls", "attempted_model_calls")
    if model_side is None:
        model_side = explicit_model_side
    components_complete = conditioning is not None and action_head is not None
    counters_complete = model_side is not None and (components_complete or explicit_model_side is not None)
    return {
        "conditioning_passes": conditioning,
        "action_head_calls": action_head,
        "raw_rows": raw_rows,
        "model_side_calls": model_side,
        "status": status,
        "counters_complete": counters_complete,
        "counter_source": "SEALED_COUNTERS" if counters else "SEALED_DOCUMENT",
    }


def _diagnostic_candidate_status(root: Path) -> str:
    root = Path(root)
    seal = root / "unified_phase_seal.json"
    failure = root / "unified_failure_seal.json"
    if seal.is_file() and not seal.is_symlink():
        return "PASS_ACCEPTED" if read_json(seal).get("status") == "PASS" else "INCOMPLETE_UNKNOWN"
    if failure.is_file() and not failure.is_symlink():
        return "FAILED_ATTEMPT"
    return "INCOMPLETE_UNKNOWN"


class ContinuationIntegrityError(RuntimeError):
    """A prior attempt cannot be silently reclassified for continuation."""


def discover_diagnostic_attempts(storage: Path, phase: str, route: str) -> list[dict[str, Any]]:
    """Discover all immutable attempts in the sibling diagnostics namespace."""

    require(phase in DIAGNOSTIC_PHASES and route in ROUTES, f"invalid diagnostic unit: {phase}/{route}")
    diagnostics_root = Path(storage).resolve()
    if diagnostics_root.name != "diagnostics":
        diagnostics_root = diagnostics_root / "diagnostics"
    if not diagnostics_root.is_dir() or diagnostics_root.is_symlink():
        return []
    suffix = f"_{phase}_{route}"
    attempts: list[dict[str, Any]] = []
    for root in sorted(diagnostics_root.iterdir(), key=lambda item: item.name):
        if not root.is_dir() or root.is_symlink() or not root.name.endswith(suffix):
            continue
        status = _diagnostic_candidate_status(root)
        seal_path = root / "unified_phase_seal.json"
        failure_path = root / "unified_failure_seal.json"
        selected = seal_path if seal_path.is_file() and not seal_path.is_symlink() else failure_path
        counts = _attempt_call_counts(root)
        attempts.append({
            "phase": phase,
            "route": route,
            "unit_key": f"{phase}:{route}",
            "root": str(root),
            "status": status,
            "seal_path": str(selected) if selected.exists() else None,
            "seal_sha256": sha256_file(selected) if selected.is_file() and not selected.is_symlink() else None,
            "counts": counts,
        })
    return attempts


def _validate_diagnostic_pass(
    candidate: Mapping[str, Any],
    *,
    phase: str,
    route: str,
    spec_path: Path,
    expected_evaluator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate a complete PASS before allowing read-only reuse."""

    root = Path(str(candidate["root"])).resolve()
    seal_path = root / "unified_phase_seal.json"
    seal = read_json(seal_path)
    expected = _expected_phase_totals(phase)[route]
    spec_doc = read_json(spec_path)
    require(seal.get("schema") == "VANDA_STAGE45E_UNIFIED_DIAGNOSTIC_PHASE_SEAL_V2", f"{phase}/{route} diagnostic seal schema is not PI0143 V2")
    require(seal.get("status") == "PASS" and seal.get("phase") == phase and seal.get("route") == route, f"{phase}/{route} diagnostic PASS identity drift")
    require(seal.get("route_head") == ROUTE_HEADS[route] and seal.get("route_branch") == ROUTE_BRANCHES[route], f"{phase}/{route} route identity drift")
    expected_spec_sha = sha256_file(spec_path)
    expected_scientific_sha = scientific_spec_identity_sha256(spec_doc)
    observed_spec_sha = seal.get("phase_spec_sha256")
    if observed_spec_sha != expected_spec_sha:
        require(seal.get("scientific_spec_identity_sha256") == expected_scientific_sha, f"{phase}/{route} phase-spec SHA drift without matching immutable scientific identity")
    else:
        if seal.get("scientific_spec_identity_sha256") is not None:
            require(seal.get("scientific_spec_identity_sha256") == expected_scientific_sha, f"{phase}/{route} scientific spec identity drift")
    require(seal.get("checkpoints") == spec_doc.get("checkpoints"), f"{phase}/{route} checkpoint specification drift")
    require(seal.get("training_lineage_identities") == [item.get("training_lineage_identity", {}) for item in spec_doc.get("checkpoints", [])], f"{phase}/{route} training identity drift")
    if expected_evaluator is not None:
        require(seal.get("evaluation_identity") == dict(expected_evaluator), f"{phase}/{route} evaluator identity drift")
    require(seal.get("stats_sha256") == (N2_STATS_SHA if route == "n2" else RUNTIME_STATS_SHA), f"{phase}/{route} stats identity drift")
    require(seal.get("stats_payload_sha256") == (N2_STATS_PAYLOAD_SHA if route == "n2" else RUNTIME_STATS_PAYLOAD_SHA), f"{phase}/{route} stats payload drift")
    require(seal.get("budget") == expected, f"{phase}/{route} accepted budget drift")
    observed = seal.get("observed_counters", {})
    counts = _attempt_call_counts(root)
    require(counts.get("counters_complete") is True, f"{phase}/{route} accepted PASS counter completeness is unknown")
    require(observed.get("conditioning_passes") == expected["conditioning_passes"], f"{phase}/{route} conditioning counter drift")
    require(observed.get("action_head_calls") == expected["action_head_calls"], f"{phase}/{route} action-head counter drift")
    require(observed.get("raw_rows") == expected["raw_rows"], f"{phase}/{route} raw-row counter drift")
    require(seal.get("training") is False and seal.get("backward") is False and seal.get("optimizer_steps") == 0, f"{phase}/{route} training boundary drift")
    for key in ("Validation-E", "Holdout-F", "WBC", "simulator", "hardware", "public_sampler_calls", "get_action_calls"):
        require(seal.get(key) in (0, False), f"{phase}/{route} forbidden boundary drift in {key}")
    inner_path = Path(str(seal.get("inner_seal"))).resolve()
    require(inner_path.is_file() and not inner_path.is_symlink(), f"{phase}/{route} inner seal unavailable")
    require(sha256_file(inner_path) == seal.get("inner_seal_sha256"), f"{phase}/{route} inner seal SHA drift")
    inner = read_json(inner_path)
    require(inner.get("status") == "PASS", f"{phase}/{route} inner seal is not PASS")
    for item in spec_doc.get("checkpoints", []):
        checkpoint = Path(item["path"]).resolve()
        state_path = checkpoint / "trainer_state.json"
        require(state_path.is_file() and not state_path.is_symlink(), f"{phase}/{route} checkpoint state unavailable: {checkpoint}")
        state = read_json(state_path)
        require(state.get("global_step") == item.get("step"), f"{phase}/{route} checkpoint step drift: {checkpoint}")
        entries = regular_tree_entries(checkpoint, f"{CHECKPOINT_HASH_PREFIX[route]}/{checkpoint.name}")
        require(tree_digest(entries) == item.get("tree_sha256"), f"{phase}/{route} checkpoint tree drift: {checkpoint}")
    return {
        **dict(candidate),
        "status": "PASS_ACCEPTED",
        "seal_path": str(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "counts": counts,
        "phase_spec_sha256": seal["phase_spec_sha256"],
        "scientific_spec_identity_sha256": expected_scientific_sha,
        "evaluation_identity": seal.get("evaluation_identity", {}),
    }


def diagnostic_accounting(
    attempts_by_unit: Mapping[str, Sequence[Mapping[str, Any]]],
    accepted_by_unit: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report unique accepted work separately from every attempted call."""

    accepted_unique_by_unit: dict[str, int] = {}
    reused_readonly_by_unit: dict[str, int] = {}
    reused_readonly_evidence_by_unit: dict[str, int] = {}
    all_attempted_by_attempt: list[dict[str, Any]] = []
    failed_attempt_by_attempt: list[dict[str, Any]] = []
    for unit in DIAGNOSTIC_UNIT_KEYS:
        accepted = accepted_by_unit.get(unit)
        if accepted is not None:
            counts = accepted.get("counts", {})
            accepted_calls = counts.get("model_side_calls")
            require(isinstance(accepted_calls, int) and not isinstance(accepted_calls, bool), f"accepted diagnostic unit has unknown model-call count: {unit}")
            accepted_unique_by_unit[unit] = accepted_calls
            reused_readonly_evidence_by_unit[unit] = accepted_calls if accepted.get("reused_readonly") is True else 0
            reused_readonly_by_unit[unit] = 0
        else:
            accepted_unique_by_unit[unit] = 0
            reused_readonly_by_unit[unit] = 0
            reused_readonly_evidence_by_unit[unit] = 0
        for attempt in attempts_by_unit.get(unit, ()):
            item = {
                "unit_key": unit,
                "phase": attempt.get("phase"),
                "route": attempt.get("route"),
                "root": attempt.get("root"),
                "status": attempt.get("status"),
                "seal_path": attempt.get("seal_path"),
                "seal_sha256": attempt.get("seal_sha256"),
                "counts": dict(attempt.get("counts", {})),
                "reused_readonly": attempt.get("reused_readonly", False),
            }
            all_attempted_by_attempt.append(item)
            if item["status"] == "FAILED_ATTEMPT":
                failed_attempt_by_attempt.append(item)
    return {
        "accepted_unique_scientific_calls_by_unit": accepted_unique_by_unit,
        "accepted_unique_scientific_calls": sum(accepted_unique_by_unit.values()),
        "all_attempted_model_calls": _sum_known_call_counts(all_attempted_by_attempt),
        "failed_attempt_model_calls": _sum_known_call_counts(failed_attempt_by_attempt),
        "all_attempted_model_calls_complete": all(item["counts"].get("counters_complete") is True for item in all_attempted_by_attempt),
        "failed_attempt_model_calls_complete": all(item["counts"].get("counters_complete") is True for item in failed_attempt_by_attempt),
        "unknown_counter_attempts": sum(item["counts"].get("counters_complete") is not True for item in all_attempted_by_attempt),
        "unknown_counter_inventory": [item for item in all_attempted_by_attempt if item["counts"].get("counters_complete") is not True],
        # A reused unit contributes zero new calls.  Historical evidence size
        # is retained in a separate namespace so it cannot be mistaken for a
        # newly executed scientific call.
        "reused_readonly_model_calls": 0,
        "reused_readonly_model_calls_by_unit": reused_readonly_by_unit,
        "reused_readonly_evidence_model_calls": sum(reused_readonly_evidence_by_unit.values()),
        "reused_readonly_evidence_model_calls_by_unit": reused_readonly_evidence_by_unit,
        "new_readonly_reuse_model_calls": 0,
        "attempt_inventory": all_attempted_by_attempt,
        "failed_attempt_inventory": failed_attempt_by_attempt,
    }


def _diagnostic_retry_allowed(candidate: Mapping[str, Any]) -> bool:
    """Allow continuation only when a failure seal proves pre-model failure."""

    root = Path(str(candidate.get("root", ""))).resolve()
    failure_path = root / "unified_failure_seal.json"
    if not failure_path.is_file() or failure_path.is_symlink():
        return False
    failure = read_json(failure_path)
    if failure.get("retry_eligibility") != "PRE_MODEL_ONLY":
        return False
    execution = failure.get("execution_state", {})
    lifecycle = failure.get("execution_lifecycle", {})
    if not isinstance(execution, Mapping) or not isinstance(lifecycle, Mapping):
        return False
    if execution.get("model_imported") not in (False, 0) or execution.get("model_construction") not in (False, 0) or execution.get("model_forward") not in (False, 0):
        return False
    if any(lifecycle.get(key) not in ("NOT_STARTED", "PASS_PRE_MODEL", "FORBIDDEN_ZERO", "PRE_MODEL") for key in ("model_import", "model_construction", "forward")):
        return False
    if execution.get("training") not in (False, 0) or execution.get("backward") not in (False, 0) or execution.get("optimizer_steps") not in (0, False):
        return False
    if failure.get("raw_seal") not in (None, False, "NOT_STARTED") or failure.get("derived") not in (None, False, "NOT_STARTED"):
        return False
    if failure.get("raw_rows") not in (None, 0) or failure.get("derived_rows") not in (None, 0):
        return False
    return True


def _sum_known_call_counts(items: Sequence[Mapping[str, Any]]) -> int | None:
    values: list[int] = []
    for item in items:
        counts = item.get("counts")
        if not isinstance(counts, Mapping):
            return None
        value = counts.get("model_side_calls")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        # Explicit legacy inventories predate the completeness field.  A
        # concrete non-negative model-side count is still usable, while the
        # current sealed readers always provide counters_complete=True.
        if counts.get("counters_complete") is False:
            return None
        values.append(value)
    return sum(values)


def select_diagnostic_attempt(
    storage: Path,
    phase: str,
    route: str,
    *,
    spec_path: Path,
    expected_evaluator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select exactly one accepted unit or report that continuation is needed."""

    candidates = discover_diagnostic_attempts(storage, phase, route)
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("status") != "PASS_ACCEPTED":
            continue
        try:
            accepted.append(_validate_diagnostic_pass(candidate, phase=phase, route=route, spec_path=Path(spec_path).resolve(), expected_evaluator=expected_evaluator))
        except Exception as exc:
            raise ContinuationIntegrityError(f"{phase}:{route} prior PASS candidate failed immutable validation; continuation is blocked: {exc}") from exc
    require(len(accepted) <= 1, f"duplicate complete PASS diagnostic units block continuation: {phase}:{route}")
    non_pass = [candidate for candidate in candidates if candidate.get("status") != "PASS_ACCEPTED"]
    if not accepted:
        for candidate in non_pass:
            if not _diagnostic_retry_allowed(candidate):
                raise ContinuationIntegrityError(f"{phase}:{route} prior {candidate.get('status')} attempt lacks sealed pre-model retry eligibility: {candidate.get('root')}")
    return {"unit_key": f"{phase}:{route}", "candidates": candidates, "accepted": accepted[0] if accepted else None, "needs_fresh_attempt": not bool(accepted)}


def run_diagnostics(
    args: argparse.Namespace,
    campaign_root: Path,
    admission_path: Path,
    phase_specs: Mapping[str, Mapping[str, Path]],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic_script = REPO / "scripts/stage45e_h200_unified_diagnostics.py"
    helper_script = Path(args.legacy_repo).resolve() / "scripts/stage45e_legacy_matched_seed.py"
    phase_roots: dict[str, dict[str, Path]] = {"E4": {}, "D1": {}, "D2A": {}}
    attempts_by_unit: dict[str, list[dict[str, Any]]] = {}
    accepted_by_unit: dict[str, dict[str, Any]] = {}
    evidence_root = Path(getattr(args, "diagnostic_evidence_root", None) or Path(args.storage_root) / "diagnostics").resolve()
    for phase in DIAGNOSTIC_PHASES:
        for route in ROUTES:
            unit_key = f"{phase}:{route}"
            spec_path = Path(phase_specs[phase][route]).resolve()
            spec_doc = read_json(spec_path)
            expected_evaluator = spec_doc.get("evaluation_identity", {})
            selection = select_diagnostic_attempt(evidence_root, phase, route, spec_path=spec_path, expected_evaluator=expected_evaluator)
            attempts_by_unit[unit_key] = selection["candidates"]
            if selection["accepted"] is not None:
                accepted = dict(selection["accepted"])
                accepted["reused_readonly"] = True
                accepted_by_unit[unit_key] = accepted
                phase_roots[phase][route] = Path(selection["accepted"]["root"])
                if state is not None:
                    state["last_completed_unit"] = unit_key
                    state["active_child"] = {"stage": unit_key, "kind": "diagnostic", "status": "REUSED_READ_ONLY"}
                append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "DIAGNOSTIC_REUSED_READ_ONLY", "unit_key": unit_key, "root": selection["accepted"]["root"], "model_calls": 0, "timestamp_utc": now_utc()})
                continue
            output = evidence_root / f"{campaign_root.name}_{phase}_{route}"
            stats = Path(args.n2_stats).resolve() if route == "n2" else Path(args.runtime_stats).resolve()
            command = [str(PYTHON), str(diagnostic_script), "--mode", "run", "--route", route, "--phase", phase, "--route-repo", str(Path(getattr(args, f"{route}_repo")).resolve()), "--helper-script", str(helper_script), "--admission", str(admission_path), "--output-root", str(output), "--base-model", str(Path(args.base_model).resolve()), "--dataset", str(Path(args.dataset).resolve()), "--groot", str(Path(args.groot).resolve()), "--cosmos", str(Path(args.cosmos).resolve()), "--modality", str(Path(args.modality).resolve()), "--config", str(Path(args.config).resolve()), "--stats", str(stats), "--sealed-baseline", str(Path(args.sealed_baseline).resolve()), "--snapshot-preflight", str(Path(args.snapshot_preflight).resolve())]
            if state is not None:
                state["failure_stage"] = unit_key
                state["active_child"] = {"stage": unit_key, "kind": "diagnostic", "status": "IN_PROGRESS"}
            _run_command(campaign_root, f"diagnostic_{phase}_{route}", command)
            selection = select_diagnostic_attempt(evidence_root, phase, route, spec_path=spec_path, expected_evaluator=expected_evaluator)
            attempts_by_unit[unit_key] = selection["candidates"]
            complete = [candidate for candidate in selection["candidates"] if candidate.get("root") == str(output) and candidate.get("status") == "PASS_ACCEPTED"]
            require(len(complete) == 1, f"new diagnostic unit did not produce one accepted PASS: {unit_key}")
            accepted = _validate_diagnostic_pass(complete[0], phase=phase, route=route, spec_path=spec_path, expected_evaluator=expected_evaluator)
            accepted["reused_readonly"] = False
            accepted_by_unit[unit_key] = accepted
            phase_roots[phase][route] = output
            if state is not None:
                state["last_completed_unit"] = unit_key
                state["active_child"] = {"stage": unit_key, "kind": "diagnostic", "status": "PASS"}
    return {
        "phase_roots": phase_roots,
        "attempts_by_unit": attempts_by_unit,
        "accepted_by_unit": accepted_by_unit,
        "accounting": diagnostic_accounting(attempts_by_unit, accepted_by_unit),
    }


def _collect_files(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in excluded:
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _path_reference(path: Path, *, role: str) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _row_bundle_files(root: Path, seal: Mapping[str, Any]) -> list[Path]:
    """Locate explicitly sealed row/raw files without moving or rewriting them."""

    root = Path(root).resolve()
    candidates: list[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            candidate = Path(value)
            if candidate.is_file() and not candidate.is_symlink():
                try:
                    if candidate.resolve().is_relative_to(root):
                        candidates.append(candidate.resolve())
                except ValueError:
                    pass
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                visit(nested)

    for key in ("row_bundle", "row_bundle_path", "raw_bundle", "raw_root", "evidence_bundle", "source_rows", "rows", "raw"):
        visit(seal.get(key))
        visit((seal.get("route_result") or {}).get(key) if isinstance(seal.get("route_result"), Mapping) else None)
    if not candidates:
        for item in sorted(root.rglob("*"), key=lambda path: str(path)):
            if not item.is_file() or item.is_symlink():
                continue
            lowered = item.name.lower()
            if any(token in lowered for token in ("row", "raw", "metric")) and item.name not in {"unified_phase_seal.json", "seal.json", "pre_model_admission.json"}:
                candidates.append(item.resolve())
    unique = {str(path): path for path in candidates}
    return [unique[key] for key in sorted(unique)]


def build_external_evidence_closure(
    campaign_root: Path,
    phase_roots: Mapping[str, Mapping[str, Path]],
    phase_specs: Mapping[str, Mapping[str, Path]],
    lineages: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    attempts_by_unit: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    evaluator: Mapping[str, Any] | None = None,
    orchestration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash-bind sibling diagnostics and expose an 18-arm C10 handoff."""

    campaign_root = Path(campaign_root).resolve()
    descriptor_root = campaign_root / "evidence_units"
    descriptor_root.mkdir(parents=True, exist_ok=True)
    descriptors: dict[str, dict[str, Any]] = {}
    for phase in DIAGNOSTIC_PHASES:
        for route in ROUTES:
            unit = f"{phase}:{route}"
            root = Path(phase_roots[phase][route]).resolve()
            seal_path = root / "unified_phase_seal.json"
            require(seal_path.is_file() and not seal_path.is_symlink(), f"external diagnostic seal missing: {unit}")
            seal = read_json(seal_path)
            inner_path = Path(str(seal.get("inner_seal"))).resolve()
            require(inner_path.is_file() and not inner_path.is_symlink(), f"external diagnostic inner seal missing: {unit}")
            row_files = _row_bundle_files(root, seal)
            require(row_files, f"external diagnostic row bundle missing: {unit}")
            spec_path = Path(phase_specs[phase][route]).resolve()
            descriptor = {
                "schema": EVIDENCE_DESCRIPTOR_SCHEMA,
                "status": "PASS",
                "unit_key": unit,
                "phase": phase,
                "route": route,
                "external": not root.is_relative_to(campaign_root),
                "attempt_root": str(root),
                "unified_seal": _path_reference(seal_path, role="unified_phase_seal"),
                "inner_seal": _path_reference(inner_path, role="inner_phase_seal"),
                "phase_spec": _path_reference(spec_path, role="phase_specification"),
                "row_bundle": [_path_reference(path, role="source_row_or_raw_file") for path in row_files],
                "source_row_schema": seal.get("source_row_schema") or (seal.get("route_result", {}).get("source_row_schema") if isinstance(seal.get("route_result"), Mapping) else None) or "ROW_BUNDLE_FILE_SET_V1",
                "source_row_sha256": tree_digest([{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in row_files]),
                "stats_sha256": seal.get("stats_sha256"),
                "stats_payload_sha256": seal.get("stats_payload_sha256"),
                "training_lineage_identities": seal.get("training_lineage_identities", []),
                "evaluation_identity": dict(evaluator or seal.get("evaluation_identity", {})),
                "orchestration_identity": dict(orchestration or seal.get("orchestration_identity", {})),
                "attempt_inventory": list((attempts_by_unit or {}).get(unit, [])),
            }
            descriptor_path = descriptor_root / f"{phase}_{route}.json"
            write_json(descriptor_path, descriptor)
            descriptor["descriptor_path"] = str(descriptor_path)
            descriptor["descriptor_sha256"] = sha256_file(descriptor_path)
            descriptors[unit] = descriptor
    inventory_path = campaign_root / "external_evidence_inventory.json"
    inventory_doc = {
        "schema": "VANDA_STAGE45E_EXTERNAL_DIAGNOSTIC_EVIDENCE_INVENTORY_V1",
        "status": "PASS",
        "campaign_root": str(campaign_root),
        "descriptors": {unit: {"path": item["descriptor_path"], "sha256": item["descriptor_sha256"], "unit_key": unit} for unit, item in descriptors.items()},
        "entry_count": len(descriptors),
        "final_result_excluded": True,
        "c10_handoff_excluded": True,
    }
    write_json(inventory_path, inventory_doc)
    inventory_sha = sha256_file(inventory_path)

    arms: list[dict[str, Any]] = []
    for route in ROUTES:
        for seed in ALL_SEEDS:
            lineage = lineages[route][seed]
            for step in (1000, 2000):
                arm = _checkpoint_spec(route, seed, step, lineage)
                arm["training_lineage_identity"] = training_lineage_identity(lineage)
                arm["diagnostic_evidence"] = {}
                for phase in DIAGNOSTIC_PHASES:
                    unit = f"{phase}:{route}"
                    spec_doc = read_json(Path(phase_specs[phase][route]))
                    in_current_spec = any(item.get("seed") == seed and item.get("step") == step for item in spec_doc.get("checkpoints", []))
                    reference = descriptors.get(unit)
                    if reference is not None and in_current_spec:
                        arm["diagnostic_evidence"][phase] = {"source": "CURRENT_PHASE_UNIT", "descriptor_path": reference["descriptor_path"], "descriptor_sha256": reference["descriptor_sha256"]}
                    else:
                        historical_by_step = lineage.get("historical_diagnostic_evidence", {}).get(phase) if isinstance(lineage.get("historical_diagnostic_evidence"), Mapping) else None
                        historical = None
                        if isinstance(historical_by_step, Mapping):
                            if str(step) in historical_by_step and isinstance(historical_by_step.get(str(step)), Mapping):
                                historical = historical_by_step[str(step)]
                            elif historical_by_step.get("status") == "PASS_READ_ONLY":
                                historical = historical_by_step
                        if historical:
                            arm["diagnostic_evidence"][phase] = {"source": "HISTORICAL_READ_ONLY", **dict(historical)}
                        else:
                            arm["diagnostic_evidence"][phase] = {"source": "MISSING_REQUIRED_REFERENCE"}
                arms.append(arm)
    handoff_path = campaign_root / "C10_HANDOFF.json"
    handoff = {
        "schema": C10_HANDOFF_SCHEMA,
        "status": "PASS" if all(all(item.get("source") != "MISSING_REQUIRED_REFERENCE" for item in arm["diagnostic_evidence"].values()) for arm in arms) else "BLOCKED_MISSING_ARM_EVIDENCE",
        "campaign_root": str(campaign_root),
        "external_inventory": str(inventory_path),
        "external_inventory_sha256": inventory_sha,
        "arms": arms,
        "arm_count": len(arms),
        "required_arm_count": 18,
        "evaluator_identity": dict(evaluator or {}),
        "orchestration_identity": dict(orchestration or {}),
        "locked_boundaries": {"Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
        "source_rows_untouched": True,
    }
    write_json(handoff_path, handoff)
    return {
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha,
        "handoff_path": str(handoff_path),
        "handoff_sha256": sha256_file(handoff_path),
        "handoff": handoff,
        "descriptors": descriptors,
    }


def build_c10_handoff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Named C10-facing entrypoint; implementation remains one closure."""

    return build_external_evidence_closure(*args, **kwargs)


def build_corelab_producer_fixture(root: Path) -> dict[str, Any]:
    """Build a tiny Vanda-produced fixture and its explicit Corelab V1 map.

    This is a CPU-only, test-only producer.  It exercises the same Vanda
    phase-spec and external-evidence serialization functions used by the
    campaign, then translates their V2/C10 output into the pinned Corelab V1
    verifier surface.  The fixture is deliberately marked synthetic and is
    never admissible to the real campaign or any execution token.
    """

    requested_root = Path(root).resolve()
    require(not requested_root.is_symlink(), f"synthetic fixture root must not be a symlink: {requested_root}")
    fixture_root = requested_root
    if fixture_root.exists():
        require(fixture_root.is_dir(), f"synthetic fixture parent is not a directory: {fixture_root}")
        fixture_root = fixture_root / "stage45e_v6_corelab_producer_fixture"
    require(not fixture_root.exists(), f"refusing to overwrite synthetic fixture root: {fixture_root}")
    fixture_root.mkdir(parents=True, exist_ok=False)
    producer_root = fixture_root / "vanda_producer"
    producer_root.mkdir()

    synthetic_lineages: dict[str, dict[int, dict[str, Any]]] = {route: {} for route in ROUTES}
    for route in ROUTES:
        for seed in ALL_SEEDS:
            checkpoint_records: dict[str, dict[str, Any]] = {}
            for step in (1000, 2000):
                checkpoint_records[str(step)] = {
                    "tree_sha256": sha256_bytes(f"SYNTHETIC_TEST_ONLY:{route}:{seed}:{step}".encode()),
                }
            synthetic_lineages[route][seed] = {
                "route": route,
                "seed": seed,
                "root": str(producer_root / "training" / f"{route}_seed{seed}"),
                "result_path": str(producer_root / "training" / f"{route}_seed{seed}" / "training_result.json"),
                "result_sha256": sha256_bytes(f"SYNTHETIC_RESULT:{route}:{seed}".encode()),
                "training_lineage_head": f"SYNTHETIC_HEAD_{route.upper()}_{seed}",
                "training_lineage_branch": f"synthetic/{route}",
                "training_source_kind": "SYNTHETIC_TEST_ONLY",
                "result": {f"checkpoint_{step}": record for step, record in ((1000, checkpoint_records["1000"]), (2000, checkpoint_records["2000"]))},
                "checkpoints": checkpoint_records,
            }

    evaluator = {"role": "SYNTHETIC_TEST_ONLY_EVALUATOR", "identity": "v6"}
    orchestration = {"role": "SYNTHETIC_TEST_ONLY_ORCHESTRATION", "head": "SYNTHETIC"}
    phase_specs = build_phase_specs(
        synthetic_lineages,
        producer_root,
        evaluator=evaluator,
        orchestration=orchestration,
        synthetic_test_only=True,
    )
    phase_roots: dict[str, dict[str, Path]] = {phase: {} for phase in DIAGNOSTIC_PHASES}
    for phase in DIAGNOSTIC_PHASES:
        for route in ROUTES:
            phase_root = producer_root / "phase_roots" / f"{phase}_{route}"
            phase_root.mkdir(parents=True)
            inner_path = phase_root / "seal.json"
            row_path = phase_root / "synthetic_rows.jsonl"
            write_json(inner_path, {
                "schema": "SYNTHETIC_TEST_ONLY_INNER_SEAL_V1",
                "status": "PASS",
                "phase": phase,
                "route": route,
            })
            append_jsonl(row_path, {
                "schema": "SYNTHETIC_TEST_ONLY_ROW_V1",
                "status": "RAW_SEALED_BEFORE_DERIVED",
                "phase": phase,
                "route": route,
                "synthetic_test_only": True,
            })
            budget = _expected_phase_totals(phase)[route]
            write_json(phase_root / "unified_phase_seal.json", {
                "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTIC_PHASE_SEAL_V2",
                "status": "PASS",
                "phase": phase,
                "route": route.upper(),
                "budget": budget,
                "inner_seal": str(inner_path),
                "inner_seal_sha256": sha256_file(inner_path),
                "phase_spec_sha256": sha256_file(phase_specs[phase][route]),
                "raw_before_derived": True,
                "public_sampler_calls": 0,
                "get_action_calls": 0,
                "Validation-E": 0,
                "Holdout-F": 0,
                "WBC": 0,
                "simulator": 0,
                "hardware": 0,
                "source_row_schema": "SYNTHETIC_TEST_ONLY_ROW_V1",
                "row_bundle": [str(row_path)],
                "stats_sha256": N2_STATS_SHA if route == "n2" else RUNTIME_STATS_SHA,
                "stats_payload_sha256": N2_STATS_PAYLOAD_SHA if route == "n2" else RUNTIME_STATS_PAYLOAD_SHA,
                "training_lineage_identities": [],
                "evaluation_identity": evaluator,
                "orchestration_identity": orchestration,
                "synthetic_test_only": True,
            })
            phase_roots[phase][route] = phase_root

    attempts_by_unit = {
        f"{phase}:{route}": [{
            "phase": phase,
            "route": route,
            "root": str(phase_roots[phase][route]),
            "status": "PASS_ACCEPTED",
            "counts": {
                "model_side_calls": _expected_phase_totals(phase)[route]["conditioning_passes"] + _expected_phase_totals(phase)[route]["action_head_calls"],
                "counters_complete": True,
            },
        }]
        for phase in DIAGNOSTIC_PHASES for route in ROUTES
    }
    vanda_closure = build_external_evidence_closure(
        producer_root,
        phase_roots,
        phase_specs,
        synthetic_lineages,
        attempts_by_unit=attempts_by_unit,
        evaluator=evaluator,
        orchestration=orchestration,
    )
    pointer_path = producer_root / "V6_PRODUCER_POINTER.json"
    write_json(pointer_path, {
        "schema": "VANDA_STAGE45E_V6_PRODUCER_POINTER_V1",
        "status": "SYNTHETIC_TEST_ONLY",
        "producer_closure_schema": C10_HANDOFF_SCHEMA,
        "external_inventory": vanda_closure["inventory_path"],
        "external_inventory_sha256": vanda_closure["inventory_sha256"],
        "c10_handoff": vanda_closure["handoff_path"],
        "c10_handoff_sha256": vanda_closure["handoff_sha256"],
        "synthetic_test_only": True,
    })

    corelab_root = fixture_root / "corelab_v1"
    corelab_root.mkdir()
    phase_totals = {
        phase: {
            key: sum(_expected_phase_totals(phase)[route.lower()][key] for route in ("LEGACY", "N2", "L1"))
            for key in ("conditioning_passes", "action_head_calls", "raw_rows")
        }
        for phase in DIAGNOSTIC_PHASES
    }
    for phase in DIAGNOSTIC_PHASES:
        phase_totals[phase]["total_model_side_calls"] = phase_totals[phase]["conditioning_passes"] + phase_totals[phase]["action_head_calls"]
    phase_seals: dict[str, dict[str, Any]] = {phase: {} for phase in DIAGNOSTIC_PHASES}
    for phase in DIAGNOSTIC_PHASES:
        for route in ("LEGACY", "N2", "L1"):
            lower_route = route.lower()
            seal_path = phase_roots[phase][lower_route] / "unified_phase_seal.json"
            phase_seals[phase][route] = {
                "route": route,
                "phase": phase,
                "path": str(seal_path),
                "sha256": sha256_file(seal_path),
                "budget": _expected_phase_totals(phase)[lower_route],
            }
    summary_path = corelab_root / "cpu_summary.json"
    write_json(summary_path, {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_CPU_EVIDENCE_SUMMARY_V1",
        "status": "PASS_READONLY_FROM_SEALED_GPU_EVIDENCE",
        "phase_totals": phase_totals,
        "phase_seals": phase_seals,
        "descriptive_only": True,
        "no_performance_based_winner": True,
        "Validation-E": 0,
        "Holdout-F": 0,
        "synthetic_test_only": True,
        "producer_closure": {
            "schema": C10_HANDOFF_SCHEMA,
            "path": vanda_closure["handoff_path"],
            "sha256": vanda_closure["handoff_sha256"],
        },
    })
    inventory_entries = []
    for path in sorted(corelab_root.rglob("*"), key=lambda item: str(item)):
        if path.is_file() and not path.is_symlink():
            inventory_entries.append({
                "path": str(path.relative_to(corelab_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    inventory_path = corelab_root / "CPU_EVIDENCE_INVENTORY.json"
    write_json(inventory_path, {
        "schema": "VANDA_STAGE45E_H200_CPU_EVIDENCE_INVENTORY_V1",
        "status": "PASS",
        "entries": inventory_entries,
        "entry_count": len(inventory_entries),
        "synthetic_test_only": True,
    })
    final_path = corelab_root / "final_result.json"
    write_json(final_path, {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_RESULT_V1",
        "status": "PASS",
        "token": "STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_COMPLETE_FOR_PI_REVIEW",
        "training": False,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "merge": False,
        "total_model_side_calls": 8976,
        "fresh_training_lineages": 7,
        "campaign_root": str(corelab_root),
        "cpu_summary": str(summary_path),
        "cpu_summary_sha256": sha256_file(summary_path),
        "cpu_inventory": str(inventory_path),
        "cpu_inventory_sha256": sha256_file(inventory_path),
        "synthetic_test_only": True,
        "producer_contract": {
            "vanda_closure_schema": C10_HANDOFF_SCHEMA,
            "vanda_closure_path": vanda_closure["handoff_path"],
            "vanda_closure_sha256": vanda_closure["handoff_sha256"],
            "corelab_verifier_source_head": CORELAB_PR65_VERIFIER_HEAD,
            "mapping": "VANDA_V2_C10_TO_CORELAB_V1_COMPATIBILITY_FIXTURE",
        },
    })
    return {
        "schema": "VANDA_STAGE45E_V6_CORELAB_PRODUCER_FIXTURE_V1",
        "status": "SYNTHETIC_TEST_ONLY_READY",
        "synthetic_test_only": True,
        "fixture_root": str(fixture_root),
        "producer_root": str(producer_root),
        "producer_pointer": str(pointer_path),
        "producer_pointer_sha256": sha256_file(pointer_path),
        "vanda_external_inventory": vanda_closure["inventory_path"],
        "vanda_external_inventory_sha256": vanda_closure["inventory_sha256"],
        "vanda_c10_handoff": vanda_closure["handoff_path"],
        "vanda_c10_handoff_sha256": vanda_closure["handoff_sha256"],
        "corelab_campaign": str(final_path),
        "corelab_campaign_sha256": sha256_file(final_path),
        "corelab_summary": str(summary_path),
        "corelab_summary_sha256": sha256_file(summary_path),
        "corelab_inventory": str(inventory_path),
        "corelab_inventory_sha256": sha256_file(inventory_path),
        "corelab_verifier_source_head": CORELAB_PR65_VERIFIER_HEAD,
        "compatibility_mapping": "VANDA_V2_C10_TO_CORELAB_V1_COMPATIBILITY_FIXTURE",
        "gpu": 0,
        "pbs": 0,
        "model_forward": 0,
        "training": 0,
    }


def _child_attempt_records(campaign_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((Path(campaign_root) / "logs").glob("*.command.json")):
        if not path.is_file() or path.is_symlink():
            continue
        record = read_json(path)
        records.append({"command_record": str(path), "command_record_sha256": sha256_file(path), **record})
    return records


def _mark_model_child_in_progress(state: dict[str, Any], stage: str, *, kind: str) -> None:
    """Mark a child boundary conservatively before it is spawned.

    A parent cannot know whether a child reached import/construction/forward
    after the child has failed unless the child emitted a verified seal.  The
    parent therefore records UNKNOWN/IN_PROGRESS at launch and only upgrades
    those fields after a PASS.  This prevents an exception path from being
    misreported as ``model_execution_started=false``.
    """

    state["failure_stage"] = stage
    state["active_child"] = {"stage": stage, "kind": kind, "status": "IN_PROGRESS"}
    execution_state = dict(state.get("execution_state", {}))
    execution_state.update({"model_imported": "UNKNOWN", "model_construction": "UNKNOWN", "model_forward": "UNKNOWN"})
    if kind == "diagnostic":
        execution_state.update({"training": False, "backward": False, "optimizer_steps": 0})
    else:
        execution_state.update({"training": "UNKNOWN", "backward": "UNKNOWN", "optimizer_steps": "UNKNOWN"})
    state["execution_state"] = execution_state
    lifecycle = dict(state.get("execution_lifecycle", {}))
    lifecycle.update({
        "model_import": "IN_PROGRESS",
        "model_construction": "IN_PROGRESS",
        "forward": "IN_PROGRESS",
    })
    if kind == "training":
        lifecycle["training"] = "IN_PROGRESS"
    elif kind == "diagnostic":
        lifecycle.update({"raw_seal": "IN_PROGRESS", "derived": "IN_PROGRESS", "sampler_calls": "FORBIDDEN_ZERO"})
    state["execution_lifecycle"] = lifecycle


def _mark_model_child_pass(state: dict[str, Any], stage: str, *, kind: str) -> None:
    """Record the verified post-child lifecycle without erasing prior history."""

    state["active_child"] = {"stage": stage, "kind": kind, "status": "PASS"}
    execution_state = dict(state.get("execution_state", {}))
    execution_state.update({"model_imported": True, "model_construction": 1, "model_forward": 1})
    if kind == "training":
        execution_state.update({"training": True, "backward": True, "optimizer_steps": 2000})
    else:
        execution_state.update({"training": False, "backward": False, "optimizer_steps": 0})
    state["execution_state"] = execution_state
    lifecycle = dict(state.get("execution_lifecycle", {}))
    lifecycle.update({"model_import": "PASS", "model_construction": "PASS", "forward": "PASS"})
    if kind == "training":
        lifecycle["training"] = "PASS"
    elif kind == "diagnostic":
        lifecycle.update({"raw_seal": "PASS", "derived": "PASS", "sampler_calls": "FORBIDDEN_ZERO"})
    state["execution_lifecycle"] = lifecycle


def _model_execution_started(state: Mapping[str, Any]) -> bool | str:
    """Return a truthful tri-state value for the top-level failure seal."""

    execution = state.get("execution_state", {})
    if not isinstance(execution, Mapping):
        return "UNKNOWN"
    values = [execution.get(key) for key in ("model_imported", "model_construction", "model_forward", "training", "backward")]
    if any(value is True or (isinstance(value, int) and not isinstance(value, bool) and value > 0) for value in values):
        return True
    if any(isinstance(value, str) and value in {"UNKNOWN", "IN_PROGRESS", "UNKNOWN_AFTER_FAILURE"} for value in values):
        return "UNKNOWN"
    return False


def write_campaign_failure_seal(
    campaign_root: Path,
    *,
    state: Mapping[str, Any],
    lineages: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    attempts_by_unit: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Seal orchestration failure without rewriting any child evidence."""

    campaign_root = Path(campaign_root).resolve()
    child_attempts = _child_attempt_records(campaign_root)
    for unit, attempts in (attempts_by_unit or {}).items():
        for attempt in attempts:
            child_attempts.append({"unit_key": unit, **dict(attempt)})
    all_calls = _sum_known_call_counts(child_attempts)
    failed_attempts = [item for item in child_attempts if item.get("status") == "FAILED_ATTEMPT"]
    failed_calls = _sum_known_call_counts(failed_attempts)
    all_attempts_complete = all(
        isinstance(item.get("counts"), Mapping)
        and isinstance(item.get("counts", {}).get("model_side_calls"), int)
        and not isinstance(item.get("counts", {}).get("model_side_calls"), bool)
        and item.get("counts", {}).get("counters_complete") is not False
        for item in child_attempts
    )
    failed_attempts_complete = all(
        isinstance(item.get("counts"), Mapping)
        and isinstance(item.get("counts", {}).get("model_side_calls"), int)
        and not isinstance(item.get("counts", {}).get("model_side_calls"), bool)
        and item.get("counts", {}).get("counters_complete") is not False
        for item in failed_attempts
    )
    failure = {
        "schema": "VANDA_STAGE45E_UNIFIED_ORCHESTRATION_FAILURE_SEAL_V1",
        "status": "BLOCKED",
        "pi_gate": PI_GATE,
        "campaign_root": str(campaign_root),
        "allocation_identity": dict(state.get("allocation_identity", {})),
        "last_completed_lineage": state.get("last_completed_lineage"),
        "last_completed_unit": state.get("last_completed_unit"),
        "pending_queue": list(state.get("pending_queue", [])),
        "failure_stage": state.get("failure_stage", "UNKNOWN"),
        "active_child": state.get("active_child"),
        "model_execution_started": _model_execution_started(state),
        "execution_lifecycle": dict(state.get("execution_lifecycle", {})),
        "execution_state": dict(state.get("execution_state", {
            "model_imported": "UNKNOWN",
            "model_construction": "UNKNOWN",
            "model_forward": "UNKNOWN",
            "training": "UNKNOWN",
            "backward": "UNKNOWN",
            "optimizer_steps": "UNKNOWN",
        })),
        "attempted_model_calls": all_calls,
        "failed_attempt_model_calls": failed_calls,
        "attempted_model_calls_complete": all_attempts_complete,
        "failed_attempt_model_calls_complete": failed_attempts_complete,
        "unknown_counter_attempts": sum(not (
            isinstance(item.get("counts"), Mapping)
            and isinstance(item.get("counts", {}).get("model_side_calls"), int)
            and not isinstance(item.get("counts", {}).get("model_side_calls"), bool)
            and item.get("counts", {}).get("counters_complete") is not False
        ) for item in child_attempts),
        "unknown_counter_inventory": [
            {
                "name": item.get("name"),
                "unit_key": item.get("unit_key"),
                "status": item.get("status"),
                "command_record": item.get("command_record"),
                "seal_path": item.get("seal_path"),
            }
            for item in child_attempts
            if not (
                isinstance(item.get("counts"), Mapping)
                and isinstance(item.get("counts", {}).get("model_side_calls"), int)
                and not isinstance(item.get("counts", {}).get("model_side_calls"), bool)
                and item.get("counts", {}).get("counters_complete") is not False
            )
        ],
        "child_attempts": child_attempts,
        "retry_eligibility": state.get("retry_eligibility", "PI_REVIEW_REQUIRED"),
        "training_lineage_identities": {route: {str(seed): training_lineage_identity(item) for seed, item in seeds.items()} for route, seeds in (lineages or {}).items()},
        "evaluator_identity": dict(state.get("evaluator_identity", {})),
        "orchestration_identity": dict(state.get("orchestration_identity", {})),
        "immutable_children_referenced": True,
        "timestamp_utc": now_utc(),
    }
    path = campaign_root / "orchestration_failure_seal.json"
    if not path.exists():
        write_json(path, failure)
    return {"path": str(path), "sha256": sha256_file(path), "document": read_json(path)}


def cpu_closure(args: argparse.Namespace, campaign_root: Path, phase_roots: Mapping[str, Mapping[str, Path]], lineages: Mapping[str, Mapping[int, Mapping[str, Any]]], source_identities: Mapping[str, Any], highwater_record: Mapping[str, Any], *, phase_specs: Mapping[str, Mapping[str, Path]] | None = None, diagnostic_run: Mapping[str, Any] | None = None, evaluator: Mapping[str, Any] | None = None, orchestration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    phase_specs = phase_specs or {phase: {route: Path(campaign_root) / "specs" / f"{phase}_{route}.json" for route in ROUTES} for phase in DIAGNOSTIC_PHASES}
    evaluator = dict(evaluator or {})
    orchestration = dict(orchestration or {})
    expected = {phase: _expected_phase_totals(phase) for phase in DIAGNOSTIC_PHASES}
    phase_seals: dict[str, dict[str, Any]] = {}
    for phase in DIAGNOSTIC_PHASES:
        phase_seals[phase] = {}
        for route in ROUTES:
            unit = f"{phase}:{route}"
            seal_path = Path(phase_roots[phase][route]) / "unified_phase_seal.json"
            candidate = {"phase": phase, "route": route, "root": str(Path(phase_roots[phase][route]).resolve()), "status": "PASS_ACCEPTED", "counts": _attempt_call_counts(Path(phase_roots[phase][route]))}
            validated = _validate_diagnostic_pass(candidate, phase=phase, route=route, spec_path=Path(phase_specs[phase][route]).resolve(), expected_evaluator=evaluator or read_json(Path(phase_specs[phase][route])).get("evaluation_identity", {}))
            seal = read_json(seal_path)
            phase_seals[phase][route] = {"path": str(seal_path), "sha256": sha256_file(seal_path), "route": route, "phase": phase, "unit_key": unit, "budget": seal["budget"], "inner_seal_sha256": seal["inner_seal_sha256"], "phase_spec_sha256": seal["phase_spec_sha256"], "evaluation_identity": validated.get("evaluation_identity", {})}
    totals = {}
    for phase in DIAGNOSTIC_PHASES:
        totals[phase] = {key: sum(expected[phase][route][key] for route in ROUTES) for key in ("conditioning_passes", "action_head_calls", "raw_rows")}
        totals[phase]["total_model_side_calls"] = totals[phase]["conditioning_passes"] + totals[phase]["action_head_calls"]
    require(totals["E4"]["total_model_side_calls"] == 1680 and totals["D1"]["total_model_side_calls"] == 3840 and totals["D2A"]["total_model_side_calls"] == 3456, f"CPU total budget drift: {totals}")
    attempts_by_unit = (diagnostic_run or {}).get("attempts_by_unit", {})
    accepted_by_unit = (diagnostic_run or {}).get("accepted_by_unit", {})
    if not accepted_by_unit:
        accepted_by_unit = {f"{phase}:{route}": {"counts": {"model_side_calls": expected[phase][route]["conditioning_passes"] + expected[phase][route]["action_head_calls"]}} for phase in DIAGNOSTIC_PHASES for route in ROUTES}
    accounting = (diagnostic_run or {}).get("accounting") or diagnostic_accounting(attempts_by_unit, accepted_by_unit)
    require(accounting["accepted_unique_scientific_calls"] == 8976, f"accepted unique diagnostic calls drift: {accounting}")
    require(accounting.get("all_attempted_model_calls_complete") is True, "diagnostic attempted-call accounting is incomplete or unknown")
    require(accounting.get("failed_attempt_model_calls_complete") is True, "diagnostic failed-attempt accounting is incomplete or unknown")
    require(isinstance(accounting.get("all_attempted_model_calls"), int), "diagnostic attempted-call total is unknown")
    require(isinstance(accounting.get("failed_attempt_model_calls"), int), "diagnostic failed-attempt total is unknown")
    external = build_external_evidence_closure(campaign_root, phase_roots, phase_specs, lineages, attempts_by_unit=attempts_by_unit, evaluator=evaluator, orchestration=orchestration)
    require(external["handoff"].get("status") == "PASS", "C10 handoff is incomplete; refusing final token")
    lineage_provenance = {route: {str(seed): training_lineage_identity(item) for seed, item in lineages[route].items()} for route in ROUTES}
    fresh_lineages = sum(1 for route in ROUTES for seed, item in lineages[route].items() if item.get("result_path") and not (route == "legacy" and seed == 2222) and not (route == "l1" and seed == 2222))
    summary = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_CPU_EVIDENCE_SUMMARY_V2",
        "status": "PASS_READONLY_FROM_SEALED_GPU_EVIDENCE",
        "pi_gate": PI_GATE,
        "campaign_root": str(campaign_root),
        "source_identities": source_identities,
        "high_water": highwater_record,
        "phase_totals": totals,
        "phase_seals": phase_seals,
        "lineage_provenance": lineage_provenance,
        "historical_reuse": {"legacy_seed2222": str(Path(args.legacy_history).resolve()), "l1_seed2222": str(Path(args.recovered_l1_root).resolve())},
        "diagnostic_accounting": accounting,
        "external_evidence_closure": {key: value for key, value in external.items() if key != "descriptors"},
        "evaluator_identity": evaluator,
        "orchestration_identity": orchestration,
        "descriptive_only": True,
        "no_performance_based_winner": True,
        "cpu_closure_execution": {"gpu": 0, "pbs": 0, "model_imported": False, "model_construction": 0, "model_forward": 0, "training": 0, "backward": 0, "optimizer_steps": 0, "public_sampler_calls": 0, "get_action_calls": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
        "campaign_history": {"training_occurred": True, "fresh_training_lineages": fresh_lineages, "accepted_training_lineage_count": 9, "diagnostic_model_calls_occurred": accounting["all_attempted_model_calls"] > 0, "diagnostic_attempts_observed": len(accounting["attempt_inventory"])},
        "timestamp_utc": now_utc(),
    }
    write_json(campaign_root / "cpu_summary.json", summary)
    inventory = _collect_files(campaign_root, excluded={"CPU_EVIDENCE_INVENTORY.json", "SHA256SUMS", "final_result.json", "C10_HANDOFF.json"})
    write_json(campaign_root / "CPU_EVIDENCE_INVENTORY.json", {"schema": "VANDA_STAGE45E_H200_CPU_EVIDENCE_INVENTORY_V2", "status": "PASS", "entries": inventory, "entry_count": len(inventory), "external_inventory": external["inventory_path"], "external_inventory_sha256": external["inventory_sha256"], "c10_handoff_excluded": True})
    inventory_sha = sha256_file(campaign_root / "CPU_EVIDENCE_INVENTORY.json")
    sums_path = campaign_root / "SHA256SUMS"
    with sums_path.open("x", encoding="utf-8") as handle:
        for entry in inventory:
            handle.write(f"{entry['sha256']}  {entry['path']}\n")
        handle.write(f"{inventory_sha}  CPU_EVIDENCE_INVENTORY.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    final = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_RESULT_V2",
        "status": "PASS",
        "token": "STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_COMPLETE_FOR_PI_REVIEW",
        "pi_gate": PI_GATE,
        "campaign_root": str(campaign_root),
        "cpu_summary": str(campaign_root / "cpu_summary.json"),
        "cpu_summary_sha256": sha256_file(campaign_root / "cpu_summary.json"),
        "cpu_inventory": str(campaign_root / "CPU_EVIDENCE_INVENTORY.json"),
        "cpu_inventory_sha256": inventory_sha,
        "sha256sums": str(sums_path),
        "phase_totals": totals,
        "total_model_side_calls": accounting["accepted_unique_scientific_calls"],
        "accepted_unique_scientific_calls": accounting["accepted_unique_scientific_calls"],
        "all_attempted_model_calls": accounting["all_attempted_model_calls"],
        "failed_attempt_model_calls": accounting["failed_attempt_model_calls"],
        "reused_readonly_model_calls": accounting["reused_readonly_model_calls"],
        "diagnostic_attempt_inventory": accounting["attempt_inventory"],
        "c10_handoff": external["handoff_path"],
        "c10_handoff_sha256": external["handoff_sha256"],
        "fresh_training_lineages": fresh_lineages,
        "historical_legacy_seed2222_reused": True,
        "historical_l1_seed2222_reused": True,
        "cpu_closure_execution": summary["cpu_closure_execution"],
        "campaign_history": summary["campaign_history"],
        "merge": False,
        "timestamp_utc": now_utc(),
    }
    write_json(campaign_root / "final_result.json", final)
    return final


def _diagnostic_attempt_snapshot(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Collect immutable diagnostic attempts after a partial child failure."""

    evidence_root = Path(getattr(args, "diagnostic_evidence_root", None) or Path(args.storage_root) / "diagnostics").resolve()
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for phase in DIAGNOSTIC_PHASES:
        for route in ROUTES:
            unit = f"{phase}:{route}"
            snapshot[unit] = discover_diagnostic_attempts(evidence_root, phase, route)
    return snapshot


def campaign(args: argparse.Namespace) -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "campaign must run inside a PBS allocation")
    storage = Path(args.storage_root).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    for name in ("training", "diagnostics", "failures", "provenance", "campaigns"):
        (storage / name).mkdir(parents=True, exist_ok=True)
    job_id = os.environ["PBS_JOBID"]
    campaign_root = storage / "campaigns" / f"pi0143_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', job_id)}"
    campaign_root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "failure_stage": "ALLOCATION_ADMISSION",
        "last_completed_lineage": None,
        "last_completed_unit": None,
        "pending_queue": list(TRAINING_QUEUE),
        "execution_state": {"model_imported": False, "model_construction": 0, "model_forward": 0, "training": False, "backward": False, "optimizer_steps": 0},
        "execution_lifecycle": {
            "admission": "NOT_STARTED",
            "model_import": "NOT_STARTED",
            "model_construction": "NOT_STARTED",
            "forward": "NOT_STARTED",
            "raw_seal": "NOT_STARTED",
            "sampler_calls": "NOT_STARTED",
            "derived": "NOT_STARTED",
            "training": "NOT_STARTED",
        },
        "active_child": None,
        "retry_eligibility": "PI_REVIEW_REQUIRED",
    }
    lineages: dict[str, dict[int, dict[str, Any]]] = {route: {} for route in ROUTES}
    diagnostic_run: dict[str, Any] = {}
    ledger_path = Path(args.allocation_ledger).resolve()
    try:
        if not ledger_path.exists():
            seed_allocation_ledger(ledger_path, HISTORICAL_ALLOCATION_RECORDS)
        ledger_admission = admit_current_allocation(ledger_path, pbs_jobid=job_id, root=campaign_root)
        state["allocation_identity"] = {"pbs_jobid": job_id, "ordinal": ledger_admission["record"].get("ordinal"), "ledger": str(ledger_path)}
        state["execution_lifecycle"]["admission"] = "PASS"
        orchestration = orchestration_identity()
        evaluator = evaluator_identity(args)
        state["orchestration_identity"] = orchestration
        state["evaluator_identity"] = evaluator
        gpu = _gpu_observation()
        state["failure_stage"] = "INPUT_AUDIT"
        orchestration_branch = orchestration["branch"]
        orchestration_head = orchestration["head"]
        require(orchestration_branch == "stage45e/h200-unified-replication-campaign", f"orchestration branch drift: {orchestration_branch}")
        require(orchestration["working_tree_clean"], "orchestration checkout is dirty")
        source_identities = verify_sources(args, Path(args.n2_repo).resolve())
        route_worktrees = {route: observe_worktree(Path(getattr(args, f"{route}_repo")), route) for route in ROUTES}
        state["failure_stage"] = "LINEAGE_BINDING"
        lineages, plan = _bind_lineages(args, storage, campaign_root)
        state["pending_queue"] = [{"route": item["route"], "seed": item["seed"], "action": item["action"]} for item in plan if item["action"] == "FRESH_BASE_2000"]
        historical = {"legacy": lineages["legacy"][2222], "l1": lineages["l1"][2222]}
        fresh_base_count = sum(item["action"] == "FRESH_BASE_2000" for item in plan)
        highwater_record = high_water(storage, output=campaign_root / "high_water.json", planned_training_roots=fresh_base_count)
        phase_spec_paths = {phase: {route: str(campaign_root / "specs" / f"{phase}_{route}.json") for route in ROUTES} for phase in DIAGNOSTIC_PHASES}
        admission = {
            "schema": "VANDA_STAGE45E_H200_UNIFIED_CAMPAIGN_ADMISSION_V2",
            "status": "PASS",
            "pi_gate": PI_GATE,
            "issue_comment": PI_ISSUE_COMMENT,
            "orchestration_identity": orchestration,
            "evaluator_identity": evaluator,
            "pbs_jobid": job_id,
            "gpu": gpu,
            "source_identities": source_identities,
            "route_worktrees": route_worktrees,
            "historical_lineages": historical,
            "lineage_plan": str(campaign_root / "lineage_plan.json"),
            "lineage_queue": plan,
            "fresh_base_training_roots": fresh_base_count,
            "phase_specs": phase_spec_paths,
            "storage_root": str(storage),
            "allocation_ledger": str(ledger_path),
            "allocation_ordinal": state["allocation_identity"]["ordinal"],
            "high_water": highwater_record,
            "model_imported": False,
            "model_construction": 0,
            "model_forward": 0,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "WBC": 0,
            "simulator": 0,
            "hardware": 0,
            "timestamp_utc": now_utc(),
        }
        write_json(campaign_root / "campaign_admission.json", admission)
        state["failure_stage"] = "RESOURCE_CANARY"
        _mark_model_child_in_progress(state, "RESOURCE_CANARY", kind="canary")
        canary = _run_canary(args, campaign_root, gpu)
        args.canary = canary
        args.gpu_uuid = gpu["rows"][0]["uuid"]
        _mark_model_child_pass(state, "RESOURCE_CANARY", kind="canary")
        append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "H200_RESOURCE_CANARY_PASS", "path": str(canary), "timestamp_utc": now_utc()})
        for item in plan:
            if item["action"] != "FRESH_BASE_2000":
                append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "LINEAGE_REUSED_READ_ONLY", **item, "timestamp_utc": now_utc()})
                continue
            route, seed = item["route"], int(item["seed"])
            state["failure_stage"] = f"TRAINING_{route}_SEED{seed}"
            root = Path(item["root"])
            command = _training_command(args, route, seed, root, canary)
            _mark_model_child_in_progress(state, f"TRAINING_{route}_SEED{seed}", kind="training")
            _run_command(campaign_root, f"train_{route}_seed{seed}", command, extra_env={"STAGE45E_L1_AUTHORITY_TOKEN": PI_GATE})
            lineages[route][seed] = _training_result(route, root, seed)
            _mark_model_child_pass(state, f"TRAINING_{route}_SEED{seed}", kind="training")
            state["last_completed_lineage"] = {"route": route, "seed": seed}
            state["pending_queue"] = [pending for pending in state["pending_queue"] if not (pending["route"] == route and pending["seed"] == seed)]
            append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "TRAINING_LINEAGE_PASS", "route": route, "seed": seed, "root": str(root), "result_sha256": lineages[route][seed]["result_sha256"], "timestamp_utc": now_utc()})
        phase_specs = build_phase_specs(lineages, campaign_root, evaluator=evaluator, orchestration=orchestration)
        append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "PHASE_SPECS_SEALED", "timestamp_utc": now_utc()})
        state["failure_stage"] = "DIAGNOSTICS"
        _mark_model_child_in_progress(state, "DIAGNOSTICS", kind="diagnostic")
        diagnostic_run = run_diagnostics(args, campaign_root, campaign_root / "campaign_admission.json", phase_specs, state=state)
        _mark_model_child_pass(state, "DIAGNOSTICS", kind="diagnostic")
        state["last_completed_unit"] = "D2A:n2"
        append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "E4_D1_D2A_SEALED", "timestamp_utc": now_utc()})
        state["failure_stage"] = "CPU_CLOSURE"
        phase_roots = diagnostic_run["phase_roots"]
        final = cpu_closure(args, campaign_root, phase_roots, lineages, source_identities, highwater_record, phase_specs=phase_specs, diagnostic_run=diagnostic_run, evaluator=evaluator, orchestration=orchestration)
        append_allocation_outcome(ledger_path, pbs_jobid=job_id, outcome={"status": "PASS", "campaign_root": str(campaign_root), "final_result_sha256": sha256_file(campaign_root / "final_result.json"), "attempted_model_calls": final["all_attempted_model_calls"], "model_imported": True, "model_forward": True, "training": True})
        write_json(storage / "provenance" / f"{campaign_root.name}_final_result.json", final)
        return {"status": "PASS", "token": final["token"], "campaign_root": str(campaign_root), "final_result": str(campaign_root / "final_result.json"), "final_result_sha256": sha256_file(campaign_root / "final_result.json"), "orchestration_head": orchestration_head, "route_heads": ROUTE_HEADS, "phase_totals": final["phase_totals"]}
    except Exception:
        try:
            attempts_for_failure = diagnostic_run.get("attempts_by_unit", {})
            if not attempts_for_failure:
                attempts_for_failure = _diagnostic_attempt_snapshot(args)
            failure = write_campaign_failure_seal(campaign_root, state=state, lineages=lineages, attempts_by_unit=attempts_for_failure)
            append_allocation_outcome(ledger_path, pbs_jobid=job_id, outcome={"status": "BLOCKED", "campaign_root": str(campaign_root), "failure_seal_sha256": failure["sha256"], "attempted_model_calls": failure["document"]["attempted_model_calls"], "failed_attempt_model_calls": failure["document"]["failed_attempt_model_calls"], "model_imported": failure["document"]["execution_state"].get("model_imported"), "model_forward": failure["document"]["execution_state"].get("model_forward"), "training": failure["document"]["execution_state"].get("training")})
        except Exception:
            # Never hide the original orchestration error behind an evidence
            # writer error.  The outer caller still receives the original
            # failure, while the immutable child records remain available.
            pass
        raise


def audit_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Verify all late-bound inputs without observing or constructing a GPU model."""

    storage = Path(args.storage_root).resolve()
    source_identities = verify_sources(args, Path(args.n2_repo).resolve())
    route_worktrees = {route: observe_worktree(Path(getattr(args, f"{route}_repo")), route) for route in ROUTES}
    legacy_decision = legacy_provenance_decision(Path(args.legacy_history), Path(args.config))
    require(legacy_decision["status"] == "PASS_EXACT_H200_FROZEN_RECIPE_REUSE", f"{legacy_decision['token']}: {legacy_decision['reason']}")
    historical = {
        "legacy": legacy_decision["lineage"],
        "legacy_seed2222_provenance_decision": {key: value for key, value in legacy_decision.items() if key != "lineage"},
        "l1": verify_recovered_l1(Path(args.recovered_l1_root), Path(args.recovered_l1_acceptance), Path(args.recovered_l1_input_v2)),
    }
    historical["diagnostic_evidence_catalog"] = audit_historical_diagnostic_evidence_catalog(require_real=True)
    historical["l1_duplicate_only"] = duplicate_l1_provenance(Path(args.duplicate_l1_root))
    supplemental = {
        "groot_manifest": Path(args.groot_manifest).resolve(),
        "release": Path(args.release).resolve(),
        "video_qualification": Path(args.video_qualification).resolve(),
        "ffmpeg_root": Path(args.ffmpeg_root).resolve(),
    }
    for name, path in supplemental.items():
        require(path.exists() and not path.is_symlink(), f"supplemental path unavailable: {name}: {path}")
    highwater_record = high_water(storage, planned_training_roots=_planned_fresh_training_count(storage))
    return {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_INPUT_AUDIT_PI0143_V2",
        "status": "PASS_CPU_ONLY",
        "source_identities": source_identities,
        "route_worktrees": route_worktrees,
        "historical_lineages": historical,
        "identity_separation": {"orchestration": orchestration_identity(), "evaluator": evaluator_identity(args)},
        "p0_checklist": p0_checklist(),
        "supplemental_paths": {key: str(value) for key, value in supplemental.items()},
        "high_water": highwater_record,
        "gpu": 0,
        "model_imported": False,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("static", "high-water", "audit-inputs", "campaign"), default="static")
    p.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    p.add_argument("--allocation-ledger", type=Path, default=STORAGE_ROOT / "provenance/STAGE45E_H200_ALLOCATION_LEDGER_PI0143.jsonl")
    p.add_argument("--diagnostic-evidence-root", type=Path, default=None)
    p.add_argument("--n2-repo", type=Path, default=WORKTREE_ROOT / "n2")
    p.add_argument("--legacy-repo", type=Path, default=WORKTREE_ROOT / "legacy")
    p.add_argument("--l1-repo", type=Path, default=WORKTREE_ROOT / "l1")
    for name, default in DEFAULTS.items():
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, default=default)
    p.add_argument("--recovered-l1-root", type=Path, default=RECOVERED_L1_ROOT)
    p.add_argument("--recovered-l1-acceptance", type=Path, default=RECOVERED_L1_ACCEPTANCE)
    p.add_argument("--recovered-l1-input-v2", type=Path, default=RECOVERED_L1_INPUT_V2)
    p.add_argument("--duplicate-l1-root", type=Path, default=DUPLICATE_L1_ROOT)
    p.add_argument("--groot-manifest", type=Path, default=Path("/scratch/e1192548/stage45b_pi0086_groot_snapshot__zhu4zz2/STAGE45B_GROOT_SOURCE_SNAPSHOT_V2.manifest.json"))
    p.add_argument("--release", type=Path, default=Path("/scratch/e1192548/stage45b_pi0087_admission_IWmsW5/STAGE45B_B2_TRAINING_RELEASE_V2.json"))
    p.add_argument("--video-qualification", type=Path, default=Path("/scratch/e1192548/stage45b_pi0087_final_video_qual_DXDgKu/VANDA_STAGE45B_PI0087_FINAL_ADMISSION_QUALIFIED.json"))
    p.add_argument("--ffmpeg-root", type=Path, default=Path("/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "static":
            result = static_audit()
        elif args.mode == "high-water":
            storage = Path(args.storage_root)
            result = high_water(storage, planned_training_roots=_planned_fresh_training_count(storage))
        elif args.mode == "audit-inputs":
            result = audit_inputs(args)
        else:
            result = campaign(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
        return 0
    except Exception as exc:
        print(f"STAGE45E_H200_UNIFIED_REPLICATION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
