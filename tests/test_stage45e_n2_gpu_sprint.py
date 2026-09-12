"""CPU/static regression checks for the PI0133 H200 N2 replication lane."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45e_n2_gpu_sprint as sprint  # noqa: E402


def test_pi0133_manifest_and_budget_are_frozen() -> None:
    manifest = json.loads(sprint.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["authority"]["issue_comment"] == 5642852068
    assert manifest["repository"]["branch"] == sprint.BRANCH
    assert manifest["repository"]["base_head"] == sprint.BASE_HEAD
    assert manifest["training"]["seeds"] == [2222, 2223, 2224]
    assert manifest["evaluation"]["arms"] == 6
    assert manifest["evaluation"]["action_head_calls"] == 576
    assert manifest["evaluation"]["conditioning_passes"] == 144
    assert manifest["evaluation"]["total_model_side_calls"] == 720
    assert manifest["diagnostics"]["d1"]["action_head_calls"] == 1296
    assert manifest["diagnostics"]["d1"]["conditioning_passes"] == 144
    assert manifest["diagnostics"]["d1"]["total_model_side_calls"] == 1440
    assert manifest["diagnostics"]["d2a"]["action_head_calls"] == 1152
    assert manifest["diagnostics"]["d2a"]["conditioning_passes"] == 144
    assert manifest["diagnostics"]["d2a"]["total_model_side_calls"] == 1296
    assert manifest["legacy_exact_checkpoint_relay"]["bundle_bytes"] == 41779394560
    assert manifest["legacy_exact_checkpoint_relay"]["bundle_sha256"] == "a022480f3882988aee3831f14f817b265525d87ec6486d879d9d525025771a05"
    assert manifest["n2_intervention"]["mask_active_scalars"] == 190
    assert manifest["n2_intervention"]["target_xyz_in_model_input"] is False


def test_sprint_static_audit_has_no_model_imports() -> None:
    result = sprint.static_audit()
    assert result["status"] == "PASS"
    assert result["top_level_forbidden_imports"] == []
    source = (ROOT / "scripts/stage45e_n2_gpu_sprint.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    assert not set(imports).intersection({"torch", "transformers", "gr00t", "isaacgym", "omni"})


def test_legacy_comparator_is_late_bound() -> None:
    source = (ROOT / "scripts/stage45e_n2_gpu_sprint.py").read_text(encoding="utf-8")
    assert "legacy_full24_2000" in source
    assert "def _legacy_checkpoint_hash" in source
    assert "/scratch/e1192548/stage45b_pi0087" not in source
    assert "no scratch-path substitution" in source


def test_h200_gpu_contract_is_generic_and_seed_complete() -> None:
    manifest = json.loads(sprint.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["gpu_policy"]["host_role"] == "Vanda Hopper H200"
    assert manifest["gpu_policy"]["preferred_uuid"] is None
    assert manifest["gpu_policy"]["minimum_vram_mib"] == 81920
    assert sprint.N2_SEEDS == (2222, 2223, 2224)
    assert sprint.MIN_H200_VRAM_MIB == 81920
    source = (ROOT / "scripts/stage45e_n2_gpu_sprint.py").read_text(encoding="utf-8")
    assert 'choices=("static", "admission", "train", "g2", "g3", "g4", "g5d1", "g5d2a", "n2-e4", "n2-d1", "n2-d2a", "continue-g3-g4-g5", "sprint")' in source
    assert "H200" in source
    assert "RTX4080S" not in source


def test_pi0133_diagnostic_modes_are_present_without_top_level_model_import() -> None:
    source = (ROOT / "scripts/stage45e_n2_gpu_sprint.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    assert not set(imports).intersection({"torch", "transformers", "gr00t", "isaacgym", "omni"})
    assert "def n2_e4_eval" in source
    assert "def _diagnostic_runtime" in source
    assert "N2_D1" in source and "N2_D2A" in source
