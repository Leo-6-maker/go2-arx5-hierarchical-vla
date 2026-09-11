"""CPU/static regression checks for the PI0125 Corelab sprint."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45e_n2_gpu_sprint as sprint  # noqa: E402


def test_pi0125_manifest_and_budget_are_frozen() -> None:
    manifest = json.loads(sprint.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["authority"]["issue_comment"] == 5635727857
    assert manifest["repository"]["branch"] == sprint.BRANCH
    assert manifest["repository"]["base_head"] == sprint.BASE_HEAD
    assert manifest["training"]["seeds"] == [2223, 2224]
    assert manifest["g2_train24_evaluation"]["total_model_side_calls"] == 480
    assert manifest["g3_d1_teacher_forced_local_flow"]["total_model_side_calls"] == 1152
    assert manifest["g4_d2a_tf_vs_fr"]["total_model_side_calls"] == 864
    assert manifest["g5_d1_n2_at_1000"]["action_head_calls"] == 648
    assert manifest["g5_d1_n2_at_1000"]["conditioning_passes"] == 216
    assert manifest["g5_d1_n2_at_1000"]["total_model_side_calls"] == 864
    assert manifest["g5_d2a_n2_at_1000"]["action_head_calls"] == 576
    assert manifest["g5_d2a_n2_at_1000"]["conditioning_passes"] == 72
    assert manifest["g5_d2a_n2_at_1000"]["total_model_side_calls"] == 648
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


def test_g5_is_n2_1000_only_and_has_no_2000_rerun() -> None:
    manifest = json.loads(sprint.MANIFEST_PATH.read_text(encoding="utf-8"))
    for key in ("g5_d1_n2_at_1000", "g5_d2a_n2_at_1000"):
        assert manifest[key]["checkpoints"] == ["N2(seed2222)@1000", "N2(seed2223)@1000", "N2(seed2224)@1000"]
        assert manifest[key]["rerun_2000"] is False
    source = (ROOT / "scripts/stage45e_n2_gpu_sprint.py").read_text(encoding="utf-8")
    assert "def g5_d1_eval" in source
    assert "def g5_d2a_eval" in source
    assert "CORELAB_STAGE45E_N2_G5_D1_RAW_STAGE_V1" not in source  # schemas are phase-derived
