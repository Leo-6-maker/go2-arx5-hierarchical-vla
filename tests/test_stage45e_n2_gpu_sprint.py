"""CPU/static regression checks for the PI0124 Corelab sprint."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45e_n2_gpu_sprint as sprint  # noqa: E402


def test_pi0124_manifest_and_budget_are_frozen() -> None:
    manifest = json.loads(sprint.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["authority"]["issue_comment"] == 5634960729
    assert manifest["repository"]["branch"] == sprint.BRANCH
    assert manifest["repository"]["base_head"] == sprint.BASE_HEAD
    assert manifest["training"]["seeds"] == [2223, 2224]
    assert manifest["g2_train24_evaluation"]["total_model_side_calls"] == 480
    assert manifest["g3_d1_teacher_forced_local_flow"]["total_model_side_calls"] == 1152
    assert manifest["g4_d2a_tf_vs_fr"]["total_model_side_calls"] == 864
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
    assert "/scratch/e1192548/stage45b_pi0087" not in source
    assert "no scratch-path substitution" in source
