"""Static regression checks for the PI0153 first-action diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = ROOT / "scripts" / "stage45f_pi0153_first_action_semantics.py"
PROBE = ROOT / "scripts" / "stage45f_pi0153_generation_probe.py"


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_pi0153_scripts_are_syntax_valid_and_lazy_runtime_imports() -> None:
    for path in (SEMANTICS, PROBE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not _top_level_imports(tree).intersection({"torch", "transformers", "gr00t", "isaacgym"})


def test_generation_probe_keeps_fixed_budget_and_exact_worker_boundary() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert "MAX_QUERIES = 4" in source
    assert "MAX_CONDITIONING = 4" in source
    assert "MAX_HEAD_CALLS = 16" in source
    assert "MAX_MODEL_CALLS = 20" in source
    assert "model.get_action(**worker_inputs)" in source
    assert "value.detach().to(device=\"cpu\", dtype=torch.float32)" in source
    assert "worker._build_processor" not in source
    assert "CHECKPOINT_TREE" in source
    assert "N2_STATS_PAYLOAD_SHA" in source


def test_cpu_semantics_packet_keeps_correct_world_bridge_and_zero_forward_boundary() -> None:
    source = SEMANTICS.read_text(encoding="utf-8")
    assert "measured_eef_world + (decoded_relative - StateV1[:3])" in source
    assert '"new_model_forwards": 0' in source
    assert '"generation_probe_decision": {"required": True' in source
