"""Static regression checks for the PI0153 first-action diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = ROOT / "scripts" / "stage45f_pi0153_first_action_semantics.py"
PROBE = ROOT / "scripts" / "stage45f_pi0153_generation_probe.py"
SUPPLEMENT = ROOT / "scripts" / "stage45f_pi0155_cpu_supplement.py"


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
    assert "MAX_QUERIES = 5" in source
    assert "MAX_CONDITIONING = 5" in source
    assert "MAX_HEAD_CALLS = 20" in source
    assert "MAX_MODEL_CALLS = 25" in source
    assert "NOMINAL_QUERIES = 4" in source
    assert "NOMINAL_MODEL_CALLS = 20" in source
    assert "model.get_action(**worker_inputs)" in source
    assert "torch.inference_mode()" in source
    assert "_inference_query(torch" in source
    assert "value.detach().to(device=\"cpu\", dtype=torch.float32)" in source
    # PI0155 requires the real deployment builder in a separate CPU process;
    # the model process itself still reuses the validated C1 instance to avoid
    # duplicate modality registration.
    assert "WORKER_BUILDER" in source
    assert "processor child" in source
    assert "CHECKPOINT_TREE" in source
    assert "N2_STATS_PAYLOAD_SHA" in source


def test_pi0155_repairs_are_explicit_and_supplement_is_cpu_only() -> None:
    source = PROBE.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    assert 'PI_GATE = "PI_GATE_STATE_0156"' in source
    assert "isinstance(sampled, tuple) and len(sampled) == 2" in source
    assert 'noise_injection_status": "EXACTLY_ONE' in source
    assert 'model_side_calls_unknown' in source
    assert 'AUTHORIZED_GPU_UUID' in source
    assert "REFERENCE_ORIGIN_DERIVED_NOT_PROVEN_MEASURED_EEF" in supplement
    assert "UNKNOWN_NOT_RECORDED" in supplement
    assert '"new_model_forwards": 0' in supplement
    assert '"validation_e_read": False' in supplement


def test_cpu_semantics_packet_keeps_correct_world_bridge_and_zero_forward_boundary() -> None:
    source = SEMANTICS.read_text(encoding="utf-8")
    assert "measured_eef_world + (decoded_relative - StateV1[:3])" in source
    assert '"new_model_forwards": 0' in source
    assert '"generation_probe_decision": {"required": True' in source
