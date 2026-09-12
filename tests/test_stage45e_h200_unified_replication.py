from __future__ import annotations

import ast
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45e_h200_unified_diagnostics as diagnostics
import stage45e_h200_unified_replication as campaign


def test_static_surfaces_are_cpu_only() -> None:
    result = campaign.static_audit()
    assert result["status"] == "PASS"
    assert result["gpu"] == 0
    assert result["model_forward"] == 0
    assert result["training"] == 0
    assert diagnostics.static_audit()["status"] == "PASS"


def test_manifest_has_exact_pi0133_budget_and_queue() -> None:
    manifest = json.loads((ROOT / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0133_V1.json").read_text())
    assert manifest["authority"]["pi_gate"] == "PI_GATE_STATE_0133"
    assert manifest["training_contract"]["training_lineages"] == 7
    assert len(manifest["training_contract"]["queue"]) == 7
    assert manifest["diagnostic_contract"]["e4"]["total_model_side_calls"] == 1680
    assert manifest["diagnostic_contract"]["d1"]["total_model_side_calls"] == 3840
    assert manifest["diagnostic_contract"]["d2a"]["total_model_side_calls"] == 3456
    assert manifest["diagnostic_contract"]["total_model_side_calls"] == 8976


def test_phase_budget_partition_is_exact() -> None:
    assert campaign._expected_phase_totals("E4") == {
        "legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96},
        "l1": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96},
        "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 576, "raw_rows": 144},
    }
    for phase, expected in (("D1", {"conditioning_passes": 384, "action_head_calls": 3456, "raw_rows": 3456}), ("D2A", {"conditioning_passes": 384, "action_head_calls": 3072, "raw_rows": 1536})):
        observed = {key: sum(value[key] for value in campaign._expected_phase_totals(phase).values()) for key in expected}
        assert observed == expected


def test_no_top_level_model_imports_or_scheduler_mutation() -> None:
    for path in (ROOT / "scripts/stage45e_h200_unified_replication.py", ROOT / "scripts/stage45e_h200_unified_diagnostics.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert not imports.intersection({"torch", "gr00t", "transformers", "jax", "tensorflow"})
    source = (ROOT / "scripts/stage45e_h200_unified_replication.py").read_text()
    assert "subprocess.run" in source
    assert "qsub(" not in source
    assert "git merge" not in source


def test_high_water_calculation_keeps_margin() -> None:
    expected = campaign.ESTIMATED_TRAINING_ROOT_BYTES * len(campaign.TRAINING_QUEUE)
    assert int(expected * 1.15) + campaign.MIN_FREE_BYTES > expected


def test_pbs_uses_cluster_routing_queue_and_single_h200() -> None:
    pbs = (ROOT / "hpc/pbs/stage45e_h200_unified_replication.pbs").read_text()
    assert "#PBS -q auto" in pbs
    assert "#PBS -l select=1:ngpus=1:ncpus=12:mem=225gb" in pbs
    assert "#PBS -q gpu" not in pbs
