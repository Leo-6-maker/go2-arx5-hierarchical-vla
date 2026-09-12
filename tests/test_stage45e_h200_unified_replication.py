from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest


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


def test_manifest_has_exact_pi0134_budget_queue_and_continuation_ceiling() -> None:
    manifest = json.loads((ROOT / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0134_V1.json").read_text())
    assert manifest["authority"]["pi_gate"] == "PI_GATE_STATE_0134"
    assert manifest["authority"]["issue_comment"] == 5643644410
    assert manifest["gpu_policy"]["max_science_allocations"] == 8
    assert manifest["gpu_policy"]["previous_science_allocations_consumed"] == 3
    assert manifest["gpu_policy"]["additional_science_allocations_authorized"] == 5
    assert manifest["gpu_policy"]["target_additional_allocations"] <= 2
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
    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source
    assert '"HF_DATASETS_OFFLINE": "1"' in source
    assert "qsub(" not in source
    assert "git merge" not in source


def test_high_water_calculation_keeps_margin() -> None:
    expected = campaign.ESTIMATED_TRAINING_ROOT_BYTES * len(campaign.TRAINING_QUEUE)
    assert int(expected * 1.15) + campaign.MIN_FREE_BYTES > expected
    continuation_expected = campaign.ESTIMATED_TRAINING_ROOT_BYTES * 5
    assert int(continuation_expected * 1.15) + campaign.MIN_FREE_BYTES > continuation_expected


def test_only_authorized_l1_pre_model_failure_can_be_retried(tmp_path: Path) -> None:
    root = tmp_path / "l1_seed2223_failed"
    root.mkdir()
    (root / "stage45e_l1_failure.json").write_text(json.dumps({
        "status": "FAIL",
        "training_seed": 2223,
        "model_imported": False,
        "model_construction_started": False,
        "model_construction_completed": False,
        "training_started": False,
        "backward_started": False,
        "optimizer_steps_observed": 0,
        "checkpoint_writes_observed": 0,
        "action_head_forwards": 0,
    }))
    evidence = campaign._authorized_pre_model_retry("l1", 2223, root)
    assert evidence is not None
    assert evidence["model_imported"] is False
    assert campaign._authorized_pre_model_retry("legacy", 2223, root) is None
    (root / "stage45e_l1_failure.json").write_text(json.dumps({"status": "FAIL", "training_seed": 2223, "model_imported": True}))
    assert campaign._authorized_pre_model_retry("l1", 2223, root) is None


def test_checkpoint_acceptance_reconciles_state_and_tree(tmp_path: Path) -> None:
    root = tmp_path / "legacy_seed2223"
    checkpoint = root / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    entries = campaign.regular_tree_entries(checkpoint, "stage45e_legacy/checkpoint-1000")
    record = {
        "status": "PASS",
        "expected_step": 1000,
        "global_step": 1000,
        "required_state_files_present": True,
        "missing_state_files": [],
        "tree_sha256": campaign.tree_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
    }
    accepted = campaign._verify_checkpoint_artifact("legacy", root, 1000, record)
    assert accepted["tree_sha256"] == record["tree_sha256"]
    (checkpoint / "scheduler.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint tree hash drift|checkpoint byte-count drift"):
        campaign._verify_checkpoint_artifact("legacy", root, 1000, record)


def test_pbs_uses_cluster_routing_queue_and_single_h200() -> None:
    pbs = (ROOT / "hpc/pbs/stage45e_h200_unified_replication.pbs").read_text()
    assert "#PBS -q auto" in pbs
    assert "#PBS -l select=1:ngpus=1:ncpus=12:mem=225gb" in pbs
    assert "#PBS -q gpu" not in pbs
    assert "/hpctmp/e1192548/stage45e_h200_unified_replication_20260912" in pbs


def test_pbs_binds_qualified_native_video_runtime_before_python() -> None:
    pbs = (ROOT / "hpc/pbs/stage45e_h200_unified_replication.pbs").read_text()
    assert "FFMPEG_RUNTIME_ROOT=/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__" in pbs
    assert 'TORCHCODEC_PACKAGE_ROOT="${TORCH_SITE}/torchcodec"' in pbs
    assert 'TORCH_LIBRARY_ROOT="${TORCH_SITE}/torch/lib"' in pbs
    assert 'export LD_LIBRARY_PATH="${FFMPEG_RUNTIME_ROOT}/lib:${FFMPEG_RUNTIME_ROOT}/compat:${TORCHCODEC_PACKAGE_ROOT}:${TORCH_LIBRARY_ROOT}:${AMBIENT_LD_LIBRARY_PATH}"' in pbs
    assert pbs.index("export LD_LIBRARY_PATH=") < pbs.index("exec \"$ORCH_PY\"")
