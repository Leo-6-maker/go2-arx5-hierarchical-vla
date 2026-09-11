"""CPU/static regression tests for the Stage45E N2 intervention."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import stage45e_n2_statistics as n2  # noqa: E402
import stage45e_n2_train24_eval as train24_eval  # noqa: E402


SOURCE = Path(
    "/var/cache/users/liuyu-nfs/stage45c_pi0100_train24_eval/"
    "evidence_completion_v3_pi0101_track_s/payload/decode_stats/"
    "STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2.json"
)
DATASET = Path(
    "/var/cache/users/liuyu-nfs/stage45c_pi0100_train24_eval/"
    "evidence_completion_v3_pi0101_track_s/payload/full24_dataset/lerobot_train"
)
N2_MANIFEST = ROOT / "manifests/STAGE45E_N2_BALANCED_XYZ_NORMALIZATION_V1.json"


def test_manifest_binds_exact_one_factor_n2() -> None:
    manifest = json.loads(N2_MANIFEST.read_text(encoding="utf-8"))
    intervention = manifest["intervention"]
    assert manifest["schema"] == "CORELAB_STAGE45E_N2_BALANCED_XYZ_NORMALIZATION_V1"
    assert intervention["xyz_universal_dimensions"] == [0, 1, 2]
    assert intervention["common_half_range_m"] == n2.HALF_RANGE_M
    assert intervention["non_xyz_selected_bounds_unchanged"] is True
    assert intervention["all_raw_statistics_unchanged"] is True
    assert manifest["source_runtime_v2"]["outer_sha256"] == n2.SOURCE_STATS_SHA256
    assert manifest["source_runtime_v2"]["payload_sha256"] == n2.SOURCE_PAYLOAD_SHA256
    assert manifest["source_runtime_v2"]["materialization_path_embedded"] is False
    assert manifest["authority"]["pi_gate"] == "PI_GATE_STATE_0122"
    assert manifest["authority"]["issue_comment"] == 5633959841
    assert manifest["return_tokens"]["resource_blocked"] == "STAGE45E_N2_CORELAB_4080_RESOURCE_BLOCKER_FOR_PI_REVIEW"
    assert manifest["return_tokens"]["train24_complete"] == "STAGE45E_N2_CORELAB_4080_TRAIN24_COMPLETE_FOR_PI_REVIEW"


def test_source_identity_and_n2_artifact_if_local_fixture_exists(tmp_path: Path) -> None:
    if not SOURCE.is_file() or not DATASET.is_dir():
        return
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    check = n2.validate_source_document(source, source_path=SOURCE)
    assert check["outer_sha256"] == n2.SOURCE_STATS_SHA256
    assert check["payload_sha256"] == n2.SOURCE_PAYLOAD_SHA256
    result = n2.build_from_paths(
        source_path=SOURCE,
        dataset_root=DATASET,
        output_path=tmp_path / "n2.json",
        repository_head="a843294780c1392369bfc987e468857c63fc13a5",
    )
    assert result["status"] == "PASS"
    assert result["clipping_audit"]["active_scalar_count_observed"] == 4560
    assert result["clipping_audit"]["xyz_scalar_count_observed"] == 1080
    artifact = json.loads((tmp_path / "n2.json").read_text(encoding="utf-8"))
    verified = n2.verify_n2_document(artifact, source=source)
    assert verified["execution_boundary_zero"] is True
    assert artifact["intervention"]["center_xyz"] == [
        0.07283134344877157,
        0.0026564446538982454,
        -0.003348907255348331,
    ]
    assert np.allclose(
        artifact["intervention"]["selected_lower_xyz"],
        [-0.034518127551228434, -0.10469302634610175, -0.11069837825534834],
        rtol=0.0,
        atol=1e-15,
    )


def test_n2_builder_has_no_model_runtime_imports() -> None:
    source = (ROOT / "scripts/stage45e_n2_statistics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    assert not any(name == "torch" or name.startswith("gr00t") for name in top_level_imports)
    assert "model_constructed" in source
    assert "gpu" in source
    assert "wbc_steps" in source


def test_inference_canary_drops_supervised_fields() -> None:
    source = (ROOT / "scripts/stage45e_n2_corelab.py").read_text(encoding="utf-8")
    assert 'inference_inputs.pop("action", None)' in source
    assert 'inference_inputs.pop("action_mask", None)' in source
    assert '(output_root / "experiment_cfg").mkdir' in source
    assert 'dict(_LAST_EXECUTION_BOUNDARY)' in source
    assert source.count('torch.autocast(device_type="cuda", dtype=torch.bfloat16)') >= 2


def test_train24_eval_static_boundary() -> None:
    source = (ROOT / "scripts/stage45e_n2_train24_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    assert not any(name.split(".")[0] in {"torch", "gr00t", "transformers", "isaacgym", "omni"} for name in top_level_imports)
    assert "STAGE45E_N2_CORELAB_TRAIN24_EVALUATION_COMPLETE_FOR_PI_REVIEW" in source
    assert "same_noise_across_checkpoints" in source
    assert "sealed_aggregate_reproduction" in source
    assert "automatic_winner" in source


def test_train24_eval_statistics_bridge_is_bounded_and_exact_structural() -> None:
    expected = {
        "selected_lower": [0.0, 1.0],
        "active_mask": [1, 0],
        "nested": {"q99": [0.5, 2.0]},
    }
    roundoff = {
        "selected_lower": [0.0, 1.0 + 5e-16],
        "active_mask": [1, 0],
        "nested": {"q99": [0.5, 2.0]},
    }
    binding = train24_eval._compare_statistics(roundoff, expected)
    assert binding["match"] is True
    assert binding["mismatch_count"] == 0
    assert binding["tolerance_atol"] == 1e-12

    structural_drift = {
        **roundoff,
        "active_mask": [1, 1],
    }
    blocked = train24_eval._compare_statistics(structural_drift, expected)
    assert blocked["match"] is False
    assert blocked["mismatch_count"] == 1
