from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

import stage45e_pi0153_selection_packet as packet  # noqa: E402


def _row(route: str, seed: int, tier: int, index: int) -> dict:
    window_id = packet.EXPECTED_WINDOWS[index]
    metrics = {
        "final_target_distance_m": float(tier) / 1000.0 + index / 10000.0,
        "absolute_progress_m": 0.1 + index / 1000.0,
        "minimum_target_distance_m": 0.05 + index / 1000.0,
        "prediction_path_length_m": 1.0 + index / 100.0,
        "roughness_ratio": 2.0 + index / 10.0,
        "endpoint_target_vector_cosine": 0.5,
        "endpoint_error_m": 0.02 + index / 10000.0,
        "position_rmse_m": 0.01 + index / 10000.0,
        "best_progress_m": 0.2 + index / 1000.0,
    }
    return {
        "arm": packet.label_for(route, seed, tier),
        "route": route,
        "seed": seed,
        "tier": tier,
        "window_index": index,
        "window_id": window_id,
        "target_id": window_id.split(":", 1)[0].removeprefix("stage22_expert_"),
        "source_kind": "CURRENT_H200_SEALED_E4",
        "stats_route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "checkpoint_tree_sha256": "a" * 64,
        "progress_sign": "POSITIVE",
        "metrics": metrics,
    }


def test_pair_specs_have_exact_pi0153_shape() -> None:
    specs = packet._pair_specs()
    assert len(specs) == 27
    assert {name: sum(item["comparison_class"] == name for item in specs) for name in ("intervention_vs_legacy", "exposure", "l1_vs_n2")} == {
        "intervention_vs_legacy": 12,
        "exposure": 9,
        "l1_vs_n2": 6,
    }
    for item in specs:
        assert item["left_seed"] == item["right_seed"]
        if item["comparison_class"] != "exposure":
            assert item["left_tier"] == item["right_tier"]
        else:
            assert (item["left_tier"], item["right_tier"]) == (2000, 1000)


def test_pairing_is_left_minus_right_and_target_stratified() -> None:
    rows = []
    for route in packet.ROUTES:
        for seed in packet.SEEDS:
            for tier in packet.TIERS:
                rows.extend(_row(route, seed, tier, index) for index in range(24))
    result = packet.build_paired_comparisons(rows)
    assert result["pairing_contract"] == {
        "intervention_vs_legacy": 12,
        "exposure": 9,
        "l1_vs_n2": 6,
        "total": 27,
        "same_seed_window_tier_required": True,
        "tiers_mixed_within_comparison": False,
        "new_weighted_score": False,
        "significance_test": False,
    }
    exposure = next(item for item in result["comparisons"] if item["comparison_class"] == "exposure" and item["left_route"] == "legacy" and item["left_seed"] == 2222)
    assert exposure["window_count"] == 24
    assert exposure["overall"]["row_count"] == 24
    assert exposure["by_target"]["red_target_train_a"]["row_count"] == 6
    assert exposure["overall"]["delta_left_minus_right"]["final_target_distance_m"]["mean"] == pytest.approx((2000 - 1000) / 1000.0)


def test_selection_packet_script_has_no_model_side_top_level_import() -> None:
    path = ROOT / "scripts" / "stage45e_pi0153_selection_packet.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    assert not imports.intersection({"torch", "gr00t", "transformers", "jax", "tensorflow", "cuda"})


def test_historical_row_compatibility_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    jsonl = tmp_path / "historical.jsonl"
    jsonl.write_text(
        '{"window_id":"stage22_expert_red_target_train_a:expert-0000","identity":{"target_id":"red_target_train_a","target_xyz_in_model_input":false}}\n'
        '{"window_id":"stage22_expert_red_target_train_a:expert-0001","identity":{"target_id":"red_target_train_a","target_xyz_in_model_input":false}}\n',
        encoding="utf-8",
    )
    rows = packet._read_rows(jsonl)
    assert len(rows) == 2
    assert packet._canonical_target_id(
        rows[0]["window_id"], packet._source_row_target_id(rows[0], rows[0]["window_id"]), "fixture"
    ) == "red_target_train_a"
    assert packet._source_target_exclusion(rows[0]) is False

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"valid":true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(packet.SelectionPacketError):
        packet._read_rows(malformed)
