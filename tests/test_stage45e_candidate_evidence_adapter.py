"""CPU/static tests for PI0128 sealed-row evidence normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage45e_candidate_evidence_adapter as adapter


def _row(window_id: str, target_id: str, checkpoint_label: str, checkpoint_sha: str, index: int) -> dict:
    return {
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
        "window_id": window_id,
        "target_id": target_id,
        "checkpoint": {"label": checkpoint_label, "sha256": checkpoint_sha},
        "metrics": {
            "target": {
                "final_target_distance_m": 0.1 + index * 1e-4,
                "absolute_progress_m": 0.04 - index * 1e-4,
                "minimum_target_distance_m": 0.08 + index * 1e-4,
                "endpoint_target_vector_cosine": None if index == 0 else 0.5,
                "path_length_m": 1.0 + index * 1e-3,
                "roughness_position_second_difference_m": {
                    "count": 38,
                    "min": 0.001,
                    "median": 0.02,
                    "mean": 0.03,
                    "p95": 0.05,
                    "max": 0.06,
                },
                "progress_sign": "POSITIVE" if index % 2 else "NEGATIVE",
            }
        },
        "target_xyz_in_model_input": False,
        "finite": True,
    }


def _index(tmp_path: Path) -> dict:
    cfg = adapter.manifest()
    candidates = {}
    for candidate_index, canonical_label in enumerate(cfg["candidate_order"]):
        checkpoint_label = cfg["source_checkpoint_label"][canonical_label]
        checkpoint_sha = f"{candidate_index + 1:064x}"
        rows = [
            _row(f"window-{index:02d}", f"target-{index // 6}", checkpoint_label, checkpoint_sha, index)
            for index in range(cfg["window_count"])
        ]
        rows_path = tmp_path / f"rows-{candidate_index}.json"
        rows_path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
        lineage_kind = "SEALED_LEGACY" if canonical_label.startswith("LEGACY") else (
            "RECOVERY_ACCEPTANCE" if canonical_label.startswith("L1") else "TRAINING_RESULT"
        )
        candidates[canonical_label] = {
            "rows_json": str(rows_path),
            "rows_sha256": adapter.sha256_file(rows_path),
            "provenance": {
                "upstream_evidence_seal_sha256": f"{candidate_index + 11:064x}",
                "checkpoint_tree_sha256": checkpoint_sha,
                "lineage_artifact_sha256": f"{candidate_index + 21:064x}",
                "lineage_kind": lineage_kind,
                "normalization_identity": "N2_BALANCED_XYZ" if canonical_label.startswith("N2") else "FULL24_RUNTIME_V2",
                "sampler_identity": "TRAIN24_E4_SEED4242",
                "decoder_identity": "STAGE40_STRUCTURED_DECODE_ACCEPTED",
                "metric_identity": "STAGE44R_TRAJECTORY_METRICS_ACCEPTED",
            },
        }
    return {
        "schema": cfg["adapter_input_schema"],
        "split": "TRAIN24",
        "validation_e_read": False,
        "holdout_f_read": False,
        "candidates": candidates,
    }


def test_plan_is_static_and_row_only() -> None:
    plan = adapter.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["accepted_row_schema"] == "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1"
    assert "checkpoint_tree_sha256" in plan["required_candidate_provenance"]["sha256_fields"]
    assert plan["required_candidate_provenance"]["row_checkpoint_sha_must_equal_checkpoint_tree_sha256"] is True
    assert plan["static_guard"]["model_construction"] == 0
    assert plan["static_guard"]["gpu"] == 0
    assert plan["static_guard"]["pbs"] == 0
    assert plan["static_guard"]["validation_e_read"] is False
    assert plan["static_guard"]["holdout_f_read"] is False


def test_adapter_normalizes_all_candidates_and_preserves_optional_cosine(tmp_path: Path) -> None:
    normalized = adapter.adapt(_index(tmp_path))
    cfg = adapter.manifest()
    assert normalized["schema"] == cfg["normalized_input_schema"]
    assert normalized["window_ids"] == [f"window-{index:02d}" for index in range(24)]
    assert list(normalized["candidates"]) == cfg["candidate_order"]
    first_candidate = normalized["candidates"]["LEGACY FULL24@2000"]
    first = first_candidate["rows"][0]
    assert first["metrics"]["endpoint_target_vector_cosine"] is None
    assert first["metrics"]["roughness_position_second_difference_m"] == pytest.approx(0.03)
    assert first_candidate["provenance"]["row_checkpoint_tree_binding"] == "EXACT"
    assert first_candidate["provenance"]["lineage_kind"] == "SEALED_LEGACY"
    assert normalized["adapter"]["model_forward"] == 0
    assert normalized["adapter"]["gpu"] == 0


def test_adapter_rejects_rows_hash_or_checkpoint_label_drift(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index["candidates"]["L1@2000"]["rows_sha256"] = "0" * 64
    with pytest.raises(adapter.CandidateEvidenceAdapterError, match="SHA mismatch"):
        adapter.adapt(index)

    index = _index(tmp_path)
    spec = index["candidates"]["L1@2000"]
    rows_path = Path(spec["rows_json"])
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows[0]["checkpoint"]["label"] = "WRONG"
    rows_path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
    spec["rows_sha256"] = adapter.sha256_file(rows_path)
    with pytest.raises(adapter.CandidateEvidenceAdapterError, match="checkpoint label drift"):
        adapter.adapt(index)


def test_adapter_rejects_checkpoint_tree_or_lineage_provenance_drift(tmp_path: Path) -> None:
    index = _index(tmp_path)
    spec = index["candidates"]["N2(seed2223)@2000"]
    spec["provenance"]["checkpoint_tree_sha256"] = "f" * 64
    with pytest.raises(adapter.CandidateEvidenceAdapterError, match="row checkpoint SHA does not match"):
        adapter.adapt(index)

    index = _index(tmp_path)
    del index["candidates"]["L1@1000"]["provenance"]["upstream_evidence_seal_sha256"]
    with pytest.raises(adapter.CandidateEvidenceAdapterError, match="upstream_evidence_seal_sha256"):
        adapter.adapt(index)

    index = _index(tmp_path)
    index["candidates"]["L1@1000"]["provenance"]["lineage_kind"] = "BEST_RETRY"
    with pytest.raises(adapter.CandidateEvidenceAdapterError, match="lineage_kind invalid"):
        adapter.adapt(index)


def test_adapt_from_file_writes_fresh_normalized_artifact(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index_path = tmp_path / "index.json"
    output_path = tmp_path / "normalized.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    result = adapter.adapt_from_file(index_path, output_path)
    assert output_path.is_file()
    assert result["adapter"]["index_sha256"] == adapter.sha256_file(index_path)
    with pytest.raises(adapter.CandidateEvidenceAdapterError, match="output must be fresh"):
        adapter.adapt_from_file(index_path, output_path)
