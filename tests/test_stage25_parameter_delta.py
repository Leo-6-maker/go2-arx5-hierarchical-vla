from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage25_o1_parameter_delta_audit import (
    ACTION_ENCODER_W1_KEY,
    CATEGORY_PROJECTOR_KEYS,
    EXPECTED_CATEGORY_IDS,
    EXPECTED_O1_INDEX_SHA256,
    EXPECTED_O1_SEAL_SHA256,
    _audit_checkpoint_pair,
    _module_group,
    _o1_seal_sha256_status,
    _o1_seal_rows,
    _require_o1_seal_sha256,
    _verify_o1_seal_against_checkpoint,
    audit_exit_code,
    safe_relative_path,
    stage24_tree_sha256,
)


def test_stage25_freezes_fourteen_category_projector_keys_and_rows() -> None:
    assert len(CATEGORY_PROJECTOR_KEYS) == 14
    assert ACTION_ENCODER_W1_KEY in CATEGORY_PROJECTOR_KEYS
    assert EXPECTED_CATEGORY_IDS == (4, 5, 6, 7, 10, 11, 24, 25, 26, 27)
    assert EXPECTED_O1_INDEX_SHA256 == "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
    assert EXPECTED_O1_SEAL_SHA256 == "7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806"


def test_stage25_module_rules_cover_expected_recipe_groups() -> None:
    assert _module_group("action_head.action_encoder.W1.W")["group"] == "CATEGORY_PROJECTORS"
    assert _module_group("action_head.position_embedding.weight")["group"] == "CATEGORY_PROJECTORS"
    assert _module_group("action_head.model.proj_out_1.weight")["group"] == "DIFFUSION_ACTION_HEAD_SHARED_TRAINABLE"
    assert _module_group("backbone.vision_model.weight")["group"] == "VISUAL_LLM_VLLN_EXPECTED_FROZEN"
    assert _module_group("action_head.vlln.weight")["expected_trainable"] is False
    assert _module_group("unrecognized.serialized.key")["group"] == "OTHER_EXPECTED_FROZEN"


def test_stage25_seal_parser_reconstructs_candidate_200_paths(tmp_path: Path) -> None:
    seal = tmp_path / "checkpoint_file_seal.tsv"
    digest = hashlib.sha256(b"fixture").hexdigest()
    seal.write_text(
        "candidate_199\tmodel.safetensors\tregular file\t7\t" + digest + "\n"
        "candidate_200\tmodel.safetensors.index.json\tregular file\t7\t" + digest + "\n",
        encoding="utf-8",
    )
    rows = _o1_seal_rows(seal)
    assert [row["relative_path"] for row in rows] == ["model.safetensors.index.json"]
    assert rows[0]["sealed_path"] == "candidate_200/model.safetensors.index.json"


def test_stage25_altered_but_self_consistent_seal_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    shard = checkpoint / "model.safetensors"
    shard.write_bytes(b"synthetic shard")
    shard_digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    seal = tmp_path / "checkpoint_file_seal.tsv"
    seal.write_text(
        "candidate_200\tmodel.safetensors\tregular file\t"
        f"{shard.stat().st_size}\t{shard_digest}\n",
        encoding="utf-8",
    )

    # The per-file contents and hash are internally consistent, but the
    # whole seal is not the frozen Stage23 seal and must not be accepted.
    per_file = _verify_o1_seal_against_checkpoint(checkpoint, seal)
    assert per_file["errors"] == []
    assert per_file["pass"] is False
    status = _o1_seal_sha256_status(seal)
    assert status["verified"] is False
    with pytest.raises(ValueError, match="seal SHA256 mismatch"):
        _require_o1_seal_sha256(seal)


def test_stage25_audit_rejects_bad_seal_before_tensor_read(tmp_path: Path) -> None:
    seal = tmp_path / "checkpoint_file_seal.tsv"
    seal.write_text(
        "candidate_200\tmodel.safetensors\tregular file\t1\t"
        + hashlib.sha256(b"x").hexdigest()
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "rejection.json"
    with pytest.raises(ValueError, match="seal SHA256 mismatch"):
        _audit_checkpoint_pair(
            tmp_path / "missing-base",
            tmp_path / "missing-o1",
            seal,
            output,
            base_tree_prefix="payload/GR00T-N1.7-3B",
        )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["o1_seal_sha256"]["expected_o1_seal_sha256"] == EXPECTED_O1_SEAL_SHA256
    assert record["o1_seal_sha256"]["verified"] is False
    assert record["execution_boundary"]["tensor_read"] is False


def test_stage25_frozen_delta_is_nonzero_and_pass_is_zero() -> None:
    assert audit_exit_code({"status": "STOP_UNEXPECTED_FROZEN_DELTA"}) != 0
    assert audit_exit_code({"status": "PASS"}) == 0
    assert audit_exit_code({"status": "UNEXPECTED_STATUS"}) != 0


def test_stage25_safe_paths_and_tree_digest_are_deterministic() -> None:
    assert safe_relative_path("model.safetensors") == "model.safetensors"
    with pytest.raises(ValueError):
        safe_relative_path("../outside")
    entries = [{"path": "a", "bytes": 1, "sha256": "0" * 64}]
    first = stage24_tree_sha256(entries, "payload/model")
    second = stage24_tree_sha256(list(reversed(entries)), "payload/model")
    assert first == second
