from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.stage25_o1_parameter_delta_audit import (
    ACTION_ENCODER_W1_KEY,
    CATEGORY_PROJECTOR_KEYS,
    EXPECTED_CATEGORY_IDS,
    EXPECTED_O1_INDEX_SHA256,
    EXPECTED_O1_SEAL_SHA256,
    SUPPLEMENTAL_METADATA_ALLOWLIST,
    _audit_checkpoint_pair,
    _module_group,
    _o1_seal_sha256_status,
    _o1_seal_rows,
    _partition_o1_checkpoint_scope,
    _require_o1_seal_sha256,
    _verify_supplemental_historical_binding,
    _verify_o1_seal_against_checkpoint,
    audit_exit_code,
    safe_relative_path,
    stage24_tree_sha256,
    stage25_supplemental_manifest_sha256,
    walk_regular_files,
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

    # The core row is internally consistent, but the whole seal is not the
    # frozen Stage23 seal and must be rejected before rows or checkpoint files
    # are parsed for scope.
    per_file = _verify_o1_seal_against_checkpoint(checkpoint, seal)
    assert per_file["errors"] == ["O1 checkpoint seal SHA256 mismatch"]
    assert per_file["entries"] == []
    assert per_file["scope"] is None
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


def _scope_entry(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "regular_file": True,
        "symlink": False,
    }


def _scope_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    core = [
        _scope_entry("config.json", b"core-config"),
        _scope_entry("model.safetensors", b"core-tensor"),
    ]
    sealed_rows = [
        {
            "sealed_path": f"candidate_200/{entry['path']}",
            "relative_path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in core
    ]
    supplemental = [
        _scope_entry(path, f"supplemental:{path}".encode("utf-8"))
        for path in SUPPLEMENTAL_METADATA_ALLOWLIST
    ]
    historical_binding = {
        str(entry["path"]): {
            "status": "EXACT_HISTORICAL_HASH_FOUND",
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "evidence": ["synthetic historical evidence"],
        }
        for entry in supplemental
    }
    return core + supplemental, sealed_rows, historical_binding


def test_stage25_two_layer_scope_accepts_exact_core_plus_five_supplementals() -> None:
    entries, sealed_rows, historical_binding = _scope_fixture()
    result = _partition_o1_checkpoint_scope(
        entries,
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert result["pass"] is True
    supplemental = result["allowed_supplemental_metadata"]
    assert supplemental["expected_file_count"] == 5
    assert supplemental["observed_file_count"] == 5
    assert supplemental["used_for_tensor_identity"] is False
    assert supplemental["SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS"] is True
    assert (
        supplemental["SUPPLEMENTAL_METADATA_HISTORY_BINDING_STATUS"]
        == "EXACT_MATCH_TO_STAGE23_HISTORICAL_EVIDENCE"
    )
    assert supplemental["SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES"] == []
    assert all(
        item["exact_match"]
        for item in supplemental["per_file"]
    )
    assert supplemental["STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256"] == (
        stage25_supplemental_manifest_sha256(
            [entry for entry in entries if entry["path"] in SUPPLEMENTAL_METADATA_ALLOWLIST]
        )
    )


def test_stage25_seal_scope_accepts_sealed_core_plus_five_supplementals(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    payloads = {
        "config.json": b"core-config",
        "model.safetensors": b"core-tensor",
    }
    payloads.update(
        {
            path: f"supplemental:{path}".encode("utf-8")
            for path in SUPPLEMENTAL_METADATA_ALLOWLIST
        }
    )
    entries: list[dict[str, object]] = []
    for relative, payload in payloads.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(_scope_entry(relative, payload))

    core_entries = [
        entry for entry in entries if entry["path"] not in SUPPLEMENTAL_METADATA_ALLOWLIST
    ]
    sealed_rows = [
        {
            "sealed_path": f"candidate_200/{entry['path']}",
            "relative_path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in core_entries
    ]
    historical_binding = {
        str(entry["path"]): {
            "status": "EXACT_HISTORICAL_HASH_FOUND",
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "evidence": ["synthetic historical evidence"],
        }
        for entry in entries
        if entry["path"] in SUPPLEMENTAL_METADATA_ALLOWLIST
    }
    seal = tmp_path / "checkpoint_file_seal.tsv"
    seal.write_text("synthetic seal\n", encoding="utf-8")
    seal_status = {
        "expected_o1_seal_sha256": EXPECTED_O1_SEAL_SHA256,
        "observed_o1_seal_sha256": EXPECTED_O1_SEAL_SHA256,
        "verified": True,
    }

    with (
        patch(
            "scripts.stage25_o1_parameter_delta_audit._o1_seal_sha256_status",
            return_value=seal_status,
        ),
        patch(
            "scripts.stage25_o1_parameter_delta_audit._o1_seal_rows",
            return_value=sealed_rows,
        ),
        patch(
            "scripts.stage25_o1_parameter_delta_audit.SUPPLEMENTAL_METADATA_PRIOR_BINDING",
            historical_binding,
        ),
    ):
        result = _verify_o1_seal_against_checkpoint(checkpoint, seal)

    assert result["pass"] is True
    assert result["expected_file_count"] == 2
    assert result["actual_file_count"] == 7
    assert result["scope"]["sealed_core"]["observed_file_count"] == 2
    assert result["scope"]["allowed_supplemental_metadata"][
        "SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS"
    ] is True
    assert result["scope"]["forbidden_unsealed_extra"]["paths"] == []


def test_stage25_two_layer_scope_rejects_core_missing_or_modified() -> None:
    entries, sealed_rows, historical_binding = _scope_fixture()
    missing = _partition_o1_checkpoint_scope(
        [entry for entry in entries if entry["path"] != "config.json"],
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert missing["pass"] is False
    assert "config.json" in missing["sealed_core"]["missing_paths"]

    modified = [
        {**entry, "sha256": "0" * 64}
        if entry["path"] == "config.json"
        else entry
        for entry in entries
    ]
    changed = _partition_o1_checkpoint_scope(
        modified,
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert changed["pass"] is False
    assert any("sealed core SHA256 mismatch: config.json" in error for error in changed["errors"])


def test_stage25_two_layer_scope_rejects_sixth_or_arbitrary_extra_file() -> None:
    entries, sealed_rows, historical_binding = _scope_fixture()
    missing_supplemental = _partition_o1_checkpoint_scope(
        [entry for entry in entries if entry["path"] != "experiment_cfg/config.yaml"],
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert missing_supplemental["pass"] is False
    assert "experiment_cfg/config.yaml" in missing_supplemental["allowed_supplemental_metadata"]["missing_paths"]

    sixth = _partition_o1_checkpoint_scope(
        entries + [_scope_entry("experiment_cfg/sixth.yaml", b"sixth")],
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert sixth["pass"] is False
    assert sixth["forbidden_unsealed_extra"]["paths"] == ["experiment_cfg/sixth.yaml"]

    arbitrary = _partition_o1_checkpoint_scope(
        entries + [_scope_entry("unexpected.txt", b"extra")],
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert arbitrary["pass"] is False
    assert arbitrary["forbidden_unsealed_extra"]["paths"] == ["unexpected.txt"]


def test_stage25_two_layer_scope_rejects_symlink_and_path_traversal(tmp_path: Path) -> None:
    entries, sealed_rows, historical_binding = _scope_fixture()
    symlink_entries = [
        {**entry, "symlink": True}
        if entry["path"] == SUPPLEMENTAL_METADATA_ALLOWLIST[0]
        else entry
        for entry in entries
    ]
    symlink_result = _partition_o1_checkpoint_scope(
        symlink_entries,
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert symlink_result["pass"] is False
    assert any("symlink file is not allowed" in error for error in symlink_result["errors"])

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    experiment_cfg = checkpoint / "experiment_cfg"
    experiment_cfg.mkdir()
    target = tmp_path / "outside.txt"
    target.write_bytes(b"outside")
    os.symlink(target, experiment_cfg / "config.yaml")
    with pytest.raises(ValueError):
        walk_regular_files(checkpoint)

    traversal = _partition_o1_checkpoint_scope(
        entries + [_scope_entry("experiment_cfg/../escape", b"escape")],
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert traversal["pass"] is False
    assert "experiment_cfg/../escape" in traversal["forbidden_unsealed_extra"]["invalid_entries"]


def test_stage25_supplemental_history_sha_mismatch_fails_closed() -> None:
    entries, sealed_rows, historical_binding = _scope_fixture()
    changed = [
        {**entry, "sha256": "0" * 64}
        if entry["path"] == "experiment_cfg/config.yaml"
        else entry
        for entry in entries
    ]
    result = _partition_o1_checkpoint_scope(
        changed,
        sealed_rows,
        historical_binding=historical_binding,
    )
    supplemental = result["allowed_supplemental_metadata"]
    assert result["pass"] is False
    assert supplemental["SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS"] is False
    assert supplemental["per_file"][1]["exact_match"] is False
    assert any(
        item["path"] == "experiment_cfg/config.yaml"
        for item in supplemental["SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES"]
    )


def test_stage25_supplemental_history_byte_mismatch_fails_closed() -> None:
    entries, sealed_rows, historical_binding = _scope_fixture()
    changed = [
        {**entry, "bytes": entry["bytes"] + 1}
        if entry["path"] == "experiment_cfg/dataset_statistics.json"
        else entry
        for entry in entries
    ]
    result = _partition_o1_checkpoint_scope(
        changed,
        sealed_rows,
        historical_binding=historical_binding,
    )
    assert result["pass"] is False
    assert any(
        item["path"] == "experiment_cfg/dataset_statistics.json"
        for item in result["SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES"]
    )


def test_stage25_missing_or_nonexact_supplemental_history_fails_closed() -> None:
    entries, _, historical_binding = _scope_fixture()
    missing_binding = dict(historical_binding)
    del missing_binding["experiment_cfg/config.yaml"]
    missing = _verify_supplemental_historical_binding(
        [entry for entry in entries if entry["path"] in SUPPLEMENTAL_METADATA_ALLOWLIST],
        missing_binding,
    )
    assert missing["SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS"] is False
    assert any(
        item["path"] == "experiment_cfg/config.yaml"
        for item in missing["SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES"]
    )

    nonexact_binding = {
        **historical_binding,
        "experiment_cfg/config.yaml": {
            **historical_binding["experiment_cfg/config.yaml"],
            "status": "HISTORICAL_COPY_FOUND_BUT_NOT_IMMUTABLY_BOUND",
        },
    }
    nonexact = _verify_supplemental_historical_binding(
        [entry for entry in entries if entry["path"] in SUPPLEMENTAL_METADATA_ALLOWLIST],
        nonexact_binding,
    )
    assert nonexact["SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS"] is False
    assert any(
        item["path"] == "experiment_cfg/config.yaml"
        for item in nonexact["SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES"]
    )


def test_stage25_safe_paths_and_tree_digest_are_deterministic() -> None:
    assert safe_relative_path("model.safetensors") == "model.safetensors"
    with pytest.raises(ValueError):
        safe_relative_path("../outside")
    entries = [{"path": "a", "bytes": 1, "sha256": "0" * 64}]
    first = stage24_tree_sha256(entries, "payload/model")
    second = stage24_tree_sha256(list(reversed(entries)), "payload/model")
    assert first == second
