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
    OBSERVED_HEADER_DTYPE_AUDIT,
    SERIALIZED_DTYPE_CAST_RULES,
    SERIALIZED_TIED_ALIAS_RULES,
    SUPPLEMENTAL_METADATA_ALLOWLIST,
    _audit_checkpoint_pair,
    _module_group,
    _o1_seal_sha256_status,
    _o1_seal_rows,
    _partition_o1_checkpoint_scope,
    _reconcile_serialized_key_set,
    _compare_streamed_key,
    _compare_tensor_values,
    _require_o1_seal_sha256,
    _serialized_dtype_comparison,
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


def _serialized_key_fixture() -> tuple[dict[str, str], dict[str, str]]:
    rule = SERIALIZED_TIED_ALIAS_RULES[0]
    base = {
        rule["base_only_key"]: rule["base_shard"],
        rule["paired_key"]: rule["base_shard"],
        "ordinary.key": "model-00002-of-00002.safetensors",
    }
    o1 = {
        rule["paired_key"]: "model-00001-of-00002.safetensors",
        "ordinary.key": "model-00002-of-00002.safetensors",
    }
    return base, o1


def test_stage25_reconciles_only_the_proven_tied_alias_omission() -> None:
    base, o1 = _serialized_key_fixture()
    result = _reconcile_serialized_key_set(base, o1)
    rule = SERIALIZED_TIED_ALIAS_RULES[0]

    assert result["status"] == "PASS_EXPECTED_TIED_ALIAS_OMISSION"
    assert result["base_key_count"] == 3
    assert result["o1_key_count"] == 2
    assert result["common_key_count"] == 2
    assert result["allowed_base_only_keys"] == [rule["base_only_key"]]
    assert result["unexpected_missing_keys"] == []
    assert result["unexpected_extra_keys"] == []
    assert rule["base_only_key"] not in result["comparison_keys"]
    assert rule["paired_key"] in result["comparison_keys"]


def test_stage25_tied_alias_reconciliation_rejects_unlisted_missing_or_extra() -> None:
    base, o1 = _serialized_key_fixture()
    base["unexplained.missing"] = "model-00002-of-00002.safetensors"
    with pytest.raises(ValueError, match="unexpected missing keys"):
        _reconcile_serialized_key_set(base, o1)

    base, o1 = _serialized_key_fixture()
    o1["unexplained.extra"] = "model-00002-of-00002.safetensors"
    with pytest.raises(ValueError, match="unexpected extra keys"):
        _reconcile_serialized_key_set(base, o1)


def test_stage25_tied_alias_reconciliation_requires_pair_and_provenance_shard() -> None:
    base, o1 = _serialized_key_fixture()
    rule = SERIALIZED_TIED_ALIAS_RULES[0]
    del o1[rule["paired_key"]]
    with pytest.raises(ValueError, match="requires paired key"):
        _reconcile_serialized_key_set(base, o1)

    base, o1 = _serialized_key_fixture()
    base[rule["base_only_key"]] = "unexpected-shard.safetensors"
    with pytest.raises(ValueError, match="unexpected BASE shard"):
        _reconcile_serialized_key_set(base, o1)


def test_stage25_dtype_cast_rule_is_exact_and_fail_closed() -> None:
    assert OBSERVED_HEADER_DTYPE_AUDIT["common_key_count"] == 1030
    assert OBSERVED_HEADER_DTYPE_AUDIT["mismatched_common_key_count"] == 537
    assert OBSERVED_HEADER_DTYPE_AUDIT["matching_common_key_count"] == 493
    assert OBSERVED_HEADER_DTYPE_AUDIT["backbone_mismatched_key_count"] == 0
    assert sum(OBSERVED_HEADER_DTYPE_AUDIT["mismatch_group_counts"].values()) == 537

    scope_examples = {
        "CATEGORY_PROJECTORS": "action_head.action_decoder.layer1.W",
        "DIFFUSION_ACTION_HEAD_SHARED_TRAINABLE": (
            "action_head.model.transformer_blocks.0.attn1.to_q.weight"
        ),
        "ACTION_HEAD_VLLN_EXPECTED_FROZEN": "action_head.vlln.weight",
    }
    rules = {rule["scope"]: rule for rule in SERIALIZED_DTYPE_CAST_RULES}
    for scope, test_key in scope_examples.items():
        rule = rules[scope]
        result = _serialized_dtype_comparison(
            test_key, rule["base_dtype"], rule["o1_dtype"]
        )

        assert result["stored_dtype_match"] is False
        assert result["dtype_comparison"] == "EXPLICIT_SEMANTIC_CAST_RULE"
        assert result["changed_element_basis"] == "decoded_float64_equality"
        assert result["dtype_cast_rule"] == rule

    rule = rules["CATEGORY_PROJECTORS"]
    test_key = scope_examples["CATEGORY_PROJECTORS"]

    with pytest.raises(ValueError, match="dtype mismatch"):
        _serialized_dtype_comparison(
            "ordinary.key", rule["base_dtype"], rule["o1_dtype"]
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        _serialized_dtype_comparison(
            "backbone.model.weight", rule["base_dtype"], rule["o1_dtype"]
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        _serialized_dtype_comparison(
            "action_head.unlisted_module.weight",
            rule["base_dtype"],
            rule["o1_dtype"],
        )
    with pytest.raises(ValueError, match="unapproved dtype cast"):
        _serialized_dtype_comparison(
            test_key, "torch.float16", rule["o1_dtype"]
        )


def test_stage25_allowed_dtype_cast_compares_decoded_values() -> None:
    torch = pytest.importorskip("torch")
    rule = SERIALIZED_DTYPE_CAST_RULES[0]
    test_key = "action_head.action_decoder.layer1.W"
    base = torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.bfloat16)
    other = base.to(dtype=torch.float32)

    result = _compare_tensor_values(
        base,
        other,
        torch,
        label=test_key,
        key=test_key,
    )

    assert result["dtype_comparison"] == "EXPLICIT_SEMANTIC_CAST_RULE"
    assert result["changed_element_basis"] == "decoded_float64_equality"
    assert result["changed_element_count"] == 0
    assert result["base_tensor_sha256"] != result["other_tensor_sha256"]
    with pytest.raises(ValueError, match="dtype mismatch"):
        _compare_tensor_values(
            base,
            other,
            torch,
            label="ordinary.key",
            key="ordinary.key",
        )


def test_stage25_streamed_dtype_cast_rule_is_cpu_and_shape_strict(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe_open = pytest.importorskip("safetensors").safe_open
    save_file = pytest.importorskip("safetensors.torch").save_file
    rule = SERIALIZED_DTYPE_CAST_RULES[0]
    test_key = "action_head.action_decoder.layer1.W"
    base_path = tmp_path / "base.safetensors"
    o1_path = tmp_path / "o1.safetensors"
    base = torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.bfloat16)
    other = base.to(dtype=torch.float32)
    save_file({test_key: base}, str(base_path))
    save_file({test_key: other}, str(o1_path))

    with safe_open(str(base_path), framework="pt", device="cpu") as base_handle:
        with safe_open(str(o1_path), framework="pt", device="cpu") as o1_handle:
            result = _compare_streamed_key(base_handle, o1_handle, test_key, torch)

    assert result["dtype_comparison"] == "EXPLICIT_SEMANTIC_CAST_RULE"
    assert result["changed_element_count"] == 0
    assert result["base_tensor_sha256"] != result["other_tensor_sha256"]

    bad_path = tmp_path / "bad.safetensors"
    save_file({test_key: torch.zeros((3, 2), dtype=torch.float32)}, str(bad_path))
    with safe_open(str(bad_path), framework="pt", device="cpu") as bad_handle:
        with safe_open(str(o1_path), framework="pt", device="cpu") as o1_handle:
            with pytest.raises(ValueError, match="shape mismatch"):
                _compare_streamed_key(bad_handle, o1_handle, test_key, torch)


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
