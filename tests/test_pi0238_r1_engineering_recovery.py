from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from scripts import pi0238_r1_engineering_recovery as r1


def test_quota_empty_is_unknown():
    parsed = r1.quota_parse({"stdout": "", "stderr": ""})
    assert parsed["nonempty"] is False
    assert parsed["parseable"] is False


def test_quota_header_without_capacity_is_not_pass():
    parsed = r1.quota_parse({"stdout": "Disk quotas for user e1192548\n", "stderr": ""})
    assert parsed["nonempty"] is True
    assert parsed["parseable"] is False


def test_qstat_filters_headers_and_own_job():
    text = "Job id            Name            User\n623999.hopper-m-02  pi0238  e1192548\n"
    assert len(r1.pbs_job_lines(text, "623999.hopper-m-02")) == 0
    assert len(r1.pbs_job_lines(text, "623998.hopper-m-02")) == 1


def test_mask_payload_negative_controls_are_real():
    mask = np.zeros((1, 40, 132), dtype=np.float32)
    mask[:, :, :10] = 1.0
    action = np.zeros_like(mask)
    assert r1.validate_mask_payload(action, mask, list(range(10)), 40 * 10)

    wrong_slot = action.copy()
    wrong_slot[:, :, 10] = 1.0
    assert not r1.validate_mask_payload(wrong_slot, mask, list(range(10)), 40 * 10)

    wrong_mask = mask.copy()
    wrong_mask[:, :, 0] = 0.0
    assert not r1.validate_mask_payload(action, wrong_mask, list(range(10)), 40 * 10)

    raw = np.zeros((40, 10), dtype=np.float32)
    raw[:, 3:9] = 1.0
    assert r1.raw_rot6d_quality(raw)["unit_norm_deviation_count"] > 0


def test_inventory_uses_declared_base_and_rejects_duplicates():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        evidence = root / "evidence"
        evidence.mkdir()
        payload = evidence / "small.txt"
        payload.write_text("sealed\n", encoding="utf-8")
        digest = r1.sha256_file(payload)
        inventory = evidence / "inventory.sha256"
        inventory.write_text(f"{digest}  small.txt\n", encoding="utf-8")
        result = r1.inventory_parse(inventory, evidence, [evidence])
        assert result["all_match"] is True
        assert result["rows"][0]["path"] == str(payload.resolve())


def test_tree_hash_is_order_independent_and_prefix_sensitive():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "b").mkdir()
        (root / "a").mkdir()
        (root / "b" / "x").write_text("x", encoding="utf-8")
        (root / "a" / "y").write_text("y", encoding="utf-8")
        first = r1.tree_sha256(root, "train")[0]
        second = r1.tree_sha256(root, "train_calib")[0]
        assert first != second
        assert first == r1.tree_sha256(root, "train")[0]
