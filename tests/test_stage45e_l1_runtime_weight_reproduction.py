"""CPU/static tests for the one-shot runtime coefficient reproduction canary."""

from __future__ import annotations

from pathlib import Path

from scripts import stage45e_l1_runtime_weight_reproduction as reproduction


ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_plan_is_single_no_update_forward() -> None:
    plan = reproduction.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["one_and_only_one_canary"] is True
    assert plan["model_construction"] == 0
    assert plan["model_forward"] == 0
    assert plan["action_head_forwards"] == 0
    assert plan["backward"] == 0
    assert plan["optimizer_steps"] == 0
    assert plan["checkpoint_writes"] == 0
    assert plan["training"] == 0
    assert plan["automatic_retry"] is False


def test_reproduction_source_has_no_update_or_training_surface() -> None:
    source = (ROOT / "scripts/stage45e_l1_runtime_weight_reproduction.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source
    assert "gr00t_run" in source  # the static guard must explicitly reject it if called
    assert "capture_effective_weight_evidence" in source
    assert "REPRODUCED_RUNTIME_COEFFICIENT_EVIDENCE" in source


def test_reproduction_pbs_is_absolute_and_single_shot() -> None:
    source = (ROOT / "hpc/pbs/stage45e_l1_runtime_weight_reproduction.pbs").read_text(encoding="utf-8")
    assert "ngpus=1" in source
    assert "unset PYTHONPATH" in source
    assert "stage45e_l1_runtime_weight_reproduction.py" in source
    assert "--expected-head" in source
    assert "--output-root" in source
    assert "qsub" not in source
    assert "backward" in source.lower()
