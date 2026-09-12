"""CPU/static tests for the recovered-primary eval-only launcher."""

from __future__ import annotations

from pathlib import Path

from scripts import stage45e_l1_recovered_eval_launcher as launcher


ROOT = Path(__file__).resolve().parents[1]


def test_recovered_launcher_plan_is_eval_only() -> None:
    plan = launcher.plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["recovered_input_v2_required"] is True
    assert plan["training_child"] is False
    assert plan["automatic_retry"] is False
    assert plan["call_budget"]["total_model_side_calls"] == 360


def test_recovered_launcher_has_no_training_surface() -> None:
    source = (ROOT / "scripts/stage45e_l1_recovered_eval_launcher.py").read_text(encoding="utf-8")
    assert "stage45e_l1_h200_train.py" not in source
    assert "--recovered-eval-input" in source
    assert '"PYTHONPATH": ""' in source
    assert "cwd=str(Path(os.getenv" in source
    assert "automatic_retry" in source and "False" in source


def test_eval_only_pbs_uses_absolute_recovered_launcher() -> None:
    source = (ROOT / "hpc/pbs/stage45e_l1_recovered_eval.pbs").read_text(encoding="utf-8")
    assert "#PBS -l select=1:ngpus=1" in source
    assert "unset PYTHONPATH" in source
    assert "stage45e_l1_recovered_eval_launcher.py" in source
    assert "--recovered-eval-input" in source
    assert "qsub" not in source
    assert "training" in source.lower()
