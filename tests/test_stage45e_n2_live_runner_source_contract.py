"""Synthetic CPU tests for the future corrected N2 live-runner source contract."""

import json
from pathlib import Path

import pytest

from scripts import stage45e_n2_live_runner_source_contract as contract


def _manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "corrected_budgets": contract.CORRECT_BUDGETS,
                "forbidden_continuation_phases": [
                    "G0_ADMISSION",
                    "G1_REPLICATION_TRAINING",
                    "G2_TRAIN24_E4",
                ],
            }
        ),
        encoding="utf-8",
    )


def _good_source() -> str:
    return f'''\
LEGACY = "{contract.LEGACY_TREE}"
N2_STATS_SHA = "{contract.N2_STATS_SHA}"
N2_STATS_PAYLOAD_SHA = "{contract.N2_STATS_PAYLOAD_SHA}"
LOGICAL_PREFIX = "checkpoint-2000"
MODE = "continue-g3-g4-g5"

def _prepare_model_conditioning(model, inputs):
    return None

def run_continuation_g3_g4_g5(args):
    phase_local = {{"conditioning_passes": 0}}
    FAILURE_SEAL = "FAILURE_SEAL"
    for window in range(24):
        conditioning = _prepare_model_conditioning(None, window)
        for noise_seed in (2223, 2224, 2225):
            for t in (0.1, 0.2, 0.3):
                _ = (conditioning, noise_seed, t)
    return phase_local, FAILURE_SEAL
'''


def test_corrected_source_contract_passes_synthetic_runner(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    manifest = tmp_path / "manifest.json"
    runner.write_text(_good_source(), encoding="utf-8")
    _manifest(manifest)
    result = contract.audit(runner, manifest)
    assert result["status"] == "PASS_STATIC_SOURCE_CONTRACT"
    assert result["d1_conditioning_reuse"]["conditioning_inside_noise_seed_loop"] is False
    assert result["continuation_driver_present"] is True
    assert result["execution_authorized"] is False


def test_source_contract_rejects_conditioning_inside_noise_seed_loop(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    manifest = tmp_path / "manifest.json"
    source = _good_source().replace(
        "conditioning = _prepare_model_conditioning(None, window)\n        for noise_seed in (2223, 2224, 2225):",
        "for noise_seed in (2223, 2224, 2225):\n            conditioning = _prepare_model_conditioning(None, window)",
    )
    runner.write_text(source, encoding="utf-8")
    _manifest(manifest)
    with pytest.raises(contract.LiveRunnerSourceContractError, match="noise_seed"):
        contract.audit(runner, manifest)


def test_source_contract_rejects_old_d1_budget(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    manifest = tmp_path / "manifest.json"
    runner.write_text(_good_source(), encoding="utf-8")
    bad = dict(contract.CORRECT_BUDGETS)
    bad["G3_D1"] = dict(bad["G3_D1"])
    bad["G3_D1"]["conditioning_passes"] = 288
    manifest.write_text(
        json.dumps(
            {
                "corrected_budgets": bad,
                "forbidden_continuation_phases": ["G1_REPLICATION_TRAINING", "G2_TRAIN24_E4"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(contract.LiveRunnerSourceContractError, match="budget drift"):
        contract.audit(runner, manifest)


def test_source_contract_rejects_missing_continuation_entrypoint(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    manifest = tmp_path / "manifest.json"
    runner.write_text(_good_source().replace("run_continuation_g3_g4_g5", "old_sprint"), encoding="utf-8")
    _manifest(manifest)
    with pytest.raises(contract.LiveRunnerSourceContractError, match="driver function"):
        contract.audit(runner, manifest)
