from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import stage45e_n2_support as support  # noqa: E402


def _synthetic_full() -> dict[str, object]:
    lower = np.zeros(support.ACTION_DIM, dtype=np.float64)
    upper = np.zeros(support.ACTION_DIM, dtype=np.float64)
    lower[:3] = [0.0, 1.0, 2.0]
    upper[:3] = [2.0 * support.COMMON_HALF_RANGE_M, 1.1, 2.1]
    lower[3:] = np.arange(support.ACTION_DIM - 3, dtype=np.float64) * 0.01
    upper[3:] = lower[3:] + 0.2
    active_mask = np.zeros((support.ACTION_HORIZON, support.ACTION_DIM), dtype=np.float64)
    active_mask.reshape(-1)[: support.ACTIVE_SCALARS] = 1.0
    return {"lower": lower, "upper": upper, "active_mask": active_mask}


def _n2_payload(full: dict[str, object], science_head: str = "1" * 40) -> dict[str, object]:
    n2 = support.derive_n2_bounds(full)
    execution = dict(support.FORBIDDEN_EXECUTION_FIELDS)
    execution.update(
        {
            "model_execution": False,
            "training": False,
            "validation_e_read": False,
            "holdout_f_read": False,
        }
    )
    return {
        "schema": "CORELAB_STAGE45E_N2_BALANCED_XYZ_STATISTICS_V1",
        "status": "READY_FOR_TRAINING_NOT_EXECUTED",
        "pi_gate": support.PI_GATE,
        "science_commit": science_head,
        "source_full24_statistics": {
            "sha256": support.FULL_STATS_SHA256,
            "statistics_payload_sha256": support.FULL_STATS_PAYLOAD_SHA256,
        },
        "n2_contract": {
            "common_half_range_m": support.COMMON_HALF_RANGE_M,
            "xyz_centers_m": n2["centers"].tolist(),
            "selected_lower": n2["lower"].tolist(),
            "selected_upper": n2["upper"].tolist(),
            "active_mask": n2["active_mask"].tolist(),
            "normalization": "MINMAX_TO_NEG1_POS1",
            "selected_lower_bound": "q01",
            "selected_upper_bound": "q99",
            "timing_mode": "T16",
            "active_scalar_count": support.ACTIVE_SCALARS,
            "mask_binary": True,
            "loss_weighting": False,
            "n1_historical_bounds": False,
            "l1_physical_loss": False,
            "non_xyz_bounds_unchanged": True,
            "centers_source": "accepted FULL24 q01/q99 midpoint per XYZ axis",
            "dataset_unchanged": True,
            "label_construction_unchanged": True,
            "decoder_unchanged": True,
            "target_xyz_in_model_input": False,
        },
        "dataset_identity": dict(support.DATASET_IDENTITY),
        "clipping_audit": support.expected_n2_clipping_audit(),
        "execution_boundary": execution,
    }


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    support.write_json(path, payload)


def test_static_templates_and_plan() -> None:
    template_result = support.validate_static_templates(ROOT)
    assert template_result["status"] == "PASS_STATIC_TEMPLATES"
    plan = support.build_static_plan()
    assert plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert plan["corelab_commit_audit"] == "PENDING_NONEMPTY_CORELAB_N2_COMMIT"
    assert plan["execution_boundary"] == {
        "h200": 0,
        "pbs": 0,
        "model_execution": 0,
        "training": 0,
        "evaluation": 0,
        "validation_e_read": 0,
        "holdout_f_read": 0,
    }


@pytest.mark.parametrize(
    "script_name,token",
    [
        (
            "stage45e_n2_evaluation_surface.py",
            "STAGE45E_N2_VANDA_EVAL_SURFACE_READY_FOR_PI_GATE",
        ),
        (
            "stage45e_vanda_h200_n2_fallback.py",
            "STAGE45E_N2_VANDA_H200_2000_TRAINING_COMPLETE_FOR_PI_REVIEW",
        ),
    ],
)
def test_entrypoints_resolve_repo_from_foreign_cwd_with_empty_pythonpath(
    tmp_path: Path, script_name: str, token: str
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), "--plan-only"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["return_token"] == token
    assert result["repo_root_resolution"] == str(ROOT)
    assert result["entrypoint"] == str(SCRIPTS / script_name)
    assert result["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert result["execution_boundary"]["model_execution"] == 0


def test_n2_derivation_preserves_centers_non_xyz_and_mask() -> None:
    full = _synthetic_full()
    n2 = support.derive_n2_bounds(full)
    np.testing.assert_array_equal(n2["centers"], [support.COMMON_HALF_RANGE_M, 1.05, 2.05])
    assert n2["common_half_range"] == support.COMMON_HALF_RANGE_M
    np.testing.assert_array_equal(n2["lower"][3:], full["lower"][3:])
    np.testing.assert_array_equal(n2["upper"][3:], full["upper"][3:])
    np.testing.assert_array_equal(n2["active_mask"], full["active_mask"])


def test_n2_artifact_validation_is_exact_and_model_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = _synthetic_full()
    monkeypatch.setattr(support, "load_accepted_full_stats", lambda path: full)
    artifact = tmp_path / "n2.json"
    payload = _n2_payload(full)
    _write_payload(artifact, payload)

    result = support.validate_n2_statistics_artifact(
        artifact,
        tmp_path / "accepted-full.json",
        expected_science_head="1" * 40,
    )
    assert result["status"] == "PASS_STATIC_N2_STATS"
    assert result["science_commit"] == "1" * 40
    assert result["active_scalar_count"] == support.ACTIVE_SCALARS
    assert support.model_imports_present() == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["n2_contract"].update({"common_half_range_m": 0.107349472}),
        lambda p: p["n2_contract"]["selected_lower"].__setitem__(3, "0.0"),
        lambda p: p["n2_contract"]["active_mask"][0].__setitem__(0, True),
        lambda p: p["n2_contract"].update({"non_xyz_bounds_unchanged": False}),
        lambda p: p["n2_contract"].update({"loss_weighting": True}),
        lambda p: p["dataset_identity"].update({"window_count": 23}),
        lambda p: p["execution_boundary"].update({"model_forward": 1}),
    ],
)
def test_n2_artifact_rejects_contract_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    full = _synthetic_full()
    monkeypatch.setattr(support, "load_accepted_full_stats", lambda path: full)
    payload = _n2_payload(full)
    mutation(payload)
    artifact = tmp_path / "mutated.json"
    _write_payload(artifact, payload)
    with pytest.raises(support.N2ContractError):
        support.validate_n2_statistics_artifact(
            artifact,
            tmp_path / "accepted-full.json",
            expected_science_head="1" * 40,
        )


def test_fallback_requires_exact_corelab_blocker_token(tmp_path: Path) -> None:
    token_path = tmp_path / "blocker.token"
    token_path.write_text("wrong-token\n", encoding="utf-8")
    with pytest.raises(support.N2ContractError, match="token"):
        support.validate_h200_fallback_preflight(
            ROOT,
            "1" * 40,
            tmp_path / "n2.json",
            token_path,
            tmp_path / "accepted-full.json",
            runtime_profile={"device": "cuda:0", "offline_mode": True, "single_gpu": True},
        )


def test_corelab_audit_requires_a_non_base_commit() -> None:
    with pytest.raises(support.N2ContractError, match="non-empty"):
        support.audit_corelab_science_commit(ROOT, support.BASE_HEAD)


def test_new_entrypoints_have_no_forbidden_top_level_model_import() -> None:
    for script in (
        SCRIPTS / "stage45e_n2_support.py",
        SCRIPTS / "stage45e_n2_evaluation_surface.py",
        SCRIPTS / "stage45e_vanda_h200_n2_fallback.py",
    ):
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not [
            module
            for module in imports
            if any(module == prefix or module.startswith(prefix + ".") for prefix in support.FORBIDDEN_IMPORT_PREFIXES)
        ]


def test_model_import_guard_rejects_forbidden_modules() -> None:
    assert support.model_imports_present({"torch": object(), "torch.nn": object(), "numpy": object()}) == [
        "torch",
        "torch.nn",
    ]
