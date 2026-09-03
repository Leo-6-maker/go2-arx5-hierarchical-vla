from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "stage23_phase_c_flow_field_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("stage23_phase_c_flow_field_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_phase_c_protocol_is_static_and_exactly_nine_probes() -> None:
    plan = MODULE.protocol_plan()
    MODULE.assert_protocol_static(plan)
    assert plan["status"] == "PREPARED_NOT_EXECUTED"
    assert plan["probe_count"] == 9
    assert plan["execution_boundary"] == {
        "model_construction": False,
        "model_forward": False,
        "model_inference": False,
        "training": False,
        "optimizer_step": False,
        "pbs_submission": False,
        "wbc_execution": False,
        "holdout_access": False,
        "inactive_noise_intervention": False,
    }


def test_official_transformed_time_quantiles_are_frozen() -> None:
    values = [MODULE.official_t_quantile(q) for q in MODULE.T_QUANTILES]
    assert values == pytest.approx([0.06776241796562847, 0.36966943557751086, 0.7837719744658148])
    assert [int(value * 1000.0) for value in values] == [67, 369, 783]


def test_probe_grid_pairs_same_noise_and_time_across_models() -> None:
    probes = MODULE.build_probe_grid()
    assert {(probe["noise_seed"], probe["t_quantile"]) for probe in probes} == {
        (seed, quantile)
        for seed in MODULE.NOISE_SEEDS
        for quantile in MODULE.T_QUANTILES
    }
    assert all(probe["same_noise_across_base_and_o1"] for probe in probes)
    assert all(probe["same_t_across_base_and_o1"] for probe in probes)
    assert all(probe["noise_shape"] == [1, 40, 132] for probe in probes)


def test_mask_partition_keeps_122_dimensions_inactive() -> None:
    mask = MODULE.expected_mask_partition()
    assert mask["active_dimensions"] == 10
    assert mask["inactive_dimensions"] == 122
    assert mask["active_elements_per_probe"] == 400
    assert mask["inactive_elements_per_probe"] == 4880
    assert mask["inactive_dimensions_contribute_to_loss"] is False


def test_c2_is_locked_until_c1_and_sampler_parity() -> None:
    c2 = MODULE.protocol_plan()["c2_sampler_diagnostic"]
    assert c2["status"] == "LOCKED_UNTIL_C1_PASS_AND_SAMPLER_PARITY_PASS"
    assert c2["candidate_steps"] == [4, 8, 16, 32]


def test_velocity_boundary_rejects_wrong_shape_without_model_call() -> None:
    class ShapeOnly:
        shape = (1, 40, 10)

    with pytest.raises(ValueError, match="predicted_velocity must have shape"):
        MODULE.validate_velocity_output(ShapeOnly())
