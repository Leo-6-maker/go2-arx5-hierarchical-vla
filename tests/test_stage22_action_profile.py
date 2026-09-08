from __future__ import annotations

import numpy as np
import pytest

from go2arx5_vla.metrics import chunk_overlap_metrics, chunk_profile
from go2arx5_vla.simulation import build_same_scene_expert_chunk


def _chunk(chunk_id: str, observation: float, offset: float = 0.0):
    return build_same_scene_expert_chunk(
        chunk_id=chunk_id,
        observation_timestamp=observation,
        current_position=[offset, 0.0, 0.5],
        target_position=[offset + 0.1, 0.02, 0.55],
        current_quaternion=[0.0, 0.0, 0.0, 1.0],
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
    )


def test_profile_reports_speed_and_roughness_without_mutating_chunk():
    chunk = _chunk("c0", 0.0)
    before = chunk.eef_position.copy()
    profile = chunk_profile(chunk)
    assert profile["point_count"] == 40
    assert profile["duration_s"] == pytest.approx(1.95)
    assert profile["linear_speed_mps"]["max"] is not None
    assert np.array_equal(chunk.eef_position, before)


def test_overlap_reports_common_interval_and_tangent_difference():
    previous = _chunk("previous", 0.0)
    current = _chunk("current", 0.2, offset=0.01)
    result = chunk_overlap_metrics(previous, current)
    assert result["overlap_available"] is True
    assert result["common_interval_s"][0] == pytest.approx(0.2)
    assert result["old_prediction_at_new_observation_vs_new_first"]["position_error_m"] >= 0.0
    assert result["translational_tangent_error_mps"]["count"] > 0
    assert result["no_trajectory_correction"] is True


def test_overlap_rejects_outside_explicit_grid():
    with pytest.raises(ValueError, match="inside the common interval"):
        chunk_overlap_metrics(
            _chunk("previous", 0.0),
            _chunk("current", 0.2),
            overlap_sample_timestamps=[0.0, 0.3],
        )


def test_stage22_recipe_flags_are_the_declared_attribution_control():
    from scripts.stage22_train_recipes import _recipe_config

    r0 = _recipe_config("r0")
    r1 = _recipe_config("r1")
    assert r0["tune_projector"] is True
    assert r0["tune_diffusion_model"] is False
    assert r1["tune_projector"] is True
    assert r1["tune_diffusion_model"] is True
    assert r1["tune_llm"] is False
    assert r1["tune_visual"] is False
