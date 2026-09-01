import numpy as np
import pytest

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import (
    GateDecision,
    ThresholdProvenance,
    TrajectorySafetyGateConfig,
    TrajectorySafetyGateV1,
)


def make_chunk(positions=None, timestamps=None):
    positions = np.asarray(
        [[0.2, 0.0, 0.4], [0.25, 0.0, 0.4], [0.3, 0.0, 0.4]] if positions is None else positions,
        dtype=np.float64,
    )
    timestamps = np.asarray([0.0, 0.1, 0.2] if timestamps is None else timestamps, dtype=np.float64)
    return TrajectoryChunkV1(
        chunk_id="gate",
        source="test",
        observation_timestamp=float(timestamps[0]),
        point_timestamps=timestamps,
        eef_position=positions,
        eef_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (len(timestamps), 1)),
        source_frame="frame",
        tcp_semantic_id="tcp",
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
    )


def test_gate_accepts_finite_continuous_chunk():
    result = TrajectorySafetyGateV1().evaluate(
        make_chunk(),
        now=0.0,
        achieved_position=[0.2, 0.0, 0.4],
        achieved_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert result.decision is GateDecision.ACCEPT
    assert result.metrics["initial_position_discontinuity_m"] == pytest.approx(0.0)


def test_gate_rejects_duplicate_timestamps_without_retiming():
    result = TrajectorySafetyGateV1().evaluate(make_chunk(timestamps=[0.0, 0.1, 0.1]))
    assert result.decision is GateDecision.REJECT
    assert any(issue.code == "NON_MONOTONIC_TIMESTAMPS" for issue in result.issues)


def test_gate_rejects_initial_jump_and_stale_chunk():
    result = TrajectorySafetyGateV1().evaluate(
        make_chunk(positions=[[1.2, 0.0, 0.4], [1.2, 0.0, 0.4], [1.2, 0.0, 0.4]]),
        now=1.0,
        achieved_position=[0.2, 0.0, 0.4],
        achieved_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert result.decision is GateDecision.REJECT
    codes = {issue.code for issue in result.issues}
    assert "INITIAL_POSITION_DISCONTINUITY_REJECT" in codes
    assert "STALE_CHUNK" in codes


def test_gate_records_workspace_warning_and_rejection_provenance():
    config = TrajectorySafetyGateConfig(
        workspace_warn_min_m=np.array([-1.0, -1.0, -1.0]),
        workspace_warn_max_m=np.array([1.0, 1.0, 1.0]),
        workspace_reject_min_m=np.array([-2.0, -2.0, -2.0]),
        workspace_reject_max_m=np.array([2.0, 2.0, 2.0]),
    )
    result = TrajectorySafetyGateV1(config).evaluate(
        make_chunk(positions=[[3.0, 0.0, 0.4], [3.0, 0.0, 0.4], [3.0, 0.0, 0.4]]),
        now=0.0,
        achieved_position=[3.0, 0.0, 0.4],
        achieved_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert result.decision is GateDecision.REJECT
    assert any(issue.code == "WORKSPACE_WARNING" for issue in result.issues)
    rejection = next(issue for issue in result.issues if issue.code == "WORKSPACE_REJECTION")
    assert rejection.provenance == ThresholdProvenance.TEST_ONLY.value or rejection.provenance == ThresholdProvenance.TEST_ONLY


def test_gate_rejects_non_finite_payload_at_validation_boundary():
    chunk = make_chunk()
    object.__setattr__(chunk, "eef_position", np.array([[np.nan, 0.0, 0.4]] * 3))
    result = TrajectorySafetyGateV1().evaluate(chunk, now=0.0)
    assert result.decision is GateDecision.REJECT
    assert any(issue.code == "NON_FINITE" for issue in result.issues)
