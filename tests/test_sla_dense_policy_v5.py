import numpy as np
import pytest

from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.eval_contract import matrix10


def inputs(tmp_path, raw):
    path = tmp_path / 'ACTION.npz'
    np.savez(path, physical_H40=raw, normalized=np.zeros((1, 40, 32), np.float32))
    actual = np.eye(4)
    return path, dict(source_from_target=np.eye(4), query_state10=pose10(actual, .6),
                      actual_TCP=actual, consumed_reference_TCP=actual,
                      base_hold=np.array([0., 0., .3, 1., 0., 0., 0.]))


def curve():
    raw = np.repeat(pose10(np.eye(4), .6)[None], 40, axis=0)
    # Equal endpoints with a meaningful interior: an endpoint construction
    # would erase this path even though the complete original gate permits it.
    raw[:, 0] = .0005 * np.sin(np.linspace(0, np.pi, 40))**2
    return raw


def test_curved_path_survives_one_coordinate_mapping(tmp_path):
    raw = curve()
    path, kwargs = inputs(tmp_path, raw)
    angle = .4
    frame = np.array([[np.cos(angle), -np.sin(angle), 0., .2],
                      [np.sin(angle), np.cos(angle), 0., -.1],
                      [0., 0., 1., .05], [0., 0., 0., 1.]])
    policy = pose10(frame @ matrix10(raw), raw[:, 9])
    np.savez(path, physical_H40=policy, normalized=np.zeros((1, 40, 32), np.float32))
    kwargs.update(source_from_target=frame, query_state10=pose10(frame, .6))
    program, audit = compile_proposal(path, digest(path), **kwargs)
    np.testing.assert_allclose(program.decoded_H40, raw, rtol=0, atol=1e-15)
    assert program.decoded_H40[20, 0] > .0004
    assert audit['policy_frame_mapping_count'] == 1
    assert not audit['endpoint_basis_applied']
    assert not audit['SLA_raw_H40_consumed']


def test_unexecuted_suffix_still_has_original_full_preview_gate(tmp_path):
    raw = curve()
    raw[30, 0] = .1
    path, kwargs = inputs(tmp_path, raw)
    with pytest.raises(StudentProposalRejected, match='FULL_PREVIEW_RATE_AND_ACCELERATION'):
        compile_proposal(path, digest(path), **kwargs)
    with np.load(path) as packet:
        assert packet['physical_H40'][30, 0] == .1


def test_raw_grip_is_not_clipped_or_decoded_again(tmp_path):
    raw = curve()
    raw[-1, 9] = 1.01
    path, kwargs = inputs(tmp_path, raw)
    with pytest.raises(StudentProposalRejected, match='FINITE_FULL40_WITHOUT_CLIPPING'):
        compile_proposal(path, digest(path), **kwargs)
