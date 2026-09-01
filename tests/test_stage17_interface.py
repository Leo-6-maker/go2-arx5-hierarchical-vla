import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1
from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def test_provider_gate_identity_binding_adapter_and_parallel_gripper():
    action = np.zeros((40, 10), dtype=np.float64)
    action[:, 0] = np.linspace(0.2, 0.3, 40)
    action[:, 1:3] = [0.0, 0.4]
    action[:, 3:9] = rotation_matrix_to_rot6d_rows(np.repeat(np.eye(3)[None], 40, axis=0))
    action[:, 9] = 0.5
    provider = GR00TTrajectoryProvider(
        action,
        config=GR00TProviderConfig(
            source_frame="frame",
            tcp_semantic_id="tcp",
            action_sample_period_s=0.05,
            timestamp_provenance="TEST_ONLY",
        ),
        observation_timestamp=10.0,
    )
    chunk = provider.next_chunk(10.0)
    assert chunk is not None
    gate = TrajectorySafetyGateV1().evaluate(
        chunk,
        now=10.0,
        achieved_position=[0.2, 0.0, 0.4],
        achieved_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert gate.decision is GateDecision.ACCEPT
    binding = EpisodeReferenceBindingV1.from_external_chunk(chunk, episode_id="episode")
    bound = binding.bind_external_chunk(chunk)
    preview, gripper = GR00TToWBCAdapter(
        ReferenceAdapterConfig(
            expected_source_frame="frame",
            wbc_frame="frame",
            tcp_semantic_id="tcp",
        )
    ).adapt(bound, controller_timestamp=10.0)
    assert preview.xyz_rot6d.shape == (9, 9)
    assert np.isfinite(preview.xyz_rot6d).all()
    assert gripper is not None
    assert gripper.normalized_command == 0.5
