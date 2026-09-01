import numpy as np

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.runtime import (
    ChunkLifecycleStatus,
    ChunkLifecycleTracker,
)


def chunk(chunk_id="c0", observation=0.0):
    timestamps = np.array([observation, observation + 0.1, observation + 0.2])
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source="test",
        observation_timestamp=observation,
        point_timestamps=timestamps,
        eef_position=np.array([[0.2, 0.0, 0.4], [0.21, 0.0, 0.4], [0.22, 0.0, 0.4]]),
        eef_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        source_frame="frame",
        tcp_semantic_id="tcp",
        validity_start=observation,
        validity_end=observation + 0.2,
    )


def test_chunk_lifecycle_fresh_aging_stale_and_out_of_order():
    tracker = ChunkLifecycleTracker()
    assert tracker.submit(chunk(), receive_timestamp=0.1).status is ChunkLifecycleStatus.FRESH
    assert tracker.sample(execute_timestamp=0.2).status is ChunkLifecycleStatus.FRESH
    aging = tracker.sample(execute_timestamp=0.4)
    assert aging.status is ChunkLifecycleStatus.AGING
    stale = tracker.sample(execute_timestamp=0.9)
    assert stale.status is ChunkLifecycleStatus.STALE
    older = tracker.submit(chunk("older", observation=-1.0), receive_timestamp=0.11)
    assert older.status is ChunkLifecycleStatus.OUT_OF_ORDER


def test_chunk_lifecycle_duplicate_is_not_a_new_chunk():
    tracker = ChunkLifecycleTracker()
    first = chunk("same")
    tracker.submit(first, receive_timestamp=0.01)
    duplicate = tracker.submit(first, receive_timestamp=0.02)
    assert duplicate.status is ChunkLifecycleStatus.OUT_OF_ORDER


def test_chunk_lifecycle_rejects_receive_or_execute_before_observation():
    tracker = ChunkLifecycleTracker()
    invalid_receive = tracker.submit(chunk(observation=1.0), receive_timestamp=0.9)
    assert invalid_receive.status is ChunkLifecycleStatus.INVALID
    assert tracker.latest_chunk is None
    tracker.submit(chunk(observation=1.0), receive_timestamp=1.1)
    invalid_execute = tracker.sample(execute_timestamp=0.9)
    assert invalid_execute.status is ChunkLifecycleStatus.INVALID
