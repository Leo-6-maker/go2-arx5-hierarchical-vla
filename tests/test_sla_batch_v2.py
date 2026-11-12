import numpy as np
import pytest

from go2arx5_vla.sla_adapter.batch_v2 import accumulation_steps, drop_state, prepared_batches, draw_bank
from go2arx5_vla.sla_adapter.dataset_v1 import CAMERAS


class Dataset:
    def __getitem__(self, index):
        return dict(image={key: np.full((224, 224, 3), index, np.uint8) for key in CAMERAS},
                    image_mask=dict(zip(CAMERAS, [True, True, False])),
                    image_padding_mask={key: np.full((224, 224), i < 2, bool) for i, key in enumerate(CAMERAS)},
                    state=np.full(32, index, np.float32), actions=np.full((40, 32), index, np.float32),
                    prompt='fixed instruction')


def tokenize(prompt, state, *, drop_state):
    return np.full(200, -1 if drop_state else int(state[0]), np.int32), np.ones(200, bool)


@pytest.mark.parametrize('micro', [1, 4, 8, 16, 32])
def test_partition_and_prefetch_preserve_sample_order_and_dropout(micro):
    visits = np.tile(np.arange(16), 4)
    records = []
    for prefetch in (False, True):
        reads = []
        batches = list(prepared_batches(Dataset(), tokenize, visits, 32, micro, prefetch=prefetch,
                                        on_read=reads.append))
        assert reads.count('host_window_read_attempts') == reads.count('host_window_reads') == 32
        states = np.concatenate([row['observation']['state'] for row in batches])
        actions = np.concatenate([row['actions'] for row in batches])
        drops = [value for row in batches for value in row['state_token_dropped']]
        tokens = np.concatenate([row['observation']['tokenized_prompt'] for row in batches])
        np.testing.assert_array_equal(states[:, 0], visits[32:64])
        np.testing.assert_array_equal(actions[:, 0, 0], visits[32:64])
        assert drops == [drop_state(i) for i in range(32, 64)]
        assert all(np.all(row['observation']['image_mask'][CAMERAS[2]] == False) for row in batches)
        records.append(tokens)
    np.testing.assert_array_equal(*records)


@pytest.mark.parametrize('micro', [0, 3, 7, 64, True])
def test_invalid_effective_batch_partition_rejected(micro):
    with pytest.raises(ValueError):
        accumulation_steps(micro)


def test_partial_update_boundary_rejected():
    with pytest.raises(ValueError):
        list(prepared_batches(Dataset(), tokenize, np.arange(64), 1, 4))
    with pytest.raises(ValueError):
        list(prepared_batches(Dataset(), tokenize, np.arange(48), 32, 4))


@pytest.mark.parametrize('micro', [1, 8, 16])
def test_random_input_producer_cannot_change_shape_with_microbatch(micro):
    with pytest.raises(ValueError, match='FIXED_EFFECTIVE32'):
        draw_bank(None, np.arange(micro, dtype=np.uint32))
