import json
import pytest
from go2arx5_vla.world_model import task_learning_features_v1 as features


def test_free_generation_excludes_expert_action_and_rtc_without_mutating_training_input():
    inputs = dict(state=[1, 2], input_ids=[3, 4], pixel_values=[5], action=[999], action_mask=[1])
    inference = features.inference_inputs(inputs)
    assert inference == dict(state=[1, 2], input_ids=[3, 4], pixel_values=[5])
    assert inputs['action'] == [999] and inputs['action_mask'] == [1]


def test_cyclic_sample_is_deterministic_and_balanced_at_optimizer_boundary():
    sequence = [features.sample_index(i, 4, 20261109) for i in range(128)]
    assert sequence == [features.sample_index(i, 4, 20261109) for i in range(64)] + [
        features.sample_index(i, 4, 20261109) for i in range(64, 128)]
    for start in range(0, 128, 32):
        assert [sequence[start:start + 32].count(i) for i in range(4)] == [8] * 4


@pytest.mark.parametrize('changed', [None, 'parameters', 'optimizer', 'scheduler', 'RNG', 'cursor'])
def test_four_vs_split_requires_all_state_not_only_model(tmp_path, changed):
    base = dict(parameters={'tensor': 'same'}, optimizer={'moment': 'same'}, scheduler={'last_epoch': 4}, RNG='same', cursor=128)
    for name in ['continuous', 'resumed', 'split']:
        folder = tmp_path / name
        folder.mkdir()
        value = dict(base)
        if name == 'split':value['cursor'] = 64
        if name == 'resumed' and changed:value[changed] = 'DIFFERENT'
        (folder / 'STATE_IDENTITY.json').write_text(json.dumps(value))
    if changed:
        with pytest.raises(ValueError, match='FOUR_CONTINUOUS'):
            features.compare_four(tmp_path / 'continuous', tmp_path / 'resumed', tmp_path / 'split')
    else:
        assert features.compare_four(tmp_path / 'continuous', tmp_path / 'resumed', tmp_path / 'split')['sampler_cursor'] == 128
