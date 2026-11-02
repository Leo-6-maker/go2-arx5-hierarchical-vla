import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.durable import read, write
from scripts.pi0294.multitask.book_gpu_capacity import GIB, INFERENCE_REQUEST
from scripts.pi0294.multitask.book_phase_diagnostic import KIND, run


@pytest.mark.parametrize('qualified', [False, True])
def test_full_saved_train32_keeps_capacity_contract_separate_from_model_payload(tmp_path, monkeypatch, qualified):
    """Exercise the first and every later generation through the actual loop."""
    import sys
    from scripts.pi0294 import native_dataset
    from scripts.pi0294.e1 import model

    categories = ('grasp_establishment', 'holding_carry', 'legal_release', 'actual_early_recovery')
    windows = [dict(id=f'TRAIN{i:03}', anchor=0, split='TRAIN', category=categories[i//8]) for i in range(32)]
    plan = dict(schema='BOOK_PHASE_MATCHED_TRAIN32_DIAGNOSTIC_V1', noise_replicates=[0, 1, 2],
        phase_metadata_student_input=False, labels_changed=False, physics_integrations=0,
        optimizer_updates=0, windows=windows)
    manifest = dict(kind=KIND, limits=dict(model_loads=1, optimizer_updates=0,
        physics_integrations=0, generation_queries=96, RGB_pairs=0, source_constructors=0))
    if qualified:
        manifest['GPU_inference_qualification'] = deepcopy(INFERENCE_REQUEST)
    write(tmp_path/'RUN_MANIFEST.json', manifest)
    write(tmp_path/'TRAIN32_DIAGNOSTIC_PLAN.json', plan)
    state = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, .5])
    expert = np.tile(state, (40, 1))

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.statistics_sha = 'SAME_STATS'
            self.rows = [(row, 0) for row in windows]

        def initialize(self):
            pass

        def raw(self, index):
            return state.copy(), expert.copy(), [], 'Put book away'

    cuda = SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (2*GIB, 16*GIB),
        memory_allocated=lambda: GIB, memory_reserved=lambda: GIB,
        max_memory_allocated=lambda: GIB, max_memory_reserved=lambda: GIB,
        reset_peak_memory_stats=lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(native_dataset, 'NativeDataset', Dataset)
    monkeypatch.setattr(model, 'payload', lambda *args: dict(images=[dict(sha256='RGB0'), dict(sha256='RGB1')]))
    generated = []
    head = SimpleNamespace(num_inference_timesteps=7)

    def generate(payload, seed):
        assert set(payload) == {'images'}
        assert head.num_inference_timesteps == 4
        generated.append(seed)
        return dict(physical_H40=expert.copy(), normalized=expert.copy(), noise=dict(seed=seed))

    instance = SimpleNamespace(bindings=dict(diagnostic_dataset='unused', source='unused', cosmos='unused'),
        identity=dict(statistics_sha256='SAME_STATS'), model=SimpleNamespace(action_head=head), generate=generate)
    result = run(tmp_path, instance, lambda: None)
    assert result['generation_queries'] == len(generated) == 96
    assert result['physics_integrations'] == result['optimizer_updates'] == 0
    assert head.num_inference_timesteps == 7
    assert read(tmp_path/'DIAGNOSTIC_PROGRESS.json')['completed_generations'] == 96
    capacity = tmp_path/'GPU_INFERENCE_CAPACITY.json'
    assert capacity.exists() is qualified
    if qualified:
        measured = read(capacity)
        assert measured['capacity_pass'] and measured['completed_generations'] == 96
        assert len(measured['measurements']) == 97
        assert measured['minimum_free_bytes'] == 2*GIB
