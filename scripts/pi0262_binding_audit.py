"""Read actual model, processor/decoder/gate sources and noise bytes; no model load."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pi0260_r3_eval as r3
import pi0260_r3_inputs as inputs
import pi0260_r2_inputs as original_inputs
import pi0242_train_infer_bridge as bridge
import stage21_same_scene_groot_inference as native
from go2arx5_vla.safety import TrajectorySafetyGateV1
from pi0262_old_reorg import ROOT, OLD

VIEWS = {
    'Donor': r3.R3 / 'models/Base6000/arms/PI0260/R3/Base/checkpoint-6000',
    'C2000': Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0261/models/C2000/arms/PI0261/C/checkpoint-2000'),
    'P2000': Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0261/models/P2000/arms/PI0261/P/checkpoint-2000'),
}


def run():
    models = {}
    for point, view in VIEWS.items():
        binding_path = r3.R3 / 'model_bindings/Base_6000.json' if point == 'Donor' else OLD / 'model_bindings' / (point + '.json')
        binding = r3.read(binding_path)
        assert binding['view_type'] == 'model-only'
        for name, expected in binding['model_files_sha256'].items():
            assert r3.sha(view / name) == expected, (point, name)
        config = r3.read(view / 'config.json')
        assert (config['num_inference_timesteps'], config['action_horizon'], config['dtype']) == (4, 40, 'bfloat16')
        models[point] = dict(binding=binding, binding_sha256=r3.sha(binding_path), view=str(view))
    noise = r3.read(r3.R3 / 'noise_manifest.json')
    assert noise['status'] == 'R3_GLOBAL_B0_B1_FROZEN'
    for row in noise['rows']:
        path = r3.R3 / 'noise' / (row['name'] + '.npz')
        assert r3.sha(path) == row['file_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert z['float32'].shape == (1, 40, 132)
            assert hashlib.sha256(z['float32'].tobytes()).hexdigest() == row['input_float32_sha256']
            assert hashlib.sha256(z['bfloat16_bits'].tobytes()).hexdigest() == row['cast_bfloat16_bits_sha256']
    code = {}
    for name, obj in (('processor_boundary', inputs), ('processor', original_inputs),
                      ('native_query_and_decode_bridge', bridge), ('native_s4_entry', native),
                      ('original_gate', TrajectorySafetyGateV1)):
        path = Path(inspect.getsourcefile(obj))
        code[name] = dict(path=str(path), sha256=r3.sha(path))
    statistics = r3.ROOT / 'dataset/statistics.json'
    assert r3.sha(statistics) == r3.read(ROOT / 'CACHE_SOURCE_INDEX.json')['statistics_sha256']
    result = dict(status='PASS_CURRENT_MODEL_PROCESSOR_DECODER_GATE_NOISE_BINDINGS',
        models=models, code=code, noise=noise, statistics_sha256=r3.sha(statistics),
        model_loads=0, model_queries=0, optimizer_updates=0, selected_checkpoint='NONE',
        deployment_selected_checkpoint='NONE', full_protocol_binding='PENDING_PROTOCOL_LOCATION')
    r3.write(ROOT / 'SOURCE_BINDING_AUDIT.json', result)
    print(json.dumps(dict(status=result['status'], model_points=list(models),
                         files_verified=sum(len(row['binding']['model_files_sha256']) for row in models.values()))))


if __name__ == '__main__':
    run()
