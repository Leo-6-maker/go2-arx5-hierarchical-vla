"""Freeze actual collection packets and exact TRAIN-only component reuse."""
import hashlib
import json
import os
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np
import torch
from admission import BASE, ROOT, OLD, RAW as OLD_RAW, read, sha, write, RELEASE
from plan import collection_cases, noise_seed, self_check

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def main():
    assert read(ROOT / 'evidence/A0_PREDECESSOR_CLOSEOUT.json')['status'] == 'PASS_CONTINUE_PI0270_A1'
    assert not (ROOT / 'evidence/model_attempts.jsonl').exists()
    torch.set_num_threads(4)
    check = self_check()
    check.update(admission='A0_ACTUALLY_CLOSED', source_sha256=sha(ROOT / 'code/plan.py'))
    write(ROOT / 'evidence/COLLECTION_PLAN_CPU_CHECK.json', check)
    source = BASE / 'PI0267/design/SCENE_CARD.json'
    scene = read(source)
    cases = collection_cases(scene)
    for relative in ('design/PAIRING_PLAN.json', 'contracts/ROT/source/dataset/statistics.json'):
        old = OLD / relative; new = ROOT / relative
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists(): assert new.read_bytes() == old.read_bytes()
        else: new.write_bytes(old.read_bytes())
    assert sha(ROOT / 'design/PAIRING_PLAN.json') == 'fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef'
    noise = []
    for case in cases:
        path = RAW / 'noise' / (case['case'] + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {}; keys = []
        for k in range(100):
            namespace = f"PI0270:COLLECT:27101:{case['family']}:{case['task']}:{case['command']}:{case['backend']}:{k}"
            seed = noise_seed(namespace)
            tensor = torch.randn((1, 40, 132), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32)
            array = tensor.numpy(); name = f'q{k:03d}'; values[name] = array
            keys.append(dict(name=name, namespace=namespace, seed=seed,
                             noise_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                             actual_bf16_initial_float32_sha256=hashlib.sha256(tensor.to(torch.bfloat16).float().numpy().tobytes()).hexdigest()))
        if path.exists():
            with np.load(path) as prior:
                assert set(prior.files) == set(values)
                assert all(np.array_equal(prior[k], v) for k, v in values.items())
        else:
            tmp = path.with_suffix('.partial.npz')
            np.savez_compressed(tmp, **values)
            with tmp.open('rb') as f: os.fsync(f.fileno())
            os.replace(tmp, path)
        noise.append(dict(file=str(path), sha256=sha(path), keys=keys, shape=[1,40,132], dtype='numpy.float32'))
        print('FROZEN_PACKET', case['case'], flush=True)
    component_card = read(OLD / 'design/COMPONENT_CARD.json')
    families = [f['family_id'] for f in scene['families'] if f['split'] == 'FEEDBACK_TRAIN'][:2]
    reuse = []
    assert len(component_card['rows']) == 32
    for row in component_card['rows']:
        assert row['family'] in families
        folder = OLD_RAW / 'component/PREFIX4096' / row['case']
        result = read(folder / 'result.json')
        assert result['validity'] in ('VALID', 'PREQUERY_EXCLUDED')
        assert result['family'] == row['family'] and result['component_start'] == row
        if result['validity'] == 'VALID': assert sha(folder / 'trajectory.npz') == result['arrays_sha256']
        observations = []
        for query in result['queries']:
            k = query['replan']; path = folder / 'queries' / f'{k:03d}'
            assert 0 <= k < 20 and sha(path / 'prediction.npz') == query['prediction_sha256']
            observations.append(dict(query_index=k, input=str(path / 'observation.npz'),
                                     input_sha256=sha(path / 'observation.npz'), prediction=str(path / 'prediction.npz'),
                                     prediction_sha256=query['prediction_sha256']))
        reuse.append(dict(source_lane='component', source_episode_key=row['case'], family=row['family'],
                          task=row['task'], command=row['command'], backend=row['backend'], start_type=row['start_type'],
                          initialization=row['initialization'], result_path=str(folder / 'result.json'),
                          result_sha256=sha(folder / 'result.json'), validity=result['validity'], observations=observations,
                          proposed_query_indices=list(range(20))))
    write(ROOT / 'design/COLLECTION_CARD.json', dict(schema='PI0270_FIXED_TRAIN_COLLECTOR_V1', release=RELEASE,
          cases=cases, source_scene_sha256=sha(source), source_scene=str(source), collector='PREFIX4096',
          collector_binding_sha256=sha(ROOT / 'model_bindings/PREFIX4096.json'), contracts={'ROT':sha(ROOT / 'contracts/ROT/source/dataset/statistics.json')},
          noise=noise, reused_components=reuse, original_proposals=6144, component_proposals=640,
          torch_packet_version=torch.__version__, task_or_phase_in_policy=False, teacher_used_for_control=False,
          first_valid_failures_retained=True, quality_stop_on_success=False, framepairs_hard_cap=28672,
          starts_hard_cap=304, model_attempts_hard_cap=30720, model_heads_hard_cap=122880,
          selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
    print('PASS_FROZEN_192_CASES_19200_ACTUAL_PACKETS_32_COMPONENT_REUSES', flush=True)


if __name__ == '__main__':
    main()
