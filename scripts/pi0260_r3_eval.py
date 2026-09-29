"""First-valid fixed PI0260-R3 offline inference on Corelab; no training."""
import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
R3 = ROOT / 'R3'
CODE = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
COSMOS = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1/cosmos_snapshot')
GROOT = Path('/home/liuyu-nfs/Projects/isaac-gr00t')
sys.path[:0] = [str(R3), str(CODE / 'scripts'), str(CODE / 'src'), str(GROOT)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), path
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def card():
    result = read(R3 / 'S4_FIXED_EVAL_CARD.json')
    assert result['schema'] == 'PI0260_R3_FIXED_1760_QUALITY_CARD_V1'
    assert result['planned_logical_tuples'] == len(result['rows']) == 1760
    for name, expected in result['source_sha256'].items():
        assert sha(R3 / name) == expected, name
    return result


@lru_cache(maxsize=160)
def sequence(name):
    path = ROOT / 'dataset' / name
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
        return read(path.with_suffix('.json')), z['state_v1'].copy(), z['raw_action'].copy()


def source(row, motion, original):
    if row['cohort'] == 'Motion2':
        case = motion[row['case_index']]
        rec, states, labels = sequence(case['sequence'])
        position = next(i for i, item in enumerate(rec['rows']) if item['index'] == case['index'])
        assert rec['rows'][position]['frame_sha256'] == case['frame_sha256']
        frame = ROOT / case['frame']
        assert sha(frame) == case['frame_sha256']
        with np.load(frame, allow_pickle=False) as z:
            packet = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(),
                          state_v1=states[position], language=case['language'])
        return packet, labels[position], case
    case = original[row['case_index']]
    path = Path(case['source_path'])
    assert sha(path) == case['source_sha256']
    with np.load(path, allow_pickle=False) as z:
        packet = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(),
                      state_v1=z['state'].copy(), language=case['original_language'])
        label = z['expert'].copy()
    return packet, label, case


def preflight():
    plan = card()
    motion = read(R3 / 'motion2_eval_cases.json')['cases']
    original = read(R3 / 'O64_eval_cases.json')['rows']
    assert len(motion) == 156 and len(original) == 64
    noise_rows = read(R3 / 'noise_manifest.json')['rows']
    for item in noise_rows:
        path = R3 / 'noise' / (item['name'] + '.npz')
        assert sha(path) == item['file_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert z['float32'].shape == (1, 40, 132)
            assert hashlib.sha256(z['float32'].tobytes()).hexdigest() == item['input_float32_sha256']
            assert hashlib.sha256(z['bfloat16_bits'].tobytes()).hexdigest() == item['cast_bfloat16_bits_sha256']
    assert {r['cohort'] for r in plan['rows']} == {'Motion2', 'O64'}
    for cohort in ('Motion2', 'O64'):
        row = next(r for r in plan['rows'] if r['cohort'] == cohort)
        packet, expert, _ = source(row, motion, original)
        assert packet['body_rgb'].shape == packet['wrist_rgb'].shape == (96, 96, 3)
        assert packet['state_v1'].shape == (10,) and expert.shape == (40, 10)
    print(json.dumps(dict(status='R3_EVAL_SOURCE_AND_NOISE_PREFLIGHT_PASS', rows=len(plan['rows']))))


def run(point, model_view, binding_path):
    import fcntl
    import torch
    from transformers import AutoModel
    from gr00t.data.embodiment_tags import EmbodimentTag
    from pi0260_r3_inputs import batch, build, identity
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    import stage21_same_scene_groot_inference as stage21
    import stage40_b0_model_control as stage40
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    plan = card()
    assert point in ('Base_0', 'Base_6000', 'G_0', 'G_6000')
    arm, step = point.split('_')
    rows = [r for r in plan['rows'] if r['arm'] == arm and r['step'] == int(step)]
    assert len(rows) == 440
    binding = read(binding_path)
    assert binding['point'] == point and binding['source_revision']
    for name, expected in binding['model_files_sha256'].items():
        assert sha(model_view / name) == expected, name
    output = R3 / 'eval' / point
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage40._bind_local_cosmos_processor(GROOT, COSMOS)
    processor = build(read(ROOT / 'dataset/statistics.json'), COSMOS)
    model = AutoModel.from_pretrained(model_view, local_files_only=True, trust_remote_code=True).eval().to('cuda:0', dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps == 4
    motion = read(R3 / 'motion2_eval_cases.json')['cases']
    original = read(R3 / 'O64_eval_cases.json')['rows']
    noises = {}
    for name in ('B0', 'B1'):
        with np.load(R3 / 'noise' / (name + '.npz'), allow_pickle=False) as values:
            noises[name] = values['float32'].copy()
    attempts = R3 / 'eval' / 'attempts.jsonl'
    head_log = R3 / 'eval' / 'head_forwards.jsonl'
    heads = sum(1 for _ in head_log.open()) if head_log.exists() else 0
    for row in rows:
        cell_id = f"{row['cohort']}-{row['case_index']:03d}-{row['noise']}"
        cell = output / 'cells' / cell_id
        if (cell / 'record.json').exists():
            continue
        packet, expert, source_row = source(row, motion, original)
        policy = batch(processor, packet)['inputs']
        condition_sha = identity(policy)
        noise = noises[row['noise']]
        cell.mkdir(parents=True, exist_ok=True)
        tensors = cell / 'tensors.npz'
        if tensors.exists():
            with np.load(tensors, allow_pickle=False) as values:
                assert np.array_equal(values['expert'], expert) and np.array_equal(values['state'], packet['state_v1'])
                assert np.array_equal(values['noise'], noise)
                raw, decoded = values['raw'].copy(), values['decoded'].copy()
            replay = bridge._decode_physical(processor, EmbodimentTag.NEW_EMBODIMENT, raw, packet['state_v1'])
            assert np.max(np.abs(replay - decoded)) <= 1e-7
        else:
            with attempts.open('a') as stream:
                count = sum(1 for _ in attempts.open()) + 1
                assert count <= plan['query_attempt_cap']
                stream.write(json.dumps(dict(ordinal=count, point=point, cell_id=cell_id, condition_sha256=condition_sha)) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            def count_head(_module, _inputs):
                nonlocal heads
                heads += 1
                assert heads <= plan['inference_head_cap']
                with head_log.open('a') as stream:
                    stream.write(json.dumps(dict(ordinal=heads, point=point, cell_id=cell_id)) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
            handle = model.action_head.model.register_forward_pre_hook(count_head)
            try:
                stock = bridge._stock_query(model, policy, stage21, noise)
            finally:
                handle.remove()
            assert len(stock['trace']) == 4 and np.array_equal(stock['noise'], noise)
            raw = np.asarray(stock['raw'], dtype=np.float32)
            decoded = bridge._decode_physical(processor, EmbodimentTag.NEW_EMBODIMENT, raw, packet['state_v1'])
            temp = cell / 'tensors.partial.npz'
            np.savez_compressed(temp, raw=raw, decoded=decoded, expert=expert, state=packet['state_v1'],
                                noise=noise, trace_x=np.stack([x['x'] for x in stock['trace']]),
                                trace_v=np.stack([x['v'] for x in stock['trace']]),
                                trace_t=np.stack([x['t'] for x in stock['trace']]))
            os.replace(temp, tensors)
        metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
        state = packet['state_v1']
        orientation = matrix_to_quat(rot6d_rows_to_matrix(state[None, 3:9]))[0]
        gate = TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded, chunk_id='pi0260-r3-' + cell_id),
                                                 now=0., achieved_position=state[:3], achieved_orientation=orientation)
        write(cell / 'record.json', dict(status='VALID', point=point, cell_id=cell_id, source=source_row,
                                        condition_sha256=condition_sha, noise_name=row['noise'],
                                        tensor_sha256=sha(tensors), metrics=metrics, gate=gate.as_dict(),
                                        model_binding_sha256=sha(binding_path), eval_card_sha256=sha(R3 / 'S4_FIXED_EVAL_CARD.json')))
        print(json.dumps(dict(status='R3_EVAL_CELL', point=point, cell_id=cell_id, xyz_rmse_m=metrics['xyz_rmse_m'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--point')
    parser.add_argument('--model-view', type=Path)
    parser.add_argument('--binding', type=Path)
    args = parser.parse_args()
    preflight() if args.preflight else run(args.point, args.model_view, args.binding)
