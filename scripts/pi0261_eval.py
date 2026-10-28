"""First-valid fixed PI0261 offline inference, reusing the sealed R3 Base6000 rows."""
import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

import pi0260_r3_eval as r3

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261')
R3 = r3.R3


def plan():
    card = r3.read(ROOT / 'FIXED_EVAL_CARD.json')
    pairs = r3.read(ROOT / 'PAIRING_PLAN.json')
    assert card['schema'] == 'PI0261_FIXED_2040_EVAL_CARD_V1'
    assert len(card['rows']) == 2040 and len(pairs['common_start_bank']) == 60
    assert card['pairing_plan_sha256'] == r3.sha(ROOT / 'PAIRING_PLAN.json')
    assert card['processed_pair_audit_sha256'] == r3.sha(ROOT / 'PROCESSED_PAIR_BANK_AUDIT.json')
    assert card['original_r3_card_sha256'] == r3.sha(R3 / 'S4_FIXED_EVAL_CARD.json')
    return card, pairs


def source(row, pairs, motion, original):
    if row['cohort'] != 'TRAIN_ENTRY':
        return r3.source(row, motion, original)
    pair = pairs['common_start_bank'][row['pair_index']]
    assert (pair['task'], pair['family']) == (row['task'], row['family'])
    item = pair['commands'][row['command']]
    rec, states, labels = r3.sequence(item['sequence'])
    j = next(j for j, record in enumerate(rec['rows']) if record['index'] == 0)
    assert rec['rows'][j]['frame_sha256'] == item['frame_sha256']
    frame = r3.ROOT / item['frame']
    assert r3.sha(frame) == item['frame_sha256']
    with np.load(frame, allow_pickle=False) as z:
        packet = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(),
                      state_v1=states[j].copy(), language=rec['sequence']['instruction'])
    assert labels[j].shape == (40, 10)
    return packet, labels[j].copy(), dict(**row, frame_sha256=item['frame_sha256'],
                                         sequence=item['sequence'], index=0, language=packet['language'])


def run(point, model_view, binding_path):
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

    card, pairs = plan()
    assert point in ('Donor', 'C2000', 'P2000')
    rows = [row for row in card['rows'] if row['point'] == point]
    assert len(rows) == 680
    binding = r3.read(binding_path)
    if point == 'Donor':
        assert binding['point'] == 'Base_6000'
        assert binding['checkpoint_tree_sha256'] == '24ed9d1b5a0e42e0aff6d62c54d8438d2f202d765b006d90c62d8623604e4ce5'
    else:
        assert binding['point'] == point and binding['source_revision']
    for name, digest in binding['model_files_sha256'].items():
        assert r3.sha(model_view / name) == digest, name
    output = ROOT / 'eval' / point
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage40._bind_local_cosmos_processor(r3.GROOT, r3.COSMOS)
    processor = build(r3.read(r3.ROOT / 'dataset/statistics.json'), r3.COSMOS)
    model = AutoModel.from_pretrained(model_view, local_files_only=True, trust_remote_code=True).eval().to('cuda:0', dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps == 4
    motion = r3.read(R3 / 'motion2_eval_cases.json')['cases']
    original = r3.read(R3 / 'O64_eval_cases.json')['rows']
    noises = {}
    for name in ('B0', 'B1'):
        with np.load(R3 / 'noise' / (name + '.npz'), allow_pickle=False) as z:
            noises[name] = z['float32'].copy()
    attempts = ROOT / 'eval/attempts.jsonl'
    head_log = ROOT / 'eval/head_forwards.jsonl'
    heads = sum(1 for _ in head_log.open()) if head_log.exists() else 0
    cache = {}
    if point == 'Donor':
        for row in rows:
            if row['cohort'] == 'TRAIN_ENTRY':
                continue
            cell_id = f"{row['cohort']}-{row['case_index']:03d}-{row['noise']}"
            cell = R3 / 'eval/Base_6000/cells' / cell_id
            record = r3.read(cell / 'record.json')
            assert record['status'] == 'VALID' and record['point'] == 'Base_6000'
            assert r3.sha(cell / 'tensors.npz') == record['tensor_sha256']
            assert record['model_binding_sha256'] == r3.sha(R3 / 'model_bindings/Base_6000.json')
            cache.setdefault((record['condition_sha256'], row['noise']), (cell / 'tensors.npz', cell_id))
        assert sum(row['cohort'] != 'TRAIN_ENTRY' for row in rows) == 440
        r3.write(output / 'original_440_reuse.json', dict(status='PASS_EXACT_R3_DONOR_440_REUSE',
                 source='Base_6000', source_card_sha256=card['original_r3_card_sha256'], rows=440,
                 model_queries=0)) if not (output / 'original_440_reuse.json').exists() else None
    for row in rows:
        if point == 'Donor' and row['cohort'] != 'TRAIN_ENTRY':
            continue
        if row['cohort'] == 'TRAIN_ENTRY':
            cell_id = f"TRAIN_ENTRY-{row['pair_index']:02d}-c{row['command']}-{row['noise']}"
        else:
            cell_id = f"{row['cohort']}-{row['case_index']:03d}-{row['noise']}"
        cell = output / 'cells' / cell_id
        if (cell / 'record.json').exists():
            prior = r3.read(cell / 'record.json')
            assert prior['status'] == 'VALID' and prior['cell_id'] == cell_id and prior['point'] == point
            assert prior['eval_card_sha256'] == r3.sha(ROOT / 'FIXED_EVAL_CARD.json')
            assert prior['model_binding_sha256'] == r3.sha(binding_path)
            assert r3.sha(cell / 'tensors.npz') == prior['tensor_sha256']
            cache.setdefault((prior['condition_sha256'], row['noise']), (cell / 'tensors.npz', cell_id))
            continue
        packet, expert, source_row = source(row, pairs, motion, original)
        policy = batch(processor, packet)['inputs']
        condition_sha = identity(policy)
        noise = noises[row['noise']]
        reused = cache.get((condition_sha, row['noise']))
        cell.mkdir(parents=True, exist_ok=True)
        tensors = cell / 'tensors.npz'
        if tensors.exists():
            reused = None
            with np.load(tensors, allow_pickle=False) as z:
                assert np.array_equal(z['expert'], expert) and np.array_equal(z['noise'], noise)
                raw, decoded = z['raw'].copy(), z['decoded'].copy()
            replay = bridge._decode_physical(processor, EmbodimentTag.NEW_EMBODIMENT, raw, packet['state_v1'])
            assert np.max(np.abs(replay - decoded)) <= 1e-7
        else:
            if reused:
                with np.load(reused[0], allow_pickle=False) as z:
                    assert np.array_equal(z['noise'], noise)
                    raw = z['raw'].copy()
                    trace_x, trace_v, trace_t = (z[name].copy() for name in ('trace_x', 'trace_v', 'trace_t'))
            else:
                count = sum(1 for _ in attempts.open()) + 1 if attempts.exists() else 1
                assert count <= card['new_attempt_cap']
                with attempts.open('a') as stream:
                    stream.write(json.dumps(dict(ordinal=count, point=point, cell_id=cell_id,
                                                 condition_sha256=condition_sha)) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
                def count_head(_module, _inputs):
                    nonlocal heads
                    heads += 1
                    assert heads <= card['inference_head_cap']
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
                trace_x = np.stack([x['x'] for x in stock['trace']])
                trace_v = np.stack([x['v'] for x in stock['trace']])
                trace_t = np.stack([x['t'] for x in stock['trace']])
            decoded = bridge._decode_physical(processor, EmbodimentTag.NEW_EMBODIMENT, raw, packet['state_v1'])
            temp = cell / 'tensors.partial.npz'
            np.savez_compressed(temp, raw=raw, decoded=decoded, expert=expert, state=packet['state_v1'],
                                noise=noise, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
            os.replace(temp, tensors)
        metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
        state = packet['state_v1']
        orientation = matrix_to_quat(rot6d_rows_to_matrix(state[None, 3:9]))[0]
        gate = TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded, chunk_id='pi0261-' + cell_id),
                                                 now=0., achieved_position=state[:3], achieved_orientation=orientation)
        r3.write(cell / 'record.json', dict(status='VALID', point=point, cell_id=cell_id,
                 source=source_row, condition_sha256=condition_sha, noise_name=row['noise'],
                 reused_from_cell_id=reused[1] if reused else None, tensor_sha256=r3.sha(tensors),
                 metrics=metrics, gate=gate.as_dict(), model_binding_sha256=r3.sha(binding_path),
                 eval_card_sha256=r3.sha(ROOT / 'FIXED_EVAL_CARD.json')))
        cache.setdefault((condition_sha, row['noise']), (tensors, cell_id))
        print(json.dumps(dict(status='PI0261_EVAL_CELL', point=point, cell_id=cell_id)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', choices=('Donor', 'C2000', 'P2000'), required=True)
    parser.add_argument('--model-view', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    run(args.point, args.model_view, args.binding)
