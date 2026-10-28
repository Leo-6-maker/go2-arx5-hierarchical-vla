"""Frozen M1 crossed card and first-valid stock S4 inference; no training."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pi0260_r3_eval as r3
from pi0260_r3_inputs import batch, build, identity
from pi0262_old_reorg import ROOT, OLD, TEMPLATES, wording, distance


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def packet(row):
    rec, states, labels = r3.sequence(row['sequence'])
    j = next(i for i, item in enumerate(rec['rows']) if item['index'] == 0)
    item = rec['rows'][j]
    assert r3.sha(r3.ROOT / item['frame']) == row['frame_sha256']
    assert digest(states[j]) == row['state_sha256'] and digest(labels[j]) == row['reference_sha256']
    with np.load(r3.ROOT / item['frame'], allow_pickle=False) as z:
        inputs = dict(body_rgb=z['body_rgb'].copy(), wrist_rgb=z['wrist_rgb'].copy(),
                      state_v1=states[j].copy(), language=row['language'])
    return inputs, labels[j].copy()


def processor():
    import stage40_b0_model_control as stage40
    stage40._bind_local_cosmos_processor(r3.GROOT, r3.COSMOS)
    return build(r3.read(r3.ROOT / 'dataset/statistics.json'), r3.COSMOS)


def freeze():
    assert not (ROOT / 'CROSSED_EVAL_CARD.json').exists()
    selected = r3.read(r3.ROOT / 'dataset/codec_receipt.json')['selected_post_codec']
    registry = r3.read(r3.ROOT / 'design/registry.json')
    families = {f['family_id']: f for f in registry['families']}
    seqs = {s['sequence_id']: s for s in registry['sequences']}
    old = r3.read(ROOT / 'CACHE_SOURCE_INDEX.json')
    cache = {(r['point'], r['sequence'], r['template'], r['noise']): r for r in old['rows']}
    bindings = r3.read(ROOT / 'SOURCE_BINDING_AUDIT.json')
    prep = r3.read(ROOT / 'S0_OLD_REORGANIZED.json')['source_inventory']
    proc = processor()
    sources, conditions, controls = [], {}, []
    for split, expected in (('TRAIN', 30), ('KIN-VAL', 10)):
        assert len(selected[split]) == expected
        for family in selected[split]:
            refs = []
            for command in (0, 1):
                name = f'{family}-M1-c{command}'
                rec, states, labels = r3.sequence(name)
                seq = seqs[name]
                color = families[family]['factors']['colors'][command]
                assert rec['sequence'] == seq and wording(seq['instruction'])[1] == color
                assert seq['paraphrases'] == [v.format(color=color) for v in TEMPLATES.values()]
                j = next(i for i, x in enumerate(rec['rows']) if x['index'] == 0)
                item = rec['rows'][j]
                assert item['active'] and item['usable']
                baseline = next(x for x in prep if x['sequence'] == name)
                assert baseline['sequence_json_sha256'] == r3.sha(r3.ROOT / 'dataset' / (name + '.json'))
                assert baseline['sequence_npz_sha256'] == r3.sha(r3.ROOT / 'dataset' / (name + '.npz'))
                row = dict(**baseline, frame=item['frame'], time_s=item['time'],
                           state_sha256=digest(states[j]), reference_sha256=digest(labels[j]),
                           color_mapping_source='original design registry family.factors.colors[command]',
                           registry_sha256=r3.sha(r3.ROOT / 'design/registry.json'))
                refs.append(labels[j])
                for template, text in TEMPLATES.items():
                    condition = dict(row, template=template, language=text.format(color=color))
                    inputs, _ = packet(condition)
                    condition['condition_sha256'] = identity(batch(proc, inputs)['inputs'])
                    conditions[name, template] = condition
                sources.append(row)
            d = distance(*refs)
            assert d > 0
            for command in (0, 1):
                own, opposite = refs[command], refs[1-command]
                positive = distance(own, opposite) - distance(own, own)
                negative = distance(opposite, opposite) - distance(opposite, own)
                # An exact tie uses identical reference copies; never an epsilon threshold.
                tie = distance(own, own.copy()) - distance(own, own)
                assert positive == d and negative == -d and tie == 0
                controls.append(dict(family=family, command=command, positive_margin_m=positive,
                                     negative_margin_m=negative, tie_margin_m=tie, tie_correct=tie > 0))
    exposure = []
    visits = r3.read(r3.R3 / 'train_visits.json')['visits']
    assert len(visits) == 6000
    logs = {'R3': visits}
    for arm in ('C', 'P'):
        path = OLD / 'training_evidence' / arm / 'main/training/sample_audit.jsonl'
        logs[arm] = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(logs[arm]) == 4000
    for arm, rows in logs.items():
        counts = Counter()
        for row in rows:
            if row['task'] != 'M1':
                continue
            seq = r3.sequence(row['sequence'])[0]['sequence']
            template, color = wording(seq['instruction'])
            assert seq['split'] == 'TRAIN' and template in ('T0', 'T1')
            counts[template, color, row['command'], row['family']] += 1
        exposure.extend(dict(arm=arm, template=k[0], color=k[1], command=k[2], family=k[3],
                             actual_supervised_visits=n) for k, n in sorted(counts.items()))
    r3.write(ROOT / 'S0_COMPLETION.json', dict(status='PASS_COLOR_MARGIN_ACTUAL_EXPOSURE',
        controls=controls, exposure=exposure, source_visits={k: len(v) for k, v in logs.items()},
        source_sha256={'registry': r3.sha(r3.ROOT / 'design/registry.json'),
            'R3_visits': r3.sha(r3.R3 / 'train_visits.json'),
            **{arm: r3.sha(OLD / 'training_evidence' / arm / 'main/training/sample_audit.jsonl') for arm in ('C', 'P')}},
        model_queries=0))
    r3.write(ROOT / 'SOURCE_TEMPLATE_SPLIT_MANIFEST.json', dict(splits=selected,
        templates=TEMPLATES, sources=sources, conditions=list(conditions.values()),
        common_support_rule='descriptive intersection of observed (color,command) in both splits; no row deletion'))
    rows = []
    noise = {v['name']: v for v in bindings['noise']['rows']}
    for point in ('Donor', 'C2000', 'P2000'):
        for condition in conditions.values():
            for name in ('B0', 'B1'):
                prior = cache.get((point, condition['sequence'], condition['template'], name))
                if prior:
                    assert prior['condition_sha256'] == condition['condition_sha256']
                    assert prior['color'] == condition['color']
                row_id = f"{point}-{condition['family']}-c{condition['command']}-{condition['template']}-{name}"
                rows.append(dict(**condition, point=point, noise=name, row_id=row_id,
                    model_binding_sha256=bindings['models'][point]['binding_sha256'],
                    model_source=bindings['models'][point]['binding'], noise_identity=noise[name],
                    reused_source=prior))
    assert len(rows) == len({r['row_id'] for r in rows}) == 1920
    assert sum(r['reused_source'] is not None for r in rows) == 504
    r3.write(ROOT / 'CROSSED_EVAL_CARD.json', dict(schema='PI0262_CROSSED_1920_V1',
        protocol_sha256=r3.sha(ROOT / 'PI0262_PROTOCOL.md'),
        evidence_ref='3e3983c8345d50eb322405084efd4f3c9381d97f', canonical=5909947990,
        source_manifest_sha256=r3.sha(ROOT / 'SOURCE_TEMPLATE_SPLIT_MANIFEST.json'),
        source_binding_sha256=r3.sha(ROOT / 'SOURCE_BINDING_AUDIT.json'),
        s0_completion_sha256=r3.sha(ROOT / 'S0_COMPLETION.json'),
        code_sha256=r3.sha(Path(__file__)), rows=rows, new_attempt_cap=1512, inference_head_cap=6048,
        planned_new_queries=1416, optimizer_updates=0, backward_calls=0,
        synthetic_research_candidate='NONE', selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status='PI0262_CARD_FROZEN', rows=len(rows), reused=504,
                         sha256=r3.sha(ROOT / 'CROSSED_EVAL_CARD.json'))), flush=True)


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def count(path):
    return sum(1 for _ in path.open()) if path.exists() else 0


def run(point):
    import fcntl
    import torch
    from transformers import AutoModel
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    card = r3.read(ROOT / 'CROSSED_EVAL_CARD.json')
    assert card['code_sha256'] == r3.sha(Path(__file__))
    assert card['source_binding_sha256'] == r3.sha(ROOT / 'SOURCE_BINDING_AUDIT.json')
    binding = r3.read(ROOT / 'SOURCE_BINDING_AUDIT.json')['models'][point]
    for name, value in binding['binding']['model_files_sha256'].items():
        assert r3.sha(Path(binding['view']) / name) == value
    lock = (ROOT / 'dispatcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proc = processor()
    model = None
    attempts, headlog = ROOT / 'attempts.jsonl', ROOT / 'head_forwards.jsonl'
    heads = count(headlog)
    for row in [r for r in card['rows'] if r['point'] == point]:
        cell = ROOT / 'eval' / point / 'cells' / row['row_id']
        record_path = cell / 'record.json'
        if record_path.exists():
            record = r3.read(record_path)
            assert record['eval_card_sha256'] == r3.sha(ROOT / 'CROSSED_EVAL_CARD.json')
            assert record['row'] == row and r3.sha(cell / 'tensors.npz') == record['tensor_sha256']
            continue
        inputs, expert = packet(row)
        policy = batch(proc, inputs)['inputs']
        assert identity(policy) == row['condition_sha256']
        with np.load(r3.R3 / 'noise' / (row['noise'] + '.npz'), allow_pickle=False) as z:
            noise = z['float32'].copy()
        assert digest(noise) == row['noise_identity']['input_float32_sha256']
        cell.mkdir(parents=True, exist_ok=True)
        tensor = cell / 'tensors.npz'
        reused = row['reused_source']
        src = Path(reused['cell']) / 'tensors.npz' if reused else tensor
        if reused:
            assert r3.sha(src) == reused['tensor_sha256']
            assert r3.sha(src.parent / 'record.json') == reused['record_sha256']
        if src.exists():
            with np.load(src, allow_pickle=False) as z:
                assert np.array_equal(z['expert'], expert) and np.array_equal(z['state'], inputs['state_v1'])
                assert np.array_equal(z['noise'], noise)
                values = {name: z[name].copy() for name in z.files}
        else:
            if model is None:
                model = AutoModel.from_pretrained(binding['view'], local_files_only=True,
                    trust_remote_code=True).eval().to('cuda:0', dtype=torch.bfloat16)
                assert model.action_head.num_inference_timesteps == 4
            ordinal = count(attempts) + 1
            assert ordinal <= card['new_attempt_cap']
            append(attempts, dict(ordinal=ordinal, row_id=row['row_id'], condition_sha256=row['condition_sha256']))
            def hook(_module, _inputs):
                nonlocal heads
                heads += 1
                assert heads <= card['inference_head_cap']
                append(headlog, dict(ordinal=heads, row_id=row['row_id']))
            handle = model.action_head.model.register_forward_pre_hook(hook)
            try:
                with torch.inference_mode():
                    stock = bridge._stock_query(model, policy, stage21, noise)
            finally:
                handle.remove()
            assert len(stock['trace']) == 4 and np.array_equal(stock['noise'], noise)
            values = dict(raw=np.asarray(stock['raw'], np.float32), expert=expert,
                state=inputs['state_v1'], noise=noise,
                trace_x=np.stack([x['x'] for x in stock['trace']]),
                trace_v=np.stack([x['v'] for x in stock['trace']]),
                trace_t=np.stack([x['t'] for x in stock['trace']]))
            values['decoded'] = bridge._decode_physical(proc, EmbodimentTag.NEW_EMBODIMENT,
                                                       values['raw'], inputs['state_v1'])
        decoded = bridge._decode_physical(proc, EmbodimentTag.NEW_EMBODIMENT, values['raw'], inputs['state_v1'])
        assert np.max(np.abs(decoded-values['decoded'])) <= 1e-7
        if not tensor.exists():
            temp = cell / 'tensors.partial.npz'
            np.savez_compressed(temp, **values)
            with temp.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temp, tensor)
        metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
        state = inputs['state_v1']
        gate = TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded, chunk_id=row['row_id']),
            now=0., achieved_position=state[:3],
            achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None, 3:9]))[0])
        r3.write(record_path, dict(status='VALID', row=row, metrics=metrics, gate=gate.as_dict(),
            tensor_sha256=r3.sha(tensor), eval_card_sha256=r3.sha(ROOT / 'CROSSED_EVAL_CARD.json')))
        print(json.dumps(dict(status='PI0262_VALID', row_id=row['row_id'], reused=bool(reused))), flush=True)
    print(json.dumps(dict(status='PI0262_POINT_COMPLETE', point=point, attempts=count(attempts), heads=count(headlog))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--point', choices=('Donor', 'C2000', 'P2000'))
    args = parser.parse_args()
    freeze() if args.freeze else run(args.point)
