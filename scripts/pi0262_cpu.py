"""Independent, read-only CPU replay of the frozen 1920 crossed M1 rows."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict

import numpy as np
import pi0260_r3_eval as r3
from pi0260_r3_eval_cpu import metrics
from pi0262_old_reorg import ROOT, TEMPLATES


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rmse(first, second):
    delta = first[:, :3].astype(np.float64) - second[:, :3].astype(np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def csv_save(name, rows):
    path = ROOT / name
    assert not path.exists()
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run():
    from pi0260_r3_inputs import batch, build, identity
    import pi0242_train_infer_bridge as bridge
    import stage40_b0_model_control as stage40
    import pi0228_train_calib_checkpoint_selection as selector
    from gr00t.data.embodiment_tags import EmbodimentTag
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    card_path = ROOT / 'CROSSED_EVAL_CARD.json'
    card = r3.read(card_path)
    assert len(card['rows']) == 1920 and len({r['row_id'] for r in card['rows']}) == 1920
    assert r3.sha(ROOT / 'SOURCE_TEMPLATE_SPLIT_MANIFEST.json') == card['source_manifest_sha256']
    bindings = r3.read(ROOT / 'SOURCE_BINDING_AUDIT.json')
    assert r3.sha(ROOT / 'SOURCE_BINDING_AUDIT.json') == card['source_binding_sha256']
    expected = {'Donor': '52abc8d151b67bb667e43fcf874dfe2cfa253364',
                'C2000': 'ce5122232b3577005ecabb4cdd95f252cde5b2a6',
                'P2000': '875abdde2cc7c9e6858fac008e53f2f4b3375255'}
    for point, revision in expected.items():
        assert '@' + revision + '/' in bindings['models'][point]['binding']['source_revision']
    for source in bindings['code'].values():
        assert r3.sha(source['path']) == source['sha256']
    assert r3.sha(r3.ROOT / 'dataset/statistics.json') == bindings['statistics_sha256']
    stage40._bind_local_cosmos_processor(r3.GROOT, r3.COSMOS)
    processor = build(r3.read(r3.ROOT / 'dataset/statistics.json'), r3.COSMOS)
    conditions, frames, predictions, references = {}, {}, {}, {}
    result, provenance = [], []
    for row in card['rows']:
        rec, states, labels = r3.sequence(row['sequence'])
        j = next(i for i, item in enumerate(rec['rows']) if item['index'] == 0)
        source = rec['rows'][j]
        assert source['active'] and source['usable'] and source['time'] == row['time_s'] == 0.
        assert r3.sha(r3.ROOT / 'dataset' / (row['sequence'] + '.json')) == row['sequence_json_sha256']
        assert r3.sha(r3.ROOT / 'dataset' / (row['sequence'] + '.npz')) == row['sequence_npz_sha256']
        assert sha_array(states[j]) == row['state_sha256'] and sha_array(labels[j]) == row['reference_sha256']
        frame = r3.ROOT / source['frame']
        if str(frame) not in frames:
            assert r3.sha(frame) == row['frame_sha256']
            with np.load(frame, allow_pickle=False) as z:
                frames[str(frame)] = {key: z[key].copy() for key in ('body_rgb', 'wrist_rgb')}
        packet = dict(**frames[str(frame)], state_v1=states[j], language=row['language'])
        key = row['sequence'], row['template']
        if key not in conditions:
            conditions[key] = identity(batch(processor, packet)['inputs'])
        assert conditions[key] == row['condition_sha256']
        cell = ROOT / 'eval' / row['point'] / 'cells' / row['row_id']
        record = r3.read(cell / 'record.json')
        assert record['status'] == 'VALID' and record['row'] == row
        assert record['eval_card_sha256'] == r3.sha(card_path)
        tensor = cell / 'tensors.npz'
        assert r3.sha(tensor) == record['tensor_sha256']
        with np.load(tensor, allow_pickle=False) as z:
            assert z['raw'].shape == (1, 40, 132) and z['decoded'].shape == (1, 40, 10)
            assert z['trace_x'].shape == z['trace_v'].shape == (4, 1, 40, 132)
            assert np.array_equal(z['state'], states[j]) and np.array_equal(z['expert'], labels[j])
            assert sha_array(z['noise']) == row['noise_identity']['input_float32_sha256']
            decoded = bridge._decode_physical(processor, EmbodimentTag.NEW_EMBODIMENT, z['raw'], states[j])
            assert np.max(np.abs(decoded - z['decoded'])) <= 1e-7
            prediction = z['decoded'][0].copy()
        if row['reused_source']:
            old = row['reused_source']
            assert r3.sha(old['cell'] + '/record.json') == old['record_sha256']
            assert r3.sha(old['cell'] + '/tensors.npz') == old['tensor_sha256']
            with np.load(old['cell'] + '/tensors.npz', allow_pickle=False) as z:
                assert np.array_equal(prediction, z['decoded'][0])
        opposite_seq = f"{row['family']}-M1-c{1-row['command']}"
        other, _, refs = r3.sequence(opposite_seq)
        k = next(i for i, item in enumerate(other['rows']) if item['index'] == 0)
        assert other['rows'][k]['frame_sha256'] == row['frame_sha256']
        observed = metrics(prediction, labels[j])
        assert all(abs(observed[key]-record['metrics'][key]) <= 1e-9 for key in observed)
        gate = TrajectorySafetyGateV1().evaluate(selector._build_chunk(prediction[None], chunk_id=row['row_id']),
            now=0., achieved_position=states[j, :3],
            achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(states[j:j+1, 3:9]))[0])
        assert gate.decision.value == record['gate']['decision']
        assert [v.code for v in gate.issues] == [v['code'] for v in record['gate']['issues']]
        own, opposite = rmse(prediction, labels[j]), rmse(prediction, refs[k])
        margin = opposite - own
        result.append(dict(row_id=row['row_id'], point=row['point'], split=row['split'],
            family=row['family'], command=row['command'], color=row['color'], template=row['template'],
            noise=row['noise'], correct=margin > 0, margin_m=margin, own_m=own, opposite_m=opposite,
            gate=gate.decision.value, reject=gate.decision.value == 'REJECT',
            accepted_correct=margin > 0 and gate.decision.value != 'REJECT',
            reused=bool(row['reused_source']), **observed))
        predictions[row['point'], row['family'], row['template'], row['noise'], row['command']] = prediction
        references[row['family'], row['command']] = labels[j]
        provenance.append(dict(row_id=row['row_id'], tensor_sha256=record['tensor_sha256'],
                               record_sha256=r3.sha(cell / 'record.json')))
    assert len(result) == 1920 and sum(r['reused'] for r in result) == 504
    csv_save('CROSSED_RESULTS.csv', result)
    grouped = defaultdict(list)
    for row in result:
        grouped[row['point'], row['split'], row['template'], row['noise']].append(row)
    groups = []
    for (point, split, template, noise), values in sorted(grouped.items()):
        pairs = defaultdict(list)
        for row in values:
            pairs[row['family']].append(row)
        assert all(len(pair) == 2 for pair in pairs.values())
        groups.append(dict(point=point, split=split, template=template, noise=noise,
            denominator=len(values), correct=sum(v['correct'] for v in values),
            positive=sum(v['margin_m'] > 0 for v in values), negative=sum(v['margin_m'] < 0 for v in values),
            zero=sum(v['margin_m'] == 0 for v in values), invalid=0, reject=sum(v['reject'] for v in values),
            both_commands_correct=sum(all(v['correct'] for v in pair) for pair in pairs.values()),
            families=len(pairs), median_xyz_m=float(np.median([v['xyz_rmse_m'] for v in values])),
            p95_xyz_m=float(np.percentile([v['xyz_rmse_m'] for v in values], 95)),
            median_so3_rad=float(np.median([v['so3_rmse_rad'] for v in values]))))
    csv_save('TEMPLATE_GROUPS.csv', groups)
    strata = defaultdict(list)
    for row in result:
        strata[row['point'], row['split'], row['template'], row['noise'], row['command'], row['color']].append(row)
    csv_save('COLOR_COMMAND_GROUPS.csv', [dict(point=k[0], split=k[1], template=k[2], noise=k[3],
        command=k[4], color=k[5], denominator=len(v), correct=sum(r['correct'] for r in v),
        positive=sum(r['margin_m'] > 0 for r in v), negative=sum(r['margin_m'] < 0 for r in v),
        zero=sum(r['margin_m'] == 0 for r in v), reject=sum(r['reject'] for r in v)) for k, v in sorted(strata.items())])
    by_target = defaultdict(dict)
    for row in result:
        by_target[row['point'], row['split'], row['family'], row['command'], row['noise']][row['template']] = row
    flips = []
    for key, values in sorted(by_target.items()):
        assert set(values) == set(TEMPLATES)
        for first, second in itertools.combinations(TEMPLATES, 2):
            a, b = values[first], values[second]
            transition = ('stable_correct' if a['correct'] else 'stable_wrong') if a['correct'] == b['correct'] else (
                'correct_to_wrong' if a['correct'] else 'wrong_to_correct')
            flips.append(dict(point=key[0], split=key[1], family=key[2], command=key[3], noise=key[4],
                color=a['color'], from_template=first, to_template=second, transition=transition,
                from_margin_m=a['margin_m'], to_margin_m=b['margin_m']))
    assert len(flips) == 2880
    csv_save('WORDING_FLIPS.csv', flips)
    decomposition = []
    for point, family, template, noise in sorted({key[:4] for key in predictions}):
        a0, a1 = references[family, 0], references[family, 1]
        d = rmse(a0, a1)
        assert d > 0
        e = [[rmse(predictions[point, family, template, noise, c], a) for a in (a0, a1)] for c in (0, 1)]
        s0, s1 = [(v[0]**2-v[1]**2)/d**2 for v in e]
        g, b = (s1-s0)/2, (s0+s1)/2
        both = s0 < 0 < s1
        assert both == (g > abs(b))
        category = 'both_correct' if both else 'nonpositive_ordering' if g <= 0 else 'positive_ordering_bias_dominates'
        split = next(r['split'] for r in result if r['family'] == family)
        decomposition.append(dict(point=point, split=split, family=family, template=template, noise=noise,
            reference_separation_m=d, e00_m=e[0][0], e01_m=e[0][1], e10_m=e[1][0], e11_m=e[1][1],
            s0=s0, s1=s1, g=g, b=b, both_commands_correct=both, category=category))
    assert len(decomposition) == 960
    csv_save('TARGET_CHOICE_DECOMPOSITION.csv', decomposition)
    by_noise = defaultdict(dict)
    for row in decomposition:
        by_noise[row['point'], row['split'], row['family'], row['template']][row['noise']] = row
    noise_flips = []
    for key, v in sorted(by_noise.items()):
        assert set(v) == {'B0', 'B1'}
        noise_flips.append(dict(point=key[0], split=key[1], family=key[2], template=key[3],
            b0_category=v['B0']['category'], b1_category=v['B1']['category'],
            c0_boundary_crossing=(v['B0']['s0'] < 0) != (v['B1']['s0'] < 0),
            c1_boundary_crossing=(v['B0']['s1'] > 0) != (v['B1']['s1'] > 0),
            pair_correctness_crossing=v['B0']['both_commands_correct'] != v['B1']['both_commands_correct']))
    csv_save('NOISE_BOUNDARY_CROSSINGS.csv', noise_flips)
    support = {split: {(r['color'], r['command']) for r in result if r['split'] == split} for split in ('TRAIN', 'KIN-VAL')}
    common = support['TRAIN'] & support['KIN-VAL']
    r3.write(ROOT / 'COMMON_SUPPORT.json', dict(rule='predefined intersection of color/command support; all rows retained',
        support={k: sorted(v) for k, v in support.items()}, common=sorted(common),
        counts={split: sum((r['color'], r['command']) in common for r in result if r['split'] == split) for split in support}))
    attempt_path, head_path = ROOT / 'attempts.jsonl', ROOT / 'head_forwards.jsonl'
    attempts = [json.loads(s) for s in attempt_path.read_text().splitlines()]
    heads = [json.loads(s) for s in head_path.read_text().splitlines()]
    assert len(attempts) <= 1512 and len(heads) <= 6048
    assert [r['ordinal'] for r in attempts] == list(range(1, len(attempts)+1))
    assert [r['ordinal'] for r in heads] == list(range(1, len(heads)+1))
    assert all(any(v['row_id'] == h['row_id'] for v in attempts) for h in heads)
    new_ids = {r['row_id'] for r in card['rows'] if not r['reused_source']}
    assert new_ids == {r['row_id'] for r in attempts}
    invalid_only_reserve_used = len(attempts) - len(new_ids)
    assert 0 <= invalid_only_reserve_used <= 96
    r3.write(ROOT / 'CPU_REPLAY.json', dict(status='PASS_INDEPENDENT_CPU_1920_REPLAY',
        rows=1920, reused_rows=504, new_attempts=len(attempts), action_head_forwards=len(heads),
        invalid_only_reserve_used=invalid_only_reserve_used,
        processed_conditions=len(conditions), groups=groups, provenance=provenance,
        model_queries_by_cpu=0, optimizer_updates=0, backward_calls=0, physics_steps=0,
        cpu_code_sha256=r3.sha(__file__), eval_card_sha256=r3.sha(card_path),
        selected_checkpoint='NONE', synthetic_research_candidate='NONE', deployment_selected_checkpoint='NONE'))
    lines = ['# PI0262 fixed crossed M1 quality', '',
        'Entry H40 reference discrimination only; 40 families are the clustering units. B0/B1 are separate repeated conditions.',
        'Historical PI0261 C/P full qualification FAIL is unchanged; all candidate/selection/deployment fields remain NONE.', '',
        '## Four cells (pooled templates; secondary description)', '',
        '| Point | Noise | TRAIN T0/T1 | TRAIN V0/V1 | KIN-VAL T0/T1 | KIN-VAL V0/V1 |',
        '|---|---|---:|---:|---:|---:|']
    for point in ('Donor', 'C2000', 'P2000'):
        for noise in ('B0', 'B1'):
            cells = []
            for split, templates in (('TRAIN', ('T0', 'T1')), ('TRAIN', ('V0', 'V1')),
                                      ('KIN-VAL', ('T0', 'T1')), ('KIN-VAL', ('V0', 'V1'))):
                v = [r for r in result if r['point'] == point and r['noise'] == noise and r['split'] == split and r['template'] in templates]
                cells.append(f"{sum(r['correct'] for r in v)}/{len(v)}")
            lines.append('| ' + ' | '.join([point, noise, *cells]) + ' |')
    lines += ['', '## Primary per-template results', '',
        '| Point | Split | Template | Noise | Correct | Both commands | Reject | Median XYZ m | p95 XYZ m | Median SO3 rad |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['point']} | {g['split']} | {g['template']} | {g['noise']} | {g['correct']}/{g['denominator']} | {g['both_commands_correct']}/{g['families']} | {g['reject']} | {g['median_xyz_m']:.6f} | {g['p95_xyz_m']:.6f} | {g['median_so3_rad']:.6f} |")
    lines += ['', '## Interpretation boundaries', '',
        'Strict margin > 0 is the semantic count; ties and REJECT remain in denominators, with accepted_correct separately in the row table. No invalid rows were omitted.',
        'Color/command support is explicitly listed in COMMON_SUPPORT.json and COLOR_COMMAND_GROUPS.csv. The intersection is descriptive, and no hard family is deleted.',
        'TRAIN/KIN-VAL are scene-family partitions, not randomized pure geometry. Position, appearance and color distribution may also differ.',
        'WORDING_FLIPS.csv contains every same-target template comparison. TARGET_CHOICE_DECOMPOSITION.csv records ordering g and common bias b; NOISE_BOUNDARY_CROSSINGS.csv records B0/B1 boundary changes.',
        f'Actual new attempts: {len(attempts)}; S4 head forwards: {len(heads)}. Reused old rows: 504. Optimizer/backward/physics: 0.']
    assert not (ROOT / 'FOUR_CELL_REPORT.md').exists()
    (ROOT / 'FOUR_CELL_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='PASS_INDEPENDENT_CPU_1920_REPLAY', rows=1920,
                         attempts=len(attempts), heads=len(heads), groups=groups)), flush=True)


if __name__ == '__main__':
    # ponytail: one immutable full-panel replay is the check, no separate test framework.
    run()
