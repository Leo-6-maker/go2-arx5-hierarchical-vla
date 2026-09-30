"""Re-index the unchanged six-family M1 saved outputs by actual wording and target."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pi0260_r3_eval as r3

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0262')
OLD = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261')
TEMPLATES = {
    'T0': 'Approach the {color} target and align the tool with its axes.',
    'T1': 'Bring the tool to the {color} marker with the pose shown by its axes.',
    'V0': 'Move to the {color} target, matching its displayed orientation.',
    'V1': 'Reach the {color} marker and match the marked tool frame.',
}


def wording(text):
    colors = re.findall(r'\b(red|blue|green)\b', text)
    assert len(colors) == 1
    matches = [key for key, template in TEMPLATES.items()
               if template.format(color=colors[0]) == text]
    assert len(matches) == 1
    return matches[0], colors[0]


def distance(first, second):
    return float(np.sqrt(np.mean(np.sum((first[:, :3].astype(np.float64) - second[:, :3].astype(np.float64)) ** 2, axis=1))))


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = r3.read(r3.R3 / 'motion2_eval_cases.json')['cases']
    selected = r3.read(r3.ROOT / 'dataset/codec_receipt.json')['selected_post_codec']
    inventory = []
    for split, expected in (('TRAIN', 30), ('KIN-VAL', 10)):
        assert len(selected[split]) == expected
        for family in selected[split]:
            pair = []
            for command in (0, 1):
                name = f'{family}-M1-c{command}'
                rec, states, labels = r3.sequence(name)
                seq = rec['sequence']
                assert (seq['split'], seq['task'], seq['command']) == (split, 'M1', command)
                template, color = wording(seq['instruction'])
                assert seq['paraphrases'] == [text.format(color=color) for text in TEMPLATES.values()]
                j = next(i for i, row in enumerate(rec['rows']) if row['index'] == 0)
                row = rec['rows'][j]
                assert row['active'] and row['usable']
                assert r3.sha(r3.ROOT / row['frame']) == row['frame_sha256']
                assert states[j].shape == (10,) and labels[j].shape == (40, 10)
                pair.append((row, states[j], labels[j], color))
                inventory.append(dict(split=split, family=family, command=command, color=color,
                    original_template=template, original_language=seq['instruction'],
                    frame_sha256=row['frame_sha256'], index=0, sequence=name,
                    sequence_json_sha256=r3.sha(r3.ROOT / 'dataset' / (name + '.json')),
                    sequence_npz_sha256=r3.sha(r3.ROOT / 'dataset' / (name + '.npz'))))
            assert pair[0][0]['frame_sha256'] == pair[1][0]['frame_sha256']
            assert np.array_equal(pair[0][1], pair[1][1]) and pair[0][3] != pair[1][3]
            assert distance(pair[0][2], pair[1][2]) > 1e-5
    rows = []
    for point in ('Donor', 'C2000', 'P2000'):
        cpu = r3.read(OLD / 'eval' / point / 'CPU_REPLAY.json')
        assert cpu['status'] == 'PASS_CPU_REPLAY' and cpu['research_qualified'] is False
        lookup = {(row['family'], row['kind'], row['noise'], row['command']): row
                  for row in cpu['semantic_rows'] if row['task'] == 'M1'}
        for index, case in enumerate(cases):
            if case['task'] != 'M1' or case['kind'] not in ('SEMANTIC', 'PARAPHRASE'):
                continue
            family = r3.sequence(case['sequence'])[0]['sequence']['family_id']
            command = case['command']
            template, color = wording(case['language'])
            assert template in ('V0', 'V1') and case['index'] == 0
            packet, own, _ = r3.source(dict(cohort='Motion2', case_index=index), cases, [])
            other_rec, _, other_labels = r3.sequence(f'{family}-M1-c{1-command}')
            opposite = other_labels[next(i for i, row in enumerate(other_rec['rows']) if row['index'] == 0)]
            for noise in ('B0', 'B1'):
                base = r3.R3 / 'eval/Base_6000/cells' if point == 'Donor' else OLD / 'eval' / point / 'cells'
                cell = base / f'Motion2-{index:03d}-{noise}'
                record = r3.read(cell / 'record.json')
                assert record['status'] == 'VALID' and record['source'] == case
                binding = r3.R3 / 'model_bindings/Base_6000.json' if point == 'Donor' else OLD / 'model_bindings' / (point + '.json')
                assert record['model_binding_sha256'] == r3.sha(binding)
                assert record['point'] == ('Base_6000' if point == 'Donor' else point)
                assert record['tensor_sha256'] == r3.sha(cell / 'tensors.npz')
                with np.load(cell / 'tensors.npz', allow_pickle=False) as z:
                    assert np.array_equal(z['expert'], own)
                    assert np.array_equal(z['state'], packet['state_v1'])
                    with np.load(r3.R3 / 'noise' / (noise + '.npz'), allow_pickle=False) as frozen:
                        assert np.array_equal(z['noise'], frozen['float32'])
                    prediction = z['decoded'][0].copy()
                a, b = distance(prediction, own), distance(prediction, opposite)
                previous = lookup[family, case['kind'], noise, command]
                assert abs(a-previous['own_m']) <= 1e-12 and abs(b-previous['opposite_m']) <= 1e-12
                rows.append(dict(point=point, family=family, command=command, color=color,
                    template=template, noise=noise, old_kind=case['kind'], own_m=a,
                    opposite_m=b, margin_m=b-a, correct=b>a, tensor_sha256=record['tensor_sha256']))
    assert len(rows) == 144
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['point'], row['template'], row['noise']].append(row)
    groups = []
    for (point, template, noise), values in sorted(grouped.items()):
        assert len(values) == 12
        pairs = defaultdict(list)
        for row in values:
            pairs[row['family']].append(row)
        assert len(pairs) == 6 and all(len(pair) == 2 for pair in pairs.values())
        groups.append(dict(point=point, template=template, noise=noise, rows=12,
            correct=sum(row['correct'] for row in values),
            both_commands_correct=sum(all(row['correct'] for row in pair) for pair in pairs.values())))
    by_target = defaultdict(dict)
    for row in rows:
        by_target[row['point'], row['family'], row['command'], row['noise']][row['template']] = row
    flips = []
    for (point, family, command, noise), values in sorted(by_target.items()):
        assert set(values) == {'V0', 'V1'}
        flips.append(dict(point=point, family=family, command=command, noise=noise,
            v0_margin_m=values['V0']['margin_m'], v1_margin_m=values['V1']['margin_m'],
            correctness_flip=values['V0']['correct'] != values['V1']['correct']))
    result = dict(status='PASS_OLD_SIX_FAMILY_M1_TEMPLATE_REORGANIZATION',
        source_rows=144, source_inventory=inventory, templates=TEMPLATES,
        same_template_groups=groups, same_target_wording_flips=flips,
        model_queries=0, optimizer_updates=0, selected_checkpoint='NONE',
        deployment_selected_checkpoint='NONE', protocol_binding='PENDING_FULL_PROTOCOL_LOCATION')
    r3.write(ROOT / 'S0_OLD_REORGANIZED.json', result)
    with (ROOT / 'S0_OLD_ROWS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(status=result['status'], source_rows=144, source_families=40,
                         same_template_groups=groups, wording_flips=sum(x['correctness_flip'] for x in flips))))


if __name__ == '__main__':
    run()
