"""Independent saved-array CPU replay of fixed PI0261 quality and instruction margins."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import pi0260_r3_eval as r3
from pi0260_r3_eval_cpu import metrics
from pi0261_eval import ROOT, R3, plan, source


def replay(point):
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    card, pairs = plan()
    rows = [row for row in card['rows'] if row['point'] == point]
    assert len(rows) == 680
    motion = r3.read(R3 / 'motion2_eval_cases.json')['cases']
    original = r3.read(R3 / 'O64_eval_cases.json')['rows']
    noise = {}
    for name in ('B0', 'B1'):
        with np.load(R3 / 'noise' / (name + '.npz'), allow_pickle=False) as z:
            noise[name] = z['float32'].copy()
    numeric = defaultdict(list)
    predicted = {}
    empty = {}
    train_rows = []
    train_prediction = {}
    for row in rows:
        donor_reuse = point == 'Donor' and row['cohort'] != 'TRAIN_ENTRY'
        if row['cohort'] == 'TRAIN_ENTRY':
            cell_id = f"TRAIN_ENTRY-{row['pair_index']:02d}-c{row['command']}-{row['noise']}"
        else:
            cell_id = f"{row['cohort']}-{row['case_index']:03d}-{row['noise']}"
        cell = (R3 / 'eval/Base_6000/cells' if donor_reuse else ROOT / 'eval' / point / 'cells') / cell_id
        record = r3.read(cell / 'record.json')
        assert record['status'] == 'VALID'
        assert record['point'] == ('Base_6000' if donor_reuse else point)
        assert record['cell_id'] == cell_id
        assert r3.sha(cell / 'tensors.npz') == record['tensor_sha256']
        if not donor_reuse:
            assert record['eval_card_sha256'] == r3.sha(ROOT / 'FIXED_EVAL_CARD.json')
        packet, expert, case = source(row, pairs, motion, original)
        assert case == record['source']
        with np.load(cell / 'tensors.npz', allow_pickle=False) as z:
            assert np.array_equal(z['expert'], expert) and np.array_equal(z['state'], packet['state_v1'])
            assert np.array_equal(z['noise'], noise[row['noise']])
            assert z['raw'].shape == (1, 40, 132) and z['decoded'].shape == (1, 40, 10)
            assert z['trace_x'].shape == z['trace_v'].shape == (4, 1, 40, 132)
            out = z['decoded'][0].copy()
        observed = metrics(out, expert)
        assert all(abs(observed[key] - record['metrics'][key]) <= 1e-9 for key in observed)
        state = packet['state_v1']
        orientation = matrix_to_quat(rot6d_rows_to_matrix(state[None, 3:9]))[0]
        gate = TrajectorySafetyGateV1().evaluate(selector._build_chunk(out[None], chunk_id='pi0261-' + cell_id),
                                                 now=0., achieved_position=state[:3], achieved_orientation=orientation)
        assert gate.decision.value == record['gate']['decision']
        assert [issue.code for issue in gate.issues] == [issue['code'] for issue in record['gate']['issues']]
        task = case.get('task', 'O64')
        kind = case.get('kind', 'O64')
        numeric[(task, row['noise'], kind)].append(dict(**observed, reject=gate.decision.value == 'REJECT'))
        if kind in ('SEMANTIC', 'PARAPHRASE'):
            predicted[(task, kind, case['sequence'], row['noise'])] = out[:, :3].astype(np.float64)
        if kind == 'EMPTY_LANGUAGE':
            empty[(task, case['sequence'], row['noise'])] = out[:, :3].astype(np.float64)
        if row['cohort'] == 'TRAIN_ENTRY':
            train_rows.append((row, out[:, :3].astype(np.float64)))
            train_prediction[(row['pair_index'], row['command'], row['noise'])] = out[:, :3].astype(np.float64)
    assert sum(len(values) for values in numeric.values()) == 680
    summary = {}
    for key, values in sorted(numeric.items()):
        xyz = np.asarray([x['xyz_rmse_m'] for x in values])
        so3 = np.asarray([x['so3_rmse_rad'] for x in values])
        group = dict(n=len(values), median_xyz_m=float(np.median(xyz)),
                     p95_xyz_m=float(np.percentile(xyz, 95)), median_so3_rad=float(np.median(so3)),
                     reject_count=sum(x['reject'] for x in values))
        if key[2] == 'KIN-VAL':
            assert group['n'] == 16
            group['quality_pass'] = (group['median_xyz_m'] <= .020 and group['p95_xyz_m'] <= .035
                                     and group['median_so3_rad'] <= .050 and group['reject_count'] == 0)
        summary['/'.join(key)] = group

    def distance(prediction, reference):
        return float(np.sqrt(np.mean(np.sum((prediction - reference[:, :3]) ** 2, axis=1))))

    semantic_rows = []
    cases = r3.read(R3 / 'motion2_eval_cases.json')
    for pair in cases['semantic_pairs']:
        task, family = pair['task'], pair['family']
        for kind in ('SEMANTIC', 'PARAPHRASE'):
            reference = {}
            for command in (0, 1):
                matches = [(index, case) for index, case in enumerate(cases['cases'])
                           if case['task'] == task and case['kind'] == kind
                           and case['sequence'] == f'{family}-{task}-c{command}']
                assert len(matches) == 1
                reference[command] = r3.source(dict(cohort='Motion2', case_index=matches[0][0]), cases['cases'], [])[1]
            for noise_name in ('B0', 'B1'):
                for command in (0, 1):
                    sequence = f'{family}-{task}-c{command}'
                    output = predicted[task, kind, sequence, noise_name]
                    own = distance(output, reference[command])
                    opposite = distance(output, reference[1 - command])
                    margin = opposite - own
                    semantic_rows.append(dict(task=task, family=family, kind=kind, noise=noise_name,
                                              command=command, own_m=own, opposite_m=opposite,
                                              margin_m=margin, correct=margin > 0,
                                              reference_separation_m=pair['reference_xyz_rmse_m']))
    assert len(semantic_rows) == 96
    semantic = {}
    for task in ('M1', 'M3'):
        for kind in ('SEMANTIC', 'PARAPHRASE'):
            for noise_name in ('B0', 'B1'):
                values = [r for r in semantic_rows if (r['task'], r['kind'], r['noise']) == (task, kind, noise_name)]
                assert len(values) == 12
                by_pair = defaultdict(list)
                for value in values:
                    by_pair[value['family']].append(value)
                assert len(by_pair) == 6 and all(len(v) == 2 for v in by_pair.values())
                semantic[f'{task}/{kind}/{noise_name}'] = dict(n=12, correct=sum(r['correct'] for r in values),
                   per_command={str(c): sum(r['correct'] for r in values if r['command'] == c) for c in (0, 1)},
                   full_correct_pairs=sum(all(r['correct'] for r in v) for v in by_pair.values()),
                   ties=sum(r['margin_m'] == 0 for r in values),
                   negative=sum(r['margin_m'] < 0 for r in values))
                semantic[f'{task}/{kind}/{noise_name}']['pass_10_of_12'] = semantic[f'{task}/{kind}/{noise_name}']['correct'] >= 10

    matched_input_effect = []
    for pair in cases['semantic_pairs']:
        task, family = pair['task'], pair['family']
        for noise_name in ('B0', 'B1'):
            sequence0 = f'{family}-{task}-c0'
            sequence1 = f'{family}-{task}-c1'
            first = predicted[task, 'SEMANTIC', sequence0, noise_name]
            second = predicted[task, 'SEMANTIC', sequence1, noise_name]
            blank = empty[task, sequence0, noise_name]
            matched_input_effect.append(dict(task=task, family=family, noise=noise_name,
                instruction_output_xyz_rmse_m=distance(first, second),
                empty_vs_command0_xyz_rmse_m=distance(blank, first),
                empty_vs_command1_xyz_rmse_m=distance(blank, second),
                reference_separation_m=pair['reference_xyz_rmse_m']))

    train = []
    assert len(train_rows) == 240
    for row, output in train_rows:
        pair = pairs['common_start_bank'][row['pair_index']]
        references = []
        for item in pair['commands']:
            rec, _, labels = r3.sequence(item['sequence'])
            j = next(j for j, source_row in enumerate(rec['rows']) if source_row['index'] == 0)
            references.append(labels[j])
        own = distance(output, references[row['command']])
        opposite = distance(output, references[1 - row['command']])
        train.append(dict(task=row['task'], family=row['family'], pair_index=row['pair_index'],
                          command=row['command'], noise=row['noise'], own_m=own, opposite_m=opposite,
                          margin_m=opposite - own, correct=opposite > own,
                          reference_separation_m=pair['reference_xyz_rmse_m']))
    train_summary = {}
    train_pair_effect = []
    for index, pair in enumerate(pairs['common_start_bank']):
        for noise_name in ('B0', 'B1'):
            train_pair_effect.append(dict(task=pair['task'], family=pair['family'], pair_index=index,
                noise=noise_name, instruction_output_xyz_rmse_m=distance(
                    train_prediction[index, 0, noise_name], train_prediction[index, 1, noise_name]),
                reference_separation_m=pair['reference_xyz_rmse_m']))
    for task in ('M1', 'M3'):
        for noise_name in ('B0', 'B1'):
            values = [r for r in train if (r['task'], r['noise']) == (task, noise_name)]
            assert len(values) == 60
            by_pair = defaultdict(list)
            for value in values:
                by_pair[value['pair_index']].append(value)
            assert len(by_pair) == 30 and all(len(v) == 2 for v in by_pair.values())
            train_summary[f'{task}/{noise_name}'] = dict(rows=60, correct=sum(r['correct'] for r in values),
                full_correct_pairs=sum(all(r['correct'] for r in v) for v in by_pair.values()),
                ties=sum(r['margin_m'] == 0 for r in values), negative=sum(r['margin_m'] < 0 for r in values))
    qualified = all(summary[f'{task}/{noise_name}/KIN-VAL']['quality_pass'] for task in ('M1', 'M3')
                    for noise_name in ('B0', 'B1')) and all(x['pass_10_of_12'] for x in semantic.values())
    result = dict(schema='PI0261_INDEPENDENT_SAVED_ARRAY_CPU_REPLAY_V1', status='PASS_CPU_REPLAY',
                  point=point, rows=680, numeric=summary, semantic=semantic,
                  semantic_rows=semantic_rows, matched_input_effect=matched_input_effect,
                  train_entry=train_summary, train_entry_rows=train, train_pair_effect=train_pair_effect,
                  research_qualified=qualified, model_queries=0,
                  selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    path = ROOT / 'eval' / point / 'CPU_REPLAY.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], point=point, rows=680,
                          research_qualified=qualified, sha256=r3.sha(path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', choices=('Donor', 'C2000', 'P2000'), required=True)
    replay(parser.parse_args().point)
