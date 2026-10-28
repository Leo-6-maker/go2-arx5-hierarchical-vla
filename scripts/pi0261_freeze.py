"""Freeze PI0261's original TRAIN common starts and matched C/P units."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main(dataset, visits_path, output):
    receipt = read(dataset / 'codec_receipt.json')
    families = receipt['selected_post_codec']['TRAIN']
    assert receipt['status'] == 'COMPACT_30_10' and len(families) == 30
    source = {}

    def sequence(family, task, command):
        name = f'{family}-{task}-c{command}'
        if name not in source:
            rec = read(dataset / (name + '.json'))
            with np.load(dataset / (name + '.npz'), allow_pickle=False) as z:
                states, labels = z['state_v1'].copy(), z['raw_action'].copy()
            assert rec['sequence']['split'] == 'TRAIN'
            assert (rec['sequence']['task'], rec['sequence']['command']) == (task, command)
            source[name] = rec, states, labels
        return name, *source[name]

    def row_id(name, row):
        return dict(sequence=name, index=row['index'], time_s=row['time'],
                    frame=row['frame'], frame_sha256=row['frame_sha256'])

    bank = []
    for task in ('M1', 'M3'):
        for family in families:
            both = [sequence(family, task, c) for c in (0, 1)]
            records = []
            for name, rec, states, labels in both:
                matches = [(j, row) for j, row in enumerate(rec['rows']) if row['index'] == 0]
                assert len(matches) == 1 and matches[0][1]['active'] and matches[0][1]['usable']
                j, row = matches[0]
                records.append((name, row, states[j], labels[j], rec['sequence']['instruction']))
            a, b = records
            assert a[1]['frame_sha256'] == b[1]['frame_sha256']
            assert np.array_equal(a[2], b[2]) and a[4] != b[4]
            separation = float(np.sqrt(np.mean(np.sum((a[3][:, :3] - b[3][:, :3]) ** 2, axis=1))))
            assert separation > 1e-5
            bank.append(dict(task=task, family=family, reference_xyz_rmse_m=separation,
                             commands=[row_id(item[0], item[1]) for item in records]))
    assert len(bank) == 60

    rng = np.random.default_rng(26101)
    cycles = {task: [] for task in ('M1', 'M3')}
    for task in cycles:
        for _ in range(34):
            cycles[task].extend(np.asarray(families)[rng.permutation(30)].tolist())
    used = Counter()
    units = []
    for block in range(500):
        slots = [(task, slot) for task in ('M1', 'M3') for slot in ('PAIR_SLOT', 'REPLAY_SLOT')]
        for offset in rng.permutation(4):
            task, slot = slots[int(offset)]
            visit = used[task]
            family = cycles[task][visit]
            used[task] += 1
            ordinary = []
            for command in (0, 1):
                name, rec, _, _ = sequence(family, task, command)
                quartile = (visit + command) % 4
                active = [row for row in rec['rows'] if row['active']]
                end = active[-1]['time']
                candidates = [row for row in active if row['usable'] and
                              (quartile * end / 4 <= row['time'] < (quartile + 1) * end / 4
                               or quartile == 3 and row['time'] == end)]
                assert candidates, (name, quartile)
                pick = candidates[int(rng.integers(len(candidates)))]
                ordinary.append(dict(command=command, quartile=quartile, **row_id(name, pick)))
            order = [0, 1] if len(units) % 2 == 0 else [1, 0]
            units.append(dict(unit=len(units) + 1, task=task, family=family,
                              slot=slot, order=order, ordinary=ordinary))
    assert len(units) == 2000 and all(used[t] == 1000 for t in used)
    counts = Counter((u['task'], u['slot']) for u in units)
    assert all(counts[t, s] == 500 for t in ('M1', 'M3') for s in ('PAIR_SLOT', 'REPLAY_SLOT'))
    for task in ('M1', 'M3'):
        family_counts = Counter(u['family'] for u in units if u['task'] == task)
        assert max(family_counts.values()) - min(family_counts.values()) <= 1

    old = read(visits_path)['visits']
    assert len(old) == 6000
    old_counts = Counter((x['task'], x['command'], x['family'], x['quartile']) for x in old)
    common = defaultdict(set)
    for x in old:
        _, rec, states, _ = sequence(x['family'], x['task'], x['command'])
        row = next((j for j, r in enumerate(rec['rows']) if r['index'] == x['index']), None)
        assert row is not None
        key = (x['task'], x['family'], x['frame_sha256'], sha(states[row].tobytes()))
        common[key].add(x['command'])
    diagnosis = dict(old_visits=6000, old_index0=sum(x['index'] == 0 for x in old),
                     old_cross_command_repeated_nonlanguage_keys=sum(len(v) == 2 for v in common.values()),
                     old_visit_counts_by_task_command={f'{t}/c{c}': sum(x['task'] == t and x['command'] == c for x in old)
                                                       for t in ('M1', 'M3') for c in (0, 1)},
                     old_visit_min_family_quartile=min(old_counts.values()),
                     old_visit_max_family_quartile=max(old_counts.values()))
    frozen = dict(schema='PI0261_PAIRING_PLAN_V1', authority_commit='6f2353a718cc877dddb3124029de76007eaa7df3',
                  dataset='COMPACT_30_10', namespace='PI0261_PAIRING:26101', seed=26101,
                  source_codec_receipt_sha256=sha((dataset / 'codec_receipt.json').read_bytes()),
                  source_visits_sha256=sha(visits_path.read_bytes()),
                  common_start_bank=bank, units=units, old_exposure_diagnosis=diagnosis,
                  selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(frozen, indent=2, sort_keys=True) + '\n'
    if output.exists():
        assert output.read_text() == rendered
    else:
        output.write_text(rendered)
    print(json.dumps(dict(status='PI0261_PLAN_FROZEN', sha256=sha(rendered.encode()),
                          units=len(units), pairs=len(bank), diagnosis=diagnosis)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--old-visits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.old_visits, args.output)
