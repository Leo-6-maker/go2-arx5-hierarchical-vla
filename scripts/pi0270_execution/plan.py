"""PI0270 fixed collection and replacement ordering; metadata only, no models."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

QUERY_INDICES = tuple(99 * j // 31 for j in range(32))
BUCKETS = ('ORIGINAL_Q0', 'ORIGINAL_Q1', 'ORIGINAL_Q2', 'ORIGINAL_Q3',
           'M1_TERMINAL', 'M3_ENTRY', 'M3_MIDARC', 'M3_TERMINAL')


def noise_seed(namespace):
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], 'big') % (2**63 - 1)


def collection_cases(scene):
    families = [f for f in scene['families'] if f['split'] == 'FEEDBACK_TRAIN']
    assert len(families) == 24 and len({f['family_id'] for f in families}) == 24
    return [dict(family=f['family_id'], task=task, command=c, backend=backend,
                 case=f"{f['family_id']}-{task}-c{c}-{backend}", query_indices=list(QUERY_INDICES))
            for f in families for task in ('M1', 'M3') for c in (0, 1) for backend in ('E0', 'E1')]


def bucket(record):
    if record['source_lane'] == 'original':
        assert record['query_index'] in QUERY_INDICES
        return f"ORIGINAL_Q{min(3, record['query_index'] * 4 // 100)}"
    assert record['source_lane'] == 'component' and 0 <= record['query_index'] < 20
    assert record['start_type'] in BUCKETS[4:]
    return record['start_type']


def replacements(plan, records):
    """Return selected source records per recovery micro; never rank by quality."""
    cells = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record['eligible']:
            cell = record['family'], record['task'], record['command']
            cells[cell][bucket(record)].append(record)
    for bins in cells.values():
        for values in bins.values():
            values.sort(key=lambda r: (r['source_episode_key'], r['query_index']))
            keys = [(r['source_episode_key'], r['query_index']) for r in values]
            assert len(keys) == len(set(keys)), 'DUPLICATE_PROPOSAL'
    occurrences = defaultdict(int)
    result = []
    for unit in plan['units']:
        assert unit['slot'] in ('COMMON_NOMINAL', 'RECOVERY_INTERVENTION')
        if unit['slot'] == 'COMMON_NOMINAL':
            continue
        for micro, command in enumerate(unit['order']):
            cell = unit['family'], unit['task'], command
            bins = cells[cell]
            order = [b for b in BUCKETS if bins.get(b)]
            assert order, ('EMPTY_RECOVERY_CELL', cell)
            n = occurrences[cell]
            selected_bucket = order[n % len(order)]
            choices = bins[selected_bucket]
            selected = choices[(n // len(order)) % len(choices)]
            result.append(dict(unit=unit['unit'], micro=micro, command=command,
                               template=unit['template'], bucket=selected_bucket, source=selected))
            occurrences[cell] += 1
    return result


def self_check():
    assert len(QUERY_INDICES) == 32 and QUERY_INDICES[0] == 0 and QUERY_INDICES[-1] == 99
    assert QUERY_INDICES[27] == 86 and len(set(QUERY_INDICES)) == 32
    assert noise_seed('PI0270:COLLECT:27101:f:M1:0:E0:0') == 4492558428037799223
    scene = dict(families=[dict(family_id=f'f{i}', split='FEEDBACK_TRAIN') for i in range(24)])
    cases = collection_cases(scene)
    assert len(cases) == 192 and sum(len(r['query_indices']) for r in cases) == 6144
    records = [dict(family='f0', task='M1', command=c, eligible=True, source_lane='original',
                    source_episode_key=f'e{c}', query_index=q) for c in (0, 1) for q in (0, 3, 25)]
    records.append(dict(records[0], eligible=False, query_index=6))
    units = [dict(unit=i + 1, family='f0', task='M1', order=[0, 1], template='T0',
                  slot='RECOVERY_INTERVENTION') for i in range(4)]
    picked = replacements(dict(units=units), records)
    assert [r['source']['query_index'] for r in picked[::2]] == [0, 25, 3, 25]
    assert picked == replacements(dict(units=units), list(reversed(records)))
    assert not replacements(dict(units=[dict(units[0], slot='COMMON_NOMINAL')]), records)
    try:
        replacements(dict(units=units), [])
    except AssertionError:
        pass
    else:
        raise AssertionError('EMPTY_CELL_ACCEPTED')
    return dict(status='PASS_METADATA_ORDER_AND_DENOMINATORS_ONLY', collection_cases=192,
                original_label_proposals=6144, new_model_teacher_render_optimizer_calls=0,
                launch_allowed=False, scientific_admission='A0_NOT_YET_CLOSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = self_check()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding='utf-8')
    print(text, end='')
