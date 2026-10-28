"""PI0263 fixed source selection and language assignment; no model computation."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

AUTHORITY = '1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28'
OLD_SHA = '208f0bfbc97095369ce84329899bbbde67eb5f694ca8c9022f0566dc61f28477'
TEMPLATES = {
    'T0': 'Approach the {color} target and align the tool with its axes.',
    'T1': 'Bring the tool to the {color} marker with the pose shown by its axes.',
}
NONLANGUAGE = ('state', 'embodiment_id', 'pixel_values', 'image_grid_thw')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, ('FROZEN_FILE_CHANGED', str(path))
    else:
        path.write_text(text)


def seed_for(unit):
    return int.from_bytes(hashlib.sha256(f'PI0263:26301:{unit}'.encode()).digest()[:8], 'big') % 2**31


def language(arm, unit, sequence, color):
    assert arm in ('F', 'X')
    original = sequence['instruction']
    if unit['task'] == 'M3':
        return original, 'ORIGINAL_M3'
    assert original in [s.format(color=color) for s in TEMPLATES.values()]
    template = next(t for t, s in TEMPLATES.items() if s.format(color=color) == original)
    if arm == 'X':
        template = unit['x_template']
    assert template in TEMPLATES
    return TEMPLATES[template].format(color=color), template


def validate(plan):
    units = plan['units']
    assert len(units) == 2000 and [u['unit'] for u in units] == list(range(1, 2001))
    counts = Counter((u['task'], u['slot']) for u in units)
    assert set(counts.values()) == {500} and len(counts) == 4
    assert Counter(u['x_template'] for u in units if u['task'] == 'M1') == {'T0': 500, 'T1': 500}
    original_m1 = sorted({u['original_unit'] for u in units if u['task']=='M1'},
        key=lambda uid:(hashlib.sha256(f'PI0263:26301:template-order:{uid}'.encode()).hexdigest(),uid))
    assert len(original_m1)==500
    first_t0=set(original_m1[:250])
    for k in range(1000):
        a, b = units[2*k:2*k+2]
        assert a['original_unit'] == b['original_unit'] and a['selected'] == b['selected']
        assert (a['visit'], b['visit'], a['order'], b['order']) == (0, 1, [0, 1], [1, 0])
        if a['task'] == 'M1':
            assert {a['x_template'], b['x_template']} == {'T0', 'T1'}
            assert a['x_template']==('T0' if a['original_unit'] in first_t0 else 'T1')
    for task in ('M1', 'M3'):
        for slot in ('PAIR_SLOT', 'REPLAY_SLOT'):
            assert len({u['family'] for u in units if (u['task'], u['slot']) == (task, slot)}) == 30


def freeze(old_path, output):
    assert sha(old_path) == OLD_SHA
    old = read(old_path)
    strata = defaultdict(list)
    bank = {(r['task'], r['family']): r for r in old['common_start_bank']}
    for unit in old['units']:
        strata[unit['task'], unit['slot']].append(unit)
    chosen = []
    for rows in strata.values():
        rows = sorted(rows, key=lambda u: u['unit'])
        first = {}
        for unit in rows:
            first.setdefault(unit['family'], unit)
        assert len(first) == 30
        keep = {u['unit'] for u in first.values()}
        for unit in rows:
            if len(keep) == 250:
                break
            keep.add(unit['unit'])
        chosen.extend(u for u in rows if u['unit'] in keep)
    chosen.sort(key=lambda u: u['unit'])
    m1 = sorted((u['unit'] for u in chosen if u['task'] == 'M1'),
                key=lambda uid: (hashlib.sha256(f'PI0263:26301:template-order:{uid}'.encode()).hexdigest(), uid))
    first_t0 = set(m1[:250])
    units = []
    for source in chosen:
        selected = bank[source['task'], source['family']]['commands'] if source['slot'] == 'PAIR_SLOT' else source['ordinary']
        for visit in (0, 1):
            template = ('T0' if (source['unit'] in first_t0) == (visit == 0) else 'T1') if source['task'] == 'M1' else 'ORIGINAL_M3'
            units.append(dict(source, unit=len(units)+1, original_unit=source['unit'], visit=visit,
                              order=[0, 1] if visit == 0 else [1, 0], selected=selected, x_template=template))
    plan = dict(schema='PI0263_COUNTERBALANCED_PLAN_V1', authority_commit=AUTHORITY,
                source_plan_sha256=OLD_SHA, namespace='PI0263:26301', seed=26301,
                common_start_bank=old['common_start_bank'], units=units,
                templates=TEMPLATES, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    validate(plan)
    write(output, plan)
    counts = Counter((u['task'], u['slot'], u['family']) for u in chosen)
    phases = Counter((u['task'], u['slot'], c.get('quartile', 'COMMON_START'))
                     for u in units for c in u['selected'])
    write(Path(output).with_name('INHERITED_EXPOSURE.json'), dict(
        base_units=1000, optimizer_units=2000, examples_per_arm=4000,
        family=[dict(task=t, slot=s, family=f, base_units=n) for (t,s,f),n in sorted(counts.items())],
        phase=[dict(task=t, slot=s, phase=str(q), examples=n) for (t,s,q),n in sorted(phases.items(), key=str)],
        forced_uniformity=False))
    print(json.dumps(dict(status='PI0263_PLAN_FROZEN', sha256=sha(output), units=2000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.old, args.output)
