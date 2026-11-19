"""Read the sealed D4 training labels, predictions, and durable visit ledgers."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


MANIFEST = '4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de'
VISITS = 'a151bfca90511a5fee70398aa50616472f989cbf41cdd899ca93d569a24040af'
B_INITIAL = '515e5200a2e756e7ddcf7e415c3f8d6177485ec897d6b345ec3f10b5751975e3'
PREDICTIONS = {
    'D4-A': '6cd43f099e3955116da83168a4af9af94784fa082fdf0a91c72c6788315b33df',
    'D4-B': '8a12460f790a9f3ddeb2da74b1a36f3a608db0a813f57f8b0d91e0c5d42fbda7',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(path, wanted):
    if sha(path) != wanted:
        raise ValueError(f'Archived evidence changed: {Path(path).name}')
    return json.loads(Path(path).read_bytes())


def first_hit(width):
    crossings = np.flatnonzero(width <= .022)
    return int(crossings[0]) if len(crossings) else None


def partition(k):
    return 'first_or_seam' if k == 0 else 'P10_after_first' if k < 10 else 'tail30'


def labels(spec):
    root = Path(spec['dataset'])
    data = checked(root/'MANIFEST.private.json', MANIFEST)
    records = []
    for episode in data['episodes']:
        pin = episode['labels']
        file = root/pin['file']
        if sha(file) != pin['sha256']:
            raise ValueError('Frozen C20 labels changed')
        with np.load(file, allow_pickle=False) as archive:
            plans = archive['nominal_policy_H40']
            for anchor, (phase, plan) in enumerate(zip(archive['phase'], plans, strict=True)):
                width = .002 + .040 * plan[:, 9].astype(np.float64)
                crossing = first_hit(width) if width[0] > .022 and width[-1] <= .022 else None
                records.append((episode['id'], anchor, str(phase), width, crossing))
    if len(records) != 1376:
        raise ValueError('Unexpected C20 window count')
    return records


def coverage(rows, indices):
    windows = [rows[i] for i in indices]
    return dict(visits=len(windows), unique_windows=len(set(indices)),
                by_phase=dict(Counter(row[2] for row in windows)),
                expert_crossing_visits=sum(row[4] is not None for row in windows),
                distinct_crossing_windows=len({i for i in indices if rows[i][4] is not None}))


def read_schedule(arm_root, target):
    log = arm_root/'work/updates.private.jsonl'
    updates = [json.loads(line) for line in log.read_text().splitlines()]
    visits, aux, dropout, rates = [], [], [], []
    if len(updates) != 500:
        raise ValueError('Expected 500 durable updates')
    for step, row in enumerate(updates):
        cursor = len(visits)
        if row['step'] != step+1 or row['cursor'] != cursor+32 or len(row['micro']) != 8:
            raise ValueError('Wrong durable update or microbatch boundary')
        for micro in row['micro']:
            if (micro['first_absolute_cursor'] != len(visits) or len(micro['indices']) != 4
                    or len(micro['state_token_dropped']) != 4):
                raise ValueError('Wrong absolute visit cursor')
            visits.extend(micro['indices'])
            dropout.extend(micro['state_token_dropped'])
        start = cursor+4*(step % 8)
        if [a['absolute_cursor'] for a in row['aux']] != list(range(start, start+4)):
            raise ValueError('Wrong auxiliary slot order')
        if [a['index'] for a in row['aux']] != visits[start:start+4]:
            raise ValueError('Auxiliary slot visit mismatch')
        aux.extend(visits[start:start+4])
        rates.append(float(row['used_lr']).hex())
    visit_sha = hashlib.sha256(np.asarray(visits, dtype='<i8').tobytes()).hexdigest()
    spec = json.loads((arm_root/'EXECUTION_SPEC.json').read_bytes())
    rate_sha = hashlib.sha256(('\n'.join(rates)+'\n').encode('ascii')).hexdigest()
    if visit_sha != VISITS or rate_sha != spec['learning_rates_sha256']:
        raise ValueError('Visit or learning rate table differs from original D4')
    return dict(ledger_sha256=sha(log), visit_bank_sha256=visit_sha, learning_rate_bank_sha256=rate_sha,
                dropout_sha256=hashlib.sha256(bytes(dropout)).hexdigest(), dropped=sum(dropout),
                target=coverage(target, visits), auxiliary=coverage(target, aux))


def read_predictions(arm_root, target):
    reference = arm_root/'TRAIN48_PREDICTIONS.private.json'
    bank = checked(reference, PREDICTIONS[arm_root.name])
    if len(bank) != 96 or len({(x['index'], x['noise']) for x in bank}) != 96:
        raise ValueError('Wrong saved prediction denominator')
    rows = []
    for record in bank:
        idx = record['index']
        episode, anchor, _, truth, goal = target[idx]
        if (episode, anchor) != (record['episode'], record['anchor']):
            raise ValueError('Archived prediction no longer matches its label')
        pin = record['output']
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('Archived prediction changed')
        with np.load(pin['path'], allow_pickle=False) as archive:
            width = .002 + .040 * archive['physical_H40'][:, 9].astype(np.float64)
        context = record['previous_reference']
        if abs(context['query_native_s']-context['source_native_s']-.05) > 1e-8:
            raise ValueError('Archived previous-command clock is not 50 ms')
        prev = context['width_m']
        observed = first_hit(width)
        if (record['width']['expert_crossing_in_window'], record['width']['predicted_first_crossing']) != (goal, observed):
            raise ValueError('Recomputed crossings disagree with preserved metric')
        excess = np.flatnonzero(np.abs(np.diff(np.r_[prev, width])) / .05 > .040 + 1e-10)
        envelope = record['envelope']
        first = envelope['first_temporal_exceedance']
        rows.append(dict(index=idx, noise=record['noise'], label_k=goal, prediction_k=observed,
            signed_error_s=None if goal is None or observed is None else round(.05*(observed-goal), 8),
            non_crossing_open_label_window=goal is None and bool(np.all(truth > .022)),
            prediction_contraction_P10_mm=1000*(prev-width[9]), label_contraction_P10_mm=1000*(prev-truth[9]),
            original_H40_pass=envelope['original_full_H40_pass'], original_H40_first=first,
            gripper_speed_violation_bins=sorted({partition(int(k)) for k in excess}),
            original_gripper_speed_pass=not bool(len(excess))))
    events = [r for r in rows if r['label_k'] is not None]
    open_windows = [r for r in rows if r['non_crossing_open_label_window']]
    errors = [r['signed_error_s'] for r in events if r['signed_error_s'] is not None]
    return dict(rows=rows, H40_pass=sum(r['original_H40_pass'] for r in rows),
        H40_and_grip_pass=sum(r['original_H40_pass'] and r['original_gripper_speed_pass'] for r in rows),
        grip_violation_bins=dict(Counter(bin for r in rows for bin in r['gripper_speed_violation_bins'])),
        events=dict(predictions=len(events), unique_windows=len({r['index'] for r in events}),
            misses=len(events)-len(errors), early=sum(e < 0 for e in errors), late=sum(e > 0 for e in errors),
            on_tick=errors.count(0.), errors_s=errors, predicted_P10=sum(
                (r['prediction_k'] is not None and r['prediction_k'] < 10) for r in events),
            label_P10=sum(r['label_k'] < 10 for r in events),
            label_P10_median_mm=float(np.median([r['label_contraction_P10_mm'] for r in events])),
            predicted_P10_median_mm=float(np.median([r['prediction_contraction_P10_mm'] for r in events]))),
        false_closes_on_open_label_windows=dict(eligible=len(open_windows),
            count=sum(r['prediction_k'] is not None for r in open_windows)))


def verify_step_zero(root):
    initial = checked(root/'INITIAL.private.json', B_INITIAL)
    pin = initial['checkpoint']
    directory = Path(pin['path'])
    manifest = checked(directory/'COMPLETE.json', pin['sha256'])
    if (manifest['step'], manifest['cursor'], len(manifest['arrays'])) != (0, 0, 87):
        raise ValueError('Archived B initialization is not the full step-zero state')
    for row in manifest['arrays']:
        if sha(directory/row['file']) != row['sha256']:
            raise ValueError('Archived B step-zero array changed')
    return dict(checkpoint_sha256=pin['sha256'], arrays_verified=87, optimizer_step=0, cursor=0,
                initial_noise_sha256=initial['initial_noise_sha256'],
                initial_flow_time_sha256=initial['initial_flow_time_sha256'])


def audit(root):
    root = Path(root)
    target = labels(json.loads((root/'D4-B/EXECUTION_SPEC.json').read_bytes()))
    arms = {arm:dict(predictions=read_predictions(root/arm, target), schedule=read_schedule(root/arm, target))
            for arm in PREDICTIONS}
    a, b = (arms[arm]['schedule'] for arm in PREDICTIONS)
    if {k:v for k,v in a.items() if k != 'ledger_sha256'} != {k:v for k,v in b.items() if k != 'ledger_sha256'}:
        raise ValueError('A/B observed training exposure differs')
    return dict(arms=arms, historical_B_step_zero=verify_step_zero(root/'D4-B'), new_updates=0,
                new_model_generations=0, new_native_requests=0, new_native_ticks=0)


if __name__ == '__main__':
    import sys
    print(json.dumps(audit(sys.argv[1]), allow_nan=False))
