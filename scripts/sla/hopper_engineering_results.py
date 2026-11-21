"""Seal candidate costs and select the fastest numerically qualified H200 backend."""
import argparse
from collections import Counter
from contextlib import closing
import math
from pathlib import Path
from statistics import median

import numpy as np

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_engineering import FM_WIDTH, backend
from scripts.sla.hopper_equivalence import COMPONENTS
from scripts.sla.hopper_inputs import BATCH, PARENT, PLAN
from scripts.sla.hopper_panels import ENGINEERING_PANEL
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned

CANDIDATES = tuple(f'E{n}' for n in range(1, 7))


def pin(path):
    return dict(path=str(path.resolve(strict=True)), sha256=digest(path))


def ordinary_arrays(directory, row):
    from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha

    name = f"{row['kind']}-{row['ordinal']:03d}.npz"
    if row['file'] != name or digest(directory/name) != row['file_sha256']:
        raise ValueError('Changed output file or panel identity')
    with np.load(directory/name, allow_pickle=False) as f:
        if set(f.files) != {'normalized', 'decoded'}:
            raise ValueError('Unexpected archived output format')
        normalized, decoded = f['normalized'], f['decoded']
        if (bytes_sha(decoded) != row['decoded_sha256'] or bytes_sha(normalized) != row['normalized_sha256']
                or not np.isfinite(decoded).all() or not np.isfinite(normalized).all()):
            raise ValueError('Decoded or normalized archive does not match the sealed output')
        return decoded


def successful(folder, role):
    ordinary_dir, restored_dir = folder/'ordinary', folder/'restore'
    ordinary = pin(ordinary_dir/'RESULT.private.json')
    restored = pin(restored_dir/'RESULT.private.json')
    o, r = pinned(ordinary), pinned(restored)
    train = pin(folder/'train/TRAIN.private.json')
    t = pinned(train)
    site = read(folder/'SITE.private.json')
    first_pin = pin(folder/'train/first-step/COMPLETE.json')
    first = pinned(first_pin)
    bound = t['binding']
    if (t['role'] != role or t['parent'] != PARENT or t['optimizer_updates'] != 16
            or t['visits'] != {d: n*16 for d, n in BATCH.items()}
            or (t['attempted_aux'], t['completed_aux']) != (64, 64)
            or first_pin['sha256'] != t['first_step_sha256'] or not first['passed'] or not first['inputs_equal']
            or set(first['components']) != set(COMPONENTS) or not all(c['passed'] for c in first['components'].values())
            or any(c['loss_relative'] > 1e-4 or c['gradient_relative_L2'] > 1e-3 for c in first['components'].values())
            or site['source'] != bound['source'] or site['role'] != role or site['parent'] != PARENT
            or bound['parent'] != PARENT or bound['role'] != role or site['readback_arrays'] != 140):
        raise ValueError('Incomplete first-step, optimizer, data or H200 state readback')
    for record in (o, r):
        if (record['panel'] != ENGINEERING_PANEL or record['parent'] != PARENT or record['role'] != role
                or record['training'] != train or record['checkpoint'] != t['checkpoint']
                or not record['attempted'] == record['completed'] == len(record['rows']) == 48
                or record['site'] != {k: site['site'][k] for k in ('host','gpu_name','gpu_uuid')}):
            raise ValueError('Engineering generations do not match their training job and GPU')
    if not o['ordinary'] or r['ordinary'] or not r['qualified'] or r['exact_restores'] != 48:
        raise ValueError('Missing independent same-device restore')
    clocks = []
    updates = Journal(folder/'train/updates', role)
    for step in range(1, 17):
        row = updates.completed(f'step-{step:02d}', dict(step=step, binding=bound))
        if (not row or row['phase_step'] != step or row['visits'] != BATCH or row['full_H40'] != 4
                or not math.isfinite(row['wall_s']) or row['wall_s'] <= 0):
            raise ValueError('Missing measured full-update timing or wrong visits')
        clocks.append(row['wall_s'])
    warm = median(clocks[4:])
    if clocks != t['update_wall_s'] or warm != t['warm_p50_s'] or warm != r['warm_p50_s']:
        raise ValueError('Candidate timing differs from the committed update ledger')
    common = ('kind','ordinal','identity','input_sha256','noise_sha256','normalized_sha256','decoded_sha256')
    seen = set()
    gate = []
    for original, restored_row in zip(o['rows'], r['rows'], strict=True):
        identity = (original['kind'], original['ordinal'])
        if identity in seen or any(original[k] != restored_row[k] for k in common):
            raise ValueError('Duplicate or mismatched restore input/output')
        seen.add(identity)
        if not np.array_equal(ordinary_arrays(ordinary_dir, original), ordinary_arrays(restored_dir, restored_row)):
            raise ValueError('Independent decoded arrays differ')
        if original['kind'] == 'target':
            gate.append(bool(original['metrics']['H40_and_seam_pass']))
    if Counter(k for k,_ in seen) != dict(target=32, robodojo=8, libero=8):
        raise ValueError('Unexpected 48-input population')
    return dict(plan=PLAN, role=role, source=bound['source'], parent=PARENT, qualified=True,
                training=train, ordinary_record=ordinary, restore=restored, first_step=first_pin,
                checkpoint=t['checkpoint'], gpu=site['site']['gpu_uuid'], job=site['site']['job'],
                reference_sha256=first['reference_sha256'], warm_p50_s=warm,
                warm_p90_s=float(np.percentile(clocks[4:],90)), first_update_s=clocks[0],
                warm_updates_per_hour=3600/warm, warm_FM_windows_per_s=sum(BATCH.values())/warm,
                H40_and_seam_pass=[sum(gate),len(gate)], optimizer_updates=16,
                visits=t['visits'], training_generations=64, panel_attempts=96)


def physical_delta(candidate, reference):
    c, r = pinned(candidate), pinned(reference)
    if len(c['rows']) != len(r['rows']):
        raise ValueError('Different physical comparison panels')
    result = {}
    for row, ref in zip(c['rows'], r['rows'], strict=True):
        if any(row[k] != ref[k] for k in ('kind','ordinal','identity','input_sha256','noise_sha256')):
            raise ValueError('Cross-candidate physical comparison uses different examples')
        a = ordinary_arrays(Path(candidate['path']).parent,row)
        b = ordinary_arrays(Path(reference['path']).parent,ref)
        if a.shape != b.shape or a.ndim != 2:
            raise ValueError('Decoded time axis or action dimensions disagree')
        delta = np.max(np.abs(a.astype(np.float64)-b),axis=0)
        kind = row['kind']
        result[kind] = np.maximum(result.get(kind, np.zeros_like(delta)), delta)
    return {k:v.tolist() for k,v in result.items()}


def winner(candidates, source):
    if set(candidates) != set(CANDIDATES):
        raise ValueError('Six original candidate dispositions are required')
    eligible = []
    for role in CANDIDATES:
        result = pinned(candidates[role])
        if result['plan'] != PLAN or result['parent'] != PARENT or result['role'] != role or result['source'] != source:
            raise ValueError('Candidate source, parent or identity changed')
        if result['qualified']:
            if not math.isfinite(result['warm_p50_s']) or result['warm_p50_s'] <= 0:
                raise ValueError('Invalid complete-update p50')
            eligible.append((result['warm_p50_s'], role))
        elif result['status'] not in ('NUMERICAL_REJECT','GPU_MEMORY_EXHAUSTED','NO_ENGINEERING_BUDGET'):
            raise ValueError('Unreconciled candidate disposition')
    if not eligible:
        raise ValueError('No qualified H200 backend')
    return min(eligible)[1]


def select(engineering_root, source):
    with closing(OwnerLock(engineering_root/'selection.lock')):
        candidates = {r: pin(engineering_root/f'{r}.outcome.json') for r in CANDIDATES}
        fastest = winner(candidates, source)
        reference = pinned(candidates['E1'])
        fm = {k:candidates[k] for k in FM_WIDTH}
        deltas = {}
        for role in CANDIDATES:
            result = pinned(candidates[role])
            if not result['qualified']:
                continue
            t = pinned(result['training'])
            if (t['binding']['fm_micro'], t['binding']['aux_micro']) != backend(role, fm):
                raise ValueError('Backend width differs from the registered candidate')
            if role != 'E1' and result['reference_sha256'] != reference['first_step']['sha256']:
                raise ValueError('Candidate was not compared with the original E1 backward pass')
            if reference.get('ordinary_record'):
                deltas[role] = physical_delta(result['ordinary_record'], reference['ordinary_record'])
        chosen = pinned(candidates[fastest])
        record = dict(plan=PLAN, source=source, parent=PARENT, winner=fastest, candidates=candidates,
                      warm_p50_s=chosen['warm_p50_s'], warm_p90_s=chosen['warm_p90_s'],
                      speedup_vs_E1=reference['warm_p50_s']/chosen['warm_p50_s'] if reference['qualified'] else None,
                      max_abs_decoded_by_domain_channel=deltas)
        write_once(engineering_root/'SELECTION.json',record)
        return record


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('engineering_root',type=Path)
    cli.add_argument('source')
    args=cli.parse_args()
    print(select(args.engineering_root,args.source)['winner'])
