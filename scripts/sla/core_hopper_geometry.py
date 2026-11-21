"""Compile new split-bound scenes on Core using the reviewed MuJoCo compiler."""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

from scripts.pi0292.durable import write
from scripts.wm02.multitask_binding_audit_v6 import pinned
from scripts.wm02.multitask_d2_population import position_id
from scripts.wm02.multitask_population_v6 import compile_population

PLAN = '171b5bedb290258954aac140157743e92e3b152e'
POPULATION = '9e483c85eb90cc2e9d1da66b9f3f8dff6606c6173b66755bcffcc4c7e5f91279'
BASE_SPEC = '4185a00c6ea6f868394ed8e550c9a720b2a0f3490c0765b8014df3c14c3d5ef9'
OLD_HOLD = '297b21cc67baa4b2cbdd78af68d244b550337b4cb9213fd27b26818e3958bec0'


def checked(ref):
    return json.loads(pinned(ref).read_bytes())


def binding(path, digest):
    return {'path': str(path), 'sha256': digest}


def build(rows_by_split, prototypes, history, excluded_offsets):
    # Offsets in mm lie within the reviewed target-adaptation envelope [-13,12].
    # Both cubes translate with the scene. The instruction never changes the layout.
    grid = list(product(range(-13, 13, 2), repeat=2))
    old = [c for bank in history for c in bank['clusters']]
    occupied = {(c['family'], position_id(c['card'])) for c in old}
    resets = {c['reset'] for c in old}
    scenes, conditions = {}, {}
    for split, rows in rows_by_split.items():
        conditions[split] = []
        for row in rows:
            sid, family, reset = row['scene_family'], row['family'], row['reset_seed']
            if sid not in scenes:
                seed = row['scene_seed']
                offsets = sorted(grid, key=lambda xy: hashlib.sha256(
                    f"{PLAN}/{seed}/{xy[0]}/{xy[1]}".encode()).digest())
                for dx, dy in offsets:
                    if (family, (dx, dy)) in excluded_offsets:
                        continue
                    card = deepcopy(prototypes[family])
                    for xyz in card['object_positions_world'].values():
                        xyz[0] += dx / 1000
                        xyz[1] += dy / 1000
                    key = (family, position_id(card))
                    if key not in occupied:
                        break
                else:
                    raise ValueError('No disjoint placement within reviewed bounds')
                if reset in resets:
                    raise ValueError('Predeclared reset overlaps another scene')
                occupied.add(key)
                resets.add(reset)
                scenes[sid] = dict(id=sid, stage=split, split=split, family=family,
                                   reset=reset, scene_seed=seed, card=card, offset_mm=[dx, dy])
            c = scenes[sid]
            if (c['family'], c['reset'], c['split'], c['scene_seed']) != (
                    family, reset, split, row['scene_seed']):
                raise ValueError('Scene family reused inconsistently')
            conditions[split].append(dict(row, id=row['condition_id'], cluster=sid, reset=reset))
    for split, n in {'TRAIN':240, 'VAL':24, 'DEV':24, 'CONF':24, 'TECH':2}.items():
        rs = conditions[split]
        if len(rs) != n or len({r['id'] for r in rs}) != n:
            raise ValueError('Condition membership mismatch')
        for count in Counter(r['cluster'] for r in rs).values():
            if count != (1 if split == 'TECH' else 6):
                raise ValueError('Scene instruction count mismatch')
    return dict(schema='SLA_CORE_HOPPER_NUMERIC_POPULATIONS_V1', plan=PLAN, conditions=conditions,
                clusters=list(scenes.values()), old_TEST_reused=False, assignment_uses_outcomes=False,
                native_dispatch_qualified=False)


def run(run_dir, baseline, source):
    mount = json.loads((run_dir/'MOUNT.private.json').read_text())
    if mount['plan'] != PLAN or time.time() >= mount['deadline_epoch']:
        raise ValueError('Campaign identity or deadline')
    spec = checked(binding(baseline/'COLD_INPUT.private.json', BASE_SPEC))
    logic = checked(binding(run_dir/'POPULATIONS.json', POPULATION))
    inventory = checked(spec['split_inventory'])
    refs = list(inventory['original_populations']) + [inventory['new_evaluation_population']]
    refs.extend(ref for key,ref in spec.items() if key.endswith('_population'))
    refs.append(binding(baseline/'population/COMPILED_POPULATION.private.json', OLD_HOLD))
    history = [checked(ref) for ref in {r['sha256']:r for r in refs}.values()]
    c20 = checked(inventory['C20_scene_catalog'])
    v52 = checked(inventory['V52_split'])['layouts']
    protected = set(inventory['protected_scene_hashes'])
    protected.update(c['scene']['sha256'] for bank in history for c in bank['clusters'])
    protected.update(c['sha256'] for c in c20)
    protected.update(c['scene']['sha256'] for c in v52)
    excluded = {(c['family'], tuple(c['xy_mm'])) for c in c20}
    excluded.update((c['family'], tuple(c['xy_offset_mm'])) for c in v52)
    proposed = build(logic['populations'], spec['prototype_cards'], history, excluded)
    root = run_dir/'population'
    root.mkdir(mode=0o700)
    spec.update(registered_root=str(root), deadline_epoch=min(time.time()+3600,mount['deadline_epoch']),
                physics=False, new_physical_requests_budget=0, new_model_generations_budget=0)
    report = compile_population(spec, proposed=proposed)
    result = json.loads((root/'COMPILED_POPULATION.private.json').read_text())
    if any(c['scene']['sha256'] in protected for c in result['clusters']):
        raise ValueError('Compiled scene matches a protected historical scene')
    if report['protection_fingerprints'] != ['67d4b804c438beabec9790ecf40b60584d2b45ec0c5f1799f377bf784f316bc8']:
        raise ValueError('Frozen robot fingerprint mismatch')
    write(run_dir/'RESULT.private.json',
          dict(report, status='COLD_NUMERIC_GEOMETRY_VERIFIED', goal_plan=PLAN, source_sha=source,
               historical_hashes=len(protected), condition_membership_sha256=POPULATION,
               new_native_requests=0, new_native_ticks=0, new_generations=0, new_updates=0))


if __name__ == '__main__':
    root, baseline, source = sys.argv[1:]
    try:
        run(Path(root), Path(baseline), source)
    except Exception as exc:
        write(Path(root)/'FAILURE.private.json', dict(first_failure=str(exc), new_native_requests=0,
                                                     new_native_ticks=0))
        raise
