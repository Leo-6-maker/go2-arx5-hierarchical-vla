"""Cold-register six missing Go2 PLACE cells for the next TRAIN version."""
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.target_adaptation_teacher import CELLS, PLAN as REVIEW_PLAN, STAGE
from scripts.pi0292.durable import read, write
from scripts.wm02.multitask_binding_audit_v6 import pinned
from scripts.wm02.multitask_d2_population import position_id
from scripts.wm02.multitask_population_v6 import compile_population


OFFSETS_MM = sorted(product((-13, -8, -3, 2, 7, 12), repeat=2),
                    key=lambda offset: (sum(axis**2 for axis in offset), offset))


def population(cards, history, catalog):
    if set(cards) != {'F1', 'F2'}:
        raise ValueError('Both original scene families are required')
    archived = [cluster for bank in history for cluster in bank['clusters']]
    occupied = {(c['family'], position_id(c['card'])) for c in archived}
    offsets = {(c['family'], tuple(c['xy_mm'])) for c in catalog}
    resets = {c['reset'] for c in archived}
    considered = {family: iter(OFFSETS_MM) for family in cards}
    clusters, rows = [], []
    lineage = REVIEW_PLAN + ':' + STAGE
    for index, (family, instruction) in enumerate(CELLS):
        for dx, dy in considered[family]:
            if (family, (dx, dy)) in offsets:
                continue
            card = deepcopy(cards[family])
            for xyz in card['object_positions_world'].values():
                xyz[0] += dx / 1000
                xyz[1] += dy / 1000
            identity = family, position_id(card)
            if identity not in occupied:
                break
        else:
            raise ValueError(f'No unused geometry for {family} {instruction}')
        occupied.add(identity)
        reset = 731000 + index
        if reset in resets:
            raise ValueError('TRAIN reset overlaps a registered condition')
        cluster = f'{STAGE}-{family}-{index+1:02}'
        clusters.append(dict(id=cluster, stage=STAGE, split='TRAIN', family=family, card=card,
                             reset=reset, offset_mm=[dx, dy], collection_lineage=lineage))
        rows.append(dict(id=f'{cluster}-{instruction}', cluster=cluster, instruction=instruction,
                         family=family, reset=reset, noise_seed=2026111900+16*reset))
    return dict(schema='SLA_SCALE_TARGET_ADAPTATION_POPULATION_V1', plan=REVIEW_PLAN,
                conditions={STAGE: rows}, clusters=clusters, old_TEST_reused=False,
                assignment_uses_outcomes=False, native_dispatch_qualified=False)


def prepare(spec):
    if spec['review_plan'] != REVIEW_PLAN:
        raise ValueError('Expected the fixed SCALE adaptation review')
    split = read(pinned(spec['split_inventory']))
    histories = [read(pinned(binding)) for binding in (
        *split['original_populations'], split['new_evaluation_population'])]
    catalog = read(pinned(split['C20_scene_catalog']))
    v52 = read(pinned(split['V52_split']))
    protected = set(split['protected_scene_hashes'])
    protected.update(c['sha256'] for c in catalog)
    protected.update(c['scene']['sha256'] for c in v52['layouts'])
    protected.update(c['scene']['sha256'] for bank in histories for c in bank['clusters'])
    root = Path(spec['registered_root'])
    if root.exists():
        raise ValueError('The cold output must have a new identity')
    proposed = population(spec['prototype_cards'], histories, catalog)
    root.mkdir(mode=0o700)
    start = compile_population(spec, proposed=proposed)
    compiled = read(root/'COMPILED_POPULATION.private.json')
    if any(c['scene']['sha256'] in protected for c in compiled['clusters']):
        raise ValueError('New TRAIN geometry overlaps a historical compiled scene')
    result = dict(status='COLD_TRAIN_PLACE_FIRST6_COMPILED',
                  plan=REVIEW_PLAN, source_sha=spec['execution_source_sha'],
                  split_inventory_sha256=spec['split_inventory']['sha256'],
                  compiled_population_sha256=start['population_sha256'],
                  compared_populations=len(histories),
                  compared_historical_scene_hashes=len(protected),
                  scene_bytes=start['actual_scene_bytes'],
                  robot_fingerprints=start['protection_fingerprints'],
                  conditions=[dict(family=c['family'], instruction=r['instruction'], reset=c['reset'],
                                   offset_mm=c['offset_mm'], scene_sha256=c['scene']['sha256'])
                              for c, r in zip(compiled['clusters'], compiled['conditions'][STAGE], strict=True)],
                  old_TEST_rollouts_read=False, native_requests=0, native_ticks=0, model_generations=0,
                  optimizer_updates=0, admitted_demonstrations=0, native_dispatch_qualified=False)
    write(root/'ACTUAL.private.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    spec = Path(args.spec)
    if digest(spec) != args.sha256:
        raise ValueError('The adaptation preparation spec changed')
    print(json.dumps(prepare(read(spec)),allow_nan=False))


if __name__ == '__main__':
    main()
