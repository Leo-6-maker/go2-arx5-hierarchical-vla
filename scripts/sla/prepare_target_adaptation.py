"""Cold-register disjoint Go2 TRAIN conditions on the original collector."""
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model import target_adaptation_teacher as adaptation
from scripts.pi0292.durable import read, write
from scripts.wm02.multitask_binding_audit_v6 import pinned
from scripts.wm02.multitask_d2_population import position_id
from scripts.wm02.multitask_population_v6 import compile_population


OFFSETS_MM = sorted(product((-13, -8, -3, 2, 7, 12), repeat=2),
                    key=lambda offset: (sum(axis**2 for axis in offset), offset))


def population(cards, history, catalog, *, scope=adaptation.SCOPE):
    if set(cards) != {'F1', 'F2'}:
        raise ValueError('Both original scene families are required')
    archived = [cluster for bank in history for cluster in bank['clusters']]
    occupied = {(c['family'], position_id(c['card'])) for c in archived}
    offsets = {(c['family'], tuple(c['xy_mm'])) for c in catalog}
    resets = {c['reset'] for c in archived}
    considered = {family: iter(OFFSETS_MM) for family in cards}
    clusters, rows = [], []
    plan, _, stage = adaptation.SCOPES[scope]
    lineage = plan + ':' + stage
    reset_base = {adaptation.SCOPE:731000,adaptation.SECOND_SCOPE:731100,adaptation.RELEASE_SCOPE:731200,
                  adaptation.HOLD_SCOPE:731300,adaptation.GOAL_HOLD_SCOPE:731400}[scope]
    for index, (family, instruction) in enumerate(adaptation.batch_cells(scope)):
        obstacle = {}
        if scope == adaptation.GOAL_HOLD_SCOPE:
            distractor = 'blue' if instruction == 'HOLD-red' else 'red'
            shift = -8 if index < 2 else 8
            obstacle = dict(distractor=distractor,distractor_delta_y_mm=shift)
        for dx, dy in considered[family]:
            if (family, (dx, dy)) in offsets:
                continue
            card = deepcopy(cards[family])
            for xyz in card['object_positions_world'].values():
                xyz[0] += dx / 1000
                xyz[1] += dy / 1000
            if obstacle:
                card['object_positions_world'][distractor][1] += shift / 1000
            identity = family, position_id(card)
            if identity not in occupied:
                break
        else:
            raise ValueError(f'No unused geometry for {family} {instruction}')
        occupied.add(identity)
        reset = reset_base + index
        if reset in resets:
            raise ValueError('TRAIN reset overlaps a registered condition')
        cluster = f'{stage}-{family}-{index+1:02}'
        clusters.append(dict(id=cluster, stage=stage, split='TRAIN', family=family, card=card,
                             reset=reset, offset_mm=[dx, dy], collection_lineage=lineage,**obstacle))
        rows.append(dict(id=f'{cluster}-{instruction}', cluster=cluster, instruction=instruction,
                         family=family, reset=reset, noise_seed=2026111900+16*reset))
    return dict(schema='SLA_SCALE_TARGET_ADAPTATION_POPULATION_V1', plan=plan,
                conditions={stage: rows}, clusters=clusters, old_TEST_reused=False,
                assignment_uses_outcomes=False, native_dispatch_qualified=False)


def prepare(spec):
    scope = spec.get('scope',adaptation.SCOPE)
    plan, _, stage = adaptation.SCOPES[scope]
    if spec['review_plan'] != plan:
        raise ValueError('Expected the fixed SCALE adaptation collection plan')
    split = read(pinned(spec['split_inventory']))
    histories = [read(pinned(binding)) for binding in (
        *split['original_populations'], split['new_evaluation_population'])]
    if scope != adaptation.SCOPE:
        prior = read(pinned(spec['first_train_population']))
        adaptation.conditions(prior)
        histories.append(prior)
    if scope in (adaptation.RELEASE_SCOPE,adaptation.HOLD_SCOPE,adaptation.GOAL_HOLD_SCOPE):
        prior = read(pinned(spec['second_train_population']))
        adaptation.conditions(prior,adaptation.SECOND_SCOPE)
        histories.append(prior)
    if scope in (adaptation.HOLD_SCOPE,adaptation.GOAL_HOLD_SCOPE):
        release = read(pinned(spec['release_train_population']))
        adaptation.conditions(release,adaptation.RELEASE_SCOPE)
        histories.append(release)
        from scripts.sla.prepare_adapt5k_native_population import validate

        evaluation = read(pinned(spec['ADAPT5K_evaluation_population']))
        validate(evaluation)
        histories.append(evaluation)
    if scope == adaptation.GOAL_HOLD_SCOPE:
        d2 = read(pinned(spec['previous_hold_train_population']))
        adaptation.conditions(d2,adaptation.HOLD_SCOPE)
        histories.append(d2)
        for key, identity in (('SCALE_evaluation_population','86ad22180a0e1fd937fdff28302e190fef65928c0d8b724de282b6c8e0eb0c4a'),
                              ('CLOCK_evaluation_population','f80d61a58c2ad20e2d2f59ccba645bf0c5a61cfc8a19d3f53862651e5c76d353')):
            if spec[key]['sha256'] != identity:
                raise ValueError('Expected the fixed SCALE and CLOCK protected populations')
            histories.append(read(pinned(spec[key])))
    catalog = read(pinned(split['C20_scene_catalog']))
    v52 = read(pinned(split['V52_split']))
    protected = set(split['protected_scene_hashes'])
    protected.update(c['sha256'] for c in catalog)
    protected.update(c['scene']['sha256'] for c in v52['layouts'])
    protected.update(c['scene']['sha256'] for bank in histories for c in bank['clusters'])
    root = Path(spec['registered_root'])
    if root.exists():
        raise ValueError('The cold output must have a new identity')
    proposed = population(spec['prototype_cards'], histories, catalog, scope=scope)
    root.mkdir(mode=0o700)
    start = compile_population(spec, proposed=proposed)
    compiled = read(root/'COMPILED_POPULATION.private.json')
    if any(c['scene']['sha256'] in protected for c in compiled['clusters']):
        raise ValueError('New TRAIN geometry overlaps a historical compiled scene')
    result = dict(status={adaptation.SCOPE:'COLD_TRAIN_PLACE_FIRST6_COMPILED',
                         adaptation.SECOND_SCOPE:'COLD_TRAIN_PLACE_SECOND6_COMPILED',
                         adaptation.RELEASE_SCOPE:'COLD_TRAIN_PLACE_RELEASE_OFFSET_COMPILED',
                         adaptation.HOLD_SCOPE:'COLD_TRAIN_HOLD_PAIR_COMPILED',
                         adaptation.GOAL_HOLD_SCOPE:'COLD_GOAL_V2_TRAIN_HOLD4_COMPILED'}[scope],
                  plan=plan, source_sha=spec['execution_source_sha'],
                  split_inventory_sha256=spec['split_inventory']['sha256'],
                  compiled_population_sha256=start['population_sha256'],
                  compared_populations=len(histories),
                  compared_historical_scene_hashes=len(protected),
                  scene_bytes=start['actual_scene_bytes'],
                  robot_fingerprints=start['protection_fingerprints'],
                  conditions=[dict(family=c['family'], instruction=r['instruction'], reset=c['reset'],
                                   offset_mm=c['offset_mm'], scene_sha256=c['scene']['sha256'])
                              for c, r in zip(compiled['clusters'], compiled['conditions'][stage], strict=True)],
                  old_TEST_rollouts_read=False, native_requests=0, native_ticks=0, model_generations=0,
                  optimizer_updates=0, admitted_demonstrations=0, native_dispatch_qualified=False)
    if scope == adaptation.GOAL_HOLD_SCOPE:
        for row, cluster in zip(result['conditions'],compiled['clusters'],strict=True):
            row.update(distractor=cluster['distractor'],distractor_delta_y_mm=cluster['distractor_delta_y_mm'])
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
