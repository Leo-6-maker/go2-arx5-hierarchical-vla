"""Cold compile fresh paired CLOCK DEV and fixed B-confirmation worlds."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

from go2arx5_vla.sla_adapter.clock500 import ARMS, PLAN
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write
from scripts.sla.prepare_adapt5k_native_population import layout_population, pin, read_pin, scene_identity
from scripts.wm02.multitask_population_v6 import compile_population


SCHEMA = 'SLA_CLOCK500_FROZEN_PAIRED_NATIVE_POPULATION_V1'
STAGES = {'CLOCK_DEV': ((12,14),(-16,9)), 'CLOCK_B_CONFIRM': ((14,16),(-9,-14))}
RESET_BASE = 736000


def proposed(prototypes, protected, *, scoring_source):
    conditions,clusters = layout_population(prototypes,protected,stages=STAGES,reset_base=RESET_BASE,plan=PLAN)
    order = []
    for ordinal,row in enumerate(conditions['CLOCK_DEV']):
        arms = ARMS[ordinal%2:]+ARMS[:ordinal%2]
        order.extend(dict(stage='CLOCK_DEV',condition=row['id'],arm=arm) for arm in arms)
    order.extend(dict(stage='CLOCK_B_CONFIRM',condition=row['id'],arm=ARMS[1])
                 for row in conditions['CLOCK_B_CONFIRM'])
    return dict(schema=SCHEMA,plan=PLAN,conditions=conditions,clusters=clusters,native_order=order,
        scoring_source=scoring_source,assignment_uses_outcomes=False,scene_instruction_independent=True,
        uses_existing_TEST=False,native_dispatch_qualified=False)


def validate(frozen):
    clusters = frozen['clusters']
    conditions = frozen['conditions']
    dev,confirm = conditions['CLOCK_DEV'],conditions['CLOCK_B_CONFIRM']
    if (frozen['schema']!=SCHEMA or frozen['plan']!=PLAN or frozen['uses_existing_TEST']
            or frozen['native_dispatch_qualified'] or frozen['assignment_uses_outcomes']
            or not frozen['scene_instruction_independent'] or len(clusters)!=8 or set(conditions)!=set(STAGES)
            or len({r['scene']['sha256'] for r in clusters})!=8 or len({r['reset'] for r in clusters})!=8
            or len(dev)!=24 or len(confirm)!=24 or len({row['noise_seed'] for row in dev+confirm})!=48):
        raise ValueError('CLOCK500_REQUIRES_FRESH_24_DEV_PAIRS_AND_24_CONFIRMATIONS')
    expected = [(row['id'],arm,'CLOCK_DEV') for i,row in enumerate(dev) for arm in ARMS[i%2:]+ARMS[:i%2]]
    expected.extend((row['id'],ARMS[1],'CLOCK_B_CONFIRM') for row in confirm)
    rows = [(r['condition'],r['arm'],r['stage']) for r in frozen['native_order']]
    if rows!=expected or len({r['id'] for r in dev+confirm})!=48:
        raise ValueError('CLOCK500_PAIRED_ORDER_OR_CONFIRMATION_MODEL_CHANGED')
    for stage,selection in conditions.items():
        clusters_for_stage = {c['id']:c for c in clusters if c['stage']==stage}
        if (Counter(c['family'] for c in clusters_for_stage.values())!=Counter(F1=2,F2=2)
                or Counter((r['cluster'],r['instruction']) for r in selection)!=Counter(
                    (cluster,instruction) for cluster in clusters_for_stage for instruction in INSTRUCTIONS)
                or any((r['reset'],r['family'])!=(clusters_for_stage[r['cluster']]['reset'],
                                              clusters_for_stage[r['cluster']]['family']) for r in selection)):
            raise ValueError('CLOCK500_LAYOUTS_MUST_MATCH_THE_FROZEN_PAIR_KEYS')


def run(spec):
    root = Path(spec['registered_root'])
    if (spec['plan']!=PLAN or spec['physics'] is not False or spec['new_physical_requests_budget']
            or spec['new_model_generations_budget'] or root.exists()
            or root.parent.stat().st_uid!=os.getuid() or spec['deadline_epoch']<time.time()+90):
        raise PermissionError('CLOCK500_ONLY_COLD_CPU_WORLD_COMPILATION_ALLOWED')
    inventory = read_pin(spec['split_inventory'])
    members = [*inventory['original_populations'],inventory['new_evaluation_population'],
                *spec['new_train_populations'],spec['ADAPT_original_population']]
    banks = [read_pin(member) for member in members]
    protected = {c['scene']['sha256'] for bank in banks for c in bank['clusters']}
    protected.update(inventory['protected_scene_hashes'])
    protected.update(x['sha256'] for x in read_pin(inventory['C20_scene_catalog']))
    protected.update(x['scene']['sha256'] for x in read_pin(inventory['V52_split'])['layouts'])
    geometry = {scene_identity(c) for bank in banks for c in bank['clusters']}
    scorer = spec['original_scoring_source']
    if scorer['sha256']!=digest(scorer['path']):
        raise ValueError('CLOCK500_ORIGINAL_SCORER_CHANGED')
    frozen = proposed(spec['prototype_cards'],banks,scoring_source=scorer)
    root.mkdir(mode=0o700)
    compiled = compile_population(spec,proposed=frozen)
    package = pin(root/'COMPILED_POPULATION.private.json')
    sealed = read_pin(package)
    validate(sealed)
    if (compiled['protection_fingerprints']!=[spec['original_robot_fingerprint']]
            or any(row['scene']['sha256'] in protected or scene_identity(row) in geometry for row in sealed['clusters'])):
        raise ValueError('CLOCK500_FROZEN_NATIVE_LAYOUTS_OVERLAP_OR_BREACH_ROBOT_IDENTITY')
    receipt = dict(status='CLOCK500_COLD_PAIRED_POPULATION_FROZEN',plan=PLAN,source_sha=spec['source_sha'],
        population=package,protected_scene_hash_count=len(protected),protected_geometry_count=len(geometry),
        compiled=compiled,DEV_cases_per_arm=24,B_confirmation_cases=24,future_native_cap=72,
        future_tick_cap=3290616,native_requests=0,native_ticks=0,optimizer_updates=0,model_generations=0)
    write(root/'ACTUAL.private.json',receipt)
    return receipt


if __name__=='__main__':
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    print(json.dumps(run(read_pin(dict(path=args.spec,sha256=args.sha256))),allow_nan=False))
