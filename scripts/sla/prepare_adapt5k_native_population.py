"""Freeze the independent ADAPT5K native population without launching a world."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from scripts.pi0292.durable import write
from scripts.wm02.multitask_d2_population import position_id
from scripts.wm02.multitask_population_v6 import compile_population


PLAN = '5734a73cbda5c333cc3d7d7e0f032e679a4c08b6'
SCHEMA = 'SLA_ADAPT5K_FROZEN_NATIVE_POPULATION_V1'
STAGES = {'ADAPT_DEV': ((10,18),(-18,7)), 'ADAPT_CONFIRM': ((18,4),(-7,-18))}


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def scene_identity(cluster):
    return cluster['family'],position_id(cluster['card'])


def population(prototypes, banks):
    if set(prototypes) != {'F1','F2'}:
        raise ValueError('ADAPT5K_NATIVE_REQUIRES_BOTH_FROZEN_FAMILIES')
    occupied = {scene_identity(row) for bank in banks for row in bank['clusters']}
    resets = {row['reset'] for bank in banks for row in bank['clusters']}
    seeds = {row['noise_seed'] for bank in banks for rows in bank['conditions'].values() for row in rows}
    conditions, clusters = {}, []
    for stage_index, (stage, offsets) in enumerate(STAGES.items()):
        conditions[stage] = []
        for layout_index, (dx,dy) in enumerate(offsets):
            for family_index,family in enumerate(('F1','F2')):
                card = deepcopy(prototypes[family])
                for place in card['object_positions_world'].values():
                    place[0] += dx/1000
                    place[1] += dy/1000
                reset = 732000 + 1000*stage_index + 10*layout_index + family_index
                cluster = dict(id=f'{stage}-{family}-{layout_index+1}',stage=stage,family=family,card=card,
                               reset=reset,split='DEV' if stage == 'ADAPT_DEV' else 'CONFIRM',
                               collection_lineage=f'{PLAN}:{stage}')
                key = scene_identity(cluster)
                if reset in resets or key in occupied:
                    raise ValueError('ADAPT5K_NATIVE_GEOMETRY_OR_RESET_OVERLAPS_AN_EXISTING_SPLIT')
                resets.add(reset)
                occupied.add(key)
                clusters.append(cluster)
                for n,instruction in enumerate(INSTRUCTIONS):
                    seed = 2026112000 + 16*reset + n
                    if seed in seeds:
                        raise ValueError('ADAPT5K_NATIVE_REUSES_GRADER_NOISE_IDENTITY')
                    seeds.add(seed)
                    conditions[stage].append(dict(id=f"{cluster['id']}-{instruction}",cluster=cluster['id'],
                        instruction=instruction,family=family,reset=reset,noise_seed=seed))
    return dict(schema=SCHEMA,plan=PLAN,conditions=conditions,clusters=clusters,
                dev_phase_steps=[0,2000,5000],confirm_phase_step=5000,
                scene_instruction_independent=True,assignment_uses_outcomes=False,old_TEST_reused=False,
                native_dispatch_qualified=False)


def validate(frozen):
    if (frozen['schema'] != SCHEMA or frozen['plan'] != PLAN or frozen['dev_phase_steps'] != [0,2000,5000]
            or frozen['confirm_phase_step'] != 5000 or frozen['scene_instruction_independent'] is not True
            or frozen['assignment_uses_outcomes'] or frozen['old_TEST_reused'] or frozen['native_dispatch_qualified']
            or set(frozen['conditions']) != set(STAGES) or len(frozen['clusters']) != 8
            or len({r['scene']['sha256'] for r in frozen['clusters']}) != 8
            or len({r['reset'] for r in frozen['clusters']}) != 8):
        raise ValueError('ADAPT5K_FIXED_DEV_AND_CONFIRM_NATIVE_POPULATION_REQUIRED')
    for stage in STAGES:
        rows = frozen['conditions'][stage]
        groups = {c['id']:c for c in frozen['clusters'] if c['stage'] == stage}
        if (Counter(c['family'] for c in groups.values()) != Counter(F1=2,F2=2)
                or Counter((r['cluster'],r['instruction']) for r in rows) != Counter(
                    (group,instruction) for group in groups for instruction in INSTRUCTIONS)
                or len({r['id'] for r in rows}) != 24 or len({r['noise_seed'] for r in rows}) != 24
                or any((r['family'],r['reset']) != (groups[r['cluster']]['family'],groups[r['cluster']]['reset']) for r in rows)):
            raise ValueError('ADAPT5K_REQUIRES_FOUR_LAYOUTS_X_SIX_INSTRUCTIONS_PER_SPLIT')


def run(spec):
    began = time.monotonic()
    root = Path(spec['registered_root'])
    if (spec['plan'] != PLAN or spec['physics'] is not False or spec['new_physical_requests_budget']
            or spec['new_model_generations_budget'] or root.exists() or spec['deadline_epoch'] < time.time()+90
            or len(spec['new_train_populations']) != 3):
        raise PermissionError('ADAPT5K_FRESH_COLD_NATIVE_SPLIT_ONLY')
    split = read_pin(spec['split_inventory'])
    banks = [read_pin(member) for member in (*split['original_populations'],split['new_evaluation_population'],
                                             *spec['new_train_populations'])]
    old_geometry = {scene_identity(c) for bank in banks for c in bank['clusters']}
    protected = {c['scene']['sha256'] for bank in banks for c in bank['clusters']}
    protected.update(split['protected_scene_hashes'])
    protected.update(x['sha256'] for x in read_pin(split['C20_scene_catalog']))
    protected.update(x['scene']['sha256'] for x in read_pin(split['V52_split'])['layouts'])
    plan = population(spec['prototype_cards'],banks)
    root.mkdir(mode=0o700)
    result = compile_population(spec,proposed=plan)
    sealed = pin(root/'COMPILED_POPULATION.private.json'); frozen = read_pin(sealed)
    validate(frozen)
    if any(row['scene']['sha256'] in protected or scene_identity(row) in old_geometry for row in frozen['clusters']):
        raise ValueError('ADAPT5K_COMPILED_SPLIT_MUST_REMAIN_DISJOINT')
    if result['protection_fingerprints'] != [spec['original_robot_fingerprint']]:
        raise ValueError('ADAPT5K_COLD_SPLIT_MUST_KEEP_ORIGINAL_ROBOT_FINGERPRINT')
    actual = dict(schema='SLA_ADAPT5K_COLD_NATIVE_POPULATION_ACTUAL_V1',status='COLD_SPLIT_FROZEN',
        population=sealed, compiled=dict(result,status='COLD_SPLIT_FROZEN'), plan=PLAN, source_sha=spec['source_sha'],
        protected_scene_hash_count=len(protected),protected_geometry_count=len(old_geometry),
        registered_native_science_cap=96, registered_episode_tick_cap=4387488, student_query_cap=15360,
        scored_native_requests=0, native_ticks=0, model_generations=0, optimizer_updates=0,
        old_TEST_rollouts_read=False, assignment_uses_outcomes=False, wall_s=time.monotonic()-began)
    write(root/'ACTUAL.private.json',actual)
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    print(json.dumps(run(read_pin(dict(path=args.spec,sha256=args.sha256))),allow_nan=False))
