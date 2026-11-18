"""Freeze D4 development layouts without starting a native world."""
from copy import deepcopy

from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from .multitask_d2_population import position_id
from .multitask_population_v6 import compile_population


def compile_d4(spec, earlier):
    occupied = {(c['family'], position_id(c['card'])) for bank in earlier for c in bank['clusters']}
    prior_resets = {row['reset'] for bank in earlier for rows in bank['conditions'].values() for row in rows}
    prior_hashes = {c['scene']['sha256'] for bank in earlier for c in bank['clusters']}
    clusters, matches = [], []
    for family_index, family in enumerate(('F1', 'F2')):
        for layout_index, (dx, dy) in enumerate(((.002,.006),(-.007,-.013))):
            card = deepcopy(spec['prototype_cards'][family])
            for position in card['object_positions_world'].values():
                position[0] += dx
                position[1] += dy
            reset = 690000 + 10*family_index + layout_index
            key = (family, position_id(card))
            if key in occupied or reset in prior_resets:
                raise ValueError('D4_DEV_GEOMETRY_OR_RESET_REUSES_HISTORICAL_POPULATION')
            occupied.add(key)
            cluster = f'D4-DEV-{family}-{layout_index+1}'
            clusters.append(dict(id=cluster, stage='D4_DEV', family=family, card=card, reset=reset))
            for task_index, instruction in enumerate(INSTRUCTIONS):
                arms = ['D4-A','D4-B'] if (layout_index+family_index+task_index)%2==0 else ['D4-B','D4-A']
                matches.append(dict(id=f'{cluster}-{instruction}', cluster=cluster, family=family,
                    reset=reset, instruction=instruction, noise_seed=2026111820+16*reset+task_index,
                    arm_order=arms))
    proposed = dict(schema='SLA_D4_C20_G7_FRESH_DEV_LAYOUTS', clusters=clusters,
        conditions={'D4_DEV':matches}, new_science_request_planned=48, historical_TEST_reused=False,
        scene_instruction_independent=True, native_dispatch_qualified=False)
    result = compile_population(spec, proposed=proposed)
    if prior_hashes.intersection(row['scene']['sha256'] for row in clusters):
        raise ValueError('D4_DEV_COMPILED_SCENE_REUSES_HISTORICAL_BYTES')
    result['status'] = 'COLD_D4_DEV_POPULATION_SEALED'
    return result
