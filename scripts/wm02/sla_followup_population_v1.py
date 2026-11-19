"""Cold, separated layouts and arm ordering for the D5 and J panels."""
from collections import Counter
from copy import deepcopy

from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from .multitask_d2_population import position_id
from .multitask_population_v6 import compile_population

PLAN = 'cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb'
SCHEMA = 'SLA_D5_AND_J_FROZEN_NEW_LAYOUTS_V1'
OFFSETS_MM = {
    'D5_GRASP24': ((4, -12), (-10, 12)),
    'D5_CONFIRM32': ((1, -14), (12, 8), (-11, -7), (9, -5)),
    'J_SCREEN48': ((-5, 10), (6, 12)),
}
ARMS = {
    'D5_GRASP24': ('D4-B', 'D5-W', 'D5-T'),
    'D5_CONFIRM32': ('D4-B', 'SELECTED_D5'),
    'J_SCREEN48': ('J0', 'J1'),
}


def population(cards, prior):
    occupied = {(c['family'], position_id(c['card'])) for bank in prior for c in bank['clusters']}
    resets = {c['reset'] for bank in prior for c in bank['clusters']}
    clusters, conditions = [], {}
    for stage_number, (stage, offsets) in enumerate(OFFSETS_MM.items()):
        arms = ARMS[stage]
        rows = conditions[stage] = []
        instructions = INSTRUCTIONS if stage == 'J_SCREEN48' else INSTRUCTIONS[:2]
        for index, (dx, dy) in enumerate(offsets):
            for family_number, family in enumerate(('F1', 'F2')):
                card = deepcopy(cards[family])
                for xyz in card['object_positions_world'].values():
                    xyz[0] += dx / 1000
                    xyz[1] += dy / 1000
                geometry = family, position_id(card)
                reset = 700000+1000*stage_number+10*index+family_number
                if geometry in occupied or reset in resets:
                    raise ValueError('FOLLOWUP_LAYOUT_OR_RESET_OVERLAPS_AN_ARCHIVED_POPULATION')
                occupied.add(geometry)
                resets.add(reset)
                layout = f'{stage}-{family}-{index+1}'
                clusters.append(dict(id=layout, stage=stage, family=family, card=card, reset=reset))
                for prompt, instruction in enumerate(instructions):
                    cyclic = len(rows) % len(arms)
                    rows.append(dict(id=f'{layout}-{instruction}', cluster=layout, instruction=instruction,
                                     family=family, reset=reset, noise_seed=2026111900+16*reset+prompt,
                                     arm_order=arms[cyclic:]+arms[:cyclic]))
    return dict(schema=SCHEMA, plan_commit=PLAN, clusters=clusters, conditions=conditions,
                scene_instruction_independent=True, old_TEST_reused=False, development_in_TRAIN=False,
                native_dispatch_qualified=False)


def validate_frozen(bank):
    if (bank['schema'] != SCHEMA or bank['plan_commit'] != PLAN or bank['old_TEST_reused']
            or bank['development_in_TRAIN'] or not bank['scene_instruction_independent']):
        raise ValueError('D5_AND_J_REQUIRE_THE_PREDECLARED_NEW_POPULATION')
    clusters = {c['id']:c for c in bank['clusters']}
    if (len(clusters) != 16 or len({c['reset'] for c in clusters.values()}) != 16
            or len({c['scene']['sha256'] for c in clusters.values()}) != 16):
        raise ValueError('FOLLOWUP_REQUIRES_16_DISTINCT_FROZEN_SCENES')
    for stage, arms in ARMS.items():
        instructions = INSTRUCTIONS if stage == 'J_SCREEN48' else INSTRUCTIONS[:2]
        rows = bank['conditions'][stage]
        stage_layouts = {c['id']:c for c in clusters.values() if c['stage'] == stage}
        expected = {(layout, task) for layout in stage_layouts for task in instructions}
        if (Counter(c['family'] for c in stage_layouts.values()) != {family:len(OFFSETS_MM[stage]) for family in ('F1','F2')}
                or len(expected) != len(rows) or {(r['cluster'],r['instruction']) for r in rows} != expected):
            raise ValueError('WRONG_FOLLOWUP_LAYOUT_OR_TASK_DENOMINATOR')
        for index, row in enumerate(rows):
            cyclic = index % len(arms)
            c = stage_layouts[row['cluster']]
            if (list(row['arm_order']) != list(arms[cyclic:]+arms[:cyclic])
                    or (row['family'],row['reset']) != (c['family'],c['reset'])):
                raise ValueError('FROZEN_FOLLOWUP_ARM_ORDER_OR_SCENE_MISMATCH')


def freeze(spec, prior):
    import json
    from pathlib import Path

    earlier = {c['scene']['sha256'] for bank in prior for c in bank['clusters']}
    result = compile_population(spec, proposed=population(spec['prototype_cards'], prior))
    bank = json.loads((Path(spec['registered_root'])/'COMPILED_POPULATION.private.json').read_bytes())
    validate_frozen(bank)
    if earlier.intersection(c['scene']['sha256'] for c in bank['clusters']):
        raise ValueError('FOLLOWUP_COLD_SCENE_BYTES_REUSE_PRIOR_POPULATION')
    return result
