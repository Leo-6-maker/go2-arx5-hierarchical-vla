"""Preregister disjoint D3 scenes and one-mechanism paired diagnostics."""

from copy import deepcopy
from itertools import product
import json

from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from .multitask_binding_audit_v6 import pinned
from .multitask_d2_population import position_id
from .multitask_population_v6 import compile_population

PAIR_TASKS = (
    ('F1', 'PLACE-red-left', 'INTERNAL_RELATIVE_CLEARANCE'),
    ('F2', 'HOLD-red', 'APPROACH_TOPOLOGY'),
    ('F2', 'PLACE-blue-right', 'EXTERNAL_CARRY_ROUTE'),
    ('F1', 'PLACE-red-right', 'SUPPORTED_RELEASE'),
    ('F1', 'HOLD-blue', 'DESCEND_CLOSE_PREFIX'),
    ('F2', 'HOLD-blue', 'MOBILE_HOLD_SIMULTANEITY'),
)
SCENES_PER_FAMILY = {'R2': 1, 'QUAL': 4, 'M2_SCREEN': 2, 'M2_CONFIRM': 2, 'M2_DEV': 5, 'M2_TEST': 5}

# Deterministic, small changes applied equally to both objects. No score,
# controller, path geometry, feasibility or image is consulted for assignment.
GRID_MM = tuple(sorted(product((-14, -9, -4, 1, 6, 11, 15), repeat=2),
                       key=lambda xy: (xy[0]**2 + xy[1]**2, xy[0], xy[1])))


def population(cards, previous):
    if set(cards) != {'F1', 'F2'}:
        raise ValueError('D3_REQUIRES_ORIGINAL_BOTH_SCENE_FAMILIES')
    used = {(group['family'], position_id(group['card']))
            for archive in previous for group in archive['clusters']}
    clusters = []
    conditions = {'R1': []}

    def scene(stage, family, descriptor):
        for x, y in GRID_MM:
            card = deepcopy(cards[family])
            for xyz in card['object_positions_world'].values():
                xyz[0] += x / 1000.
                xyz[1] += y / 1000.
            signature = family, position_id(card)
            if signature not in used:
                used.add(signature)
                break
        else:
            raise ValueError('D3_PREDECLARED_DISJOINT_SCENE_LATTICE_EXHAUSTED')
        identifier = f'D3-{stage}-{family}-{descriptor}'
        reset = 678000 + 10 * len(clusters)
        clusters.append(dict(id=identifier, stage=stage, family=family, card=card, reset=reset))
        return identifier, reset

    for index, (family, instruction, mechanism) in enumerate(PAIR_TASKS, 1):
        pair = f'P{index}'
        cluster, reset = scene('R1', family, pair)
        order = ('CONTROL', 'TREATMENT') if index % 2 else ('TREATMENT', 'CONTROL')
        for arm in order:
            conditions['R1'].append(dict(id=f'{cluster}-{arm}', cluster=cluster, instruction=instruction,
                                        family=family, reset=reset, noise_seed=2026111800 + reset * 16,
                                        arm=arm, pair=pair, mechanism=mechanism,
                                        ablated=mechanism if arm == 'CONTROL' else None))
    for stage, groups in SCENES_PER_FAMILY.items():
        rows = conditions[stage] = []
        for ordinal, family in product(range(1, groups+1), ('F1', 'F2')):
            cluster, reset = scene(stage, family, ordinal)
            for number, instruction in enumerate(INSTRUCTIONS):
                row = dict(id=f'{cluster}-{instruction}', cluster=cluster, instruction=instruction,
                           family=family, reset=reset, noise_seed=2026111800 + reset * 16 + number)
                if stage == 'M2_SCREEN':
                    row['arm_order'] = ['M2-BASE', 'M2-SRC'] if len(rows) % 2 == 0 else ['M2-SRC', 'M2-BASE']
                rows.append(row)
    if len(clusters) != 44:
        raise ValueError('D3_EXACT_PREREGISTERED_LAYOUT_COUNT')
    return dict(schema='SLA_D3_DISJOINT_MULTITASK_POPULATION', clusters=clusters, conditions=conditions,
                scene_instruction_independent=True, old_TEST_reused=False,
                development_in_TRAIN=False, native_dispatch_qualified=False)


def compile_d3(spec):
    archives = [json.loads(pinned(pin).read_bytes()) for pin in spec['sealed_prior_populations']]
    proposed = population(spec['prototype_cards'], archives)
    result = compile_population(spec, proposed=proposed)
    result['status'] = 'PREREGISTERED_D3_POPULATION_COMPILED_WITH_ZERO_INTEGRATION'
    return result
