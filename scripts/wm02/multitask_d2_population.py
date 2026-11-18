"""Cold, disjoint Teacher development, qualification and M2 evaluation layouts."""

from copy import deepcopy

from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from .multitask_population_v6 import compile_population

# Fixed before inspecting any new score. A scene has both objects and both
# destinations, and its six instructions share the exact same scene bytes.
OFFSETS = {
    'DEVELOPMENT': ((.004,.008),(-.005,-.011)),
    'QUAL': ((-.012,-.002),(.012,-.002),(.003,.007),(-.003,.009)),
    'M2_SCREEN': ((-.006,.004),(.011,.003)),
    'M2_CONFIRM': ((-.009,-.005),(.006,-.007)),
    'M2_DEV': ((-.014,.001),(-.011,.010),(.007,.013),(.013,-.011),(.002,-.015)),
    'M2_TEST': ((-.015,-.003),(-.012,.013),(.015,-.004),(.010,.014),(-.002,.016)),
}


def position_id(card):
    return tuple(tuple(round(axis,10) for axis in card['object_positions_world'][color])
                 for color in ('red','blue'))


def population(cards, prior):
    if set(cards)!={'F1','F2'}:
        raise ValueError('D2_REQUIRES_BOTH_FROZEN_SCENE_FAMILIES')
    previous = {(c['family'],position_id(c['card'])) for c in prior['clusters']}
    conditions = {}
    clusters = []
    for stage_number,(stage,offsets) in enumerate(OFFSETS.items()):
        conditions[stage] = rows = []
        for i,(dx,dy) in enumerate(offsets):
            for family_number,family in enumerate(('F1','F2')):
                card = deepcopy(cards[family])
                for xyz in card['object_positions_world'].values():
                    xyz[0] += dx
                    xyz[1] += dy
                position=(family,position_id(card))
                if position in previous:
                    raise ValueError('D2_SCENE_GEOMETRY_CROSSES_SPLITS')
                previous.add(position)
                cluster=f'D2-{stage}-{family}-{i+1}'
                reset=671000+1000*stage_number+10*i+family_number
                clusters.append(dict(id=cluster,stage=stage,family=family,card=card,reset=reset))
                for cell,instruction in enumerate(INSTRUCTIONS):
                    row = dict(id=f'{cluster}-{instruction}',cluster=cluster,
                               instruction=instruction,family=family,reset=reset,
                               noise_seed=2026111800+16*reset+cell)
                    if stage=='DEVELOPMENT':
                        row['block']=i+1
                    if stage=='M2_SCREEN':
                        row['arm_order']=['M2-BASE','M2-SRC'] if len(rows)%2==0 else ['M2-SRC','M2-BASE']
                    rows.append(row)
    return dict(schema='SLA_D2_NEW_DISJOINT_MULTITASK_POPULATIONS',clusters=clusters,conditions=conditions,
                scene_instruction_independent=True,old_TEST_reused=False,development_in_TRAIN=False,
                native_dispatch_qualified=False)


def compile_d2(spec):
    import json
    from .multitask_binding_audit_v6 import pinned

    original = json.loads(pinned(spec['prior_compiled_population']).read_bytes())
    result = population(spec['prototype_cards'],original)
    return compile_population(spec,proposed=result)
