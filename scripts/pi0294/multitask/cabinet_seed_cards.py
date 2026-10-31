"""Four prospective TRAIN translations of the successful custom PORT08 fixture.

Only fixture XY changes. These cards identify candidates, not demonstrations,
native qualification, an original RoboCasa benchmark or target-WBC results.
"""
from .cabinet_braked_turn import cards as original_cards

VERSION = 'CUSTOM_CABINET_PORT08_FIXTURE_XY_TRAIN_SEEDS_V1'
OFFSETS_M = ((-.004, 0.), (.004, 0.), (0., -.004), (0., .004))


def seed_card(index):
    if type(index) is not int or not 1 <= index <= len(OFFSETS_M):
        raise ValueError('EXPLICIT_DECLARED_CUSTOM_CABINET_TRAIN_SEED')
    card = original_cards()['right']
    offset = OFFSETS_M[index-1]
    for frame in (card['cabinet']['cabinet_world'], card['door']['hinge_world']):
        for axis, amount in enumerate(offset):
            frame[axis][3] += amount
    for axis, amount in enumerate(offset):
        card['source_base_card']['goal_xy'][axis] += amount
    card.update(id=card['id']+f'_SEED{index:03}', cabinet_seed_index=index,
                source_seed_set=VERSION, fixture_translation_world_m=[*offset, 0.])
    return card
