"""Predeclared WM01 scenes. No native construction and no score selection."""
import copy
import numpy as np


def definitions():
    from scripts.pi0294.scene_cards import scene_group, task_card
    result = {}
    for split, count, start in (('TRAIN', 64, 110300), ('CALIB', 16, 110400),
                                ('DEV', 20, 110500), ('TEST', 50, 110600)):
        cards = []
        for i in range(count):
            group = scene_group('TRAIN', 'F0', start+i)
            group.update(group_id=f'WM01/{split}/{i:03d}', scene_group=f'WM01/{split}/{i:03d}', split=split)
            card = task_card(group, 'T1')
            card['parent_id'] = card['id']
            cards.append(card)
        result[split] = cards
    return result


def compare_geometry(cards, historical):
    from scripts.pi0294.scene_cards import near_duplicate_report
    exact = []
    for card in cards:
        for old in historical:
            if np.allclose([*card['source_xy'], *card['goal_xy'], card['cube_yaw']],
                           [*old['source_xy'], *old['goal_xy'], old['cube_yaw']], rtol=0, atol=1e-12):
                exact.append([card['id'], old.get('id', old.get('group_id'))])
    return dict(exact_duplicates=exact, near_duplicates=near_duplicate_report(cards, historical))


def validate_parent_splits(rows):
    parents = {}
    scenes = {}
    for row in rows:
        for key, mapping in (('parent_id', parents), ('scene_id', scenes)):
            if row['split'] not in ('TRAIN', 'CALIB', 'DEV', 'TEST'):
                raise ValueError('Unknown split')
            if mapping.setdefault(row[key], row['split']) != row['split']:
                raise ValueError('Parent/scene crosses data split')
