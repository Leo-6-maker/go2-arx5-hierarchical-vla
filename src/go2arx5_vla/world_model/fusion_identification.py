"""Predeclared whole-parent, balanced local response identification (no VLA)."""
from collections import Counter
import random
from .fusion_contract import OFFSETS

SCHEMA = 'WM02_FUSION_BALANCED_RESPONSE_PANEL_V1'


def panel(scenes, seed=366201):
    """Two independent restarts per action, balanced across the two CPU slots."""
    rows = []; rng = random.Random(seed)
    for scene in sorted(scenes, key=lambda s: (s['split'] != 'TRAIN', s['scene_id'])):
        slot0 = [(a, r) for a in range(4) for r in range(2) if (a+r) % 2 == 0]
        slot1 = [(a, r) for a in range(4) for r in range(2) if (a+r) % 2 == 1]
        rng.shuffle(slot0); rng.shuffle(slot1)
        for pair in zip(slot0, slot1):
            index = len(rows)//2
            for slot, (action, repeat) in enumerate(pair):
                rows.append(dict(parent=scene['parent'], scene_id=scene['scene_id'], split=scene['split'],
                    pair_index=index, slot=slot, nominal_action=action, repeat_index=repeat,
                    method='B0_DIRECT', desired_delta_xy_m=OFFSETS[action, :2].tolist(),
                    purpose='TRAIN_FEEDBACK' if scene['split'] == 'TRAIN' else 'EVALUATION',
                    inference_mode='CALIBRATION_NO_VLA', sampling_source='PREDECLARED_BALANCED_ACTION_IDENTIFICATION'))
    value = dict(schema=SCHEMA, seed=seed, rows=rows, TRAIN_parents=4, DEV_parents=2,
        planned_evaluations=48, planned_qualifications=6, actual_counterfactual_forks=False,
        task_failure_exclusion=False, eligibility='COMPLETE_NATIVE_PASS_AND_RESPONSE_EVIDENCE_WHOLE_PAIR',
        four_environment_execution=False, TEST_read=False)
    validate(value)
    return value


def validate(value):
    rows = value['rows']
    if (value.get('schema') != SCHEMA or len(rows) != 48
            or value.get('eligibility') != 'COMPLETE_NATIVE_PASS_AND_RESPONSE_EVIDENCE_WHOLE_PAIR'
            or value.get('task_failure_exclusion') is not False or value.get('TEST_read') is not False):
        raise ValueError('EXACT_48_BALANCED_TRAIN_DEV_RESPONSE_CONDITIONS_REQUIRED')
    parents = {}
    for row in rows:
        identity = (row['scene_id'], row['split'])
        if row['parent'] in parents and parents[row['parent']] != identity:
            raise ValueError('WHOLE_PHYSICAL_PARENT_SPLIT_AND_SCENE_REQUIRED')
        parents[row['parent']] = identity
        a = row['nominal_action']
        if (type(a) is not int or a not in range(4) or row['repeat_index'] not in (0, 1)
                or row['split'] not in ('TRAIN', 'DEV') or row['method'] != 'B0_DIRECT'
                or row['purpose'] != ('TRAIN_FEEDBACK' if row['split'] == 'TRAIN' else 'EVALUATION')
                or row['inference_mode'] != 'CALIBRATION_NO_VLA'
                or row['desired_delta_xy_m'] != OFFSETS[a, :2].tolist()
                or row.get('sampling_source') != 'PREDECLARED_BALANCED_ACTION_IDENTIFICATION'):
            raise ValueError('ONLY_ACTUAL_PREDECLARED_LOCAL_ACTION_RESPONSE_LABELS_ALLOWED')
    if Counter(s for _, s in parents.values()) != dict(TRAIN=4, DEV=2) or len({s for s,_ in parents.values()}) != 6:
        raise ValueError('FOUR_NEW_TRAIN_TWO_NEW_DEV_PARENTS_REQUIRED')
    for parent in parents:
        subset = [r for r in rows if r['parent'] == parent]
        if (Counter((r['nominal_action'], r['repeat_index']) for r in subset)
                != Counter((a, r) for a in range(4) for r in range(2))
                or Counter((r['nominal_action'], r['slot']) for r in subset)
                != Counter((a, s) for a in range(4) for s in range(2))):
            raise ValueError('EVERY_ACTION_TWO_RESTARTS_AND_BOTH_CPU_SLOTS_REQUIRED')
    for at in range(0, len(rows), 2):
        pair = rows[at:at+2]
        if ([r['slot'] for r in pair] != [0, 1]
                or any(r['pair_index'] != at//2 for r in pair)
                or len({r['parent'] for r in pair}) != 1):
            raise ValueError('EXACT_TWO_SLOT_SAME_PARENT_FINITE_PAIR_REQUIRED')
    if any(r['split'] == 'TRAIN' for r in rows[32:]) or any(r['split'] != 'TRAIN' for r in rows[:32]):
        raise ValueError('TRAIN_IDENTIFICATION_MUST_PRECEDE_DEV')
    return rows
