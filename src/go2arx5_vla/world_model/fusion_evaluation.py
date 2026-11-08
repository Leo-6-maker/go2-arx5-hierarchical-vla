"""Frozen four-arm calibration panel, whole new DEV parents and original goals."""
from collections import Counter
import random
from .fusion_contract import OFFSETS
from .fusion_corrector import ARMS

SCHEMA='WM02_FUSION_FOUR_ARM_FROZEN_DEV_PANEL_V1'


def panel(scenes,seed=366501):
    rng=random.Random(seed);rows=[]
    for scene in sorted(scenes,key=lambda s:s['scene_id']):
        goals=list(range(4));rng.shuffle(goals)
        for goal in goals:
            slots=[[(m,r)for i,m in enumerate(ARMS)for r in range(2)if (i+r+goal)%2==s]for s in range(2)]
            for slot in slots:rng.shuffle(slot)
            for pair in zip(*slots):
                index=len(rows)//2
                for slot,(method,repeat)in enumerate(pair):
                    rows.append(dict(parent=scene['parent'],scene_id=scene['scene_id'],split=scene['split'],
                        pair_index=index,slot=slot,goal_index=goal,nominal_action=goal,repeat_index=repeat,
                        method=method,desired_delta_xy_m=OFFSETS[goal,:2].tolist(),purpose='EVALUATION',
                        inference_mode='CALIBRATION_NO_VLA',sampling_source='FROZEN_FOUR_ARM_NEW_DEV_PANEL'))
    value=dict(schema=SCHEMA,seed=seed,rows=rows,split='DEV',physical_parents=6,goal_indices=list(range(4)),
        methods=list(ARMS),primary_method=ARMS[3],baseline=ARMS[1],repeats=2,inference_minimum_parents=6,
        planned_evaluations=192,planned_qualifications=6,task_failure_exclusion=False,
        eligibility='COMPLETE_NATIVE_PASS_AND_RESPONSE_EVIDENCE_WHOLE_PAIR',TEST_read=False)
    validate(value);return value


def validate(value):
    rows=value['rows'];parents={}
    if (value.get('schema')!=SCHEMA or len(rows)!=192 or value.get('split')!='DEV'
            or value.get('methods')!=list(ARMS) or value.get('goal_indices')!=list(range(4))
            or value.get('baseline')!=ARMS[1] or value.get('primary_method')!=ARMS[3]
            or value.get('eligibility')!='COMPLETE_NATIVE_PASS_AND_RESPONSE_EVIDENCE_WHOLE_PAIR'
            or value.get('task_failure_exclusion')is not False or value.get('TEST_read')is not False):
        raise ValueError('EXACT_FROZEN_192_FOUR_ARM_DEV_CONDITIONS_REQUIRED')
    for row in rows:
        identity=(row['scene_id'],row['split'])
        if row['parent']in parents and parents[row['parent']]!=identity:raise ValueError('WHOLE_NEW_DEV_PHYSICAL_PARENT_REQUIRED')
        parents[row['parent']]=identity;g=row['goal_index']
        if (type(g)is not int or g not in range(4) or type(row['nominal_action'])is not int
                or row['nominal_action']!=g or row['desired_delta_xy_m']!=OFFSETS[g,:2].tolist()
                or row['split']!='DEV' or row['purpose']!='EVALUATION' or row['method']not in ARMS
                or row['repeat_index']not in (0,1) or row['inference_mode']!='CALIBRATION_NO_VLA'
                or row['sampling_source']!='FROZEN_FOUR_ARM_NEW_DEV_PANEL'):
            raise ValueError('SAME_KNOWN_INTENT_AND_FROZEN_FOUR_METHODS_REQUIRED')
    if len(parents)!=6 or len({s for s,_ in parents.values()})!=6:raise ValueError('EXACT_SIX_DISTINCT_NEW_DEV_PARENTS_REQUIRED')
    for p in parents:
        r=[r for r in rows if r['parent']==p]
        if (Counter((x['goal_index'],x['method'],x['repeat_index'])for x in r)
                !=Counter((g,m,i)for g in range(4)for m in ARMS for i in range(2))
                or Counter((x['goal_index'],x['method'],x['slot'])for x in r)
                !=Counter((g,m,i)for g in range(4)for m in ARMS for i in range(2))):
            raise ValueError('ALL_GOALS_METHODS_TWO_RESTARTS_AND_BOTH_SLOTS_REQUIRED')
    for at in range(0,len(rows),2):
        pair=rows[at:at+2]
        if ([r['slot']for r in pair]!=[0,1] or any(r['pair_index']!=at//2 for r in pair)
                or len({(r['parent'],r['goal_index'])for r in pair})!=1):
            raise ValueError('EXACT_TWO_SLOT_SAME_PARENT_INTENT_PAIR_REQUIRED')
    return rows
