"""Finite supported-policy comparisons: single qualification before paired repeat."""
import numpy as np
from .pilot2_supported import GOALS


def panel(scenes, variants, *, seed, split='DEV'):
    if (split not in ('DEV','TEST')or not scenes or any(s['split']!=split for s in scenes)
            or len({s['parent']for s in scenes})!=len(scenes)
            or len(variants)!=5 or len({v['name']for v in variants})!=5
            or sorted(v['method']for v in variants)!=sorted([
                'GOAL_FSM','CONSTANT_KEEP','CONSTANT_RELEASE','TRAINED_GOAL_POLICY','TRAINED_GOAL_POLICY'])):
        raise ValueError('FROZEN_PARENT_SPLIT_AND_FIVE_MATCHED_POLICY_VARIANTS_REQUIRED')
    rng=np.random.default_rng(seed);singles=[];repeat=[]
    for i,scene in enumerate(scenes):
        for g,goal in enumerate(GOALS):
            for j,variant in enumerate(variants):
                slot=(i*2+g+j)%2
                row=dict(parent=scene['parent'],scene_id=scene['scene_id'],split=split,
                    task_goal=goal,variant=variant['name'],method=variant['method'])
                singles.append(dict(row,repeat_index=0,slot=slot,single_qualification=True))
                repeat.append(dict(row,repeat_index=1,slot=1-slot,single_qualification=False))
    # Odd-method groups balance globally because each parent has both goals.
    by_slot=[[r for r in repeat if r['slot']==slot]for slot in range(2)]
    for rows in by_slot:rng.shuffle(rows)
    pairs=[]
    for i,(a,b)in enumerate(zip(*by_slot)):
        pairs.extend([dict(a,pair_index=i),dict(b,pair_index=i)])
    rng.shuffle(singles)
    return dict(schema='WM02_PILOT2_REPEATED_SUPPORTED_POLICY_PANEL_V1',seed=seed,split=split,
        physical_parents=len(scenes),variants=variants,goals=list(GOALS),
        single_rows=singles,paired_repeat_rows=pairs,planned_evaluations=len(singles)+len(pairs),
        same_case_second_repeat_uses_opposite_slot=True,
        qualification_requires_native_safe_eligible_prefix_not_task_success=True,
        exact_hidden_state_fork=False,native_integrations=0)


def summarize(plan, observations):
    def key(r):return(r['parent'],r['task_goal'],r['variant'],r['repeat_index'])
    rows=plan['single_rows']+plan['paired_repeat_rows'];expected={key(r):r for r in rows};found={}
    if len(expected)!=len(rows):raise ValueError('UNIQUE_FROZEN_SUPPORTED_POLICY_CONDITIONS_REQUIRED')
    for r in observations:
        k=key(r)
        if k in found or k not in expected or r['slot']!=expected[k]['slot']:
            raise ValueError('UNIQUE_PREREGISTERED_SUPPORTED_POLICY_CASE_AND_SLOT_REQUIRED')
        found[k]=r
    result=dict(schema='WM02_PILOT2_REPEATED_SUPPORTED_POLICY_SUMMARY_V1',planned=len(rows),
        attempted=sum(bool(r['attempted'])for r in found.values()),
        not_run=len(rows)-sum(bool(r['attempted'])for r in found.values()),
        native_failed=0,invalid_evidence=0,ineligible_prefix=0,safety_stops=0,
        variants={},CONTACT_CONTROL_GAIN=False,frame_counts_are_not_independent_trials=True)
    valid={}
    for k,r in found.items():
        if not r['attempted']:continue
        result['safety_stops']+=int(bool(r['safety_stop']))
        if not r['native_passed']:result['native_failed']+=1
        elif not r['evidence_valid']:result['invalid_evidence']+=1
        elif not r['decision_prefix_eligible']:result['ineligible_prefix']+=1
        else:valid[k]=r
    for v in plan['variants']:
        name=v['name'];attempted=[r for k,r in found.items()if k[2]==name and r['attempted']]
        measured=[r for k,r in valid.items()if k[2]==name]
        successes=sum(bool(r['task_success'])for r in measured)
        result['variants'][name]=dict(attempted=len(attempted),valid_eligible=len(measured),
            strict_successes=successes,strict_success_rate=successes/len(attempted)if attempted else None,
            successes_per_planned_case=successes/sum(row['variant']==name for row in rows),
            by_parent={p:dict(valid_eligible=sum(r['parent']==p for r in measured),
                strict_successes=sum(r['parent']==p and bool(r['task_success'])for r in measured))
                for p in sorted({r['parent']for r in rows})},
            by_goal={g:dict(valid_eligible=sum(r['task_goal']==g for r in measured),
                strict_successes=sum(r['task_goal']==g and bool(r['task_success'])for r in measured))
                for g in plan['goals']})
    return result
