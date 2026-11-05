"""Causal late-decision action contract; no simulator construction."""
import numpy as np
SCHEMA='WM02_TARGET_LATE_DECISION_REPLAY_V3'
MANIFEST_SCHEMA='WM02_NIGHT1_LATE_DECISION_MANIFEST_V3'

def goals(commands,windows,start,offset):
    proposed=[p for p in commands if p['phase']=='proposed'];boundaries=[p for p in commands if p['phase']=='decision-boundary']
    if [p['native_step']for p in proposed]!=[start+25*k for k in(0,7,8,15,22)]or len(windows)!=24 or len(boundaries)!=1 or boundaries[0]['native_step']!=start+200 or boundaries[0]['future_response_observed']is not False:raise ValueError('EXACT_HOLD8_THEN_DECISION_INSTALLATION_REQUIRED')
    def goal(p):return np.r_[p['action_EEF10'][-1],p['measured_BASE_pose_hold']]
    initial=goal(proposed[0]);future=goal(proposed[2]);expected=initial.copy();expected[:3]+=np.asarray(offset)
    if initial.shape!=(17,)or not np.allclose(future,expected,atol=1e-8,rtol=0):raise ValueError('ONLY_PREDECLARED_SMALL_GOAL_OFFSET_ALLOWED')
    by_ref={p['wire_sha256']:p for p in proposed};actual=[]
    for i,p in enumerate(proposed):
        value=np.asarray(p['action_EEF10']);base=np.asarray(p['measured_BASE_pose_hold']);g=initial if i<2 else future
        action=np.repeat(g[None,:10],40,axis=0)
        if i==2:
            weight=np.minimum((np.arange(40)+1)/8,1.)[:,None];action[:,:3]=initial[:3]+weight*(future[:3]-initial[:3])
        if value.shape!=(40,10)or not np.allclose(value,action,atol=1e-8,rtol=0)or not np.allclose(base,initial[10:],atol=1e-8,rtol=0):raise ValueError('NO_DECISION_PREVIEW_IN_PREFIX_AND_EXACT_CAUSAL_RAMP_REQUIRED')
    for atom,w in enumerate(windows):
        p=by_ref.get(w['reference_sha256'])
        if p is None or p['native_step']>start+25*atom:raise ValueError('ACTUAL_INSTALLED_REFERENCE_REQUIRED_FOR_EACH_ATOM')
        g=goal(p);expected=initial if atom<8 else future
        if not np.allclose(g,expected,atol=1e-8,rtol=0):raise ValueError('DECISION_MUST_FIRST_AFFECT_ATOM8')
        actual.append(g)
    return future,np.asarray(actual)
