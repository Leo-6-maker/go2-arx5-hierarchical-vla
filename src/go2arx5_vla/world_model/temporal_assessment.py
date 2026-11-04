"""Prespecified development gates, never an Actor/target permission mechanism."""
import numpy as np

def assess(result,baseline,*,candidate='M3',promotion=None):
    rows=result['models'][candidate]['rows'];old=baseline['models']['M1']['rows'];checks=[]
    for h,limit in[(1,.0005),(8,.001),(16,.0015)]:
        group=[r for r in rows if r['partition']=='DIAGNOSTIC_DEV'and r['anchor']=='transport'and r['mode']=='FREE_CAUSAL'and r['horizon']==h]
        if {r['scene']for r in group}!=set(range(24,32)):raise ValueError('All original DEV8 parents required')
        parents=[dict(scene=r['scene'],true_m=r['object_position']['actual_toward_component_m'],predicted_m=r['object_position']['predicted_toward_component_m'],abs_error_m=abs(r['object_position']['predicted_toward_component_m']-r['object_position']['actual_toward_component_m']),direction_positive=r['object_position']['predicted_toward_component_m']>0)for r in group]
        checks.append(dict(gate='object_action_difference',horizon=h,tolerance_m=limit,parents=parents,pass_all=all(x['abs_error_m']<=limit and x['direction_positive']for x in parents)))
    for h in(1,16):
        def selected(values):return sorted([r for r in values if r['partition']=='DIAGNOSTIC_DEV'and r['anchor']=='transport'and r['mode']=='FREE_CAUSAL'and r['horizon']==h],key=lambda r:r['scene'])
        new=selected(rows);ref=selected(old);nr=float(np.mean([x['object_position']['endpoint_RMSE_m']for x in new]));br=float(np.mean([x['object_position']['endpoint_RMSE_m']for x in ref]))
        limit=promotion[f'H{h}_absolute_object_RMSE_m_no_worse_than_M1']if promotion else br
        checks.append(dict(gate='mean_parent_absolute_object_RMSE_no_worse_M1',horizon=h,candidate=candidate,candidate_RMSE_m=nr,M3_RMSE_m=nr if candidate=='M3'else None,M1_RMSE_m=br,declared_limit_m=limit,pass_all=nr<=limit,parents=[dict(scene=x['scene'],candidate_RMSE_m=x['object_position']['endpoint_RMSE_m'],M1_RMSE_m=y['object_position']['endpoint_RMSE_m'])for x,y in zip(new,ref)]))
    if promotion:
        for phase in('preclose','supported_release'):
            group=[r for r in rows if r['partition']=='DIAGNOSTIC_DEV'and r['anchor']==phase and r['mode']=='FREE_CAUSAL'and r['horizon']==16]
            if {r['scene']for r in group}!=set(range(24,32)):raise ValueError('Full DEV phase population required')
            value=float(np.mean([r['events']['native_grasp']['Brier']for r in group]));limit=promotion[f'{phase}_H16_grasp_Brier_no_worse_than_M1'];checks.append(dict(gate='H16_grasp_Brier_no_worse_M1',anchor=phase,candidate=candidate,Brier=value,declared_limit=limit,pass_all=value<=limit))
    inv=[x for x in result['invariance']if x['model']==candidate];checks.append(dict(gate='canonical_permutation_invariance',checked_parents=len(inv),max_abs=max(x['canonical_candidate_permutation_max_abs']for x in inv),pass_all=len(inv)==32 and all(x['canonical_candidate_permutation_max_abs']==0 for x in inv)))
    passed=all(c['pass_all']for c in checks)
    return dict(schema='WM02_TEMPORAL_DEV_GATE_V1',candidate=candidate,scope='Existing DEV8 development evidence only; no independent test or Actor permission',checks=checks,all_development_gates_pass=passed,new_holdout_sampling='NOT_RUN_PENDING_SEPARATE_PREREGISTRATION'if passed else'NOT_RUN_DEVELOPMENT_GATE_FAILED',Actor_updates=0,target_integrations=0,interpretation='Passing these local numeric gates alone would not establish contact/safety reliability, online decision benefit or target-domain validity')
