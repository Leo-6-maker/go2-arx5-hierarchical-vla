"""Finite pre-rollout R4 scene design inside the explicitly allowed ranges."""
import copy
import numpy as np
from .scene_cards import scene_group,task_card,near_duplicate_report,validate_groups,digest
from scripts.pi0291_arx5_data.scene import DOWNWARD_ROTATION


def build(exclusions):
    accepted=[];rejected=[];groups=[];pilot=[]
    for split,count in (('PILOT',2),('TRAIN',24)):
        for index in range(count):
            approach=(index%2==0) if split=='PILOT' else index<12
            for candidate in range(64):
                serial=9000+(1000 if split=='PILOT' else 0)+index
                group=scene_group(split,'F1',serial,candidate_index=candidate)
                group['group_id']=group['scene_group']=f'PI0294-R4/{split}/'+('APPROACH' if approach else 'COMPOSITE')+f'/{index:03d}'
                group['R4_design']='CONTACT_APPROACH' if approach else 'QUALIFIED_F1_SPACE_F2_APPEARANCE'
                group['prefreeze_candidate_index']=candidate
                if approach:
                    rng=np.random.Generator(np.random.PCG64(group['seed']))
                    offset=np.r_[rng.uniform(-.005,.005,2),rng.uniform(0,.01)]
                    initial=np.eye(4);initial[:3,:3]=DOWNWARD_ROTATION
                    initial[:3,3]=np.r_[group['source_xy'],group['table_top_m']+.10]+offset
                    group['initial_tcp']=initial.tolist();group['R4_initial_TCP_offset_m']=offset.tolist()
                else:
                    visual=scene_group(split,'F2',serial,candidate_index=candidate)['visual']
                    group['visual']=copy.deepcopy(visual);group['family']='F2'
                conflict=near_duplicate_report([group],[*exclusions,*accepted])
                if conflict['conflicts']:
                    rejected.append(dict(group_id=group['group_id'],candidate=candidate,seed=group['seed'],conflicts=conflict['conflicts']))
                    continue
                accepted.append(group);(pilot if split=='PILOT' else groups).append(group);break
            else:raise ValueError('FINITE_PRE_FREEZE_GEOMETRY_PROPOSALS_EXHAUSTED_NO_ROLLOUT_RETRY')
    uniqueness=validate_groups(accepted,exclusions)
    cards=[task_card(g,t) for g in groups for t in ('T1','T2')]
    pilot_cards=[task_card(g,t) for g in pilot for t in ('T1','T2')]
    assert len(cards)==48 and len(pilot_cards)==4 and sum(g['R4_design']=='CONTACT_APPROACH' for g in groups)==12
    for population in (cards,pilot_cards):
        for i in range(0,len(population),2):
            a,b=population[i:i+2];ignored={'id','task','task_id','instruction'}
            assert {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}
    return dict(schema='PI0294_R4_FIXED_CONTACT48_PLAN_V1',status='FROZEN_DESIGN_NOT_STATIC_OR_PHYSICS_QUALIFIED',
        groups=groups,cards=cards,pilot_groups=pilot,pilot_cards=pilot_cards,
        exclusions_sha256=digest(exclusions),geometric_uniqueness=uniqueness,prefreeze_rejected_geometry=rejected,
        parameter_search_after_rollout=False,failed_seed_replacement=False,new_teacher_algorithm=False,
        new_physics_or_render_or_model_queries=0,selected='NONE',deployment='NONE')


def self_check():
    plan=build([])
    assert digest(plan)==digest(build([]))
    for g in plan['groups']:
        if 'R4_initial_TCP_offset_m' in g:
            x,y,z=g['R4_initial_TCP_offset_m'];assert abs(x)<=.005 and abs(y)<=.005 and 0<=z<=.01
    return plan


if __name__=='__main__':self_check();print('R4_PLAN_ZERO_COMPUTE_SELF_CHECK_PASS')
