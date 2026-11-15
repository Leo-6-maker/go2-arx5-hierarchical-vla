"""Predeclared mobile train/development populations and one learning budget."""
TRAIN_XY_MM=((0,0),(-5,-5),(-5,5),(5,-5),(5,5))
DEV_XY_MM=((-5,0),(0,-5),(5,0),(0,5))
VERSION='MFREE_C20_FINAL_PLAN_V1'


def catalog():
    rows=[]
    # First two resets per layout, then reserved attempts for ordinary task
    # failures. Native unsafe conditions are isolated instead of retried.
    for family in ('F1','F2'):
        for wave in ((1,2),(3,4)):
            for layout,xy in enumerate(TRAIN_XY_MM):
                for reset in wave:
                    rows.append(dict(family=family,split='TRAIN',layout_id=f'T{layout}',
                        xy_mm=list(xy),reset=reset,episode_id=f'{family}-T{layout}-r{reset}',
                        scene_lineage=f'MFREE_C20-{family}-TRAIN-T{layout}',
                        session=f'MFREE_C20-{family}-TRAIN',
                        grasp_goal_raise_m=.008 if family=='F2' else 0.,
                        C4_member_if_success=layout==0 and reset<=2))
    dev=[dict(family=family,split='DEV',layout_id=f'D{layout}',xy_mm=list(xy),reset=reset,
        episode_id=f'{family}-D{layout}-r{reset}',scene_lineage=f'MFREE_C20-{family}-DEV-D{layout}',
        session=f'MFREE_C20-{family}-DEV') for family in ('F1','F2')
        for layout,xy in enumerate(DEV_XY_MM) for reset in (1,2)]
    return dict(schema=VERSION,train_attempts=rows,dev16=dev,train_success_target=20,
        maximum_train_attempts=40,successes_per_family=10,successes_per_layout=2,
        baseline8=[dict(model=model,family=family,reset=reset,xy_mm=[0,0])
            for model in ('PD4_EXECUTED_REF','PD4_FINAL_PLAN') for family in ('F1','F2') for reset in (101,102)],
        train8=[dict(family=family,layout_id=f'T{layout}',reset=201+reset)
            for family in ('F1','F2') for layout in (0,4) for reset in (0,1)],
        recipe=dict(id='MFREE_C20_FINAL_FM1000',parent='haofuly/SLA_base',
            revision='5bbdf3012ab643554aa2eb83ddeae9ace3dce476',updates=1000,batch=32,
            microbatch=8,accumulation=4,visits=32000,peak_lr=5e-5,warmup=50,final_lr=1e-5,
            state_token_dropout=.25,loss='FM_ONLY',spatial_loss=False,BASE_head=False,
            sampler='EXISTING_STAGE_BALANCED',checkpoint_selection='FIXED_STEP1000',
            step500='TRAIN_DIAGNOSTICS_ONLY',optional_successor_automatically_started=False),
        physics=dict(WBC_changes=False,safety_changes=False,real_robot=False,RL=False),
        modes=dict(M_FREE='EEF_ONLY',M_ASSIST='ENTIRE_MODE_QUARANTINED'),
        labels=dict(view='CURRENT_QUERY_FINAL_NOMINAL_H40',action_dt_s=.05,query_dt_s=.5,
            planned_mask=40,confirmed_consumed_normal=10,unconsumed_is_executed=False,
            gripper='GRIP_LOGIT_SIGMOID_ONCE',stats='NEW_SUCCESSFUL_TRAIN_EPISODES_ONLY'),
        old_V3_RGB='ABSENT_SEEDS_ONLY_NO_POSTHOC_REPLAY_AS_REAL_OBSERVATION')
