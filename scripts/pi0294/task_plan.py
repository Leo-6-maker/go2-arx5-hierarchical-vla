"""Four fixed full-body pilot attempts; no score-dependent seed replacement."""
import copy


def pilot_cards():
    cards=[]
    for family,shift in (('W0',0.),('W1',.15)):
        for task in ('T1','T2'):
            card=dict(id=f'TASK-PILOT-{family}-{task}',group_id='TASK-PILOT-'+family,
                task=task,task_id='FULLBODY_'+task+'_V1',split='PILOT',seed=0,
                source_xy=[.45+shift,-.08],goal_xy=[.45+shift,.08],cube_yaw=0.,
                table_center_x=.515+shift,base_translation_preference_m=shift,
                minimum_actual_BASE_translation_m=.10 if family=='W1' else 0.,
                scene_class=family,duration_s=32.,warmup_sim_s=5.,
                teacher='ACTUAL_STATE_FEEDBACK_ORIGINAL_TASK_PATH_V1',
                controls='FROZEN_NATIVE_WBC_PLUS_INDEPENDENT_FINITE_GRIPPER',
                instruction=('Pick up the red cube, place it in the blue target, release and withdraw.'
                    if task=='T1' else 'Push the red cube along the table into the blue target, then withdraw.'),
                retry_valid_failure=False,pose_gain_or_bias_search=False)
            cards.append(card)
    return copy.deepcopy(cards)
