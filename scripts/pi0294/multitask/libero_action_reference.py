"""Physical EEF reference extraction from the selected original LIBERO demo.

Original observations are post-action. The first row therefore has no saved
pre-action EEF and is excluded. Outputs are unexecuted source-robot references,
never ARX5 labels or achieved future states relabelled as commands.
"""
import math
import numpy as np
from go2arx5_vla.task_data.codec import pose10

ROBOSUITE_140='fbee5844ff5632f5b5698e204ec5357ca50be0df'


def rotation(vector):
    v=np.asarray(vector,dtype=float);angle=np.linalg.norm(v)
    if v.shape!=(3,)or not np.isfinite(v).all():raise ValueError('FINITE_AXIS_ANGLE')
    if math.isclose(angle,0.):return np.eye(3)
    x,y,z=v/angle;k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def extract(actions,post_action_ee_states,env_args):
    c=env_args['env_kwargs']['controller_configs'];env=env_args['env_kwargs']
    if (env['robots']!=['Panda'] or env['control_freq']!=20 or c['type']!='OSC_POSE'
            or c['impedance_mode']!='fixed' or c['control_delta']is not True
            or c['input_min']!=-1 or c['input_max']!=1
            or c['output_max']!=[.05,.05,.05,.5,.5,.5]
            or c['output_min']!=[-.05,-.05,-.05,-.5,-.5,-.5]
            or c['position_limits']is not None or c['orientation_limits']is not None
            or c['interpolation']is not None):
        raise ValueError('EXACT_ORIGINAL_PANDA_FIXED_OSC_DELTA_CONTROLLER')
    a=np.asarray(actions,dtype=float);s=np.asarray(post_action_ee_states,dtype=float)
    if (a.ndim!=2 or a.shape[1]!=7 or s.shape!=(len(a),6)or len(a)<2
            or not np.isfinite(a).all()or not np.isfinite(s).all()or np.any(np.abs(a)>1)):
        raise ValueError('FINITE_NORMALIZED_ACTIONS_AND_POST_ACTION_MEASUREMENTS')
    rows=[];unknown=[];previous_orientation=None;previous_gripper_intent=None
    for index in range(1,len(a)):
        command=a[index];before=s[index-1];delta=command[:6]*np.asarray(c['output_max'])
        if any(not math.isclose(x,0.)for x in delta[3:]):
            # Original set_goal_orientation: WORLD delta left-multiplies the
            # measured current rotation. Exactly zero preserves prior goal.
            previous_orientation=rotation(delta[3:])@rotation(before[3:])
        grip=int(np.sign(command[6]))
        if grip:previous_gripper_intent=grip
        if previous_orientation is None or previous_gripper_intent is None:
            unknown.append(index);continue
        target=np.eye(4);target[:3,3]=before[:3]+delta[:3];target[:3,:3]=previous_orientation
        # This is the declared ARX5 endpoint interpretation of open/close
        # intent; Panda's incremental gripper action is not a measured width.
        endpoint=.042/.044 if previous_gripper_intent<0 else .002/.044
        rows.append(dict(original_action_index=index,pre_action_observation_index=index-1,
            controller_target_world=target,normalized_original_action=command.copy(),
            gripper_direction=previous_gripper_intent,ARX5_endpoint_reference=endpoint,
            EEF10_world_reference=pose10(target,endpoint)))
    return dict(rows=rows,excluded_first_action=True,unknown_stateful_goal_indices=unknown,
        gripper_semantics='PANDA_SIGN_DIRECTION_NOT_MEASURED_APERTURE; DECLARED_ARX5_ENDPOINT_REFERENCE',
        original_action_observation_alignment='ACTION_i_USES_POST_ACTION_OBSERVATION_i_MINUS_1',
        data_role='UNEXECUTED_ORIGINAL_ROBOT_REFERENCE_ONLY_NOT_ARX5_SUPERVISION',
        training_admitted=False,new_physics=0)
