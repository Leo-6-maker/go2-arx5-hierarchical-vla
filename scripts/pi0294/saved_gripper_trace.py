"""Analyze saved commands and joint rows; never load NN or integrate physics."""
import numpy as np


def analyze(rows, commands, mapping, initial_q, *, nq, nv, nu=8):
    rows=np.asarray(rows,dtype=np.float64);initial_q=np.asarray(initial_q,dtype=np.float64)
    if (nu != 8 or rows.ndim != 2 or rows.shape[1] != 1+nq+nv+nu+7
            or not len(rows) or not np.isfinite(rows).all() or initial_q.shape != (nq,)
            or not np.isfinite(initial_q).all()):
        raise ValueError('FINITE_ACTUAL_SAVED_HIGH_RATE_LAYOUT_REQUIRED')
    if not np.allclose(rows[:,0],np.arange(1,len(rows)+1)*.002,rtol=0,atol=1e-8):
        raise ValueError('FROZEN_500HZ_CLOCK_REQUIRED')
    fingers=('left_slide','right_slide')
    if set(mapping)!=set(fingers):raise ValueError('EXACT_TWO_FINGER_METADATA_REQUIRED')
    for name in fingers:
        item=mapping[name]
        if (not 0<=item['qpos']<nq or not 0<=item['qvel']<nv or not 0<=item['ctrl']<nu):
            raise ValueError('ACTUAL_COMPILED_JOINT_ACTUATOR_MAPPING_REQUIRED')
    points=(len(rows)+24)//25
    expected=(points+7)//8
    if len(commands)!=expected:raise ValueError('ALL_ACTUAL_SENT_PREFIXES_REQUIRED')
    prefix=[];summaries=[]
    for sequence,command in enumerate(commands):
        values=np.asarray(command,dtype=np.float64)
        if values.shape!=(8,10) or not np.isfinite(values).all():
            raise ValueError('EXACT_SENT_EEF10_PREFIX8_REQUIRED')
        grip=values[:,9]
        if ((grip<0)|(grip>1)).any():raise ValueError('SAVED_GRIPPER_COMMAND_OUT_OF_CONTRACT')
        prefix.extend(grip.tolist())
        used=min(8,max(0,points-sequence*8))
        summaries.append(dict(sequence=sequence,start_tick=sequence*200,
            prefix_gripper=grip.tolist(),minimum=float(grip.min()),maximum=float(grip.max()),
            points_executed_at_least_one_tick=used))
    qpos_indices=[mapping[name]['qpos'] for name in fingers]
    qvel_indices=[mapping[name]['qvel'] for name in fingers]
    ctrl_indices=[mapping[name]['ctrl'] for name in fingers]
    achieved=rows[:,np.asarray(qpos_indices)+1]
    velocity=rows[:,np.asarray(qvel_indices)+1+nq]
    actual_ctrl=rows[:,np.asarray(ctrl_indices)+1+nq+nv]
    target=np.full(2,.044);targets=[];predicted=[]
    previous=initial_q[qpos_indices]
    for index,q in enumerate(achieved):
        requested=prefix[index//25]*.044
        target+=np.clip(requested-target,-.04*.002,.04*.002)
        predicted.append(np.clip(400*(target-previous),-8,8).copy())
        targets.append(target.copy());previous=q
    targets=np.asarray(targets);predicted=np.asarray(predicted)
    error=float(np.max(np.abs(predicted-actual_ctrl)))
    if error>1e-9:raise ValueError('SAVED_CTRL_NOT_EXPLAINED_BY_FROZEN_COMMAND_ARITHMETIC')
    details=[]
    for i,name in enumerate(fingers):
        bad=np.flatnonzero((achieved[:,i]<-.001)|(achieved[:,i]>.045))
        details.append(dict(joint=name,initial_position_m=float(initial_q[qpos_indices[i]]),
            final_position_m=float(achieved[-1,i]),final_velocity_m_s=float(velocity[-1,i]),
            final_slewed_target_m=float(targets[-1,i]),final_motor_ctrl_N=float(actual_ctrl[-1,i]),
            maximum_position_m=float(achieved[:,i].max()),minimum_position_m=float(achieved[:,i].min()),
            maximum_target_m=float(targets[:,i].max()),minimum_target_m=float(targets[:,i].min()),
            first_limit_violation_tick=int(bad[0]+1) if len(bad) else None,
            violating_rows=len(bad),final_position_above_target_m=float(achieved[-1,i]-targets[-1,i])))
    return dict(status='PASS_SAVED_PREFIX_TO_MOTOR_COMMAND_ARITHMETIC_NOT_PHYSICS_OR_CAUSAL_PROOF',
        high_rate_rows=len(rows),sent_prefixes=len(commands),executed_20Hz_points=points,
        all_sent_gripper_commands_in_unit_interval=True,prefixes=summaries,joints=details,
        motor_ctrl_max_abs_reconstruction_error_N=error,
        final_sampled_command_open_fraction=float(prefix[(len(rows)-1)//25]),
        frozen_parameters=dict(dt_s=.002,point_hold_ticks=25,travel_m=.044,target_speed_m_s=.04,
            proportional_gain_N_m=400,force_cap_N=8,effective_lower_m=-.001,effective_upper_m=.045),
        model_loads=0,physics_integrations=0,render_calls=0,correction_applied=False)
