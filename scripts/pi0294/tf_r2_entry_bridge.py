"""Actual open reset to a closed skill: explicit connector, unchanged skill values."""
import numpy as np
from .task_gripper_contract import WORKING_OPEN,WORKING_CLOSED,SPEED,COMMAND_DT

VERSION='WORKING_MIDPOINT_THEN_SOURCE_V1'
HORIZON_VERSION='WORKING_MIDPOINT_CLOSE_BEFORE_PREVIEW_CONTACT_V2'
S7_VERSION='TF_R2_S7_QUINTIC_REQUEST_095_V1'


def compose(program,entry,card):
    from .tf_r2_source import with_entry
    version=card.get('entry_gripper_bridge')
    if version not in (VERSION,HORIZON_VERSION,S7_VERSION):return with_entry(program,entry)
    if card['task']!='T2' or program['gripper'][0]!=0. or abs(card['initial_finger_q_m']-WORKING_OPEN)>1e-15:
        raise ValueError('EXACT_COMMON_OPEN_TO_CLOSED_T2_SKILL_REQUIRED')
    if version in (HORIZON_VERSION,S7_VERSION):
        close=float(entry['gripper_close_start_s']);end=float(entry['time'][-1])
        if not .5<=close<end-.55:raise ValueError('FROZEN_HORIZON_CLOSURE_SCHEDULE_REQUIRED')
        combined=with_entry(program,entry)
        n=len(entry['time'])-1
        combined['gripper'][:n]=np.where(np.asarray(entry['time'][:-1])<close,.5,0.)
        result=dict(combined,bridge_close_start_s=np.asarray(close),bridge_entry_end_s=np.asarray(end))
        if version==S7_VERSION:
            result.update(bridge_profile_duration_s=np.asarray(.95),bridge_profile_start_s=np.asarray(close+.5-.95))
        return result
    midpoint=(WORKING_OPEN+WORKING_CLOSED)/2
    close_s=abs(midpoint-WORKING_CLOSED)/SPEED+COMMAND_DT
    close_s=round(close_s/COMMAND_DT)*COMMAND_DT
    end=float(entry['time'][-1]);begin=end+close_s
    return dict(time=np.r_[entry['time'][:-1],end,begin+program['time']],
        poses=np.concatenate([entry['poses'][:-1],program['poses'][:1],program['poses']]),
        gripper=np.r_[np.full(len(entry['time'])-1,.5),program['gripper'][0],program['gripper']])


def source_begin(entry,card):
    end=float(entry['time'][-1])
    if card.get('entry_gripper_bridge')!=VERSION:return end
    midpoint=(WORKING_OPEN+WORKING_CLOSED)/2
    close_s=abs(midpoint-WORKING_CLOSED)/SPEED+COMMAND_DT
    return end+round(close_s/COMMAND_DT)*COMMAND_DT


def plan_closure(client,state,entry,joint_path,root,stage):
    """Forecast neutral-finger geometry; finish closing before its first preview.

    The original H40 collision certifier continues to use current measured
    fingers. It is deliberately not changed to assume a future closed state.
    """
    import hashlib,json,math
    from pathlib import Path
    import mujoco
    from .tf_r2_plan_runtime import reserve_static,finish_static,read,sha
    from .mp_client import PlannerFailure,contact_permitted
    from scripts.pi0292.durable import write
    if not stage or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for c in stage):
        raise ValueError('FIXED_BRIDGE_STAGE_ID_REQUIRED')
    times=np.asarray(entry['time']);q=np.asarray(joint_path);end=float(times[-1])
    if q.shape!=(len(times),6) or not 0<end<=124:raise ValueError('BOUNDED_EXISTING_ENTRY_REQUIRED')
    input_hash=hashlib.sha256(b''.join(np.asarray(x,dtype='<f8').tobytes() for x in
        (state['qpos'],state['qvel'],state['ctrl'],times,q))).hexdigest()
    out=Path(root)/'factory/entry_bridge_plans'/stage/'PLAN.json'
    if out.exists():
        proof=read(out)
        if proof['input_sha256']!=input_hash or proof['implementation_sha256']!=sha(Path(__file__)):raise ValueError('FROZEN_BRIDGE_PLAN_INPUT_CHANGED')
        if proof['status']!='PASS':raise PlannerFailure(proof['reason'],proof)
        return dict(entry,gripper_close_start_s=proof['close_start_s'],gripper_bridge_plan=proof)
    count=math.ceil(end/.002)+1;name='GRIPPER_CLOSURE_'+stage
    m,d=client.model,client.native_data
    fingers=np.asarray([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_'+side+'_slide') for side in ('left','right')])
    if (fingers<0).any():raise ValueError('ORIGINAL_INDEPENDENT_FINGERS_REQUIRED')
    reserve_static(root,name,count)
    rows=0;first_contact=None;status='PASS';reason=None
    try:
        d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl']
        for tick in range(count):
            at=min(tick*.002,end);rows+=1
            d.qpos[client.arm_q]=[np.interp(at,times,q[:,j]) for j in range(6)]
            d.qpos[m.jnt_qposadr[fingers]]=(WORKING_OPEN+WORKING_CLOSED)/2
            mujoco.mj_forward(m,d)
            for contact in d.contact:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (contact.geom1,contact.geom2)]
                pad_cube='task_cube_geom' in names and any(x in ('task_left_pad','task_right_pad') for x in names)
                if pad_cube and first_contact is None:first_contact=at
                if not pad_cube and not contact_permitted(names,contact.dist,'contact'):
                    raise PlannerFailure('ENTRY_NEUTRAL_GEOMETRY_COLLISION:'+str(names),{})
            if first_contact is not None:break
        first=end if first_contact is None else first_contact
        close_s=abs((WORKING_OPEN+WORKING_CLOSED)/2-WORKING_CLOSED)/SPEED+COMMAND_DT
        # Full 2s certifier horizon + a complete prefix ACK + original travel.
        close=math.floor((first-2.-8*COMMAND_DT-close_s)/COMMAND_DT+1e-10)*COMMAND_DT
        if close<.5 or close+close_s>=end:raise PlannerFailure('NO_ENTRY_ROOM_FOR_HORIZON_SAFE_CLOSURE',{})
    except PlannerFailure as exc:status='FAIL';reason=str(exc)
    except Exception as exc:
        status='ENGINEERING_INCOMPLETE';reason=type(exc).__name__+': '+str(exc);raise
    finally:
        proof=dict(status=status,reason=reason,input_sha256=input_hash,implementation_sha256=sha(Path(__file__)),source_sha=read(Path(root)/'RUN_MANIFEST.json')['code']['vla_commit'],
            original_entry_end_s=end,first_neutral_pad_cube_contact_s=first_contact,
            close_start_s=close if status=='PASS' else None,close_travel_and_ACK_s=close_s if status=='PASS' else None,
            forecast_points=rows,forecast_hypothetical_not_actual_observation=True,
            full_preview_s=2.,executed_prefix_ACK_s=8*COMMAND_DT,working_midpoint_m=(WORKING_OPEN+WORKING_CLOSED)/2,
            original_certifier_uses_actual_fingers_unchanged=True,new_physics=0,new_IK=0,new_RRT=0,new_render=0)
        out.parent.mkdir(parents=True,exist_ok=True);write(out,proof);finish_static(root,name,rows,status)
    if status!='PASS':raise PlannerFailure(reason,proof)
    return dict(entry,gripper_close_start_s=close,gripper_bridge_plan=proof)


def check():
    from .tf_r2_source import sample,with_entry
    from .task_gripper_contract import teacher_physical_fraction
    entry=dict(time=np.array([0.,1.,2.]),poses=np.repeat(np.eye(4)[None],3,axis=0))
    program=dict(time=np.array([0.,1.,2.]),poses=np.repeat(np.eye(4)[None],3,axis=0),gripper=np.array([0.,0.,1.]))
    card=dict(task='T2',initial_finger_q_m=.042,entry_gripper_bridge=VERSION)
    combined=compose(program,entry,card);begin=source_begin(entry,card)
    assert abs(begin-2.55)<1e-12
    assert np.array_equal(combined['poses'][-len(program['poses']):],program['poses'])
    assert np.array_equal(combined['gripper'][-len(program['gripper']):],program['gripper'])
    assert np.array_equal(combined['time'][-len(program['time']):]-begin,program['time'])
    _,opening=sample(combined,np.array([0.,1.,2.,2.5,2.55,4.55]))
    assert np.array_equal(opening,[.5,.5,0.,0.,0.,1.])
    assert np.allclose(teacher_physical_fraction(opening)*.044,[.022,.022,.002,.002,.002,.042],atol=1e-15,rtol=0)
    for k,v in with_entry(program,entry).items():assert np.array_equal(compose(program,entry,{})[k],v)
    try:compose(program,entry,dict(card,task='T1'))
    except ValueError:pass
    else:raise AssertionError('WRONG_TASK_BRIDGE_NOT_REJECTED')
    horizon=dict(entry,gripper_close_start_s=.8)
    v2=compose(program,horizon,dict(card,entry_gripper_bridge=HORIZON_VERSION))
    old=with_entry(program,entry)
    assert np.array_equal(v2['poses'],old['poses']) and np.array_equal(v2['time'],old['time'])
    assert np.array_equal(v2['gripper'][-len(program['gripper']):],program['gripper'])
    _,grip=sample(v2,np.array([0.,.75,.8,1.,2.,4.]))
    assert np.array_equal(grip,[.5,.5,0.,0.,0.,1.])
    return dict(status='PASS_COMMON_RESET_ENTRY_BRIDGE_CPU',source_skill_values_unchanged=True,
        midpoint_m=.022,closing_hold_s=.55,midpoint_is_working_stroke_mean=True,
        close_time_derived_from_original_SPEED_and_COMMAND_DT=True,legacy_exact_regression=True,new_physics=0)
