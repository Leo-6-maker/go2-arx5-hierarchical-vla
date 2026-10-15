"""Actual open reset to a closed skill: explicit connector, unchanged skill values."""
import numpy as np
from .task_gripper_contract import WORKING_OPEN,WORKING_CLOSED,SPEED,COMMAND_DT

VERSION='WORKING_MIDPOINT_THEN_SOURCE_V1'


def compose(program,entry,card):
    from .night_r1_source import with_entry
    if card.get('entry_gripper_bridge')!=VERSION:return with_entry(program,entry)
    if card['task']!='T2' or program['gripper'][0]!=0. or abs(card['initial_finger_q_m']-WORKING_OPEN)>1e-15:
        raise ValueError('EXACT_COMMON_OPEN_TO_CLOSED_T2_SKILL_REQUIRED')
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


def check():
    from .night_r1_source import sample,with_entry
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
    return dict(status='PASS_COMMON_RESET_ENTRY_BRIDGE_CPU',source_skill_values_unchanged=True,
        midpoint_m=.022,closing_hold_s=.55,midpoint_is_working_stroke_mean=True,
        close_time_derived_from_original_SPEED_and_COMMAND_DT=True,legacy_exact_regression=True,new_physics=0)
