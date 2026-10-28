import ast
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.task_data.config import layout,COMMISSIONING,PILOTS,PARAMETERS as P
from go2arx5_vla.task_data.scene import xml_scene
from go2arx5_vla.task_data.codec import relative_action,decode
from go2arx5_vla.task_data.teacher import Teacher
from go2arx5_vla.trajectory.so3 import quat_to_matrix,geodesic_angle
from go2arx5_vla.target_sim.scorer import score_episode,BOOL_FIELDS


def test_frozen_ids_and_disjoint_layout_families():
    cards=[layout(x) for x in COMMISSIONING+PILOTS]
    assert cards==[layout(x) for x in COMMISSIONING+PILOTS]
    assert len({x['seed'] for x in cards})==12
    assert {x['family'] for x in cards[:4]}=={x['family'] for x in cards[4:8]}
    assert not {x['family'] for x in cards[:4]}&{x['family'] for x in cards[8:]}
    assert all(np.linalg.norm(np.asarray(x['source_xy'])-x['goal_xy'])>=.15 for x in cards)


def test_native_row6d_relative_roundtrip_nonidentity():
    rng=np.random.default_rng(32);anchor=np.eye(4);anchor[:3,:3]=quat_to_matrix(rng.normal(size=4));anchor[:3,3]=rng.uniform(-.2,.2,3)
    targets=[]
    for _ in range(40):
        T=np.eye(4);T[:3,:3]=quat_to_matrix(rng.normal(size=4));T[:3,3]=rng.uniform(-.2,.2,3);targets.append((T,float(rng.uniform())))
    coded=relative_action(targets,anchor)
    assert coded.shape==(40,10)
    for row,(pose,grip) in zip(coded,targets):
        decoded,opening=decode(row,anchor)
        assert np.max(np.abs(decoded-pose))<1e-12 and abs(grip-opening)<1e-12


@pytest.mark.mujoco
def test_no_mocap_constraint_or_private_assets_and_named_dofs():
    from go2arx5_vla.task_data.engine import Engine
    engine=Engine(layout('C2900'))
    assert engine.model.neq==0 and engine.model.nmocap==0
    assert engine.model.nq==16 and engine.model.nv==14 and engine.model.nu==2
    assert '<mesh' not in engine.xml and '<include' not in engine.xml
    assert np.allclose(engine.data.qpos[engine.finger_q],.044)
    assert engine.snapshot()['native_grasp'] is False
    assert engine.steps==0 and engine.data.time==0
    engine.close()


def test_no_post_reset_pose_assignment():
    source=Path(__file__).parents[2]/'src/go2arx5_vla/task_data/engine.py'
    tree=ast.parse(source.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
    for node in ast.walk(method):
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            assert not any('qpos' in ast.unparse(x) or 'xpos' in ast.unparse(x) or 'mocap' in ast.unparse(x) for x in targets)


@pytest.mark.mujoco
def test_rotation_feedback_discrete_stability_from_actual_inertia_no_integration():
    from go2arx5_vla.task_data.engine import Engine
    import mujoco
    for identifier in COMMISSIONING+PILOTS:
        engine=Engine(layout(identifier));M=np.zeros((engine.model.nv,engine.model.nv))
        mujoco.mj_fullM(engine.model,M,engine.data.qM)
        A=np.linalg.inv(M[engine.tool_v:,engine.tool_v:])[3:6,3:6]
        B=np.linalg.inv(np.eye(3)+P['dt']*P['tool_rotation_kd']*A)
        F=np.eye(3)-P['dt']*A@B*P['tool_rotation_kd']
        G=-P['dt']*A@B*P['tool_rotation_kp']
        transition=np.block([[np.eye(3)+P['dt']*G,P['dt']*F],[G,F]])
        assert np.max(np.abs(np.linalg.eigvals(transition)))<1
        assert engine.steps==0
        engine.close()


def synthetic_physics():
    n=641;t=np.arange(n)/20;obj=np.repeat(np.eye(4)[None],n,axis=0);obj[:,:3,3]=[-.1,0,.42]
    obj[40:80,2,3]=.46;obj[80:160,:3,3]=[.12,0,.46];obj[160:,:3,3]=[.12,0,.42]
    tcp=obj.copy();tcp[:,2,3]+=.13
    flags={k:np.ones(n,dtype=bool) for k in BOOL_FIELDS}
    for k in ['native_grasp','left_object_contact','right_object_contact']:flags[k][:]=False;flags[k][20:160]=True
    for k in ['native_abort','illegal_contact','joint_limit_penetration','reset','takeover']:flags[k][:]=False
    return dict(time_s=t,object_pose=obj,tcp_world=tcp,object_linear_velocity=np.zeros((n,3)),object_angular_velocity=np.zeros((n,3)),**flags)


def test_score_requires_history_complete_hold_and_real_bilateral_flags():
    d=synthetic_physics();args=dict(goal_center=[.12,0,.42],initial_object_pose=d['object_pose'][0],table_height_m=.4,evidence_source='CPU_SYNTHETIC_NOT_PHYSICS')
    assert score_episode(d,**args)['persistent_success']
    d['right_object_contact'][:]=False
    assert not score_episode(d,**args)['persistent_success']
    d=synthetic_physics();d['object_pose'][580,0,3]=.2
    assert score_episode(d,**args)['hold_pass']==40 and not score_episode(d,**args)['persistent_success']


@pytest.mark.mujoco
def test_teacher_reads_actual_contact_and_does_not_advance_on_close_command():
    from go2arx5_vla.task_data.engine import Engine
    engine=Engine(layout('C2901'));s=engine.snapshot();teacher=Teacher(layout('C2901'),s['object_pose'])
    teacher.stage='close'
    for t in np.arange(0,2,.05):s['time_s']=float(t);s['native_grasp']=False;teacher.observe(s)
    assert teacher.stage=='close' and not teacher.lifted
    s['native_grasp']=True;s['time_s']=2.;teacher.observe(s)
    assert teacher.stage=='verify_grasp'
    s['time_s']=2.35;teacher.observe(s);assert teacher.stage=='lift'
    plan=teacher.plan(s);assert plan['poses'].shape==(40,4,4) and plan['opening'].shape==(40,)
    assert np.allclose(plan['poses'][0],s['tcp'])
    assert np.max(np.linalg.norm(np.diff(plan['poses'][:,:3,3],axis=0),axis=1))<=P['reference_speed_m_s']/20+1e-12
    engine.close()
