import numpy as np
import pytest
from scripts.wm02.audit_fusion_source_phase_curriculum import placements,scan,SOURCE_VECTOR


def test_source_relative_object_uses_inverse_rotation_not_absolute_source_coordinates():
    tcp=np.eye(4);tcp[:3,3]=[.35,-.04,.65]
    frame=np.eye(4);frame[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];frame[:3,3]=[1,2,3]
    args=dict(tcp=tcp,frame=frame,table=[.515,0,.415],cube=[.45,-.08,.46],halfsize=[.17,.24,.025])
    variants=placements(**args)
    np.testing.assert_allclose(frame[:3,:3]@(variants[0]['cube_xyz']-tcp[:3,3]),SOURCE_VECTOR,atol=1e-16)
    np.testing.assert_allclose(variants[0]['table_xyz']-variants[0]['cube_xyz'],[.065,.08,-.045],atol=1e-16)
    assert variants[1]['table_halfsize'].tolist()==[.06,.06,.025]
    bad=frame.copy();bad[0,0]=4
    with pytest.raises(ValueError):placements(**dict(args,frame=bad))


@pytest.fixture
def scene():
    import mujoco
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="table" type="box" pos="2 0 .05" size=".1 .1 .05"/>
    <body pos="0 0 1"><freejoint/><geom name="robot" type="sphere" size=".1"/></body>
    <body pos="2 0 .2"><freejoint/><geom name="cube" type="box" size=".02 .02 .02"/></body>
    </worldbody></mujoco>''')
    return model,[(1,model.qpos0.copy())]


def test_forward_audit_detects_new_table_collision_without_stepping(scene):
    m,poses=scene;before=m.body_mass.copy()
    baseline=scan(m,poses,table_id=0,cube_id=2,cube_qpos_adr=7)
    assert baseline['contact_free_at_saved_poses'] and baseline['new_integrations']==0
    candidate=dict(table_xyz=[0,0,1],table_halfsize=[.1,.1,.05],cube_xyz=[2,0,.2])
    changed=scan(m,poses,table_id=0,cube_id=2,cube_qpos_adr=7,placement=candidate)
    assert changed['robot_task_interpenetration_pose_count']==1 and changed['first_interpenetration_native_step']==1
    assert changed['deepest_interpenetration_m']<0 and not changed['new_world_dynamics_or_native_qualification']
    np.testing.assert_array_equal(m.body_mass,before)


def test_accidental_integrator_and_nonfinite_saved_state_are_rejected(scene,monkeypatch):
    import mujoco
    m,poses=scene;bad=poses[0][1].copy();bad[0]=np.nan
    with pytest.raises(ValueError,match='FINITE_SAVED_QPOS'):scan(m,[(1,bad)],table_id=0,cube_id=2,cube_qpos_adr=7)
    monkeypatch.setattr(mujoco,'mj_forward',lambda model,data:mujoco.mj_step(model,data))
    with pytest.raises(RuntimeError,match='REPLAY_FORBIDS_INTEGRATION'):scan(m,poses,table_id=0,cube_id=2,cube_qpos_adr=7)
