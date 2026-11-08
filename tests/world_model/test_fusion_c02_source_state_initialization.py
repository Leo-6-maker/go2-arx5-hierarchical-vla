"""New C02 initializer boundaries; all samples/programs here are synthetic."""
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_c02_source_state_initialization_v1 as c
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as generic
from go2arx5_vla.world_model import target_sequential as target
from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def setup():
    actual=np.eye(4);actual[:3,3]=[.35,0.,.80]
    source=np.eye(4);source[:3,:3]=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    cube=np.eye(4);cube[:3,3]=[.46,-.14,.50]
    inputs=dict(raw=None,source_payload=dict(state_eef10=pose10(source,1.)),frame=np.eye(4))
    return actual,cube,inputs


def test_full_teacher_program_is_a_real_fixed_goal_and_retains_all40():
    actual,cube,inputs=setup();doc=c.episode('/new/distinct')
    program,proof=c.build(inputs,doc,actual,actual,[0.,0.,0.,1.,0.,0.,0.],cube)
    assert inputs['raw'].shape==(40,10)
    np.testing.assert_array_equal(inputs['raw'],program.desired_path[list(program.knot_indices)])
    np.testing.assert_allclose(program.desired_path[-1,:3],[.46,-.14,.58],atol=1e-15)
    assert len(program.knot_indices)==40 and len(set(program.knot_indices))==40
    assert program.end_atom<=1024 and generic.endpoint(doc)==c.E==29503
    assert proof['minimum_full_path_clearance_m']>=.07
    assert not proof['native_execution_authorized']
    assert target.native_task_bucket(c.PHASE,True)=='64'


def test_wrong_course_clearance_is_rejected_before_any_native_install():
    actual,cube,inputs=setup();cube[2,3]=.579
    with pytest.raises(ValueError,match='WHOLE_NEW_COURSE_CLEARANCE'):
        c.build(inputs,c.episode('/new/distinct'),actual,actual,[0.,0.,0.,1.,0.,0.,0.],cube)


def test_unknown_or_changed_recipe_rejected_before_backend_or_inputs():
    with pytest.raises(ValueError,match='EXACT_FINITE_C02_TEACHER_SCOPE'):
        c.config(dict(phase=c.PHASE,kind=c.KIND,WBC_changes=True))
    actual,cube,inputs=setup();doc=c.episode('/new/distinct');doc['target_source_xyz'][0]=.9
    with pytest.raises(ValueError,match='EXACT_EPISODE'):
        c.build(inputs,doc,actual,actual,[0.,0.,0.,1.,0.,0.,0.],cube)


def test_install_checks_full40_and_keeps_measured_base_and_clock():
    actual,cube,inputs=setup();base=[0.,0.,0.,1.,0.,0.,0.]
    program,_=c.build(inputs,c.episode('/new/distinct'),actual,actual,base,cube)
    a=SimpleNamespace(spec=dict(phase=c.PHASE),trajectory_program=program,trajectory_initial_tcp=actual,
        trajectory_initial_base=base,observe=lambda:dict(step=3703))
    c.validate_install(a,program.preview(0))
    wrong=program.preview(0).copy();wrong[-1,0]+=.001
    with pytest.raises(ValueError,match='FULL40_PERSISTENT'):
        c.validate_install(a,wrong)
    a.trajectory_initial_base=[.01,0.,0.,1.,0.,0.,0.]
    with pytest.raises(ValueError,match='PROGRAM_AND_BASE'):
        c.validate_install(a,program.preview(0))


def test_qualification_uses_achieved_pose_and_cannot_pad_source_state():
    actual,cube,inputs=setup();goal=np.eye(4);goal[:3,3]=[.46,-.14,.58];goal[:3,:3]=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    inputs['statistics']=dict(eef_9d=dict(min=[-1.,-1.,0.,-.03,-1.,-1.,-.03,-1.,-1.],max=[1.,1.,1.,.03,1.,1.,.03,1.,1.]),gripper_position=dict(min=[0.],max=[1.]))
    tail=[dict(native_step=i,tcp=goal.copy(),object_pose=cube.copy(),qpos=np.r_[np.zeros(3),1.,np.zeros(30)],finger_q=np.array([.042,.042]))for i in range(c.E-200,c.E+1,25)]
    good=c.qualification(tail,inputs)
    assert good['qualified'] and not good['source_inputs_replaced']
    tail[-1]['tcp'][0,3]=1.5
    bad=c.qualification(tail,inputs)
    assert not bad['qualified'] and 0 in bad['metrics']['outside_training_range_components']
    assert bad['metrics']['actual_source_state_eef10'][0]==1.5


def test_zero_or_nonunit_base_is_not_a_valid_calibration_target():
    actual,cube,inputs=setup()
    with pytest.raises(ValueError,match='MEASURED_FINITE_UNIT_BASE7'):
        c.build(inputs,c.episode('/new/distinct'),actual,actual,[0.]*7,cube)
