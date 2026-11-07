import numpy as np
import pytest
import copy,json
from go2arx5_vla.world_model.fusion_task_reference_diagnostic_v1 import build,DISTANCE_M,PRODUCER,CONFIG,TASK_CONDITION,validate_registration
from go2arx5_vla.world_model.storage import digest


def inputs():
    tcp=np.eye(4);tcp[:3,3]=[.3,.1,.6]
    obj=np.eye(4);obj[:3,3]=tcp[:3,3]+[.2,.04,-.05]
    reference=tcp.copy();reference[0,3]+=.001
    return dict(actual_TCP=tcp,initial_reference_TCP=reference,
                base_pose_hold=[0,0,.3,0,0,0,1],actual_object_pose=obj)


def test_fixed_privileged_intent_is_not_an_executed_or_model_action():
    data=inputs();before={k:np.array(v).copy() for k,v in data.items()}
    program,report=build(**data)
    direction=(data['actual_object_pose'][:3,3]-data['actual_TCP'][:3,3])
    direction/=np.linalg.norm(direction)
    np.testing.assert_allclose(program.desired_path[-1,:3]-data['actual_TCP'][:3,3],DISTANCE_M*direction,atol=1e-15)
    assert program.geometry_scale==1 and program.time_stretch_factor==2
    assert len(program.knot_indices)==40 and program.end_atom+8<=1288
    assert report['producer']==PRODUCER and report['dynamic_qualification']is False
    assert report['native_execution_authorized']is False
    assert report['actual_consumed_commands']==report['GR00T_queries']==report['Processor_decodes']==0
    assert report['original_short_intent_screen']['qualified']is True
    assert report['original_domain_screen']['eligible_for_original_native_certificate']is True
    for k in data:np.testing.assert_array_equal(data[k],before[k])


def test_task_direction_rotation_covariance_and_fixed_original_knots():
    first,report=build(**inputs());data=inputs()
    r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    for key in ('actual_TCP','actual_object_pose','initial_reference_TCP'):
        data[key][:3,3]=r@data[key][:3,3];data[key][:3,:3]=r@data[key][:3,:3]
    rotated,second=build(**data)
    np.testing.assert_allclose(rotated.desired_path[:,:3],first.desired_path[:,:3]@r.T,atol=1e-15)
    np.testing.assert_array_equal(rotated.decoded_H40,second['original_physical_H40'])


@pytest.mark.parametrize('fault',['near_object','nonfinite_object','bad_TCP_rotation','bad_base'])
def test_rejects_domain_or_observation_faults_instead_of_clipping(fault):
    data=inputs()
    if fault=='near_object':data['actual_object_pose'][:3,3]=data['actual_TCP'][:3,3]+[.144,0,0]
    elif fault=='nonfinite_object':data['actual_object_pose'][0,3]=np.nan
    elif fault=='bad_TCP_rotation':data['actual_TCP'][0,0]=2.
    else:data['base_pose_hold'][-1]=0.
    with pytest.raises(ValueError):build(**data)


def registered(tmp_path):
    path=tmp_path/'teacher-panel.json';root=tmp_path/'teacher-diagnostic'
    path.write_text(json.dumps(dict(assignments=[dict(native_run=root.name,parent='EXPLICIT_GEOMETRY_DIAGNOSTIC',split='DIAGNOSTIC',task_reference_diagnostic=CONFIG)])))
    pin=digest(path)
    return dict(registered_root=str(root),phase='collect-target-fusion-task-phase-live-skill-v9',kind='FUSION_TASK_PHASE_LIVE_SKILL_V9',
        task_condition=TASK_CONDITION,task_reference_diagnostic=copy.deepcopy(CONFIG),live_skill_geometry_scale=1.,live_skill_time_stretch_factor=2,
        response_learning_panel=str(path),response_learning_panel_sha256=pin,
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',split='DIAGNOSTIC',parent='EXPLICIT_GEOMETRY_DIAGNOSTIC',panel_sha256=pin))


def registered_short(tmp_path):
    from go2arx5_vla.world_model import fusion_task_reference_diagnostic_v1 as teacher
    s=registered(tmp_path);path=__import__('pathlib').Path(s['response_learning_panel']);p=json.loads(path.read_text())
    p['assignments'][0]['task_reference_diagnostic']=teacher.SHORT_CONFIG;path.write_text(json.dumps(p));pin=digest(path)
    s.update(task_condition=teacher.SHORT_TASK_CONDITION,task_reference_diagnostic=copy.deepcopy(teacher.SHORT_CONFIG),live_skill_start_step=teacher.SHORT_BEGIN,max_ticks=teacher.endpoint(teacher.SHORT_TASK_CONDITION),response_learning_panel_sha256=pin)
    s['response_learning_assignment']['panel_sha256']=pin
    return s


def test_short_teacher_has_its_own_complete_horizon_and_unchanged_task_limits(tmp_path):
    from go2arx5_vla.world_model import fusion_task_reference_diagnostic_v1 as teacher
    from go2arx5_vla.world_model import fusion_task_phase_live_skill_contract_v9 as c
    from go2arx5_vla.world_model.fusion_graded_short_skill_v1 import score_contract
    s=registered_short(tmp_path);assert teacher.validate_registration(s)==teacher.SHORT_CONFIG
    assert c.terminal_step(teacher.SHORT_TASK_CONDITION)==38703
    assert c.terminal_step(c.MECHANICAL_TASK_CONDITION)==66503
    score=c.score_contract(task_condition=teacher.SHORT_TASK_CONDITION)
    assert score['registered_final_step']==38703 and score['actual_short_skill_score_contract']==score_contract()
    s['max_ticks']+=25
    with pytest.raises(ValueError,match='EXACT_PREREGISTERED'):teacher.validate_registration(s)


@pytest.mark.parametrize('fault',['translation','rotation','nonfinite'])
def test_short_diagnostic_stops_at_original_standing_bounds(fault):
    from go2arx5_vla.world_model.fusion_task_reference_diagnostic_v1 import require_standing
    initial=dict(qpos=np.r_[0.,0.,.33,1.,0.,0.,0.],native_step=34303);sample=copy.deepcopy(initial);sample['native_step']+=25
    require_standing(sample,initial)
    if fault=='translation':sample['qpos'][0]=.0021
    elif fault=='rotation':sample['qpos'][3:]=[np.cos(.1/2),0,0,np.sin(.1/2)]
    else:sample['qpos'][0]=np.nan
    with pytest.raises(ValueError):require_standing(sample,initial)


@pytest.mark.parametrize('fault',['TRAIN','old_task','larger_step','WM_extrapolation','changed_panel','wrong_row'])
def test_explicit_geometric_diagnostic_cannot_be_a_hidden_vla_or_training_arm(tmp_path,fault):
    spec=registered(tmp_path);assert validate_registration(spec)==CONFIG
    if fault=='TRAIN':spec['response_learning_assignment']['split']='TRAIN'
    elif fault=='old_task':spec['task_condition']='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'
    elif fault=='larger_step':spec['task_reference_diagnostic']['step_m']=.01
    elif fault=='WM_extrapolation':spec['program_response_forecast']={'not_in_teacher_domain':True}
    elif fault=='changed_panel':__import__('pathlib').Path(spec['response_learning_panel']).write_text('{}')
    else:spec['registered_root']+='_not_declared'
    with pytest.raises(ValueError):validate_registration(spec)


def test_teacher_final_gate_retains_gr00t_not_executed_and_original_consumption(tmp_path):
    from go2arx5_vla.world_model.fusion_trajectory_qualification_native import complete_reference_consumption
    spec=registered(tmp_path);spec.update(max_ticks=66503,live_skill_start_step=34303)
    atoms=(66503-34303)//25
    suffix=dict(task_reference_producer=PRODUCER,all_40_knots_consumed=True,raw_knot_consumed=[True]*40,
        complete_suffix_atoms=atoms,windows=[{}]*atoms,solver_target_confirmed_atoms=atoms,gripper_confirmed_atoms=atoms,install_attempts=[{}])
    result=dict(task_reference_producer=PRODUCER,unused_GR00T_proposal_retained=True,all_40_knots_consumed=False,
        raw_knot_consumed=[False]*40,GR00T_raw_H40_consumed=False,Teacher_reference_all40_consumed=True,
        VLA_action_installations=0,actual_VLA_commands=0,full_H40_intent_is_an_executed_skill=False,
        actual_reference_installations=1,generated_segment=suffix)
    assert complete_reference_consumption(spec,result)
    for field,value in [('all_40_knots_consumed',True),('actual_VLA_commands',1),('GR00T_raw_H40_consumed',True),('unused_GR00T_proposal_retained',False)]:
        bad=copy.deepcopy(result);bad[field]=value
        assert not complete_reference_consumption(spec,bad)
    bad=copy.deepcopy(result);bad['generated_segment']['windows'].pop()
    assert not complete_reference_consumption(spec,bad)


def test_original_gr00t_and_hold_terminal_predicates_keep_their_distinct_meaning(tmp_path):
    from go2arx5_vla.world_model.fusion_trajectory_qualification_native import complete_reference_consumption
    from test_fusion_program_reference_intervention_v1 import document
    assert complete_reference_consumption({},dict(all_40_knots_consumed=True))
    assert not complete_reference_consumption({},dict(all_40_knots_consumed=False))
    config=document('HOLD');path=tmp_path/'hold-panel.json';root=tmp_path/'hold'
    path.write_text(json.dumps(dict(domain=dict(time_stretch_factor=2),assignments=[dict(native_run=root.name,parent='ORIGINAL_HOLD',split='TRAIN',reference_intervention=config)])))
    pin=digest(path)
    spec=dict(phase='collect-target-fusion-task-phase-live-skill-v9',task_condition='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1',live_skill_geometry_scale=1.,live_skill_time_stretch_factor=2,
        registered_root=str(root),program_reference_intervention=config,response_learning_panel=str(path),response_learning_panel_sha256=pin,
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',split='TRAIN',parent='ORIGINAL_HOLD',panel_sha256=pin),max_ticks=66503,live_skill_start_step=34303)
    atoms=1288
    result=dict(post_query_HOLD_control=True,reference_intervention=config,all_40_knots_consumed=False,raw_knot_consumed=[False]*40,VLA_action_installations=0,actual_VLA_commands=0,full_H40_intent_is_an_executed_skill=False,actual_reference_installations=1,
        generated_segment=dict(post_query_HOLD_control=True,reference_intervention=config,all_declared_reference_knot_times_consumed=True,complete_suffix_atoms=atoms,windows=[{}]*atoms,solver_target_confirmed_atoms=atoms,gripper_confirmed_atoms=atoms,install_attempts=[{}]))
    assert complete_reference_consumption(spec,result)
    result['actual_VLA_commands']=1
    assert not complete_reference_consumption(spec,result)
