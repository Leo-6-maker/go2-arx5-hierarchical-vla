"""Pure fixed-clock calibration fixtures; no native run, FK, renderer or query."""
import base64
import copy
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
import struct
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal
from scripts.pi0294.dec1.target_observation import transform_hash
from scripts.pi0294.task_gripper_contract import require_working_target, teacher_physical_fraction
from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.trajectory.so3 import geodesic_angle
from go2arx5_vla.world_model import fusion_task_phase_initialization as phase


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def pose(xyz,angle=0.):
    value=np.eye(4);value[:3,3]=xyz;value[:3,:3]=rz(angle)
    return value


def pack(state):
    values=np.concatenate([state[k] for k in ('qpos','qvel','ctrl')]).astype('<f8')
    return struct.pack('<8sQdIII',b'GRTSTATE',state['native_step'],state['time_s'],34,32,20)+values.tobytes()+struct.pack('<I',0)


@pytest.fixture
def setup(monkeypatch):
    picture=io.BytesIO();Image.new('RGB',(256,256),(19,29,43)).save(picture,format='PNG');png=picture.getvalue()
    source=pose([.45,-.08,.54],.7)
    payload=dict(snapshot_uuid='ca8cb3a0-3b6f-414f-9a20-c6369fc24d48',language='Exact saved source fixture language.',
        state_eef10=pose10(source,.95).tolist(),images=[dict(camera=k,png_base64=base64.b64encode(png).decode(),
            sha256=hashlib.sha256(png).hexdigest())for k in ('external_rgb','wrist_rgb')],
        frame='PI0291_MUJOCO_WORLD',tcp='ARX5_NATIVE_SITE_x5_ee')
    source_bytes=json.dumps(sealed(dict(payload=payload)),allow_nan=False).encode()
    monkeypatch.setattr(phase,'SOURCE_OBSERVATION',hashlib.sha256(source_bytes).hexdigest())
    monkeypatch.setattr(phase,'SOURCE_PAYLOAD',document_hash(payload))
    monkeypatch.setattr(phase,'SOURCE_STATE',document_hash(payload['state_eef10']))
    frame=pose([.31,-.17,.09],.3);monkeypatch.setattr(phase,'FRAME_MAP',transform_hash(frame))
    initial=pose([.20,.10,.70],.05);decision=pose([.205,.095,.70],.08)
    cube=pose([.35,.12,.46]);a=.2
    cube[:3,:3]=[[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]]
    qpos=np.zeros(34);qpos[3]=1.
    state=dict(native_step=3703,time_s=3703*.002,tcp=decision,object_pose=cube,
        qpos=qpos,qvel=np.zeros(32),ctrl=np.zeros(20),safety_valid=True,native_abort=False,
        robot_object_contact=False,gripper_state_valid=True,object_present=True)
    packet=pack(state)
    args=dict(source_observation_bytes=source_bytes,decision_packet_bytes=packet,
        measured_decision_state=state,T_source_from_target=frame,scene_sha256=phase.SCENE)
    inputs=phase.load_inputs(**args)
    return dict(inputs=inputs,args=args,initial=initial,decision=decision,cube=cube,source=source,frame=frame,
        base=np.array([0.,0.,0.,1.,0.,0.,0.]))


def episode(setup,alpha=1.):
    return phase.make_episode(run='/fixture/fresh-init',geometry_scale=alpha)


def program(setup,alpha=1.):
    return phase.make_program(setup['inputs'],episode(setup,alpha),setup['initial'],setup['decision'],setup['base'])


def samples(setup,alpha=1.):
    p=program(setup,alpha);end=phase.endpoint(episode(setup,alpha));result=[]
    for step in range(end-200,end+1,25):
        q=np.zeros(34);q[3]=1.
        result.append(dict(native_step=step,time_s=step*.002,tcp=p.graded_goal_TCP_target.copy(),
            object_pose=setup['cube'].copy(),qpos=q,qvel=np.zeros(32),ctrl=np.zeros(20),finger_q=np.array([.042,.042]),
            safety_valid=True,native_abort=False,robot_object_contact=False,left_object_contact=False,
            right_object_contact=False,native_grasp=False,contact_valid=True,gripper_state_valid=True,
            object_present=True,illegal_contact=False,joint_limit_penetration=False,reset=False,takeover=False))
    return result


def test_static_episode_exists_before_future_packet_and_goal_are_known(setup):
    e=episode(setup)
    assert e==phase.make_episode(run='/fixture/fresh-init',geometry_scale=1.)
    assert e['source_observation_sha256']==phase.SOURCE_OBSERVATION and e['frame_map_sha256']==phase.FRAME_MAP
    assert not any(k in json.dumps(e).lower() for k in ('packet_sha','cube_pose','goal_tcp','decision_tcp_target'))
    assert phase.endpoint(e)==29503 and e['ramp_atoms']==1024
    assert e['object_and_goal_are_policy_inputs'] is False and e['source_pose_is_expert_action_label'] is False
    assert e['thresholds_are_native_safety'] is False


def test_actual_runtime_goal_binding_uses_world_z_center_and_fixed_source_rotation(setup):
    inputs=setup['inputs'];body=unseal(inputs['frozen']);goal=np.asarray(body['full_goal_TCP_target'])
    assert np.array_equal(goal[:3,3],setup['cube'][:3,3]+[0.,0.,.14])
    assert not np.allclose(goal[:3,3],setup['cube'][:3,3]+setup['cube'][:3,:3]@[0.,0.,.14])
    assert np.allclose(goal[:3,:3],setup['frame'][:3,:3].T@setup['source'][:3,:3],rtol=0,atol=1e-14)
    assert body['decision_packet_sha256']==hashlib.sha256(setup['args']['decision_packet_bytes']).hexdigest()
    assert body['frozen_cube_pose_sha256']==transform_hash(setup['cube'])
    assert inputs['raw'].shape==(40,10) and np.array_equal(matrix10(inputs['raw'])[-1],goal)
    assert not inputs['provenance']['actual_cube_FK_verified_by_this_module']
    assert not inputs['provenance']['native_execution_authorized'] and not inputs['provenance']['raw_is_VLA_action']
    assert not inputs['provenance']['object_and_goal_are_policy_inputs']


@pytest.mark.parametrize('alpha,n,end',[(.03,128,7103),(.1,256,10303),(.25,512,16703),(.5,768,23103),(1.,1024,29503)])
def test_fixed_grade_program_has_two_starts_and_one_world_endpoint(setup,alpha,n,end):
    p=program(setup,alpha);e=episode(setup,alpha)
    assert p.end_atom==n and phase.endpoint(e)==end
    assert p.reference_path.shape==p.desired_path.shape==(n+1,10)
    assert np.array_equal(p.reference_path[0],pose10(setup['initial'],phase.OPEN))
    assert np.array_equal(p.desired_path[0],pose10(setup['decision'],phase.OPEN))
    assert np.allclose(matrix10(p.reference_path[0]),setup['initial'],rtol=0,atol=1e-15)
    assert np.allclose(matrix10(p.desired_path[0]),setup['decision'],rtol=0,atol=1e-15)
    assert np.array_equal(p.reference_path[-1],p.desired_path[-1])
    assert np.allclose(matrix10(p.reference_path[-1]),p.graded_goal_TCP_target,rtol=0,atol=1e-15)
    if alpha==1.:
        assert np.array_equal(p.graded_goal_TCP_target,p.full_goal_TCP_target)
        old_transport=setup['initial'][:3,3]+p.full_goal_TCP_target[:3,3]-setup['decision'][:3,3]
        assert not np.allclose(p.reference_path[-1,:3],old_transport,rtol=0,atol=1e-6)
    assert len(p.knot_indices)==40 and len(set(p.knot_indices))==40 and p.knot_indices[0]==0 and p.knot_indices[-1]==n
    assert np.all(p.reference_path[:,9]==float(teacher_physical_fraction(1.)))
    require_working_target(p.reference_path[:,9]);require_working_target(p.preview(n+8)[:,9])
    evidence=p.evidence();assert not evidence['endpoint_transport_applied'] and not evidence['achieved_state_reanchoring']
    assert evidence['raw_VLA_knots']==0 and not evidence['native_qualification_established']
    json.dumps(evidence,allow_nan=False)


def test_preview_is_persistent_full40_suffix_tail_and_arrays_cannot_be_replaced_or_mutated(setup):
    p=program(setup,.03)
    for offset in (0,1,7,39,100,p.end_atom-2,p.end_atom,p.end_atom+8):
        expected=p.reference_path[np.minimum(offset+np.arange(40),p.end_atom)]
        assert p.preview(offset).shape==(40,10) and np.array_equal(p.preview(offset),expected)
        assert np.array_equal(p.desired_at(offset),p.desired_path[min(offset,p.end_atom)])
    q=p.preview(0);q[0,0]=99;assert p.reference_path[0,0]!=99
    for a in (p.reference_path,p.desired_path,p.base_pose_hold,p.graded_goal_TCP_target,setup['inputs']['raw']):
        with pytest.raises(ValueError):a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):p.end_atom=1
    p.evidence()['clock']['end_atom']=0;assert p.evidence()['clock']['end_atom']==p.end_atom
    for offset in (-1,.5,True):
        with pytest.raises(ValueError):p.preview(offset)
        with pytest.raises(ValueError):p.desired_at(offset)


def test_rest_to_rest_quintic_end_rates_are_small_without_resampling_or_clipping(setup):
    p=program(setup,.03)
    for path in (p.reference_path,p.desired_path):
        v=np.linalg.norm(np.diff(path[:,:3],axis=0)/.05,axis=1)
        a=np.linalg.norm(np.diff(np.diff(path[:,:3],axis=0)/.05,axis=0)/.05,axis=1)
        assert v.max()<.05 and a.max()<.25
        assert v[0]<v.max()*.002 and v[-1]<v.max()*.002
        rotations=matrix10(path)[:,:3,:3]
        assert np.max(geodesic_angle(rotations[:-1],rotations[1:])/.05)<.25
    bad=setup['initial'].copy();bad[0,3]=-100.
    with pytest.raises(ValueError,match='NO_RETIME_NO_CLIP'):
        phase.make_program(setup['inputs'],episode(setup,.03),bad,setup['decision'],setup['base'])


def test_small_rotation_uses_exact_so3_time_law_not_nlerp_approximation(setup):
    p=program(setup,.03);rotations=matrix10(p.desired_path)[:,:3,:3]
    total=float(geodesic_angle(rotations[0],rotations[-1]))
    for i in (32,64,96):
        u=i/p.end_atom;weight=10*u**3-15*u**4+6*u**5
        assert float(geodesic_angle(rotations[0],rotations[i]))==pytest.approx(total*weight,rel=0,abs=5e-11)


def test_full_grade_actual_nine_samples_can_satisfy_new_diagnostic_but_never_grant_execution(setup):
    rows=samples(setup);value=phase.query_readiness(rows,episode(setup),setup['inputs'])
    assert value['query_ready'] and all(value['predicates'].values())
    assert value['actual_sample_steps']==list(range(29303,29504,25))
    assert value['metrics']['minimum_actual_cube_distance_m']>.12
    assert value['metrics']['relative_hold']['window_s']==pytest.approx(.4)
    assert not value['native_execution_authorized'] and not value['GPU_query_authorized']
    assert not value['actual_native_qualification_established'] and not value['full_task_success']
    assert not value['source80mm_implies_target_skill_pass']
    assert value['new_queries']==value['new_integrations']==0
    json.dumps(value,allow_nan=False)


@pytest.mark.parametrize('alpha',[.03,.1,.25,.5])
def test_small_grade_never_query_ready_even_if_actual_at_full_goal(setup,alpha):
    rows=samples(setup,alpha);full=program(setup).full_goal_TCP_target
    for row in rows:row['tcp']=full.copy()
    value=phase.query_readiness(rows,episode(setup,alpha),setup['inputs'])
    assert not value['query_ready'] and not value['predicates']['full_grade']


@pytest.mark.parametrize('fault,predicate',[
    ('position','position'),('orientation','orientation'),('distance','actual_cube_distance'),
    ('jitter','continuous_relative_hold'),('closed','actual_open'),('cube_moved','cube_still'),
    ('robot_object_contact','no_contact'),('left_object_contact','no_contact'),('native_grasp','no_contact'),
    ('safety_valid','original_safety'),('native_abort','original_safety'),('contact_valid','original_safety'),
    ('gripper_state_valid','original_safety'),('object_present','original_safety'),('illegal_contact','original_safety'),
    ('joint_limit_penetration','original_safety'),('reset','original_safety'),('takeover','original_safety')])
def test_every_actual_tail_sample_controls_readiness_no_reference_substitution(setup,fault,predicate):
    rows=samples(setup);row=rows[3]
    if fault=='position':row['tcp'][0,3]+=.011
    elif fault=='orientation':row['tcp'][:3,:3]=row['tcp'][:3,:3]@rz(.051)
    elif fault=='distance':row['object_pose'][:3,3]=row['tcp'][:3,3]-[0.,0.,.12]
    elif fault=='jitter':row['tcp'][0,3]+=.003
    elif fault=='closed':row['finger_q'][0]=.0379
    elif fault=='cube_moved':row['object_pose'][0,3]+=.00051
    else:row[fault]=fault not in ('safety_valid','contact_valid','gripper_state_valid','object_present')
    value=phase.query_readiness(rows,episode(setup),setup['inputs'])
    assert not value['query_ready'] and not value['predicates'][predicate]


def test_cube_rotation_is_recorded_without_inventing_a_new_native_safety_threshold(setup):
    rows=samples(setup)
    for row in rows:row['object_pose'][:3,:3]=row['object_pose'][:3,:3]@rz(.7)
    value=phase.query_readiness(rows,episode(setup),setup['inputs'])
    assert value['query_ready'] and value['metrics']['maximum_cube_rotation_from_frozen_rad']==pytest.approx(.7)
    assert value['thresholds_are_native_safety'] is False


@pytest.mark.parametrize('fault',['count','gap','wrong_B','future','wrong_time','missing_contact','fake_bool','bad_quat','finger_range'])
def test_no_nearest_sample_missing_native_fields_or_fabricated_clock(setup,fault):
    rows=samples(setup)
    if fault=='count':rows=rows[:-1]
    elif fault=='gap':rows[3]['native_step']+=25
    elif fault=='wrong_B':
        for r in rows:r['native_step']-=25;r['time_s']-=.05
    elif fault=='future':rows[-1]['native_step']+=25;rows[-1]['time_s']+=.05
    elif fault=='wrong_time':rows[3]['time_s']+=.0001
    elif fault=='missing_contact':rows[3].pop('contact_valid')
    elif fault=='fake_bool':rows[3]['safety_valid']=1
    elif fault=='bad_quat':rows[3]['qpos'][3]=0
    elif fault=='finger_range':rows[3]['finger_q'][0]=.045
    with pytest.raises(ValueError):phase.query_readiness(rows,episode(setup),setup['inputs'])


@pytest.mark.parametrize('fault',['source_bytes','source_state','packet_clock','state_clock','qpos','frame','scene','unsafe','contact'])
def test_runtime_binding_requires_exact_source_packet_state_scene_and_frame(setup,fault):
    args=copy.deepcopy(setup['args'])
    if fault=='source_bytes':args['source_observation_bytes']+=b' '
    elif fault=='source_state':
        record=json.loads(args['source_observation_bytes']);record['body']['payload']['state_eef10'][0]+=1
        args['source_observation_bytes']=json.dumps(sealed(record['body'])).encode()
    elif fault=='packet_clock':
        raw=bytearray(args['decision_packet_bytes']);struct.pack_into('<Q',raw,8,3728);args['decision_packet_bytes']=bytes(raw)
    elif fault=='state_clock':args['measured_decision_state']['native_step']=3728
    elif fault=='qpos':args['measured_decision_state']['qpos'][0]=1
    elif fault=='frame':args['T_source_from_target'][0,3]+=.001
    elif fault=='scene':args['scene_sha256']='0'*64
    elif fault=='unsafe':args['measured_decision_state']['safety_valid']=False
    elif fault=='contact':args['measured_decision_state']['robot_object_contact']=True
    with pytest.raises(ValueError):phase.load_inputs(**args)


@pytest.mark.parametrize('fault',['goal','cube_hash','state','claim_FK','raw','provenance'])
def test_resealed_runtime_binding_cannot_change_source_goal_derivation_or_claim_fk(setup,fault):
    inputs=copy.deepcopy(setup['inputs']);body=unseal(inputs['frozen'])
    if fault=='goal':body['full_goal_TCP_target'][0][3]+=.001
    elif fault=='cube_hash':body['frozen_cube_pose_sha256']='0'*64
    elif fault=='state':body['source_state_eef10'][0]+=.001
    elif fault=='claim_FK':body['actual_cube_binding']='ACTUALLY_NATIVE_QUALIFIED'
    elif fault=='raw':inputs['raw']=inputs['raw'].copy();inputs['raw'][0,0]+=1
    elif fault=='provenance':inputs['provenance']['object_and_goal_are_policy_inputs']=True
    inputs['frozen']=sealed(body)
    with pytest.raises(ValueError):phase.make_program(inputs,episode(setup),setup['initial'],setup['decision'],setup['base'])


@pytest.mark.parametrize('key,value',[('ramp_atoms',1023),('planned_end_atom',512),('geometry_scale',True),
    ('geometry_scale',.2),('reference_anchor','DECISION_REANCHOR'),('tail_hold_atoms',9),('physics_dt_s',.004),
    ('source_observation_sha256','0'*64),('thresholds',{}),('run','relative')])
def test_static_manifest_cannot_change_clock_old_safety_or_anchor(setup,key,value):
    e=episode(setup);e[key]=value
    with pytest.raises(ValueError):phase.episode_contract(e)


def test_program_cannot_use_a_later_achieved_decision_as_reanchor(setup):
    changed=setup['decision'].copy();changed[0,3]+=.001
    with pytest.raises(ValueError,match='SAME_FROZEN_ACTUAL_DECISION_TCP'):
        phase.make_program(setup['inputs'],episode(setup),setup['initial'],changed,setup['base'])


def install_args(setup,offset=0):
    p=program(setup,.03)
    return dict(action=p.preview(offset),initial_tcp=setup['initial'],program=p,step=3703+25*offset,
        base=setup['base'],frozen_base=setup['base'].copy(),end_step=7103)


def test_new_install_contract_keeps_full_preview_ack_certifier_and_base_requirements(setup):
    for offset in (0,7,128,135):
        proof=phase.validate_install_preview(**install_args(setup,offset))
        assert proof['points_checked']==40 and proof['original_certify_chunk_required']
        assert proof['original_install_ACK_required'] and not proof['physics_advance_authorized']
    args=install_args(setup);args.update(step=2503,program=None,
        action=np.repeat(pose10(setup['initial'],phase.OPEN)[None],40,axis=0))
    assert phase.validate_install_preview(**args)['points_checked']==40


@pytest.mark.parametrize('fault',['future_suffix','last_row','working_grip','wrong_base','wrong_init',
    'wrong_end','late_atom','wrong_type','predecision_program','early_goal','nonfinite'])
def test_new_install_contract_cannot_use_a_single_good_row_or_skip_original_join(setup,fault):
    args=install_args(setup)
    if fault=='future_suffix':args['action']=args['program'].preview(1)
    elif fault=='last_row':args['action'][-1,0]+=.001
    elif fault=='working_grip':args['action'][0,9]=1.
    elif fault=='wrong_base':args['base']=args['base'].copy();args['base'][0]+=.001
    elif fault=='wrong_init':args['initial_tcp']=args['initial_tcp'].copy();args['initial_tcp'][0,3]+=.001
    elif fault=='wrong_end':args['end_step']+=25
    elif fault=='late_atom':args['step']+=1
    elif fault=='wrong_type':args['program']=object()
    elif fault=='predecision_program':args['step']=2503
    elif fault=='early_goal':args['step']=2503;args['program']=None
    elif fault=='nonfinite':args['action'][0,0]=np.nan
    with pytest.raises(ValueError):phase.validate_install_preview(**args)


def test_import_has_no_model_renderer_native_runtime_or_authority_side_effect():
    code="import sys;from go2arx5_vla.world_model import fusion_task_phase_initialization;assert not any(k in sys.modules for k in ('torch','mujoco','gr00t','rclpy','transformers'))"
    result=subprocess.run([sys.executable,'-B','-c',code],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
