"""Synthetic record regressions only: zero engines, physics or demonstrations."""
from copy import deepcopy
import hashlib,json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.pi0294.multitask.night_cabinet_audit import audit
from scripts.pi0294.multitask.night_book_audit import audit as book_audit
from scripts.pi0294.multitask.robocasa_cabinet_cards import cards
from scripts.pi0294.multitask.robocasa_cabinet_scene import VERSION
from scripts.pi0294.multitask.robocasa_cabinet_assets import native_door_status
from scripts.pi0291_arx5_data.config import PARAMETERS as P

raw=lambda v:json.dumps(v,separators=(',',':'),allow_nan=False)
sha=lambda b:hashlib.sha256(b).hexdigest()


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    folder=tmp_path_factory.mktemp('SYNTHETIC_NOT_A_NATIVE_EPISODE')
    Image.new('RGB',(256,256)).save(folder/'frame.png');image=(folder/'frame.png').read_bytes()
    image_pin=dict(file='frame.png',bytes=len(image),sha256=sha(image))
    card=cards()['right'];cb=(json.dumps(card,indent=2)+'\n').encode();(folder/'CARD.json').write_bytes(cb)
    xml=('<mujoco model="'+VERSION+'"><worldbody><body><joint name="rc_doorhinge" range="0 1.57"/></body></worldbody></mujoco>').encode()
    (folder/'NATIVE_SCENE.private.xml').write_bytes(xml)
    ranges={'rc_doorhinge':[0.,1.57]};identity=np.eye(4).tolist()
    def state(step):
        angle=min(max((step-2500)/5000,0),1)*.95*1.57
        grasp=step==2600;finger=[.01,.01]if grasp else[P['finger_travel']]*2
        contacts=[]
        if grasp:
            for g,n in[(1,[1.,0.,0.]),(2,[-1.,0.,0.])]:
                contacts.append(dict(geom0=g,geom1=3,distance_m=0.,normal=n,normal_force_N=1.))
        if step>=8000:contacts.append(dict(geom0=3,geom1=4,distance_m=0.,normal=[0.,0.,1.],normal_force_N=1.))
        obj=np.eye(4);obj[:3,3]=[.1 if step>=8000 else-.5,0.,.5]
        tcp=np.eye(4);tcp[:3,3]=[0.,0.,.5]
        return dict(time_s=step*.002,qpos=[angle,0.,0.],qvel=[0.,0.,0.],tcp=tcp.tolist(),
            arm_q=[0.],arm_qd=[0.],finger_q=finger,safety_valid=True,native_abort=False,
            tool_linear_velocity=[0.,0.,0.],object_linear_velocity=[0.,0.,0.],object_angular_velocity=[0.,0.,0.],
            gripper_fraction=sum(finger)/2/P['finger_travel'],contacts=contacts,native_grasp=grasp,
            object_pose=obj.tolist(),hinge_angle_rad=angle,hinge_velocity_rad_s=0.,
            native_door_joint_positions={'rc_doorhinge':angle},native_door_joint_ranges=ranges,
            native_robocasa_door_status=native_door_status({'rc_doorhinge':angle},ranges),
            native_goal_site_position=[.1,0.,.5],native_goal_site_rotation=np.eye(3).tolist(),
            native_goal_site_half_size=[.05,.05,.05],native_book_body_origin=obj[:3,3].tolist())
    q=dict(source_port=VERSION,card_sha256=sha(cb),native_scene_sha256=sha(xml),passive_joint_name='rc_doorhinge',
        passive_joint_qposadr=0,passive_joint_dofadr=0,passive_joint_range=[0.,1.57],
        original_arm_bounds=[[-1.,1.]],original_finger_v_indices=[1,2],
        robot_geom_ids=[1,2],arm_geom_ids=[1,2],obstacle_geom_ids=[4],
        geom_bindings=dict(left_pad=1,right_pad=2,cube_geom=3,table=10,floor=11),
        steps=0,simulation_time=0.,new_constructor=1)
    rows=[dict(kind='qualification',value=q)]
    rows += [dict(kind='warmup_tick',value=dict(native_step=i,actual_state=state(i)))for i in range(1,2501)]
    rows.append(dict(kind='warmup_complete',value=dict(steps=2500)))
    plan=None
    for i in range(641):
        step=2500+i*25;s=state(step);t=s['time_s']
        cal={n:dict(timestamp_s=t,T_world_optical=identity,K=[[100.,0.,128.],[0.,100.,128.],[0.,0.,1.]])for n in('body_rgb','wrist_rgb')}
        rows.append(dict(kind='observation',value=dict(index=i,native_step=step,time_s=t,actual_state=s,
            policy_obs=dict(instruction=card['instruction'],body_rgb=image_pin,wrist_rgb=image_pin),calibration=cal)))
        if i==640:break
        if i%8==0:
            values=[[k*.001,0.,.5,1.,0.,0.,0.,1.,0.,1.]for k in range(40)]
            ident=sha(json.dumps(dict(index=i,source_time_s=t,rows10=values),sort_keys=True,separators=(',',':')).encode())
            plan=dict(ref_id=ident,observation_index=i,source_time_s=t,rows10=values,committed_prefix_count=0,
                role='UNEXECUTED_REFERENCE_EVIDENCE_NOT_TRAINING_LABEL')
            rows.append(dict(kind='planned_H40',value=plan))
        base=dict(ref_id=plan['ref_id'],observation_index=i,prefix_index=i%8,source_time_s=t,
            action_EEF10=plan['rows10'][i%8],start_native_step=step,training_admitted=False)
        rows.append(dict(kind='submitted_interval',value=dict(base,expected_end_native_step=step+25,completed=False)))
        rows += [dict(kind='native_tick',value=dict(native_step=j,actual_state=state(j)))for j in range(step+1,step+26)]
        rows.append(dict(kind='completed_interval',value=dict(base,end_native_step=step+25,completed=True)))
    terminal=dict(status='RAW_SOURCE_COMPLETE_UNAUDITED',error=None,actual_integrations=18500,actual_RGB_pairs=642,recorded_RGB_pairs=642)
    return folder,rows,terminal,image


def write(folder,rows,terminal,image):
    (folder/'frame.png').write_bytes(image)
    with(folder/'ACTUAL_RECORDS.jsonl').open('w')as f:
        for row in rows:f.write(raw(row)+'\n')
    (folder/'TERMINAL.json').write_text(raw(terminal))


def test_full_synthetic_cabinet_and_existing_book_integrity_routes(synthetic):
    folder,rows,terminal,image=synthetic;write(folder,rows,terminal,image)
    for check,key in[(audit,'native_ROBOCASA_goal_final'),(book_audit,'native_LIBERO_goal_final')]:
        v=check(folder)
        assert v['qualified_source_pilot']and v[key]and v['actual_integrations']==18500
        assert v['new_simulator_or_model_calls']==0 and not v['training_admitted']


@pytest.mark.parametrize('fault,expected',[
    ('late_command','COMMAND_MUST_PRECEDE'),('suffix_label','ONLY_ACTUALLY_COMPLETED'),
    ('altered_H40','ORIGINAL_H40_HASH'),('hidden_integration','FULL_5S'),('hidden_RGB','FULL_5S'),
    ('no_grasp','NO_ACTUAL_ORIGINAL_HANDLE_GRASP'),('already_open','TASK_MUST_BEGIN')])
def test_forged_timing_labels_costs_or_unsupported_success_never_qualify(synthetic,fault,expected):
    folder,original,term,image=synthetic;rows=deepcopy(original);terminal=dict(term)
    submitted=next(i for i,r in enumerate(rows)if r['kind']=='submitted_interval')
    if fault=='late_command':rows[submitted],rows[submitted+1]=rows[submitted+1],rows[submitted]
    elif fault=='suffix_label':
        for r in rows:
            if r['kind']in('submitted_interval','completed_interval')and r['value']['observation_index']==0:
                r['value']['action_EEF10']=next(x['value']['rows10'][9]for x in rows if x['kind']=='planned_H40')
    elif fault=='altered_H40':next(r['value']for r in rows if r['kind']=='planned_H40')['rows10'][31][0]+=.01
    elif fault=='hidden_integration':terminal['actual_integrations']+=1
    elif fault=='hidden_RGB':terminal['actual_RGB_pairs']+=1
    elif fault=='no_grasp':
        for r in rows:
            if r['kind']=='native_tick'and r['value']['native_step']==2600:
                r['value']['actual_state']['contacts']=[];r['value']['actual_state']['native_grasp']=False
    else:
        s=next(r['value']['actual_state']for r in rows if r['kind']=='observation')
        s['qpos'][0]=s['hinge_angle_rad']=1.5;s['native_door_joint_positions']={'rc_doorhinge':1.5}
        s['native_robocasa_door_status']=native_door_status({'rc_doorhinge':1.5},s['native_door_joint_ranges'])
    write(folder,rows,terminal,image);v=audit(folder)
    assert not v['qualified_source_pilot']and any(expected in x for x in v['integrity_problems'])


def test_goal_claim_without_raw_joint_support_raises_and_cannot_admit(synthetic):
    folder,original,terminal,image=synthetic;rows=deepcopy(original)
    s=next(r['value']['actual_state']for r in rows if r['kind']=='native_tick')
    s['native_door_joint_positions']={'rc_doorhinge':1.5}
    write(folder,rows,terminal,image)
    with pytest.raises(ValueError,match='ACTUAL_RAW_QPOS_QVEL'):audit(folder)


def test_auditor_reserves_once_on_same_native_account_before_compute(tmp_path,monkeypatch):
    from scripts.pi0294.multitask import night_scope as scope,night_book_chain as chain
    path=tmp_path/'control/NIGHT_SOURCE_ACCOUNT.json';path.parent.mkdir()
    now=scope.STOP_NEW-3600;monkeypatch.setattr(chain.time,'time',lambda:now)
    v=dict(scope=scope.SCOPE,caps=scope.CAPS,approval_sha256=scope.APPROVAL_SHA,
        stop_new_epoch=scope.STOP_NEW,deadline_epoch=scope.DEADLINE,revision=0,native_adopted=True,
        charged_upper={k:0 for k in scope.CAPS},operations=[])
    path.write_text(raw(v));chain.reserve_audit(tmp_path,'right','ROBOCASA_CABINET')
    after=path.read_bytes();value=json.loads(after)
    assert value['charged_upper']['Core_CPU_s']==40 and value['charged_upper']['attempts']==0
    with pytest.raises(PermissionError):chain.reserve_audit(tmp_path,'right','ROBOCASA_CABINET')
    assert path.read_bytes()==after
    assert chain.planned_tasks('ALL_PINNED')==(('BOOK','left'),('BOOK','right'),('ROBOCASA_CABINET','right'),('ROBOCASA_CABINET','left'))


def test_placeholder_rgb_cannot_qualify_even_with_matching_saved_hash(synthetic):
    import io
    folder,original,terminal,_=synthetic;rows=deepcopy(original);stream=io.BytesIO()
    Image.new('RGB',(1,1)).save(stream,format='PNG');image=stream.getvalue()
    for row in rows:
        if row['kind']=='observation':
            for name in('body_rgb','wrist_rgb'):
                row['value']['policy_obs'][name].update(bytes=len(image),sha256=sha(image))
    write(folder,rows,terminal,image);v=audit(folder)
    assert not v['qualified_source_pilot']and 'ACTUAL_256_RGB_PNG_NOT_A_PLACEHOLDER'in v['integrity_problems']


def test_mixed_plan_requires_all_fixed_cards_and_assets_before_first_child(tmp_path):
    from scripts.pi0294.multitask import night_book_chain as chain
    from scripts.pi0294.multitask.libero_book_cards import cards as book_cards
    fixed={'BOOK':book_cards(),'ROBOCASA_CABINET':cards()}
    path=tmp_path/'FIXED_TASK_CARDS.json';path.write_text(raw(fixed))
    assets=tmp_path/'ASSET_BINDINGS.private.json';assets.write_text(raw({'files':{}}))
    with pytest.raises(PermissionError,match='ALL_MIXED_TASK_ASSET'):chain.checked_plan(tmp_path,'ALL_PINNED')
    definitions={n:'def_'+n for n in('cabinet_single','slab','bar_handle')}
    textures={n:'tex_'+n for n in('wood','metal')}
    assets.write_text(raw(dict(files={n:{}for n in[*definitions.values(),*textures.values()]},
        robocasa_definition_members=definitions,robocasa_texture_members=textures)))
    assert chain.checked_plan(tmp_path,'ALL_PINNED')==chain.planned_tasks('ALL_PINNED')
    fixed['ROBOCASA_CABINET']['left']['task_duration_s']=31.;path.write_text(raw(fixed))
    with pytest.raises(PermissionError,match='ONE_SEALED_PREDECLARED_TASK_CARD'):chain.checked_plan(tmp_path,'ALL_PINNED')


def test_equal_valued_prefix_from_another_h40_is_not_current_command_label(synthetic):
    folder,original,terminal,image=synthetic;rows=deepcopy(original)
    future=next(r['value']for r in rows if r['kind']=='planned_H40'and r['value']['observation_index']==8)
    for r in rows:
        if r['kind']in('submitted_interval','completed_interval')and r['value']['observation_index']==0:
            r['value']['ref_id']=future['ref_id']
    write(folder,rows,terminal,image);v=audit(folder)
    assert not v['qualified_source_pilot']and 'ONLY_ACTUALLY_COMPLETED_SUBMITTED_PREFIX_CAN_LABEL'in v['integrity_problems']
