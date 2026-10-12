"""Independent saved-data admission; kinematic forward only, never integration."""
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from .config import PARAMETERS as P,INSTRUCTION
from .engine import POINTS
from .codec import relative_action,decode,pose10
from .io import file_hash
from ..trajectory.so3 import quat_to_matrix

OBS_KEYS={'instruction','actual_tcp','actual_gripper','actual_gripper_speed','timestamp_s','valid','body_rgb','wrist_rgb'}


def near(a,b,tolerance=1e-8):
    assert np.shape(a)==np.shape(b)
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert np.isfinite(error) and error<=tolerance,error
    return error


def audit_episode(folder,negative_fixture=None):
    folder=Path(folder);read=lambda n:json.loads((folder/n).read_text())
    result=read('RESULT.json');card=read('CARD.json');samples=read('samples.json');refs=read('references.json')
    files=read('FILE_MANIFEST.json')
    for row in files:
        name=row['path']
        path=folder/name
        assert path.resolve().is_relative_to(folder.resolve()) and path.is_file()
        assert file_hash(path)==row['sha256'],name
    if negative_fixture=='privileged_leak':samples[0]['policy_obs']['teacher_stage']='FORBIDDEN_ORACLE'
    elif negative_fixture=='sensor_time':samples[0]['policy_obs']['wrist_rgb']['timestamp_s']+=.05
    elif negative_fixture=='future_history':samples[0]['causal_history']['previous_committed']=1
    elif negative_fixture=='codec_rotation':refs[0]['rows10'][0][3]+=1
    elif negative_fixture is not None:raise ValueError(negative_fixture)
    with np.load(folder/'physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in f.files}
    with np.load(folder/'highrate.npz',allow_pickle=False) as f:high=f['rows'].copy()
    n=len(samples);assert n==result['paired_samples']==len(physics['time_s'])
    near(physics['time_s'],np.arange(n)/20)
    assert len(high)==result['physics_steps'];near(high[:,0],np.arange(1,len(high)+1)*P['dt'])
    assert np.isfinite(high).all()
    assert np.max(np.abs(high[:,31:33]))<=P['finger_force_max_N']+1e-10
    assert np.max(np.abs(high[:,33:36]))<=P['tool_force_max_N']+1e-10
    assert np.max(np.abs(high[:,36:39]))<=P['tool_torque_max_Nm']+1e-10
    complete=result['status']=='VALID_COMPLETE' and n==641 and len(high)==16000
    if complete:assert not high[:,-1].any()
    xml=read('scene.xml.json');assert __import__('hashlib').sha256(xml['xml'].encode()).hexdigest()==xml['sha256']
    model=mujoco.MjModel.from_xml_string(xml['xml']);data=mujoco.MjData(model)
    assert model.neq==0 and model.nmocap==0 and model.nq==16 and model.nv==14 and model.nu==2
    nameid=lambda kind,name:mujoco.mj_name2id(model,kind,name)
    tool=nameid(mujoco.mjtObj.mjOBJ_BODY,'tool');cube=nameid(mujoco.mjtObj.mjOBJ_BODY,'cube')
    qidx=[int(model.jnt_qposadr[nameid(mujoco.mjtObj.mjOBJ_JOINT,name)]) for name in ['left_slide','right_slide']]
    vindex=[int(model.jnt_dofadr[nameid(mujoco.mjtObj.mjOBJ_JOINT,name)]) for name in ['left_slide','right_slide']]
    pads=[nameid(mujoco.mjtObj.mjOBJ_GEOM,name) for name in ['left_pad','right_pad']]
    cubegeom=nameid(mujoco.mjtObj.mjOBJ_GEOM,'cube_geom')
    data.qpos[qidx]=P['finger_travel']
    max_camera_error=0.;max_codec_error=0.;contact_counts=[0,0];max_snapshot_skew=0.;wrist_rig=None;body_rig=None
    minimum_wrist_red_pixels=256*256;maximum_contact_force_error=0.
    for i,s in enumerate(samples):
        obs=s['policy_obs'];assert set(obs)==OBS_KEYS and obs['instruction']==INSTRUCTION and obs['valid']
        time=i/20;assert abs(obs['timestamp_s']-time)<=1e-8
        snapshot=f"{card['id']}:{i*25}";assert s['snapshot_id']==snapshot
        h=s['causal_history'];assert h==dict(episode_id=card['id'],segment_id=0,previous_samples=i,previous_committed=i,reset=False,abort=False,takeover=False)
        near(obs['actual_tcp'],physics['tcp_world'][i]);near(s['privileged']['object_pose'],physics['object_pose'][i])
        if i:
            row=high[i*25-1];data.qpos[:]=row[1:17];data.qvel[:]=row[17:31];data.ctrl[:]=row[31:33];data.time=time
            data.xfrc_applied[tool]=row[33:39]
        mujoco.mj_forward(model,data)
        tcp=np.eye(4);tcp[:3,:3]=data.xmat[tool].reshape(3,3);tcp[:3,3]=data.xpos[tool]
        obj=np.eye(4);obj[:3,:3]=data.xmat[cube].reshape(3,3);obj[:3,3]=data.xpos[cube]
        near(tcp,obs['actual_tcp']);near(obj,physics['object_pose'][i])
        near(data.qpos[qidx],s['execution']['actual_finger_q']);near(data.qvel[vindex],s['execution']['actual_finger_qd'])
        assert abs(obs['actual_gripper']-np.mean(data.qpos[qidx])/P['finger_travel'])<1e-8
        assert abs(obs['actual_gripper_speed']-np.mean(data.qvel[vindex])/P['finger_travel'])<1e-8
        points=obj[:3,3]+POINTS@obj[:3,:3].T;near(points,s['privileged']['object_points'])
        for body,keylinear,keyangular in [(cube,'object_linear_velocity','object_angular_velocity')]:
            velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,velocity,0)
            near(velocity[3:],physics[keylinear][i]);near(velocity[:3],physics[keyangular][i])
        assert len(s['privileged']['contacts'])==data.ncon
        for j,recorded in enumerate(s['privileged']['contacts']):
            contact=data.contact[j];force=np.zeros(6);mujoco.mj_contactForce(model,data,j,force)
            assert recorded['geom0']==contact.geom1 and recorded['geom1']==contact.geom2
            near(recorded['normal'],contact.frame[:3]);assert abs(recorded['distance_m']-contact.dist)<1e-8
            error=abs(recorded['normal_force_N']-force[0]);maximum_contact_force_error=max(maximum_contact_force_error,error)
            assert error<=1e-5,error
        normals=[[],[]]
        for contact in s['privileged']['contacts']:
            assert contact['distance_m']>=-P['penetration_stop_m'] if complete else True
            for side,pad in enumerate(pads):
                if {contact['geom0'],contact['geom1']}=={pad,cubegeom} and contact['normal_force_N']>=P['minimum_contact_force_N']:
                    normal=np.asarray(contact['normal']);normals[side].append(normal if contact['geom1']==cubegeom else -normal)
        left,right=map(bool,normals);opposing=bool(left and right and any(np.dot(a,b)<=P['opposing_contact_dot_max'] for a in normals[0] for b in normals[1]))
        assert left==bool(physics['left_object_contact'][i]) and right==bool(physics['right_object_contact'][i])
        assert opposing==s['privileged']['opposing_contact'];contact_counts[0]+=left;contact_counts[1]+=right
        if physics['native_grasp'][i]:assert left and right and opposing and s['privileged']['relative_motion_valid']
        for camera in ['body_rgb','wrist_rgb']:
            image=obs[camera];g=s['geometry'][camera]
            assert image['snapshot_id']==g['snapshot_id']==snapshot and image['frame_index']==i
            max_snapshot_skew=max(max_snapshot_skew,abs(image['timestamp_s']-g['timestamp_s']),abs(g['timestamp_s']-time))
            assert max_snapshot_skew<=1e-8 and image['valid'] and image['source']=='actual_mujoco_renderer'
            path=folder/image['path'];assert path.resolve().is_relative_to(folder.resolve())
            assert file_hash(path)==image['sha256'];rgb=np.asarray(Image.open(path))
            assert rgb.shape==(256,256,3) and rgb.dtype==np.uint8 and float(rgb.std())>8
            if camera=='wrist_rgb':
                red=int(np.sum((rgb[:,:,0].astype(float)>1.5*rgb[:,:,1])&(rgb[:,:,0].astype(float)>1.5*rgb[:,:,2])&(rgb[:,:,0]>80)))
                minimum_wrist_red_pixels=min(minimum_wrist_red_pixels,red)
                if card['id'].startswith('P'):assert red>=20,'Wrist object visibility diagnostic failed'
            cid=nameid(mujoco.mjtObj.mjOBJ_CAMERA,camera);T=np.eye(4)
            T[:3,:3]=data.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.]);T[:3,3]=data.cam_xpos[cid]
            max_camera_error=max(max_camera_error,near(T,g['T_world_optical']))
            focal=256/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2));K=np.array([[focal,0,128],[0,focal,128],[0,0,1.]])
            near(K,g['K']);cam_points=(points-T[:3,3])@T[:3,:3];projection=cam_points@K.T
            near(projection[:,:2]/projection[:,2,None],g['object_points_uv'],1e-6)
            assert g['object_point_ids']==list(range(8)) and len(g['object_point_visible'])==8
            if camera=='wrist_rgb':
                rig=np.linalg.inv(tcp)@T
                if wrist_rig is None:wrist_rig=rig
                near(rig,wrist_rig)
            else:
                if body_rig is None:body_rig=T
                near(T,body_rig)
        command=s['supervision']
        if i==640:assert command is None
        elif command is not None:
            ref=refs[i//8];k=i%8
            assert command['sequence']==i and command['point_index']==k and command['ref_id']==ref['ref_id']
            assert command['actually_committed'] and abs(command['command_time_s']-time)<1e-8
            for key in ['desired','accepted','installed','committed']:near(command[key],ref['planned_H40'][k])
            assert command['desired_gripper']==command['committed_gripper']==ref['gripper_H40'][k]
    for ref in refs:
        i=ref['decision_index']*8;assert ref['prior_committed_count']==i
        assert ref['observation_snapshot']==samples[i]['snapshot_id']
        near(ref['anchor_actual_tcp'],samples[i]['policy_obs']['actual_tcp']);near(ref['offsets_s'],np.arange(40)/20)
        poses=np.asarray(ref['planned_H40']);grips=np.asarray(ref['gripper_H40'])
        assert poses.shape==(40,4,4) and grips.shape==(40,) and np.isin(grips,[0.,1.]).all()
        near(poses[0],ref['anchor_actual_tcp'])
        assert np.max(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1))<=P['reference_speed_m_s']/20+1e-8
        for pose,grip,row,q in zip(poses,grips,ref['rows10'],ref['raw_quaternion_xyzw']):
            near(row,pose10(pose,grip));near(quat_to_matrix(q),pose[:3,:3])
        anchor=np.asarray(ref['anchor_actual_tcp']);coded=relative_action(list(zip(poses,grips)),anchor)
        for row,pose,grip in zip(coded,poses,grips):
            restored,opening=decode(row,anchor);max_codec_error=max(max_codec_error,near(restored,pose))
            assert opening==grip
        assert ref['committed_prefix_count']==min(8,max(0,min(n,640)-i))
    windows=max(0,min(n-1,len(high)//25)-40+1)
    return dict(id=card['id'],status='PASS_FULL_DATA' if complete else 'PRESERVED_INCOMPLETE_NOT_ADMITTED',
        complete=complete,paired_samples=n,physics_steps=len(high),file_count=len(files),
        contact_samples=dict(left=contact_counts[0],right=contact_counts[1]),
        max_camera_kinematic_error=max_camera_error,max_sensor_skew_s=max_snapshot_skew,max_codec_roundtrip_error=max_codec_error,
        max_forward_recomputed_contact_force_error_N=maximum_contact_force_error,
        minimum_wrist_visible_red_pixels=minimum_wrist_red_pixels,
        visual_admission=bool(card['id'].startswith('P') and minimum_wrist_red_pixels>=20),
        actual_committed_H40_windows=windows,terminal_observation_has_no_invented_command=True,
        RGB_shape=[2,256,256,3],action_shape=[40,10],privileged_policy_separation=True,
        physics_sha256=file_hash(folder/'physics.npz'),manifest_sha256=file_hash(folder/'FILE_MANIFEST.json'),
        post_reset_integrations_in_auditor=0,GR00T=0)
