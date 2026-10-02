"""Declared zero-physics virtual tool; reuse original scene and camera rasterizer."""
import isaacgym
from isaacgym import gymapi
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267'
sys.path[:0]=[str(BASE/'PI0250/runtime/reference-317a/src'),str(BASE/'PI0266/code')]
from pi0259_renderer import Renderer as OriginalRenderer
from pi0259_observation import visibility
from pi0255_motion3 import scene,write,sha

# TCP is the palm's +z face centre; fingers extend along +z from that plane.
PARTS=[('palm',[0,0,-.020],[.040,.020,.040]),
       ('finger_left',[0,.015,.015],[.008,.008,.030]),
       ('finger_right',[0,-.015,.015],[.008,.008,.030])]
TCP_FROM_LINK6=np.array([.1424,0,.0001057])


def collision(p,r,obstacle,half):
    hits=[]
    for name,offset,size in PARTS:
        center=p+r@np.array(offset);h=np.array(size)/2
        if center[2]-np.dot(abs(r[2]),h)<=0:hits.append(name+':ground')
        # Exact OBB/AABB separating-axis test, including nine edge cross products.
        axes=[*r.T,*np.eye(3),*(np.cross(a,b) for a in r.T for b in np.eye(3))]
        if all(abs(np.dot(center-obstacle,a))<=np.dot(abs(r.T@a),h)+np.dot(abs(a),half)+1e-12
               for a in axes if np.linalg.norm(a)>1e-12):hits.append(name+':obstacle')
    return hits


class Renderer(OriginalRenderer):
    def __init__(self,registry,family):
        self.reg=registry;self.family=family;self.gym=gymapi.acquire_gym();params=gymapi.SimParams()
        params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
        params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=2
        self.sim=self.gym.create_sim(0,0,gymapi.SIM_PHYSX,params);assert self.sim is not None
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);self.gym.add_ground(self.sim,plane)
        self.env=self.gym.create_env(self.sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1)
        self.actors={}
        for index,(name,offset,size) in enumerate(PARTS):
            opts=gymapi.AssetOptions();opts.fix_base_link=True;opts.disable_gravity=True
            asset=self.gym.create_box(self.sim,*size,opts)
            actor=self.gym.create_actor(self.env,asset,gymapi.Transform(),name,1,0,index+1)
            self.gym.set_rigid_body_color(self.env,actor,0,gymapi.MESH_VISUAL,gymapi.Vec3(.55,.55,.55))
            self.actors[name]=actor
        self.scene_ids={};self.add_scene(family);self.cameras={}
        props=gymapi.CameraProperties();props.width=props.height=96;props.horizontal_fov=registry['renderer']['horizontal_fov_deg']
        body=self.gym.create_camera_sensor(self.env,props);root=np.array(registry['root_pose'])
        pos=Rotation.from_quat(root[3:]).apply(registry['renderer']['rigs_base_positions'][registry['renderer']['selected_rig']])+root[:3]
        self.gym.set_camera_location(body,self.env,gymapi.Vec3(*pos),gymapi.Vec3(*registry['renderer']['look_at_world']))
        props=gymapi.CameraProperties();props.width=props.height=96
        wrist=self.gym.create_camera_sensor(self.env,props);attachment=gymapi.Transform()
        # Palm actor origin = TCP-[0,0,.020]; preserve the original link6->camera transform.
        attachment.p=gymapi.Vec3(*(np.array(registry['renderer']['wrist_attachment'])-TCP_FROM_LINK6+[0,0,.020]))
        attachment.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0),np.deg2rad(registry['renderer']['wrist_local_y_rotation_deg']))
        handle=self.gym.get_actor_rigid_body_handle(self.env,self.actors['palm'],0)
        self.gym.attach_camera_to_body(wrist,self.env,handle,attachment,gymapi.FOLLOW_TRANSFORM)
        self.cameras=dict(body=body,wrist=wrist);self.gym.prepare_sim(self.sim)

    def place(self,p,r,timestamp=0.):
        p=np.asarray(p,float);r=np.asarray(r,float)
        assert p.shape==(3,) and r.shape==(3,3) and np.isfinite(p).all() and np.isfinite(r).all()
        assert np.max(abs(r@r.T-np.eye(3)))<1e-8 and np.linalg.det(r)>.999999
        before=(repr(random.getstate()),repr(np.random.get_state()))
        errors=[]
        for name,offset,size in PARTS:
            expected=p+r@np.array(offset);tr=gymapi.Transform();tr.p=gymapi.Vec3(*expected);tr.r=gymapi.Quat(*Rotation.from_matrix(r).as_quat())
            handle=self.gym.get_actor_rigid_body_handle(self.env,self.actors[name],0);self.gym.set_rigid_transform(self.env,handle,tr)
            got=self.gym.get_actor_rigid_body_states(self.env,self.actors[name],gymapi.STATE_ALL)[0]['pose']
            errors.append(float(np.max(abs(np.array(list(got['p']))-expected))))
            assert np.max(abs(Rotation.from_quat(np.array(list(got['r']))).as_matrix()-r))<1e-6
        self.gym.step_graphics(self.sim);self.gym.render_all_camera_sensors(self.sim)
        frames={};views={};projections={}
        for name,camera in self.cameras.items():
            frames[name]=np.asarray(self.gym.get_camera_image(self.sim,self.env,camera,gymapi.IMAGE_COLOR)).reshape(96,96,4)[...,:3].copy()
            frames[name+'_segmentation']=np.asarray(self.gym.get_camera_image(self.sim,self.env,camera,gymapi.IMAGE_SEGMENTATION)).reshape(96,96).copy()
            frames[name+'_depth']=np.asarray(self.gym.get_camera_image(self.sim,self.env,camera,gymapi.IMAGE_DEPTH)).reshape(96,96).copy()
            views[name]=np.asarray(self.gym.get_camera_view_matrix(self.sim,self.env,camera)).tolist()
            projections[name]=np.asarray(self.gym.get_camera_proj_matrix(self.sim,self.env,camera)).tolist()
        assert before==(repr(random.getstate()),repr(np.random.get_state())),'capture mutated RNG'
        assert max(errors)<1e-6 and self.gym.get_sim_time(self.sim)==0.
        return frames,dict(timestamp=float(timestamp),TCP_position_world=p.tolist(),TCP_rotation_world=r.tolist(),camera_view=views,
            camera_projection=projections,scene_ids=self.scene_ids,placement_max_m=max(errors),physics_steps=0,RNG_unchanged=True)


def freeze():
    card=json.loads((ROOT/'design/SCENE_CARD.json').read_text());reg=card['registry'];path=ROOT/'R1/design/VIRTUAL_TOOL_CARD.json'
    value=dict(schema='PI0267_VEEF_KINEMATIC_V1',source_scene_sha256=sha(ROOT/'design/SCENE_CARD.json'),parts=PARTS,
        TCP_mapping='palm +z face centre; orientation is world-from-tool',camera=reg['renderer'],canonical_root=reg['root_pose'],
        TCP_from_link6_m=TCP_FROM_LINK6.tolist(),TCP_from_camera_m=(np.array(reg['renderer']['wrist_attachment'])-TCP_FROM_LINK6).tolist(),
        camera_rotation_y_deg=reg['renderer']['wrist_local_y_rotation_deg'],collision='primitive union versus original obstacle and ground; markers noncollidable',
        data_backend='CONTROLLER_INDEPENDENT_VIRTUAL_TOOL_SUPERVISION',policy_inputs=['body_rgb','wrist_rgb','state_v1','language'])
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(value))
    else:write(path,value)
    return card


if __name__=='__main__':
    card=freeze();reg=card['registry'];family=card['families'][0];renderer=Renderer(reg,family)
    try:
        pose=np.array(reg['initial_TCP_world']);frames,info=renderer.place(pose[:3,3],pose[:3,:3])
        frames2,info2=renderer.place(pose[:3,3],pose[:3,:3]);assert info==info2
        assert all(np.array_equal(frames[k],frames2[k],equal_nan=True) for k in frames)
        data=scene(reg,family);view=np.array(info['camera_view']['body']);projection=np.array(info['camera_projection']['body'])
        eye=np.linalg.inv(view)[3,:3];root_pose=np.array(reg['root_pose'])
        assert np.max(abs(eye-(Rotation.from_quat(root_pose[3:]).apply(reg['renderer']['rigs_base_positions'][reg['renderer']['selected_rig']])+root_pose[:3])))<1e-5
        projection_checks=[]
        for name,point in [('target0',data['targets'][0]),('target1',data['targets'][1])]:
            clip=np.r_[point,1]@view@projection;pixel=(clip[:2]/clip[3]*[1,-1]+1)*48
            yx=np.argwhere(frames['body_segmentation']==renderer.scene_ids[name]);assert len(yx)>0
            measured=yx[:,::-1].mean(0);assert np.linalg.norm(pixel-measured)<5
            projection_checks.append(dict(name=name,predicted_xy=pixel.tolist(),segmentation_centroid_xy=measured.tolist()))
        target=np.array(data['targets'][0]);occluder=eye+.15*(target-eye)+[0,0,.020]
        blocked,blocked_info=renderer.place(occluder,np.eye(3),7.)
        before_count=int(np.sum(frames['body_segmentation']==renderer.scene_ids['target0']))
        after_count=int(np.sum(blocked['body_segmentation']==renderer.scene_ids['target0']))
        assert after_count<before_count,('REAL_OCCLUSION_NOT_COUNTED',before_count,after_count)
        rp=pose[:3,3]+[.001,-.002,.003];rr=Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix()
        moved,moved_info=renderer.place(rp,rr,8.)
        camera_eye=np.linalg.inv(np.array(moved_info['camera_view']['wrist']))[3,:3]
        assert np.max(abs(camera_eye-(rp+rr@(np.array(reg['renderer']['wrist_attachment'])-TCP_FROM_LINK6))))<1e-5
        assert collision(np.array(data['obstacle']),np.eye(3),np.array(data['obstacle']),np.array(reg['geometry']['obstacle_half_extents_m']))
        assert any('ground' in h for h in collision(np.array([2.,2.,0.]),np.eye(3),np.array(data['obstacle']),np.array(reg['geometry']['obstacle_half_extents_m'])))
        assert not collision(np.array([2.,2.,2.]),rr,np.array(data['obstacle']),np.array(reg['geometry']['obstacle_half_extents_m']))
        root=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1');root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(root/'renderer_start.npz',**frames)
        np.savez_compressed(root/'renderer_occlusion.npz',**blocked)
        np.savez_compressed(root/'renderer_nonidentity.npz',**moved)
        checks=[dict(task=task,command=c,check=visibility(frames,renderer.scene_ids,task,c,common_start=True)) for task in ('M1','M3') for c in (0,1)]
        write(ROOT/'R1/evidence/RENDERER_FIXTURES.json',dict(status='PASS_RENDER_IDENTITY_PROJECTION_RIGID_EXTRINSIC_OCCLUSION_COLLISION',
            info=info,checks=checks,projection_checks=projection_checks,occlusion=dict(before_pixels=before_count,after_pixels=after_count),
            nonidentity=moved_info,frame_pairs_captured=4,physics_steps=0))
        print('PASS_RENDER_IDENTITY_PROJECTION_RIGID_EXTRINSIC_OCCLUSION_COLLISION',flush=True)
    finally:renderer.close()
