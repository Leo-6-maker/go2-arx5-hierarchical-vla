"""Isolated prescribed-pose IsaacGym renderer. Never constructs WBC or steps physics."""
import isaacgym  # Native import order is required.
from isaacgym import gymapi
from pathlib import Path
import argparse
import ast
import copy
import json
import math
import sys
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from go2arx5_vla.wbc.native.kinematics import rpy_to_matrix,axis_angle_to_matrix,homogeneous,NativeArmKinematics
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix

ROOT=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258')


from pi0258_geometry import full_fk, defaults

class Renderer:
    def __init__(self,registry,family=None):
        self.reg=registry;self.gym=gymapi.acquire_gym();params=gymapi.SimParams()
        params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.use_gpu_pipeline=False
        params.physx.use_gpu=False;params.physx.num_threads=2
        self.sim=self.gym.create_sim(0,0,gymapi.SIM_PHYSX,params);assert self.sim is not None
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);self.gym.add_ground(self.sim,plane)
        self.env=self.gym.create_env(self.sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1)
        p=Path(registry['urdf']['path']);self.urdf=ET.parse(p).getroot()
        opts=gymapi.AssetOptions();opts.fix_base_link=True;opts.disable_gravity=True;opts.collapse_fixed_joints=False
        opts.flip_visual_attachments=False
        # Prescribed link actors avoid articulation's deferred FK requiring a physics step.
        # These private mesh views are renderer-only, never exported or used by WBC.
        views=ROOT/'runtime/private_link_views';views.mkdir(exist_ok=True)
        self.actors={};self.names=[]
        for link in self.urdf.findall('link'):
            name=link.attrib['name'];robot=ET.Element('robot',name='PI0258_render_only_'+name)
            for material in self.urdf.findall('material'):robot.append(copy.deepcopy(material))
            piece=copy.deepcopy(link)
            for mesh in piece.iter('mesh'):mesh.attrib['filename']=str((p.parent/mesh.attrib['filename']).resolve())
            robot.append(piece);view=views/(name+'.urdf');ET.ElementTree(robot).write(str(view))
            asset=self.gym.load_asset(self.sim,str(views),view.name,opts)
            actor=self.gym.create_actor(self.env,asset,gymapi.Transform(),name,1,0,len(self.names)+1)
            assert self.gym.get_actor_rigid_body_count(self.env,actor)==1
            self.actors[name]=actor;self.names.append(name)
        source=json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256/evidence/runtime_readback.json').read_text())
        self.defaults=defaults(next(x['path'] for x in source['files'] if x['name']=='native_config'))
        self.cameras={}
        for name,link,translation in [('body',self.names[0],(1.15,-1.15,.95)),('wrist','x5_link6',(.28,-.10,.12))]:
            props=gymapi.CameraProperties();props.width=props.height=96
            c=self.gym.create_camera_sensor(self.env,props);self.cameras[name]=c
            attachment=gymapi.Transform();attachment.p=gymapi.Vec3(*translation)
            attachment.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0),math.radians(135))
            body=self.gym.get_actor_rigid_body_handle(self.env,self.actors[link],0)
            self.gym.attach_camera_to_body(c,self.env,body,attachment,gymapi.FOLLOW_TRANSFORM)
        self.scene_ids={}
        if family is not None:self.add_scene(family)
        self.gym.prepare_sim(self.sim)
        self.kin=NativeArmKinematics.from_urdf(p)

    def add_scene(self,family):
        import pi0255_motion3 as old
        data=old.scene(self.reg,family);geo=self.reg['geometry']
        def marker(position,size,color,name,box=False):
            options=gymapi.AssetOptions();options.fix_base_link=True;options.disable_gravity=True
            asset=self.gym.create_box(self.sim,*size,options) if box else self.gym.create_sphere(self.sim,float(size),options)
            pose=gymapi.Transform();pose.p=gymapi.Vec3(*position);seg=100+len(self.scene_ids)
            actor=self.gym.create_actor(self.env,asset,pose,name,1,0,seg)
            self.gym.set_rigid_body_color(self.env,actor,0,gymapi.MESH_VISUAL,gymapi.Vec3(*color));self.scene_ids[name]=seg
        colors=dict(red=(.9,.03,.03),blue=(.03,.15,.9),green=(.03,.85,.1))
        for i,p in enumerate(data['targets']):
            marker(p,geo['target_radius_m'],colors[family['factors']['colors'][i]],'target'+str(i))
            rotation=rpy_to_matrix(np.array([0.,0.,geo['target_yaw_rad'][i]]))
            for a,c in enumerate(((1.,0,0),(0,1.,0),(0,0,1.))):marker(p+rotation[:,a]*geo['axes_length_m'],geo['axes_marker_radius_m'],c,'axis%d_%d'%(i,a))
        marker(data['obstacle'],2*np.asarray(geo['obstacle_half_extents_m']),(.45,.45,.45),'obstacle',True)
        marker(data['goal'],geo['target_radius_m'],(1.,.35,.03),'goal')
        for i,a in enumerate(np.linspace(-3*math.pi/4,3*math.pi/4,geo['contour_points'])):
            marker(data['center']+[data['radius']*math.cos(a),data['radius']*math.sin(a),0],geo['contour_marker_radius_m'],(.95,.8,.05),'contour'+str(i))
        for name,a,color in [('black_A',-3*math.pi/4,(.03,.03,.03)),('white_B',3*math.pi/4,(1.,1.,1.))]:
            marker(data['center']+[data['radius']*math.cos(a),data['radius']*math.sin(a),0],geo['start_marker_radius_m'],color,name)

    def place(self,q):
        positions=dict(self.defaults);positions.update(zip(self.kin.joint_names,map(float,q)))
        root=homogeneous(quat_to_matrix(np.array(self.reg['root_pose'][3:])),np.array(self.reg['root_pose'][:3]))
        poses=full_fk(self.urdf,positions,root)
        for name,actor in self.actors.items():
            t=poses[name];transform=gymapi.Transform();transform.p=gymapi.Vec3(*t[:3,3])
            transform.r=gymapi.Quat(*matrix_to_quat(t[:3,:3]))
            body=self.gym.get_actor_rigid_body_handle(self.env,actor,0)
            self.gym.set_rigid_transform(self.env,body,transform)
        self.gym.step_graphics(self.sim);self.gym.render_all_camera_sensors(self.sim)
        observed={n:self.gym.get_actor_rigid_body_states(self.env,a,gymapi.STATE_ALL)[0] for n,a in self.actors.items()}
        error=max(float(np.max(abs(np.array(list(observed[n]['pose']['p']))-poses[n][:3,3]))) for n in self.names)
        rotation_error=max(float(np.max(abs(quat_to_matrix(np.array(list(observed[n]['pose']['r'])))-poses[n][:3,:3]))) for n in self.names)
        assert error<1e-5 and rotation_error<1e-5,('render placement readback failed',error,rotation_error)
        assert float(self.gym.get_sim_time(self.sim))==0.,'hidden integration'
        frames={n:np.asarray(self.gym.get_camera_image(self.sim,self.env,c,gymapi.IMAGE_COLOR)).reshape(96,96,4)[...,:3].copy() for n,c in self.cameras.items()}
        segmentation={n:np.asarray(self.gym.get_camera_image(self.sim,self.env,c,gymapi.IMAGE_SEGMENTATION)).reshape(96,96).copy() for n,c in self.cameras.items()}
        independent_tcp=poses['x5_link6']@homogeneous(np.eye(3),self.kin.tcp_offset_xyz)
        assert np.max(abs(independent_tcp-root@self.kin.tcp_pose(q)))<1e-10
        info=dict(body_transform_error=error,body_rotation_matrix_error=rotation_error,physics_steps=0,sim_time=0.,FK_TCP_world=independent_tcp[:3,3].tolist(),
            camera_view={n:np.asarray(self.gym.get_camera_view_matrix(self.sim,self.env,c)).tolist() for n,c in self.cameras.items()},
            camera_projection={n:np.asarray(self.gym.get_camera_proj_matrix(self.sim,self.env,c)).tolist() for n,c in self.cameras.items()})
        info['visible_pixels']={name:{camera:int((seg==idx).sum()) for camera,seg in segmentation.items()} for name,idx in self.scene_ids.items()}
        return frames,info

    def close(self):self.gym.destroy_sim(self.sim)


def development_starts():
    from pi0258_kinematic import reference
    from pi0255_motion3 import sha,write
    reg=json.loads((ROOT/'design/registry.json').read_text());seal=json.loads((ROOT/'design/generation_seal.json').read_text())
    assert seal['authority']=='8944d569d9f83baa74f0cb28c1113ba9659e528a' and seal['canonical_comment'] and seal['commit']
    assert seal['registry_sha256']==sha(ROOT/'design/registry.json')
    for name,digest in seal['source_files'].items():assert sha(ROOT/'runtime'/name)==digest,name
    results=[]
    for family in reg['families'][:6]:
        output=ROOT/'development'/family['family_id'];output.mkdir(parents=True,exist_ok=True)
        if (output/'starts.json').exists():
            results+=json.loads((output/'starts.json').read_text())['sequences'];continue
        r=Renderer(reg,family)
        try:
            frames,info=r.place(np.array(reg['arm_default']));again,again_info=r.place(np.array(reg['arm_default']))
            assert all(np.array_equal(frames[k],again[k]) for k in frames)
            state=np.r_[info['FK_TCP_world'],np.eye(3)[:2].reshape(-1),.5]
            np.savez_compressed(output/'common_start.npz',**frames,state_v1=state)
            write(output/'render_readback.json',info)
            rows=[]
            for sequence in [s for s in reg['sequences'] if s['family_id']==family['family_id']]:
                task=sequence['task'];command=sequence['command']
                required=seal['visibility_required'][task][str(command)]
                invisible=[name for name in required if sum(info['visible_pixels'][name].values())==0]
                times=np.arange(round(reg['experts'][task]['nominal_end_s']/.05)+1)*.05
                p,q=reference(reg,family,task,command,times)
                name=sequence['sequence_id']+'-reference.npz'
                np.savez_compressed(output/name,synthetic_reference_time=times,desired_position=p,desired_quaternion=q)
                row=dict(sequence=sequence,start_visible=not invisible,missing_cues=invisible,required_cues=required,
                    reference_file=name,reference_sha256=sha(output/name),common_input_sha256=sha(output/'common_start.npz'),
                    status='CONTINUE_GEOMETRY' if not invisible else 'VALID_DATA_FAIL_COMMAND_CUE_INVISIBLE_AT_SHARED_START',
                    geometry_executed=False,model_queries=0,physics_steps=0)
                rows.append(row)
            write(output/'starts.json',dict(sequences=rows,actual_pair_RGB_equal=True,actual_pair_FK_equal=True,physics_steps=0));results+=rows
        finally:r.close()
    totals={task:dict(visible=sum(x['sequence']['task']==task and x['start_visible'] for x in results),planned=12,needed=10) for task in ('M1','M2','M3')}
    write(ROOT/'evidence/development_start_visibility.json',dict(sequences=results,totals=totals,physics_steps=0,model_queries=0,
        note='Necessary observability condition only; a visible start is not complete sequence admission.'))
    print(json.dumps(totals),flush=True)


if __name__=='__main__':
    if '--development-starts' in sys.argv:
        development_starts();raise SystemExit(0)
    reg=json.loads((ROOT/'design/registry.json').read_text());r=Renderer(reg,reg['families'][0])
    try:
        a,ia=r.place(np.asarray(reg['arm_default']));b,ib=r.place(np.array([.1,.2,.3,.1,-.1,.1]));c,ic=r.place(np.asarray(reg['arm_default']))
        assert all(np.array_equal(a[k],c[k]) for k in a),'same pose rendered differently'
        assert any(not np.array_equal(a[k],b[k]) for k in a),'rendered image ignores placed joints'
        np.savez_compressed(ROOT/'evidence/renderer_fixture.npz',**{'nominal_'+k:v for k,v in a.items()},**{'changed_'+k:v for k,v in b.items()})
        (ROOT/'evidence/renderer_fixture.json').write_text(json.dumps(dict(status='PASS_ZERO_STEP_PLACEMENT_FIXTURE',nominal=ia,changed=ib,restored=ic),indent=2))
        print('PASS_ZERO_STEP_PLACEMENT_FIXTURE',flush=True)
    finally:r.close()
