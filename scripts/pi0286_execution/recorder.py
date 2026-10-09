"""Counted source physics and command bridge; no learned model or target WBC."""
import json, os, time, itertools, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import sapien
import gymnasium as gym
from scipy.spatial.transform import Rotation
from PIL import Image
import mani_skill.envs
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix
from contract import ROOT, sha, write

LIMITS = dict(starts=176, control=116160, physics=580800, rgb=116160, teacher=43000, dispatch=232320)
RAW=Path(os.environ.get('PI0286_RECORD_ROOT',str(ROOT/'source')))
if RAW!=ROOT/'source':
    tier=json.loads((ROOT/'design/RECORDING_TIER.json').read_text())
    assert tier['status']=='PASS_CLOSED_PILOT_CUMULATIVE_LEDGER_DURABLE_LOCAL_TIER'
    assert RAW.resolve()==Path(tier['record_root']).resolve()

def opening(q):
    # Native rubber-pad inner faces are 7.58mm +/- half of 15.2mm.
    return (np.asarray(q)[-2:].sum()-.00004)/.08

def array(x):
    return x.detach().cpu().numpy().copy() if isinstance(x,torch.Tensor) else np.asarray(x).copy()

def matrix(pose):
    return array(pose.to_transformation_matrix())[0]

def native_action(world, root, gripper):
    assert np.isfinite(world).all() and np.isfinite(gripper)
    assert np.max(np.abs(world[:3,:3].T@world[:3,:3]-np.eye(3))) < 1e-5
    assert abs(np.linalg.det(world[:3,:3])-1) < 1e-5
    local = np.linalg.inv(root)@world
    angles = Rotation.from_matrix(local[:3,:3]).as_euler('XYZ')
    rebuilt = array(euler_angles_to_matrix(torch.as_tensor(angles,dtype=torch.float64),'XYZ'))
    assert np.max(np.abs(rebuilt-local[:3,:3])) < 1e-5
    assert -.1 <= gripper <= 1.1
    sent = np.clip(gripper,0,1)
    return np.r_[local[:3,3],angles,2*sent-1].astype(np.float32)

def self_check():
    root=np.eye(4); root[:3,3]=[-.615,0,0]
    world=np.eye(4);world[:3,:3]=Rotation.from_euler('XYZ',[.34,-.23,1.12]).as_matrix();world[:3,3]=[.2,-.1,.3]
    a=native_action(world,root,.25)
    assert np.max(np.abs(Rotation.from_euler('XYZ',a[3:6]).as_matrix()-world[:3,:3]))<1e-5
    assert abs(a[-1]+.5)<1e-6 and np.max(np.abs(a[:3]-(world[:3,3]-root[:3,3])))<1e-6
    for g in (-.10001,1.10001,float('nan')):
        try:native_action(world,root,g)
        except AssertionError:pass
        else:raise AssertionError('invalid actuator fraction admitted')

class Ledger:
    def __init__(self):
        global BUDGET_COUNTS
        self.path=RAW/'evidence/budget.jsonl'; self.path.parent.mkdir(parents=True,exist_ok=True)
        self.counts=Counter() if BUDGET_COUNTS is None else BUDGET_COUNTS
        if BUDGET_COUNTS is None and self.path.exists():
            for line in self.path.read_text().splitlines():
                row=json.loads(line); self.counts[row['kind']]+=1
        BUDGET_COUNTS=self.counts
        self.stream=self.path.open('a',buffering=1)
    def reserve(self,kind,slot):
        from r2 import phase_allowed
        phase_allowed('physics')
        assert self.counts[kind]<LIMITS[kind], (kind,self.counts[kind])
        self.counts[kind]+=1
        self.stream.write(json.dumps(dict(kind=kind,ordinal=self.counts[kind],slot=slot,wall_ns=time.time_ns()))+'\n')
    def sync(self):
        self.stream.flush();os.fsync(self.stream.fileno())
    def close(self):
        self.sync();self.stream.close()

ACTIVE=None
BUDGET_COUNTS=None
OLD_STEP=ManiSkillScene.step
def counted_step(scene):
    rec=ACTIVE; assert rec is not None
    rec.ledger.reserve('physics',rec.slot)
    OLD_STEP(scene)
    rec.physics_completed+=1
    if rec.env is not None:rec.sample()
ManiSkillScene.step=counted_step

class Episode:
    def __init__(self,slot,seed,mode,environment,env_kwargs):
        global ACTIVE
        assert ACTIVE is None
        self.slot=slot;self.seed=seed;self.mode=mode;self.env=None;self.physics_completed=0
        parent=RAW/'episodes'/slot
        existing=sorted(parent.glob('attempt-*/RESULT.json'))
        assert not any(json.loads(p.read_text())['status'].startswith('VALID') for p in existing), 'Reuse the first valid result'
        assert all((p/'RESULT.json').exists() for p in parent.glob('attempt-*')), 'Actual incomplete handle must be audited before another start'
        if existing:
            starts=list((RAW/'episodes').glob('*/attempt-*/START.json'))
            replacements=len(starts)-len({p.parent.parent.name for p in starts})
            assert replacements<16, 'Same-condition technical replacement allowance exhausted'
        self.folder=parent/f'attempt-{len(existing):03d}'
        self.folder.mkdir(parents=True)
        self.ledger=Ledger();self.ledger.reserve('starts',slot);self.ledger.sync()
        write(self.folder/'START.json',dict(slot=slot,seed=seed,mode=mode,pid=os.getpid(),status='STARTED_BEFORE_NATIVE_CONSTRUCTION'))
        self.states=[];self.contacts=self.folder.joinpath('contacts.jsonl').open('a',buffering=1)
        self.inputs=[];self.commands=[];self.phase=[];self.installed=[];self.ik_holds=[];self.control=0
        self.initialized=False;ACTIVE=self
        try:
            self.env=gym.make(environment,robot_uids='panda_wristcam',num_envs=1,
                sim_backend='cpu',render_backend='cuda:0',obs_mode='none',reward_mode='none',
                control_mode=mode,max_episode_steps=660,**env_kwargs).unwrapped
            self.env.reset(seed=seed)
            self.initialized=True
            assert self.env.sim_freq==100 and self.env.control_freq==20
            assert not self.env.gpu_sim_enabled and self.env.device.type=='cpu'
            assert set(self.env._sensors)=={'base_camera','hand_camera'}
            self.arm=self.env.agent.controller.controllers['arm']
            if mode=='pd_ee_pose':
                assert not self.arm.config.use_delta and not self.arm.config.use_target and not self.arm.config.normalize_action
                self.fk=self.arm.kinematics.pmodel;self.fk_link=self.arm.kinematics.end_link_idx
                self.ik_original=self.arm.kinematics.compute_ik
                def observed_ik(*a,**kw):
                    result=self.ik_original(*a,**kw);self.ik_holds.append(result is None);return result
                self.arm.kinematics.compute_ik=observed_ik
            else:
                # Native FK instance is separate from the live articulation; no pose setter.
                from mani_skill.agents.controllers.utils.kinematics import Kinematics
                kin=Kinematics(self.env.agent.urdf_path,'panda_hand_tcp',self.env.agent.robot,self.arm.active_joint_indices)
                self.fk=kin.pmodel;self.fk_link=kin.end_link_idx
            self.root=matrix(self.env.agent.robot.pose)
            urdf=ET.parse(self.env.agent.urdf_path).getroot()
            offsets=[]
            for name,sign in [('panda_leftfinger',1),('panda_rightfinger',-1)]:
                tip=urdf.find(f"link[@name='{name}']").findall('collision')[-1]
                y=float(tip.find('origin').get('xyz').split()[1])
                half=float(tip.find('geometry/box').get('size').split()[1])/2
                offsets.append(y-sign*half)
            assert abs(offsets[0]-offsets[1]+.00004)<1e-10
            self.initial=tree_array(self.env.get_state_dict())
            np.savez(self.folder/'initial.npz',**flatten(self.initial))
            self.sample()
            current=array(self.env.agent.robot.get_qpos())[0]
            fk=self.command_fk(current)
            assert np.max(np.abs(fk-matrix(self.env.agent.tcp.pose)))<1e-5
            inventory=dict(mode=mode,sim_freq=self.env.sim_freq,control_freq=self.env.control_freq,
                backend=str(self.env.backend),source_fileset_sha256=sha(ROOT/'design/SOURCE_FILESET.json'),
                arm_config=repr(self.arm.config),gripper_config=repr(self.env.agent.controller.controllers['gripper'].config),
                camera_config={k:repr(s.config) for k,s in self.env._sensors.items()},
                opening_calibration=dict(urdf_sha256=sha(self.env.agent.urdf_path),pad_inner_face_offsets=offsets,
                    width_formula='q_left+q_right-0.00004 m',nominal_width_m=.08),
                FK_current_max_abs=float(np.max(np.abs(fk-matrix(self.env.agent.tcp.pose)))))
            write(self.folder/'NATIVE_INVENTORY.json',inventory)
        except BaseException:
            self.finish('ENGINEERING_INCOMPLETE');raise
    def command_fk(self,q):
        self.fk.compute_forward_kinematics(np.asarray(q,dtype=np.float64))
        return self.root@self.fk.get_link_pose(self.fk_link).to_transformation_matrix()
    def sample(self):
        e=self.env;agent=e.agent
        info={k:bool(array(v)[0]) for k,v in e.evaluate().items()}
        row=dict(time_s=self.control*.05 if not self.states else (len(self.states))*.01,
            tcp=matrix(agent.tcp.pose),qpos=array(agent.robot.get_qpos())[0],qvel=array(agent.robot.get_qvel())[0],
            cubeA_pose=array(e.cubeA.pose.raw_pose)[0],cubeB_pose=array(e.cubeB.pose.raw_pose)[0],
            cubeA_velocity=array(e.cubeA.linear_velocity)[0],cubeA_angular=array(e.cubeA.angular_velocity)[0],
            cubeB_velocity=array(e.cubeB.linear_velocity)[0],cubeB_angular=array(e.cubeB.angular_velocity)[0],
            left_force=array(e.scene.get_pairwise_contact_forces(agent.finger1_link,e.cubeA))[0],
            right_force=array(e.scene.get_pairwise_contact_forces(agent.finger2_link,e.cubeA))[0],
            left_direction=matrix(agent.finger1_link.pose)[:3,1],right_direction=-matrix(agent.finger2_link.pose)[:3,1],**info)
        self.states.append(row)
        contacts=[]
        for c in e.scene.get_contacts():
            points=[]
            for p in c.points:
                points.append({n:array(getattr(p,n)).tolist() for n in ('position','normal','impulse','separation') if hasattr(p,n)})
            contacts.append(dict(bodies=[getattr(b,'name',repr(b)) for b in c.bodies],points=points))
        self.contacts.write(json.dumps(dict(physics_sample=len(self.states)-1,contacts=contacts))+'\n')
    def observe(self):
        if len(self.inputs)>self.control:
            assert len(self.inputs)==self.control+1
            return self.inputs[-1]
        self.ledger.reserve('rgb',self.slot);self.ledger.sync()
        e=self.env;e.scene.update_render(update_sensors=True,update_human_render_cameras=False);e.capture_sensor_data()
        result={}
        for name,sensor in e._sensors.items():
            # GPU Vulkan rendering, CPU readback; no CUDA neural environment required.
            cam=sensor.camera._render_cameras[0]
            textures={n:torch.as_tensor(np.asarray(cam.get_picture(n)).copy()) for n in sensor._shader_config.texture_names}
            images={}
            for n,t in textures.items():images.update(sensor._shader_config.texture_transforms[n](t))
            rgb=array(images['rgb'])
            assert rgb.dtype==np.uint8
            assert rgb.shape==(128,128,3) and rgb.max()>0
            result[name+'_raw']=rgb
            result[name+'_96']=np.asarray(Image.fromarray(rgb).resize((96,96),Image.Resampling.BILINEAR)).copy()
            for key in ('position','depth','segmentation'):result[name+'_'+key]=array(images[key])
            for key,val in sensor.get_params().items():result[name+'_'+key]=array(val)[0]
        tcp=matrix(e.agent.tcp.pose);q=array(e.agent.robot.get_qpos())[0]
        result['state_v1']=np.r_[tcp[:3,3],tcp[:2,:3].reshape(6),opening(q)].astype(np.float32)
        result['time_s']=np.asarray(self.control*.05)
        self.inputs.append(result)
        return result
    def step(self,action,phase=3,command=None,teacher=False):
        assert self.control<660
        self.ledger.reserve('control',self.slot)
        self.ledger.reserve('dispatch',self.slot);self.ledger.reserve('dispatch',self.slot)
        if teacher:self.ledger.reserve('teacher',self.slot)
        self.ledger.sync();self.observe()
        action=np.asarray(action,dtype=np.float32);assert np.isfinite(action).all()
        if command is None:
            q=array(self.env.agent.robot.get_qpos())[0];q[:7]=action[:7]
            command=np.r_[self.command_fk(q)[:3,3],self.command_fk(q)[:2,:3].reshape(6),(action[-1]+1)/2]
        self.commands.append(np.asarray(command));self.phase.append(phase)
        output=self.env.step(action);self.control+=1
        installed=dict(action=action.copy(),arm_target=array(self.arm._target_qpos)[0])
        q=array(self.env.agent.robot.get_qpos())[0];q[:7]=installed['arm_target']
        installed['installed_target_fk']=self.command_fk(q)
        if self.mode=='pd_ee_pose':installed['base_target']=matrix(self.arm._target_pose)
        self.installed.append(installed)
        return output
    def finish(self,status,error=None):
        global ACTIVE
        if self.contacts.closed:return
        self.contacts.flush();os.fsync(self.contacts.fileno());self.contacts.close()
        if self.states:np.savez(self.folder/'physics.npz',**{k:np.stack([r[k] for r in self.states]) for k in self.states[0]})
        if self.inputs:np.savez(self.folder/'observations.npz',**{k:np.stack([r[k] for r in self.inputs]) for k in self.inputs[0]})
        np.savez(self.folder/'commands.npz',action=np.asarray(self.commands),phase=np.asarray(self.phase),ik_hold=np.asarray(self.ik_holds))
        if self.installed:np.savez(self.folder/'installed.npz',**{k:np.stack([r[k] for r in self.installed]) for k in self.installed[0]})
        persistent=False;first=None
        if self.states:
            success=np.asarray([r['success'] for r in self.states])[::5]
            at=np.flatnonzero(success[:601]);first=float(at[0]*.05) if len(at) else None
            persistent=bool(len(success)>=601 and success[560:601].all())
        max_grasp=max((sum(1 for _ in group) for flag,group in itertools.groupby(r['is_cubeA_grasped'] for r in self.states) if flag),default=0)
        result=dict(status=status,error=error,control_calls=self.control,physics_completed=self.physics_completed,
            physics_samples=len(self.states),RGB_pairs=len(self.inputs),first_success_s=first,persistent_success=persistent,
            sustained_grasp_0p3=max_grasp>=30,
            ik_holds=sum(self.ik_holds),files={p.name:sha(p) for p in self.folder.glob('*') if p.is_file() and p.name!='RESULT.json'})
        write(self.folder/'RESULT.json',result);self.ledger.close()
        if self.env is not None:self.env.close()
        ACTIVE=None
        return result

def tree_array(value):
    return {k:tree_array(v) for k,v in value.items()} if isinstance(value,dict) else array(value)

def flatten(tree,prefix=''):
    result={}
    for k,v in tree.items():
        key=prefix+k
        if isinstance(v,dict):result.update(flatten(v,key+'/'))
        else:result[key]=v
    return result
