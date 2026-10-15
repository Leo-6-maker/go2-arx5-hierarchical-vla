"""Read actual native binary state and save camera/Teacher inputs consistently."""
import os
from pathlib import Path
import struct
import numpy as np


def read_state(path):
    raw=Path(path).read_bytes()
    if len(raw)<40:raise ValueError('INCOMPLETE_ACTUAL_NATIVE_STATE')
    magic,step,t,nq,nv,nu=struct.unpack_from('<8sQdIII',raw)
    if magic!=b'GRTSTATE' or (nq,nv,nu)!=(34,32,20):raise ValueError('NATIVE_STATE_ABI')
    n=nq+nv+nu;values=np.frombuffer(raw,dtype='<f8',count=n,offset=36).copy();offset=36+8*n
    nc,=struct.unpack_from('<I',raw,offset);offset+=4
    if len(raw)!=offset+160*nc or not np.isfinite(values).all():raise ValueError('FINITE_COMPLETE_NATIVE_STATE_REQUIRED')
    contacts=[]
    for i in range(nc):
        a,b,dist=struct.unpack_from('<IId',raw,offset);offset+=16
        rest=np.frombuffer(raw,dtype='<f8',count=18,offset=offset).copy();offset+=144
        contacts.append(dict(geom0=a,geom1=b,distance_m=dist,wrench=rest[:6],normal=rest[6:9],frame=rest[6:15].reshape(3,3),point=rest[15:]))
    return dict(step=step,time_s=t,qpos=values[:nq],qvel=values[nq:nq+nv],ctrl=values[nq+nv:],contacts=contacts,contact_time_s=t-.002)


def write_gripper(path,sequence,source,opening):
    if not np.isfinite([source,opening]).all() or not 0<=opening<=1:raise ValueError('GRIPPER_INTENT_RANGE')
    raw=struct.pack('<8sQddd',b'GRTGRIP1',sequence,source,opening,source+.4)
    path=Path(path);tmp=path.with_suffix('.pending')
    with tmp.open('xb') as f:f.write(raw);f.flush()
    os.replace(tmp,path)


class ActualState:
    def __init__(self,model):
        import mujoco
        self.model=model;self.data=mujoco.MjData(model)
        self.tcp=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'x5_ee')
        self.cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'task_cube')
        self.fingers=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'task_'+s+'_slide') for s in ('left','right')]
        self.pad=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_'+s+'_pad') for s in ('left','right')]
        self.cube_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_cube_geom')
        self.geoms={'left_pad':self.pad[0],'right_pad':self.pad[1]}
        self.finger_q=np.array([model.jnt_qposadr[j] for j in self.fingers])

    def pose(self,name):
        p=np.eye(4)
        if name=='tool':p[:3,:3]=self.data.site_xmat[self.tcp].reshape(3,3);p[:3,3]=self.data.site_xpos[self.tcp]
        else:p[:3,:3]=self.data.xmat[self.cube].reshape(3,3);p[:3,3]=self.data.xpos[self.cube]
        return p

    def update(self,packet):
        import mujoco
        from .task_contacts import allowed
        m,d=self.model,self.data
        d.qpos[:]=packet['qpos'];d.qvel[:]=packet['qvel'];d.ctrl[:]=packet['ctrl'];d.time=packet['time_s']
        mujoco.mj_forward(m,d)
        sides=[[],[]];robot_contact=False;safe=True
        for c in packet['contacts']:
            a,b=c['geom0'],c['geom1']
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (a,b)]
            safe &= allowed(*names)
            for i,pad in enumerate(self.pad):
                if {a,b}=={pad,self.cube_geom} and c['wrench'][0]>=.05:
                    sides[i].append(c['normal'] if b==self.cube_geom else -c['normal'])
            robot_contact |= self.cube_geom in (a,b) and any(g in self.pad for g in (a,b)) and c['wrench'][0]>=.05
        opposing=bool(sides[0] and sides[1] and any(np.dot(a,b)<=-.5 for a in sides[0] for b in sides[1]))
        velocity=np.zeros(6);toolv=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.cube,velocity,0)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,self.tcp,toolv,0)
        relative=bool(np.linalg.norm(velocity[3:]-toolv[3:])<=.02)
        q=d.qpos[self.finger_q].copy();valid=bool((q>=0).all() and (q<=.044).all())
        return dict(time_s=float(d.time),tcp=self.pose('tool'),object_pose=self.pose('cube'),
            gripper_fraction=float(q.mean()/.044),finger_q=q,
            object_linear_velocity=velocity[3:].copy(),object_angular_velocity=velocity[:3].copy(),
            left_object_contact=bool(sides[0]),right_object_contact=bool(sides[1]),opposing_contact=opposing,
            relative_motion_valid=relative,native_grasp=bool(sides[0] and sides[1] and opposing and relative and valid),
            robot_object_contact=bool(robot_contact),safety_valid=bool(safe),object_present=True,
            contact_valid=True,gripper_state_valid=valid,native_abort=False,illegal_contact=not safe,
            joint_limit_penetration=not valid,reset=False,takeover=False,
            contact_time_s=packet['contact_time_s'],native_step=packet['step'],qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
