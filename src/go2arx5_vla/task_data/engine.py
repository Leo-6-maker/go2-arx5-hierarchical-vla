from __future__ import annotations

import itertools
import numpy as np
import mujoco
from .config import PARAMETERS as P
from .scene import xml_scene
from ..trajectory.so3 import matrix_to_quat

POINTS = np.asarray(list(itertools.product((-.02,.02), repeat=3)))


class Engine:
    def __init__(self, card, mode='normal'):
        self.card, self.mode = card, mode
        xml = xml_scene(card)
        if mode == 'empty_grasp':
            sx,sy=card['source_xy']
            xml=xml.replace(f'name="cube" pos="{sx} {sy} .42"', f'name="cube" pos="{sx+.22} {sy} .42"')
        self.xml=xml;self.model=mujoco.MjModel.from_xml_string(xml);self.data=mujoco.MjData(self.model)
        self.ids={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name) for name in ['tool','cube','left_finger','right_finger']}
        self.geoms={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ['cube_geom','left_pad','right_pad','palm','table','floor']}
        self.joints={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in ['left_slide','right_slide','tool_free','cube_free']}
        self.finger_q=[int(self.model.jnt_qposadr[self.joints[x]]) for x in ['left_slide','right_slide']]
        self.finger_v=[int(self.model.jnt_dofadr[self.joints[x]]) for x in ['left_slide','right_slide']]
        self.tool_v=int(self.model.jnt_dofadr[self.joints['tool_free']])
        self.data.qpos[self.finger_q]=P['finger_travel']
        self.finger_targets=np.full(2,P['finger_travel'])
        self.steps=0;self.safety=[];self.renderer=None
        assert self.model.neq==0 and self.model.nmocap==0
        mujoco.mj_forward(self.model,self.data)

    def pose(self,name):
        i=self.ids[name];T=np.eye(4);T[:3,:3]=self.data.xmat[i].reshape(3,3);T[:3,3]=self.data.xpos[i];return T

    def velocity(self,name):
        result=np.zeros(6);mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.ids[name],result,0)
        return result[3:].copy(),result[:3].copy()

    def contacts(self):
        sides={'left':[],'right':[]};all_contacts=[];illegal=False
        for k in range(self.data.ncon):
            c=self.data.contact[k];g0,g1=int(c.geom1),int(c.geom2);force=np.zeros(6)
            mujoco.mj_contactForce(self.model,self.data,k,force)
            normal=c.frame[:3].copy()
            entry=dict(geom0=g0,geom1=g1,distance_m=float(c.dist),normal=normal.tolist(),normal_force_N=float(force[0]))
            all_contacts.append(entry)
            for side,pad in [('left','left_pad'),('right','right_pad')]:
                if {g0,g1}=={self.geoms[pad],self.geoms['cube_geom']} and force[0]>=P['minimum_contact_force_N']:
                    n=normal if g1==self.geoms['cube_geom'] else -normal
                    sides[side].append(n)
            if (g0 in [self.geoms[x] for x in ['palm','left_pad','right_pad']] and g1 in [self.geoms['table'],self.geoms['floor']]) or (g1 in [self.geoms[x] for x in ['palm','left_pad','right_pad']] and g0 in [self.geoms['table'],self.geoms['floor']]):
                illegal=True
            if c.dist < -P['penetration_stop_m']:illegal=True
        opposing=bool(sides['left'] and sides['right'] and any(np.dot(a,b)<=P['opposing_contact_dot_max'] for a in sides['left'] for b in sides['right']))
        return bool(sides['left']),bool(sides['right']),opposing,illegal,all_contacts

    def snapshot(self):
        mujoco.mj_forward(self.model,self.data)
        tcp,obj=self.pose('tool'),self.pose('cube')
        vl,va=self.velocity('cube');tl,ta=self.velocity('tool')
        left,right,opposing,illegal,contacts=self.contacts()
        q=self.data.qpos[self.finger_q].copy();qd=self.data.qvel[self.finger_v].copy()
        relative=bool(np.linalg.norm(vl-tl)<=P['relative_grasp_speed_max_m_s'])
        valid=bool(np.isfinite(q).all() and (q>=-.001).all() and (q<=P['finger_travel']+.001).all())
        grasp=bool(left and right and opposing and relative and valid)
        return dict(time_s=float(self.data.time),tcp=tcp,object_pose=obj,object_linear_velocity=vl,object_angular_velocity=va,
            tool_linear_velocity=tl,tool_angular_velocity=ta,finger_q=q,finger_qd=qd,
            gripper_fraction=float(np.mean(q)/P['finger_travel']),gripper_speed=float(np.mean(qd)/P['finger_travel']),
            left_object_contact=left,right_object_contact=right,opposing_contact=opposing,relative_motion_valid=relative,
            native_grasp=grasp,object_present=True,contact_valid=True,gripper_state_valid=valid,
            safety_valid=not bool(self.safety) and not illegal,native_abort=bool(self.safety),illegal_contact=illegal,
            joint_limit_penetration=not valid,reset=False,takeover=False,contacts=contacts,
            actuator_force=self.data.actuator_force.copy(),qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),
            object_points=obj[:3,3]+POINTS@obj[:3,:3].T)

    def step(self,target,opening):
        tcp=self.pose('tool');lin,ang=self.velocity('tool')
        force=P['tool_position_kp']*(target[:3,3]-tcp[:3,3])-P['tool_position_kd']*lin
        force+=np.array([0,0,(P['tool_mass']+2*P['finger_mass'])*9.81])
        Rerr=target[:3,:3]@tcp[:3,:3].T
        axis=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])*.5
        torque=P['tool_rotation_kp']*axis-P['tool_rotation_kd']*ang
        force=np.clip(force,-P['tool_force_max_N'],P['tool_force_max_N'])
        torque=np.clip(torque,-P['tool_torque_max_Nm'],P['tool_torque_max_Nm'])
        self.data.xfrc_applied[:]=0
        self.data.xfrc_applied[self.ids['tool'],:3]=force
        self.data.xfrc_applied[self.ids['tool'],3:]=torque
        desired=np.full(2,float(opening)*P['finger_travel'])
        if self.mode=='single_side':desired[1]=P['finger_travel']
        slew=P['finger_target_speed_m_s']*P['dt']
        self.finger_targets+=np.clip(desired-self.finger_targets,-slew,slew)
        self.data.ctrl[:]=np.clip(P['finger_kp']*(self.finger_targets-self.data.qpos[self.finger_q])-P['finger_kd']*self.data.qvel[self.finger_v],-P['finger_force_max_N'],P['finger_force_max_N'])
        mujoco.mj_step(self.model,self.data);self.steps+=1
        mujoco.mj_forward(self.model,self.data)
        if self.contacts()[3]:self.safety.append('ILLEGAL_CONTACT_OR_PENETRATION')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():self.safety.append('NONFINITE')
        if any(w.number for w in self.data.warning):self.safety.append('MUJOCO_WARNING')
        if np.max(np.abs(self.data.qvel[self.finger_v]))>P['finger_speed_stop_m_s']:self.safety.append('FINGER_VELOCITY')
        if np.linalg.norm(self.data.qvel[self.tool_v:self.tool_v+3])>P['tool_speed_stop_m_s']:self.safety.append('TOOL_VELOCITY')
        if self.safety:raise RuntimeError(self.safety[-1])

    def render(self,snapshot):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,P['pixels'],P['pixels'])
        images={};geometry={}
        for name in ['body_rgb','wrist_rgb']:
            self.renderer.update_scene(self.data,camera=name);images[name]=self.renderer.render().copy()
            cam=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,name)
            R=self.data.cam_xmat[cam].reshape(3,3)@np.diag([1.,-1.,-1.])
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=self.data.cam_xpos[cam]
            focal=P['pixels']/(2*np.tan(np.deg2rad(self.model.cam_fovy[cam])/2))
            K=np.array([[focal,0,P['pixels']/2],[0,focal,P['pixels']/2],[0,0,1.]])
            points=(snapshot['object_points']-T[:3,3])@R
            projected=points@K.T;uv=projected[:,:2]/projected[:,2,None]
            visible=[]
            for point,pixel,depth in zip(snapshot['object_points'],uv,points[:,2]):
                direction=point-T[:3,3];direction/=np.linalg.norm(direction)
                geom=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(self.model,self.data,T[:3,3],direction,None,1,-1,geom)
                visible.append(bool(depth>0 and (pixel>=0).all() and (pixel<P['pixels']).all() and geom[0]==self.geoms['cube_geom'] and distance>=0))
            geometry[name]=dict(K=K.tolist(),T_world_optical=T.tolist(),parent='world_fixed_rig' if name=='body_rgb' else 'dynamic_tool',
                object_point_ids=list(range(8)),object_points_uv=uv.tolist(),object_point_visible=visible,
                timestamp_s=float(self.data.time),snapshot_id=f"{self.card['id']}:{self.steps}")
        return images,geometry

    def close(self):
        if self.renderer is not None:self.renderer.close()
