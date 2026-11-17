"""Explicit41/38/20 native snapshots and two-object measured contact truth."""
from pathlib import Path
import struct
import numpy as np
from scripts.pi0294.task_state import _parse_state_abi,ActualState
from .multitask_scene_v6 import OBJECTS
from .multitask_contact_v6 import allowed_pair


def parse_state(raw):
    packet=_parse_state_abi(raw,(41,38,20))
    if not np.isfinite(packet['time_s'])or abs(packet['time_s']-.002*packet['step'])>1e-8:
        raise ValueError('EXACT_NATIVE_MULTITASK_2MS_CLOCK_REQUIRED')
    for c in packet['contacts']:
        if not np.isfinite([c['distance_m'],*c['wrench'],*c['frame'].flat,*c['point']]).all():raise ValueError('FINITE_NATIVE_MULTITASK_CONTACT_PACKET_REQUIRED')
    return packet


def read_state(path):return parse_state(Path(path).read_bytes())


def packets(path):
    """Stream actual snapshots; a gap is reported to the strict task scorer."""
    previous=None
    with Path(path).open('rb')as f:
        while size:=f.read(4):
            if len(size)!=4:raise ValueError('TRUNCATED_MULTITASK_JOURNAL_LENGTH')
            count,=struct.unpack('<I',size)
            if not 40<=count<=2*1024**2:raise ValueError('BOUNDED_MULTITASK_STATE_PACKET_REQUIRED')
            raw=f.read(count)
            if len(raw)!=count:raise ValueError('TRUNCATED_MULTITASK_NATIVE_PACKET')
            row=parse_state(raw)
            if previous is not None and row['step']<=previous:raise ValueError('MONOTONE_MULTITASK_NATIVE_JOURNAL_REQUIRED')
            previous=row['step'];yield row


class TwoObjectState(ActualState):
    def __init__(self,model,selected):
        import mujoco
        if(model.nq,model.nv,model.nu)!=(41,38,20)or selected not in OBJECTS:raise ValueError('EXPLICIT_TWO_OBJECT_MODEL_AND_TASK_REQUIRED')
        super().__init__(model);self.selected=selected;self.object_ids={}
        for color,names in OBJECTS.items():
            ids=dict(body=model.body(names['body']).id,geom=model.geom(names['geom']).id,joint=model.joint(names['joint']).id)
            expected=27 if color=='red'else 34
            if model.jnt_qposadr[ids['joint']]!=expected or model.geom_bodyid[ids['geom']]!=ids['body']or model.jnt_type[ids['joint']]!=mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('TWO_DECLARED_PASSIVE_BOX_BODIES_AND_ADDRESSES_REQUIRED')
            self.object_ids[color]=ids
        self.cube=self.object_ids[selected]['body'];self.cube_geom=self.object_ids[selected]['geom']
        self.robot_bodies=set(range(1,model.nbody))-{x['body']for x in self.object_ids.values()}

    def update(self,packet):
        import mujoco
        m,d=self.model,self.data
        if packet['qpos'].shape!=(41,)or packet['qvel'].shape!=(38,)or packet['ctrl'].shape!=(20,):raise ValueError('MODEL_BOUND_TWO_OBJECT_PACKET_REQUIRED')
        d.qpos[:]=packet['qpos'];d.qvel[:]=packet['qvel'];d.ctrl[:]=packet['ctrl'];d.time=packet['time_s'];mujoco.mj_forward(m,d)
        tcp=self.pose('tool');toolv=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,self.tcp,toolv,0)
        q=d.qpos[self.finger_q].copy();valid=bool((q>=0).all()and(q<=.044).all());objects={};safe=True
        for contact in packet['contacts']:
            a,b=contact['geom0'],contact['geom1']
            if not 0<=a<m.ngeom or not 0<=b<m.ngeom:raise ValueError('CONTACT_IDS_BOUND_TO_ACTUAL_SCENE_REQUIRED')
            safe&=allowed_pair(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,a),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,b),self.selected)
        for color,ids in self.object_ids.items():
            sides=[[],[]];hand=False;support=False
            for c in packet['contacts']:
                a,b=c['geom0'],c['geom1'];force=c['wrench'][0]
                for i,pad in enumerate(self.pad):
                    if {a,b}=={pad,ids['geom']}and force>=.05:sides[i].append(c['normal']if b==ids['geom']else-c['normal'])
                if ids['geom']in(a,b):
                    other=b if a==ids['geom']else a
                    hand|=int(m.geom_bodyid[other])in self.robot_bodies and force>=.05
                    support|=other==m.geom('task_table').id and force>=.05
            velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,ids['body'],velocity,0)
            opposing=bool(sides[0]and sides[1]and any(np.dot(a,b)<=-.5 for a in sides[0]for b in sides[1]));relative=bool(np.linalg.norm(velocity[3:]-toolv[3:])<=.02)
            pose=np.eye(4);pose[:3,:3]=d.xmat[ids['body']].reshape(3,3);pose[:3,3]=d.xpos[ids['body']]
            objects[color]=dict(pose=pose,linear_velocity=velocity[3:].copy(),angular_velocity=velocity[:3].copy(),
                left_object_contact=bool(sides[0]),right_object_contact=bool(sides[1]),opposing_contact=opposing,relative_motion_valid=relative,
                native_grasp=bool(sides[0]and sides[1]and opposing and relative and valid),hand_contact=bool(hand),table_contact=bool(support))
        selected=objects[self.selected];base=np.eye(4);b=m.body('x5_base_link').id;base[:3,:3]=d.xmat[b].reshape(3,3);base[:3,3]=d.xpos[b]
        return dict(time_s=float(d.time),native_step=packet['step'],tcp=tcp,base_pose=base,objects=objects,object_pose=selected['pose'],
            object_linear_velocity=selected['linear_velocity'],object_angular_velocity=selected['angular_velocity'],gripper_fraction=float(q.mean()/.044),
            gripper_open_fraction=float(q.mean()/.044),finger_q=q,**{k:selected[k]for k in('left_object_contact','right_object_contact','opposing_contact','relative_motion_valid','native_grasp')},
            robot_object_contact=selected['hand_contact'],safety_valid=bool(safe),object_present=True,contact_valid=True,gripper_state_valid=valid,native_abort=False,
            illegal_contact=not safe,joint_limit_penetration=not valid,reset=False,takeover=False,contact_time_s=packet['contact_time_s'],
            qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),privileged_object_truth_for_audit_only=True)
