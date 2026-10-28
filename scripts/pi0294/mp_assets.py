"""Export the unchanged native kinematic/collision model for CPU planning only.

No actual mjData is modified, rendered or integrated. Capsule unions retain the
original cylinder and hemispherical volume. Visual-only meshes stay visual-only.
"""
import itertools,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from go2arx5_vla.trajectory.so3 import quat_to_matrix
from .task_capture_runtime import sha


def rpy(q):
    r=quat_to_matrix(np.r_[q[1:],q[0]]);cy=np.hypot(r[0,0],r[1,0])
    return ([np.arctan2(r[2,1],r[2,2]),np.arctan2(-r[2,0],cy),np.arctan2(r[1,0],r[0,0])]
        if cy>1e-10 else [np.arctan2(-r[1,2],r[1,1]),np.arctan2(-r[2,0],cy),0.])


def vector(x):return ' '.join(format(float(v),'.17g') for v in x)


def origin(node,pos,quat):ET.SubElement(node,'origin',xyz=vector(pos),rpy=vector(rpy(quat)))


def export(model,folder,model_sha):
    import mujoco
    from scripts.pi0292.durable import write
    folder=Path(folder);folder.mkdir(exist_ok=False)
    m=model;name=lambda kind,i:mujoco.mj_id2name(m,kind,i)
    bname=lambda i:name(mujoco.mjtObj.mjOBJ_BODY,int(i))
    native_bodies=[i for i in range(1,m.nbody) if bname(i)!='task_cube']
    if bname(1)!='base_link' or m.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('FULL_NATIVE_FREE_ROOT_REQUIRED')
    tree=ET.Element('robot',name='native_go2_arx5_planning_only');joint_names=[];joint_ids=[];gmap=[]
    for i in native_bodies:
        ET.SubElement(tree,'link',name=bname(i))
        if i==1:continue
        joints=[j for j in range(m.njnt) if m.jnt_bodyid[j]==i]
        if len(joints)>1:raise ValueError('UNSUPPORTED_MULTI_JOINT_BODY_NOT_DROPPED')
        if joints:
            j=joints[0];jn=name(mujoco.mjtObj.mjOBJ_JOINT,j)
            if np.max(np.abs(m.jnt_pos[j]))>1e-12 or m.qpos0[m.jnt_qposadr[j]]!=0:
                raise ValueError('NONZERO_PIVOT_OR_REFERENCE_REQUIRES_EXPLICIT_EXPORT')
            kind={int(mujoco.mjtJoint.mjJNT_HINGE):'revolute',int(mujoco.mjtJoint.mjJNT_SLIDE):'prismatic'}[int(m.jnt_type[j])]
            node=ET.SubElement(tree,'joint',name=jn,type=kind);ET.SubElement(node,'axis',xyz=vector(m.jnt_axis[j]))
            ET.SubElement(node,'limit',lower=str(m.jnt_range[j,0]),upper=str(m.jnt_range[j,1]),effort='100',velocity='0.7')
            joint_names.append(jn);joint_ids.append(j)
        else:node=ET.SubElement(tree,'joint',name='fixed_'+bname(i),type='fixed')
        ET.SubElement(node,'parent',link=bname(m.body_parentid[i]));ET.SubElement(node,'child',link=bname(i));origin(node,m.body_pos[i],m.body_quat[i])
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g])
        if b not in native_bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        gn='native_geom_'+str(g);link=ET.SubElement(tree,'link',name=gn)
        node=ET.SubElement(tree,'joint',name='fixed_'+gn,type='fixed');ET.SubElement(node,'parent',link=bname(b));ET.SubElement(node,'child',link=gn);origin(node,m.geom_pos[g],m.geom_quat[g])
        typ=int(m.geom_type[g]);size=m.geom_size[g]
        def shape(kind,values,pos=(0,0,0)):
            c=ET.SubElement(link,'collision');ET.SubElement(c,'origin',xyz=vector(pos),rpy='0 0 0');geom=ET.SubElement(c,'geometry');ET.SubElement(geom,kind,**values)
        if typ==int(mujoco.mjtGeom.mjGEOM_BOX):shape('box',dict(size=vector(2*size)))
        elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):shape('sphere',dict(radius=str(size[0])))
        elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):shape('cylinder',dict(radius=str(size[0]),length=str(2*size[1])))
        elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            shape('cylinder',dict(radius=str(size[0]),length=str(2*size[1])))
            for z in (-size[1],size[1]):shape('sphere',dict(radius=str(size[0])),(0,0,z))
        else:raise ValueError('UNSUPPORTED_COLLISION_SHAPE_MUST_NOT_BE_OMITTED:'+str(typ))
        gmap.append(dict(native_geom=g,link=gn,body=b,native_name=name(mujoco.mjtObj.mjOBJ_GEOM,g),type=typ))
    site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'x5_ee')
    ET.SubElement(tree,'link',name='x5_ee');node=ET.SubElement(tree,'joint',name='fixed_x5_ee',type='fixed')
    ET.SubElement(node,'parent',link=bname(m.site_bodyid[site]));ET.SubElement(node,'child',link='x5_ee');origin(node,m.site_pos[site],m.site_quat[site])
    semantic=ET.Element('robot',name=tree.get('name'));disabled=[]
    excluded={tuple(sorted((int(sig)>>16,int(sig)&65535))) for sig in m.exclude_signature}
    for a,b in itertools.combinations(gmap,2):
        ba,bb=a['body'],b['body'];wa,wb=int(m.body_weldid[ba]),int(m.body_weldid[bb])
        parent=wa==int(m.body_weldid[m.body_parentid[wb]]) or wb==int(m.body_weldid[m.body_parentid[wa]])
        excluded_pair=tuple(sorted((ba,bb))) in excluded
        masks=bool(m.geom_contype[a['native_geom']]&m.geom_conaffinity[b['native_geom']] or m.geom_contype[b['native_geom']]&m.geom_conaffinity[a['native_geom']])
        if wa==wb or parent or excluded_pair or not masks:
            ET.SubElement(semantic,'disable_collisions',link1=a['link'],link2=b['link'],reason='NativeFilter')
            disabled.append([a['link'],b['link']])
    (folder/'robot.urdf').write_text(ET.tostring(tree,encoding='unicode'))
    (folder/'robot.srdf').write_text(ET.tostring(semantic,encoding='unicode'))
    proof=dict(schema='NATIVE_FULLBODY_PLANNING_EXPORT_V1',status='EXPORTED_NOT_FK_QUALIFIED',native_model_sha256=model_sha,
        urdf_sha256=sha(folder/'robot.urdf'),srdf_sha256=sha(folder/'robot.srdf'),joint_names=joint_names,native_joint_ids=joint_ids,
        joint_qpos_addresses=[int(m.jnt_qposadr[j]) for j in joint_ids],geometry_map=gmap,native_filter_pairs=disabled,
        source_native_collidable_robot_geoms=len(gmap),native_robot_bodies=len(native_bodies),leg_joint_count=sum(not n.startswith('x5_') and not n.startswith('task_') for n in joint_names),
        physical_model_modified=False,physics_steps=0,planning_root_pose='EXTERNALLY_BOUND_TO_CURRENT_ACTUAL_FREE_ROOT',
        note='No future leg dynamics prediction; dense native clone and actual observer remain required')
    write(folder/'ASSET_BINDING.private.json',proof)
    return proof
