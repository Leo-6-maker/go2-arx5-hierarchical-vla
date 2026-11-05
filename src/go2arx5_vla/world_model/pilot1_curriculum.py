"""Compile a separate reachable contact scene; never modify robot dynamics."""
import hashlib
import json
import numpy as np


def robot_contract(original, candidate):
    import mujoco
    from numpy.testing import assert_array_equal
    for name in ('nq','nv','nu','nbody','njnt','ngeom','nsensor','nkey','neq','nmocap','nplugin'):
        if getattr(original,name)!=getattr(candidate,name):raise ValueError('COURSE_ROBOT_ABI_CHANGED:'+name)
    fields={mujoco.mjtObj.mjOBJ_BODY:('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat','body_parentid','body_jntadr','body_jntnum'),
            mujoco.mjtObj.mjOBJ_JOINT:('jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_qposadr','jnt_dofadr','jnt_stiffness'),
            mujoco.mjtObj.mjOBJ_ACTUATOR:('actuator_trntype','actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrllimited','actuator_forcerange'),
            mujoco.mjtObj.mjOBJ_SENSOR:('sensor_type','sensor_objtype','sensor_objid','sensor_adr','sensor_dim'),
            mujoco.mjtObj.mjOBJ_GEOM:('geom_bodyid','geom_type','geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity','geom_margin','geom_condim','geom_solref','geom_solimp')}
    counts={mujoco.mjtObj.mjOBJ_BODY:original.nbody,mujoco.mjtObj.mjOBJ_JOINT:original.njnt,mujoco.mjtObj.mjOBJ_ACTUATOR:original.nu,mujoco.mjtObj.mjOBJ_SENSOR:original.nsensor,mujoco.mjtObj.mjOBJ_GEOM:original.ngeom}
    document={}
    for kind,properties in fields.items():
        for i in range(counts[kind]):
            name=mujoco.mj_id2name(original,kind,i)
            if mujoco.mj_id2name(candidate,kind,i)!=name:raise ValueError('COURSE_NATIVE_ID_ORDER_CHANGED')
            for field in properties:
                # Scene-only translations/dimensions; no friction, mass or pair changes.
                allowed=(kind==mujoco.mjtObj.mjOBJ_BODY and name=='task_cube'and field=='body_pos')or(kind==mujoco.mjtObj.mjOBJ_GEOM and(name=='task_table'or(name or'').startswith('task_goal_'))and field in('geom_pos','geom_size'))
                if not allowed:assert_array_equal(getattr(original,field)[i],getattr(candidate,field)[i],err_msg='FROZEN_COURSE_PROPERTY:'+field+':'+str(name))
                if not allowed:document[str(int(kind))+':'+str(i)+':'+field]=np.asarray(getattr(original,field)[i]).tolist()
    for field in ('dof_damping','dof_armature','dof_frictionloss','dof_parentid','dof_jntid','mesh_vert','mesh_face','mesh_normal','mesh_texcoord','exclude_signature'):
        assert_array_equal(getattr(original,field),getattr(candidate,field),err_msg=field)
        document[field]=hashlib.sha256(getattr(original,field).tobytes()).hexdigest()
    for name in dir(original.opt):
        if name.startswith('_'):continue
        old=getattr(original.opt,name);new=getattr(candidate.opt,name)
        if isinstance(old,(np.ndarray,float,int)):
            assert_array_equal(old,new,err_msg='FROZEN_PHYSICS_OPTION:'+name);document['opt:'+name]=np.asarray(old).tolist()
    cube=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,'task_cube');joint=int(original.body_jntadr[cube]);address=int(original.jnt_qposadr[joint])
    for field in ('qpos0','qpos_spring'):
        a=getattr(original,field).copy();b=getattr(candidate,field).copy();a[address:address+3]=0;b[address:address+3]=0;assert_array_equal(a,b,err_msg=field);document[field]=a.tolist()
    for field in ('key_qpos','key_qvel','key_ctrl','key_time'):
        a=getattr(original,field).copy();b=getattr(candidate,field).copy()
        if field=='key_qpos':a[:,address:address+3]=0;b[:,address:address+3]=0
        assert_array_equal(a,b,err_msg=field);document[field]=a.tolist()
    pin=hashlib.sha256(json.dumps(document,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(status='COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT',sha256=pin,
                scene_variation='PRE_EPISODE_CUBE_XYZ_AND_STATIC_SUPPORT_GEOMETRY_ONLY',
                original_dt_s=original.opt.timestep,robot_mass_inertia_friction_actuators_unchanged=True,
                controller_or_safety_changes=False)


def scene_xml(spec, original):
    import mujoco
    import xml.etree.ElementTree as ET
    from scripts.pi0294.task_plant import xml_scene
    cube=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,'task_cube');j=int(original.body_jntadr[cube]);q=int(original.jnt_qposadr[j]);source=original.qpos0[q:q+3]
    if not np.array_equal(original.qpos0[q+3:q+7],[1.,0.,0.,0.]):raise ValueError('EXPLICIT_UNROTATED_CURRICULUM_CUBE_REQUIRED')
    card=dict(goal_xy=[.45,.08],source_xy=source[:2].tolist(),cube_yaw=0.,table_center_x=.515)
    xml=ET.fromstring(xml_scene(card,spec['asset_root']));world=xml.find('worldbody');table=world.find("geom[@name='task_table']")
    center=np.asarray(spec['table_center_xyz_m'],float);size=np.asarray(spec['table_half_size_m'],float);object_position=np.asarray(spec['object_initial_xyz_m'],float)
    if any(a.shape!=(3,)or not np.isfinite(a).all()for a in (center,size,object_position))or np.any(size<=0):raise ValueError('FINITE_PREDECLARED_SUPPORT_AND_CUBE_REQUIRED')
    if abs(object_position[2]-.02-(center[2]+size[2]))>1e-9 or np.any(np.abs(object_position[:2]-center[:2])+.02>size[:2]):raise ValueError('REAL_INITIAL_CUBE_SUPPORT_WITH_NO_PENETRATION_REQUIRED')
    number=lambda a:' '.join(format(float(v),'.17g')for v in a)
    table.set('pos',number(center));table.set('size',number(size));world.find("body[@name='task_cube']").set('pos',number(object_position))
    for i,(x,y,dx,dy)in enumerate(((center[0]-size[0],center[1],.0005,size[1]),(center[0]+size[0],center[1],.0005,size[1]),(center[0],center[1]-size[1],size[0],.0005),(center[0],center[1]+size[1],size[0],.0005))):
        g=world.find("geom[@name='task_goal_%d']"%i);g.set('pos',number([x,y,center[2]+size[2]+.0002]));g.set('size',number([dx,dy,.0002]))
    for key in xml.findall('./keyframe/key'):
        index=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_KEY,key.get('name'))
        if index<0:raise ValueError('ORIGINAL_HOME_KEYFRAME_ID_REQUIRED')
        value=original.key_qpos[index].copy();value[q:q+3]=object_position
        key.set('qpos',number(value));key.set('qvel',number(original.key_qvel[index]));key.set('ctrl',number(original.key_ctrl[index]))
    return ET.tostring(xml,encoding='unicode')
