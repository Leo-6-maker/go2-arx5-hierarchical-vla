"""Cold two-object scene candidate, preserving original robot/control fields.

Both physical objects and both noncontact destination markers are always in
the scene. Instruction selection is a separate case field, never scene input.
No renderer, integration, controller build or native admission occurs here.
"""
import copy,hashlib,json
import xml.etree.ElementTree as ET
import numpy as np

VERSION='MFREE_TWO_OBJECT_HOLD_PLACE_SCENE_V6'
OBJECTS={'red':dict(body='task_cube',joint='task_cube_free',geom='task_cube_geom'),
         'blue':dict(body='task_blue_cube',joint='task_blue_cube_free',geom='task_blue_cube_geom')}


def numbers(value):return ' '.join(format(float(x),'.17g')for x in value)


def overlay_xml(template_spec,original,card):
    from .pilot1_curriculum import scene_xml
    table=original.geom('task_table').id;center=original.geom_pos[table].copy();half=original.geom_size[table].copy();top=float(center[2]+half[2])
    positions={c:np.asarray(card['object_positions_world'][c],float)for c in('red','blue')};zones=card['zones']
    if set(card['object_positions_world'])!={'red','blue'}or set(zones)!={'left','right'}:raise ValueError('TWO_COLORS_AND_TWO_REGIONS_REQUIRED')
    for p in positions.values():
        if p.shape!=(3,)or not np.isfinite(p).all()or abs(p[2]-.02-top)>1e-9 or np.any(abs(p[:2]-center[:2])+.025>half[:2]):
            raise ValueError('FULL_INITIAL_OBJECT_FOOTPRINT_ON_UNCHANGED_SUPPORT_REQUIRED')
    if not np.any(abs(positions['red'][:2]-positions['blue'][:2])>=.045):raise ValueError('TWO_INITIAL_BOXES_MUST_NOT_OVERLAP')
    zone_geometry={}
    for name,z in zones.items():
        xy=np.asarray(z['center_world_XY'],float);size=np.asarray(z['half_size_XY'],float)
        if xy.shape!=(2,)or size.shape!=(2,)or not np.isfinite([xy,size]).all()or np.any(size<=.02)or np.any(abs(xy-center[:2])+size>half[:2]-.005):
            raise ValueError('COMPLETE_DESTINATION_REGION_INSIDE_UNCHANGED_TABLE_REQUIRED')
        zone_geometry[name]=dict(center_world_XY=xy.tolist(),half_size_XY=size.tolist())
    left,right=zone_geometry['left'],zone_geometry['right']
    if not np.any(abs(np.asarray(left['center_world_XY'])-right['center_world_XY'])>=np.asarray(left['half_size_XY'])+right['half_size_XY']+.005):
        raise ValueError('DISTINCT_NONOVERLAPPING_DESTINATION_REGIONS_REQUIRED')
    spec=dict(template_spec,table_center_xyz_m=center.tolist(),table_half_size_m=half.tolist(),object_initial_xyz_m=positions['red'].tolist())
    root=ET.fromstring(scene_xml(spec,original));world=root.find('worldbody');robot=world.find("body[@name='base_link']")
    protected={tag:ET.tostring(root.find(tag))if root.find(tag)is not None else None for tag in('option','actuator','sensor','contact','equality')};before_robot=ET.tostring(robot)
    red=world.find("body[@name='task_cube']")
    if red is None or len(red.findall('freejoint'))!=1 or len(red.findall('geom'))!=1 or len(red)!=2:raise ValueError('EXACT_ORIGINAL_SINGLE_FREE_BOX_TEMPLATE_REQUIRED')
    red.set('pos',numbers(positions['red']));red.find('geom').set('rgba','.82 .04 .04 1')
    blue=copy.deepcopy(red);blue.set('name',OBJECTS['blue']['body']);blue.set('pos',numbers(positions['blue']))
    blue.find('freejoint').set('name',OBJECTS['blue']['joint']);blue.find('geom').set('name',OBJECTS['blue']['geom']);blue.find('geom').set('rgba','.04 .12 .82 1');world.append(blue)
    # Reuse the four existing noncontact marker geoms so original native robot
    # geom IDs/order stay fixed. Filled regions plus small accents are visual.
    for i,(name,color)in enumerate((('left','.1 .65 .2 .45'),('right','.95 .7 .05 .45'))):
        z=zone_geometry[name];marker=world.find("geom[@name='task_goal_%d']"%i);accent=world.find("geom[@name='task_goal_%d']"%(i+2))
        for g in(marker,accent):
            if g is None or g.get('contype')!='0'or g.get('conaffinity')!='0':raise ValueError('EXISTING_NONCONTACT_MARKERS_REQUIRED')
        marker.set('pos',numbers([*z['center_world_XY'],top+.0002]));marker.set('size',numbers([*z['half_size_XY'],.0002]));marker.set('rgba',color)
        accent.set('pos',numbers([*z['center_world_XY'],top+.0005]));accent.set('size',numbers([.003,.015,.0002]));accent.set('rgba',color.rsplit(' ',1)[0]+' 1')
    for key in root.findall('./keyframe/key'):
        q=np.fromstring(key.get('qpos'),sep=' ');v=np.fromstring(key.get('qvel'),sep=' ')
        if q.shape!=(34,)or v.shape!=(32,):raise ValueError('EXACT_LEGACY_TASK_KEYFRAME_REQUIRED')
        key.set('qpos',numbers(np.r_[q,positions['blue'],1.,0.,0.,0.]));key.set('qvel',numbers(np.r_[v,np.zeros(6)]))
    if ET.tostring(robot)!=before_robot or any(value!=(ET.tostring(root.find(tag))if root.find(tag)is not None else None)for tag,value in protected.items()):
        raise ValueError('ROBOT_EXECUTOR_PHYSICS_OR_PROTECTED_XML_CHANGED')
    root.set('model',VERSION)
    return ET.tostring(root,encoding='unicode'),dict(version=VERSION,expected_ABI=dict(nq=41,nv=38,nu=20),objects=OBJECTS,
        zones=zone_geometry,table_top_world_z=top,object_half_sizes={c:[.02,.02,.02]for c in OBJECTS},direction_world_XY=card['direction_world_XY'],
        original_robot_XML_unchanged=True,scene_depends_on_instruction=False,added_actuators=0,added_mocap=0,added_equalities=0,native_dispatch_qualified=False)


def robot_contract(original,candidate):
    """Positive original robot/control inventory, independent of task DOF count."""
    import mujoco
    from numpy.testing import assert_array_equal
    if(original.nq,original.nv,original.nu)!=(34,32,20)or(candidate.nq,candidate.nv,candidate.nu)!=(41,38,20):raise ValueError('DECLARED_ONE_TO_TWO_PASSIVE_BOX_ABI_ONLY')
    if any(getattr(candidate,k)!=getattr(original,k)+n for k,n in(('nbody',1),('njnt',1),('ngeom',1),('nsensor',0),('nkey',0),('neq',0),('nmocap',0),('nplugin',0))):
        raise ValueError('EXACT_ONE_PASSIVE_FREE_BOX_ADDITION_ONLY')
    groups=[(mujoco.mjtObj.mjOBJ_BODY,original.nbody,('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat','body_parentid','body_jntadr','body_jntnum')),
        (mujoco.mjtObj.mjOBJ_JOINT,original.njnt,('jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_qposadr','jnt_dofadr','jnt_stiffness')),
        (mujoco.mjtObj.mjOBJ_ACTUATOR,original.nu,('actuator_trntype','actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrllimited','actuator_forcerange')),
        (mujoco.mjtObj.mjOBJ_SENSOR,original.nsensor,('sensor_type','sensor_objtype','sensor_objid','sensor_adr','sensor_dim')),
        (mujoco.mjtObj.mjOBJ_GEOM,original.ngeom,('geom_bodyid','geom_type','geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity','geom_margin','geom_condim','geom_solref','geom_solimp'))]
    protected={}
    for kind,count,fields in groups:
        for i in range(count):
            name=mujoco.mj_id2name(original,kind,i)
            if mujoco.mj_id2name(candidate,kind,i)!=name:raise ValueError('ORIGINAL_NATIVE_IDS_OR_NAMES_CHANGED')
            scene=name=='task_cube'or name=='task_cube_free'or name=='task_cube_geom'or(name or'').startswith('task_goal_')
            for field in fields:
                allowed=scene and((name=='task_cube'and field=='body_pos')or(name or'').startswith('task_goal_')and field in('geom_pos','geom_size'))
                if not allowed:
                    assert_array_equal(getattr(original,field)[i],getattr(candidate,field)[i],err_msg=field+':'+str(name))
                    protected[str(int(kind))+':'+str(i)+':'+field]=np.asarray(getattr(original,field)[i]).tolist()
    for field,n in(('dof_damping',32),('dof_armature',32),('dof_frictionloss',32),('dof_parentid',32),('dof_jntid',32)):
        assert_array_equal(getattr(original,field),getattr(candidate,field)[:n],err_msg=field)
    for field in('mesh_vert','mesh_face','mesh_normal','mesh_texcoord','exclude_signature'):
        assert_array_equal(getattr(original,field),getattr(candidate,field),err_msg=field);protected[field]=hashlib.sha256(getattr(original,field).tobytes()).hexdigest()
    for field,n in(('qpos0',27),('qpos_spring',27)):
        assert_array_equal(getattr(original,field)[:n],getattr(candidate,field)[:n],err_msg=field)
    for field,n in(('key_qpos',27),('key_qvel',26),('key_ctrl',20)):
        assert_array_equal(getattr(original,field)[:,:n],getattr(candidate,field)[:,:n],err_msg=field)
    assert_array_equal(original.key_time,candidate.key_time,err_msg='ORIGINAL_KEY_TIME')
    for name in dir(original.opt):
        if name.startswith('_'):continue
        value=getattr(original.opt,name)
        if isinstance(value,(np.ndarray,float,int)):assert_array_equal(value,getattr(candidate.opt,name),err_msg='PHYSICS_OPTION:'+name)
    return dict(status='ORIGINAL_ROBOT_CONTROL_IDS_AND_DYNAMICS_EXACT_WITH_ONE_NEW_PASSIVE_BOX',
        fingerprint=hashlib.sha256(json.dumps(protected,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        original_robot_nq=27,original_robot_nv=26,actuators=20,physics_dt_s=float(candidate.opt.timestep),native_dispatch_qualified=False)
