"""Add task gripper/cameras/objects to the original floating native robot.

No root/leg removal, native actuator replacement, controller tuning or physics
integration. All compiled original robot properties are checked independently.
"""
import copy
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from .task_gripper_contract import VERSION,WORKING_OPEN,MARGIN

PARAMETERS = dict(table_top=.44, table_center_x=.515, cube_edge=.04, cube_mass=.05,
    finger_travel=.044, finger_force_max_N=8., finger_target_speed_m_s=.04,
    finger_kp=400., finger_kd=3., finger_damping=3., finger_armature=.002,
    pad_friction=1.2, table_friction=.8, cube_friction=.8,
    body_camera_position=[.25,-.24,.28], body_camera_look_at=[.50,0.,.10],
    wrist_camera_position=[.1124,.10,.055], wrist_camera_look_at=[.1774,0.,.0001057],
    camera_fovy=75., pixels=256,working_margin_m=MARGIN,initial_finger_position_m=WORKING_OPEN)


def camera(parent, name, position, target, up):
    position, target, up = map(lambda v: np.asarray(v,dtype=float), (position,target,up))
    z = position-target; z /= np.linalg.norm(z)
    x = np.cross(up,z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    ET.SubElement(parent,'camera',name=name,pos=' '.join(map(str,position)),
                  xyaxes=' '.join(map(str,np.r_[x,y])),fovy=str(PARAMETERS['camera_fovy']))


def xml_scene(card, asset_root):
    asset_root = Path(asset_root).resolve()
    load = lambda p: ET.fromstring(re.sub(r'<!--.*?-->', '',p.read_text(),flags=re.DOTALL))
    robot = load(asset_root/'go2_x5.xml'); scene = load(asset_root/'scene.xml')
    if robot.find('include') is not None:
        raise ValueError('EXPLICIT_NATIVE_ASSET_SCHEMA_REQUIRED')
    robot.set('model',VERSION)
    robot.find('compiler').set('meshdir',str(asset_root/'assets'))
    world = robot.find('worldbody'); base = world.find("body[@name='base_link']")
    if base.find('freejoint') is None:
        raise ValueError('NATIVE_FREE_ROOT_REQUIRED')
    # Merge the delivered scene's floor/lighting/assets; retain robot option,
    # contact exclusions, sensors, original bodies, inertias and 18 actuators.
    for tag in ('asset','visual','statistic'):
        incoming = scene.find(tag)
        if incoming is None: continue
        existing = robot.find(tag)
        if existing is None: robot.append(copy.deepcopy(incoming))
        else:
            for element in incoming: existing.append(copy.deepcopy(element))
    for element in scene.find('worldbody'): world.append(copy.deepcopy(element))
    link = base.find(".//body[@name='x5_link6']")
    actuators = robot.find('actuator')
    if len(actuators) != 18: raise ValueError('ORIGINAL_18_ACTUATORS_REQUIRED')
    for side,body_name,axis,frame_y in (
            ('left','x5_link7','0 1 0',-.0249),('right','x5_link8','0 -1 0',.0249)):
        finger = link.find(f"body[@name='{body_name}']")
        if finger is None or finger.find('joint') is not None:
            raise ValueError('DELIVERED_FIXED_FINGER_SCHEMA_CHANGED')
        ET.SubElement(finger,'joint',name='task_'+side+'_slide',type='slide',axis=axis,
            range='0 .044',limited='true',damping='3',armature='.002',frictionloss='0')
        outward = .003 if side == 'left' else -.003
        ET.SubElement(finger,'geom',name='task_'+side+'_pad',type='box',
            pos=f'.05843 {frame_y+outward} .006376',size='.02 .003 .017',
            friction='1.2 .02 .002',condim='4',margin='0',solref='.008 1',
            rgba='.10 .16 .18 1',mass='0',contype='1',conaffinity='1',group='3')
        ET.SubElement(actuators,'motor',name='task_'+side+'_motor',joint='task_'+side+'_slide',
                      ctrllimited='true',ctrlrange='-8 8',gear='1')
    camera(base,'body_rgb',PARAMETERS['body_camera_position'],PARAMETERS['body_camera_look_at'],[0,0,1])
    camera(link,'wrist_rgb',PARAMETERS['wrist_camera_position'],PARAMETERS['wrist_camera_look_at'],[1,0,0])
    top=PARAMETERS['table_top'];gx,gy=card['goal_xy'];sx,sy=card['source_xy'];yaw=card['cube_yaw']
    # Native base nose collision capsule reaches x=.335 at startup. The proximal
    # edge is .345: measured 10mm clearance, not a new robot pose or WBC change.
    ET.SubElement(world,'geom',name='task_table',type='box',pos=f"{card.get('table_center_x',PARAMETERS['table_center_x'])} 0 {top-.025}",
                  size='.17 .24 .025',rgba='.69 .70 .66 1',friction='.8 .02 .002')
    for i,(x,y,dx,dy) in enumerate(((gx-.08,gy,.002,.08),(gx+.08,gy,.002,.08),(gx,gy-.08,.08,.002),(gx,gy+.08,.08,.002))):
        ET.SubElement(world,'geom',name=f'task_goal_{i}',type='box',pos=f'{x} {y} {top+.0002}',
            size=f'{dx} {dy} .0002',rgba='.08 .35 .80 1',contype='0',conaffinity='0')
    cube=ET.SubElement(world,'body',name='task_cube',pos=f'{sx} {sy} {top+.02}',
                      quat=f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}')
    ET.SubElement(cube,'freejoint',name='task_cube_free')
    ET.SubElement(cube,'geom',name='task_cube_geom',type='box',size='.02 .02 .02',mass='.05',
                  friction='.8 .02 .002',condim='4',rgba='.82 .04 .04 1',margin='0',solref='.008 1')
    extra_q = [WORKING_OPEN,WORKING_OPEN,sx,sy,top+.02,np.cos(yaw/2),0,0,np.sin(yaw/2)]
    for key in robot.findall('./keyframe/key'):
        for name,extra in (('qpos',extra_q),('qvel',[0]*8),('ctrl',[0]*2)):
            if name in key.attrib:
                key.set(name,key.get(name)+' '+' '.join(map(str,extra)))
    return ET.tostring(robot,encoding='unicode')


def check_frozen_robot(original, task):
    import mujoco
    from numpy.testing import assert_array_equal
    # New finger DOFs are appended after all native joints; body IDs/sensor
    # offsets and original actuator order must stay exactly compatible.
    if (task.nq != original.nq+9 or task.nv != original.nv+8 or task.nu != original.nu+2
            or task.nmocap != 0 or task.neq != original.neq or task.opt.timestep != original.opt.timestep):
        raise ValueError('TASK_OVERLAY_DIMENSION_OR_PHYSICS_CHANGE')
    properties={mujoco.mjtObj.mjOBJ_BODY:('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat'),
        mujoco.mjtObj.mjOBJ_JOINT:('jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_qposadr','jnt_dofadr'),
        mujoco.mjtObj.mjOBJ_ACTUATOR:('actuator_trntype','actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_gainprm','actuator_biasprm'),
        mujoco.mjtObj.mjOBJ_SENSOR:('sensor_type','sensor_objtype','sensor_objid','sensor_adr','sensor_dim')}
    checked={}
    for kind,fields in properties.items():
        count={mujoco.mjtObj.mjOBJ_BODY:original.nbody,mujoco.mjtObj.mjOBJ_JOINT:original.njnt,mujoco.mjtObj.mjOBJ_ACTUATOR:original.nu,mujoco.mjtObj.mjOBJ_SENSOR:original.nsensor}[kind]
        for i in range(count):
            name=mujoco.mj_id2name(original,kind,i)
            j=mujoco.mj_name2id(task,kind,name) if name else i
            if j != i: raise ValueError('NATIVE_ID_ORDER_CHANGED:'+str(name))
            for field in fields: assert_array_equal(getattr(original,field)[i],getattr(task,field)[j])
        checked[str(kind)]=dict(count=count,fields=list(fields))
    for field in ('dof_damping','dof_armature','dof_frictionloss'):
        assert_array_equal(getattr(original,field),getattr(task,field)[:original.nv])
    for body in range(original.nbody):
        old=np.flatnonzero(original.geom_bodyid==body)
        new=[int(g) for g in np.flatnonzero(task.geom_bodyid==body)
             if not (mujoco.mj_id2name(task,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '').startswith('task_')]
        if len(old)!=len(new):raise ValueError('NATIVE_GEOMETRY_COUNT_CHANGED')
        for a,b in zip(old,new):
            for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction',
                          'geom_contype','geom_conaffinity','geom_margin','geom_condim','geom_solref','geom_solimp'):
                assert_array_equal(getattr(original,field)[a],getattr(task,field)[b])
    return dict(status='COMPILED_NATIVE_PROPERTIES_EXACTLY_PRESERVED',checked=checked,
        original_robot_body_mass_kg=float(original.body_mass.sum()),
        task_robot_body_mass_kg=float(task.body_mass[:original.nbody].sum()),
        added_object_mass_kg=float(task.body_mass[original.nbody:].sum()),
        new_gripper_joints=2,new_gripper_actuators=2,zero_WBC_parameter_changes=True)
