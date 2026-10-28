"""V3 named task-only geometry/reset changes; native robot remains byte frozen."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .task_plant import xml_scene as original_xml_scene,check_frozen_robot
from .task_capture_runtime import read,sha

def xml_scene(card,asset_root):
    if card['table_top_m'] not in (.44,.47):raise PermissionError('NIGHT_REGISTERED_SCENE_HEIGHT')
    tree=ET.fromstring(original_xml_scene(card,asset_root));delta=card['table_top_m']-.44
    for name in ['task_table']+['task_goal_'+str(i) for i in range(4)]:
        item=tree.find("./worldbody/geom[@name='%s']"%name);p=np.fromstring(item.get('pos'),sep=' ');p[2]+=delta;item.set('pos',' '.join(map(str,p)))
    item=tree.find("./worldbody/body[@name='task_cube']");p=np.fromstring(item.get('pos'),sep=' ');p[2]+=delta;item.set('pos',' '.join(map(str,p)))
    for key in tree.findall('./keyframe/key'):
        if 'qpos' not in key.attrib:continue
        q=np.fromstring(key.get('qpos'),sep=' ');assert q.shape==(34,);q[29]+=delta
        q[25:27]=card['initial_finger_q_m']
        key.set('qpos',' '.join(map(str,q)))
    return ET.tostring(tree,encoding='unicode')

def initial_data(model):
    import mujoco
    d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,0)
    for name in ['x5_joint2','x5_joint3']:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);d.qpos[model.jnt_qposadr[j]]=.9
    mujoco.mj_forward(model,d);return d

def initial_audit(model,d,card):
    import mujoco
    fingers=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'task_'+s+'_slide') for s in ('left','right')]
    values=d.qpos[model.jnt_qposadr[fingers]];expected=card['initial_finger_q_m']
    if not np.allclose(values,expected,atol=1e-15,rtol=0):raise ValueError('V3_ACTUAL_INITIAL_GRIPPER_IDENTITY')
    assert np.all(d.qvel==0) and np.all(d.ctrl==0)
    limited=model.jnt_limited.astype(bool)
    for j in np.flatnonzero(limited):
        q=d.qpos[model.jnt_qposadr[j]]
        if not model.jnt_range[j,0]<=q<=model.jnt_range[j,1]:raise ValueError('V3_INITIAL_JOINT_LIMIT')
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_table');cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'task_cube')
    assert np.array_equal(model.geom_size[table],[.17,.24,.025])
    assert abs(d.geom_xpos[table,2]+.025-card['table_top_m'])<1e-14
    assert abs(d.xpos[cube,2]-(card['table_top_m']+.02))<1e-14
    return dict(status='PASS_ZERO_STEP_INIT_TASK_ONLY',finger_qpos_m=values.tolist(),finger_command_fraction=expected/.044,table_top_m=card['table_top_m'],cube_center_z_m=float(d.xpos[cube,2]),time_s=float(d.time),physics_steps=0)

def actual_zero_audit(folder,card,expected_qpos):
    from scripts.pi0292.durable import write
    rows=[json.loads(l) for l in (Path(folder)/'initialization.jsonl').read_text().splitlines()]
    actual=next(r for r in rows if r['kind']=='actual_startup_V1')
    if actual['time']!=0 or not np.allclose(actual['qpos'],expected_qpos,atol=1e-15,rtol=0):raise ValueError('V3_REAL_MJDATA_INITIALIZATION_MISMATCH')
    if any(actual['qvel']) or any(actual['ctrl']):raise ValueError('V3_REAL_INITIAL_VELOCITY_OR_CONTROL')
    write(Path(folder)/'ACTUAL_ZERO_STEP_IDENTITY.json',dict(status='PASS_REAL_MJDATA_EQUALS_PREEXECUTION',qpos_sha256=__import__('hashlib').sha256(np.asarray(actual['qpos'],dtype='<f8').tobytes()).hexdigest(),finger_qpos_m=actual['qpos'][25:27],table_top_m=card['table_top_m'],time_s=0,physics_before_audit=0))
