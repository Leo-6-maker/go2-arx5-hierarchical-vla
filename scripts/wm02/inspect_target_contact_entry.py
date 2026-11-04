"""Inspect actual native contact entry and missing approach qualification; no integration."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.storage import digest,write_json
    s=json.loads(a.spec.read_text());pin=s['frozen_files'].get(s['model_path'])
    if digest(s['model_path'])!=pin:raise ValueError('PINNED_NATIVE_TARGET_MODEL_REQUIRED')
    import mujoco,numpy as np
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0294.task_state import ActualState,read_state
    from scripts.pi0294.task_contacts import allowed
    m=mujoco.MjModel.from_binary_path(s['model_path']);packet=read_state(a.state);actual=ActualState(m)
    with forbid_integration(mujoco)as calls:state=actual.update(packet)
    if calls:raise RuntimeError('CONTACT_ENTRY_INSPECTION_MUST_NOT_INTEGRATE')
    joints=[]
    for side in('left','right'):
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_'+side+'_slide');act=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'task_'+side+'_motor')
        if j<0 or act<0:raise ValueError('ACTUAL_NATIVE_FINGER_BINDING_REQUIRED')
        joints.append(dict(side=side,joint_id=j,actuator_id=act,joint_type=int(m.jnt_type[j]),limited=bool(m.jnt_limited[j]),range_m=m.jnt_range[j].tolist(),actual_position_m=float(packet['qpos'][m.jnt_qposadr[j]]),actual_speed_m_s=float(packet['qvel'][m.jnt_dofadr[j]]),transmission_gear=m.actuator_gear[act].tolist()))
    names={name:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)for name in('task_table','task_cube_geom','task_left_pad','task_right_pad')}
    if any(i<0 for i in names.values()):raise ValueError('EXACT_NATIVE_TABLE_OBJECT_PAD_BINDINGS_REQUIRED')
    support=[]
    for contact in packet['contacts']:
        if {contact['geom0'],contact['geom1']}=={names['task_table'],names['task_cube_geom']}:
            support.append(dict(normal_force_N=float(contact['wrench'][0]),distance_m=contact['distance_m'],point_world_m=contact['point'].tolist(),contact_time_s=packet['contact_time_s']))
    clearance=[]
    for g in range(m.ngeom):
        body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or''
        if body in('base','base_link','trunk'):
            ray=np.zeros(6);distance=mujoco.mj_geomDistance(m,actual.data,g,names['task_table'],10.,ray)
            clearance.append(dict(body=body,geom=g,distance_m=float(distance)))
    value=dict(schema='WM02_ACTUAL_TARGET_CONTACT_ENTRY_INSPECTION_V1',source_sha=s['source_sha'],spec_sha256=digest(a.spec),model_sha256=pin,native_binary_sha256=digest(s['native_binary']),state_sha256=digest(a.state),native_step=packet['step'],sim_time_s=packet['time_s'],contact_time_s=packet['contact_time_s'],finger_bindings=joints,allowed_pairs={name:allowed(name,'task_cube_geom')for name in('task_table','task_left_pad','task_right_pad')},base_table_clearance=clearance,actual_support_contacts=support,actual_supported=any(c['normal_force_N']>=.05 for c in support),actual_left_contact=state['left_object_contact'],actual_right_contact=state['right_object_contact'],actual_native_grasp=state['native_grasp'],tcp_position_m=state['tcp'][:3,3].tolist(),object_position_m=state['object_pose'][:3,3].tolist(),object_minus_tcp_m=(state['object_pose'][:3,3]-state['tcp'][:3,3]).tolist(),first_blocker='NO_QUALIFIED_NATIVE_APPROACH_PREFIX_TO_CONTACT',conditions={k:'NOT_RUN_NO_QUALIFIED_APPROACH'for k in('supported_gripper','supported_grasp','supported_release','withdrawal')},contact_qualification='NOT_ESTABLISHED_BY_READONLY_GEOMETRY',physics_integrations=0,WM_updates=0,Actor_updates=0)
    write_json(a.output,value);print(json.dumps({k:value[k]for k in('schema','native_step','actual_supported','actual_native_grasp','first_blocker','physics_integrations')}));return 0
if __name__=='__main__':raise SystemExit(main())
