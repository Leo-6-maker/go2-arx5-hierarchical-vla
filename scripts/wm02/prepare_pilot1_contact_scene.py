"""Compile and audit a new reachable course with zero physics integrations."""
import argparse,json,time,ctypes
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    import numpy as np,mujoco
    from go2arx5_vla.world_model.pilot1_curriculum import scene_xml,robot_contract
    from go2arx5_vla.world_model.native_replay import packets,forbid_integration
    from go2arx5_vla.world_model.storage import digest,write_json
    from scripts.pi0294.task_contacts import allowed
    s=json.loads(a.spec.read_text());started=time.monotonic();cpu=time.process_time()
    if mujoco.__version__!='3.2.7':raise ValueError('SAME_NATIVE_MUJOCO327_REQUIRED')
    for name in ('original_model','actual_path_journal','reset_library'):
        if digest(s[name])!=s[name+'_sha256']:raise ValueError('FROZEN_SCENE_AUDIT_INPUT_CHANGED:'+name)
    a.output.mkdir(exist_ok=False);original=mujoco.MjModel.from_binary_path(s['original_model']);text=scene_xml(s,original);(a.output/'SCENE.private.xml').write_text(text)
    model=mujoco.MjModel.from_xml_string(text);contract=robot_contract(original,model);mujoco.mj_saveModel(model,str(a.output/'SCENE.private.mjb'),None);model=mujoco.MjModel.from_binary_path(str(a.output/'SCENE.private.mjb'));robot_contract(original,model)
    data=mujoco.MjData(model);handle=ctypes.CDLL(s['reset_library']);reset=handle.WM02ResetAudit;reset.argtypes=[ctypes.c_void_p,ctypes.c_void_p];reset.restype=ctypes.c_int
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'task_cube');q=int(model.jnt_qposadr[model.body_jntadr[cube]]);issues=[];touches=[];samples=0
    def contacts(step):
        for contact in data.contact:
            names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g))for g in(contact.geom1,contact.geom2)];row=dict(step=step,names=names,distance_m=float(contact.dist))
            if not allowed(*names)or('task_cube_geom'in names and contact.dist<-.004):issues.append(row)
            if'task_cube_geom'in names and any(n in names for n in('task_left_pad','task_right_pad')):touches.append(row)
    with forbid_integration(mujoco)as calls:
        if reset(model._address,data._address)!=0:raise ValueError('ORIGINAL_NATIVE_RESET_REJECTED')
        contacts(0);initial=data.qpos.copy()
        for row in packets(s['actual_path_journal']):
            if row['step']%25!=3:continue
            data.qpos[:]=row['qpos'];data.qvel[:]=0;data.qpos[q:q+7]=initial[q:q+7];mujoco.mj_forward(model,data);contacts(row['step']);samples+=1
    result=dict(status='COURSE_STATIC_PASSED_NATIVE_STILL_REQUIRED'if not issues else'COURSE_STATIC_REJECTED',scene_id=s['scene_id'],parent=s['parent'],split=s['split'],source_sha=s['source_sha'],scene_sha256=digest(a.output/'SCENE.private.mjb'),scene_xml_sha256=digest(a.output/'SCENE.private.xml'),robot_contract=contract,
        initial_cube_xyz_m=initial[q:q+3].tolist(),table_center_xyz_m=s['table_center_xyz_m'],table_half_size_m=s['table_half_size_m'],
        actual_path_journal_sha256=s['actual_path_journal_sha256'],counterfactual_static_sweep_states=samples,unsafe_geometry=issues,geometric_pad_cube_overlaps=touches,
        static_sweep_is_not_dynamic_qualification=True,real_contact_positives=0,new_physics_integrations=0,forbidden_integration_attempts=len(calls),WM_updates=0,Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-started)
    write_json(a.output/'RESULT.private.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('unsafe_geometry','geometric_pad_cube_overlaps')}));print(json.dumps(dict(unsafe_geometry=issues[:12],geometric_pad_cube_overlaps=touches[:12])));return 0 if not issues else 1


if __name__=='__main__':raise SystemExit(main())
