"""Prepare allowlisted object-layout MJB variants; no simulator integration."""
def main(argv=None):
    import argparse,json,time,hashlib
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.storage import digest,write_json
    s=json.loads(a.spec.read_text())
    if digest(s['original_model'])!=s['original_model_sha256']:raise ValueError('FROZEN_ORIGINAL_MODEL_CHANGED')
    import mujoco,numpy as np
    from go2arx5_vla.world_model.target_manifest import model_fingerprint
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0294.task_contacts import allowed
    a.output.mkdir(exist_ok=False);original=mujoco.MjModel.from_binary_path(s['original_model']);fp,document=model_fingerprint(original);body=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,'task_cube');joint=int(original.body_jntadr[body]);q=int(original.jnt_qposadr[joint]);initial=original.qpos0[q:q+3].copy();scenes=[]
    requests=[dict(scene_id='ORIGINAL',parent='SAME_ORIGINAL_TASK_SCENE_RESET_V2_ENGINEERING',split='TRAIN',offset_xy_m=[0.,0.]),*s['new_scenes']]
    if len({r['parent']for r in requests})!=len(requests):raise ValueError('UNIQUE_PHYSICAL_PARENTS_REQUIRED')
    for r in requests:
        offset=np.asarray(r['offset_xy_m'],dtype=float)
        if offset.shape!=(2,)or not np.isfinite(offset).all()or np.max(np.abs(offset))>.025:raise ValueError('FINITE_PREDECLARED_LOCAL_OBJECT_LAYOUT_REQUIRED')
        m=mujoco.MjModel.from_binary_path(s['original_model']);m.qpos0[q:q+2]+=offset
        path=Path(s['original_model'])if r['scene_id']=='ORIGINAL'else a.output/(r['scene_id']+'.mjb')
        if r['scene_id']!='ORIGINAL':mujoco.mj_saveModel(m,str(path),None)
        m=mujoco.MjModel.from_binary_path(str(path));actual_fp,_=model_fingerprint(m)
        if actual_fp!=fp:raise ValueError('FROZEN_MODEL_CHANGE_OUTSIDE_OBJECT_INITIAL_XY')
        d=mujoco.MjData(m)
        with forbid_integration(mujoco)as calls:mujoco.mj_forward(m,d)
        bad=[]
        for c in d.contact:
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))for g in(c.geom1,c.geom2)]
            if not allowed(*names)or('task_cube_geom'in names and float(c.dist)<-.004):bad.append(dict(names=names,distance_m=float(c.dist)))
        if calls:raise RuntimeError('STATIC_SCENE_PREPARATION_MUST_NOT_INTEGRATE')
        if not np.allclose(d.xpos[body],initial+np.r_[offset,0.],atol=1e-10,rtol=0):raise ValueError('ACTUAL_OBJECT_LAYOUT_NOT_APPLIED')
        scenes.append(dict(**r,model_path=str(path),model_sha256=digest(path),robot_fingerprint=fp,actual_initial_object_position_m=d.xpos[body].tolist(),static_check='REJECTED_UNSAFE_INITIAL_CONTACT'if bad else'PASSED_NO_INTEGRATION',unsafe_initial_contacts=bad,physics_integrations=0))
    result=dict(schema='WM02_NIGHT1_SCENE_ALLOWLIST_V1',robot_fingerprint=fp,original_model_sha256=s['original_model_sha256'],native_binary_sha256=s['native_binary_sha256'],scene_variation='TASK_CUBE_INITIAL_XY_ONLY_ROBOT_AND_ALL_OTHER_MODEL_ARRAYS_FROZEN',scenes=scenes,source_sha=s['source_sha'],physics_integrations=0,WM_updates=0,Actor_updates=0)
    write_json(a.output/'ROBOT_FINGERPRINT.private.json',document);write_json(a.output/'SCENE_ALLOWLIST.private.json',result);print(json.dumps(dict(scenes=len(scenes),static_pass=sum(x['static_check']=='PASSED_NO_INTEGRATION'for x in scenes),robot_fingerprint=fp,physics_integrations=0)));return 0
if __name__=='__main__':raise SystemExit(main())
