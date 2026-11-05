"""Prepare allowlisted object-layout MJB variants; no simulator integration."""
def main(argv=None):
    import argparse,json,time,hashlib,subprocess,ctypes
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
    version=s.get('layout_reset_contract','QPOS0_XY_V1')
    a.output.mkdir(exist_ok=False);original=mujoco.MjModel.from_binary_path(s['original_model']);fp,document=model_fingerprint(original,version);body=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,'task_cube');joint=int(original.body_jntadr[body]);q=int(original.jnt_qposadr[joint]);home=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_KEY,'home');initial=(original.key_qpos[home,q:q+3]if version!='QPOS0_XY_V1'else original.qpos0[q:q+3]).copy();scenes=[];reset=None
    if version in('HOME_KEYFRAME_XY_V2','HOME_ROOT_OBJECT_XY_V3'):
        for key in('native_startup_header','native_home_header','mujoco_library'):
            if digest(s[key])!=s[key+'_sha256']:raise ValueError('FROZEN_NATIVE_RESET_INPUT_CHANGED:'+key)
        cpp=a.output/'reset_audit.cpp';cpp.write_text('#include "native_startup.hpp"\nextern "C" int WM02ResetAudit(const mjModel*m,mjData*d){try{PI0289ResetStartup(m,d);return 0;}catch(...){return 1;}}\n')
        lib=a.output/'reset_audit.so';argv=['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(Path(s['native_startup_header']).parent),'-I'+s['mujoco_include'],str(cpp),s['mujoco_library'],'-Wl,-rpath,'+str(Path(s['mujoco_library']).parent),'-o',str(lib)];run=subprocess.run(argv,capture_output=True,text=True,timeout=30);write_json(a.output/'RESET_AUDIT_BUILD.private.json',dict(argv=argv,exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr));run.check_returncode()
        symbols=subprocess.check_output(['nm','-D',str(lib)],text=True)
        if any(' U '+name in symbols for name in('mj_step','mj_step1','mj_step2')):raise ValueError('STATIC_RESET_HELPER_MUST_NOT_LINK_INTEGRATION')
        handle=ctypes.CDLL(str(lib));reset=handle.WM02ResetAudit;reset.argtypes=[ctypes.c_void_p,ctypes.c_void_p];reset.restype=ctypes.c_int
    requests=[dict(scene_id='ORIGINAL',parent='ROOTSAFE_ORIGINAL_TRAIN_V3'if version=='HOME_ROOT_OBJECT_XY_V3'else'SAME_ORIGINAL_TASK_SCENE_RESET_V2_ENGINEERING',split='TRAIN',offset_xy_m=[0.,0.]),*s['new_scenes']]
    if len({r['parent']for r in requests})!=len(requests):raise ValueError('UNIQUE_PHYSICAL_PARENTS_REQUIRED')
    for r in requests:
        offset=np.asarray(r['offset_xy_m'],dtype=float)
        if offset.shape!=(2,)or not np.isfinite(offset).all()or np.max(np.abs(offset))>.025:raise ValueError('FINITE_PREDECLARED_LOCAL_OBJECT_LAYOUT_REQUIRED')
        m=mujoco.MjModel.from_binary_path(s['original_model'])
        if version!='QPOS0_XY_V1':m.key_qpos[home,q:q+2]+=offset
        else:m.qpos0[q:q+2]+=offset
        base_offset=np.asarray(r.get('base_offset_xy_m',s.get('base_offset_xy_m',[0.,0.])),dtype=float)
        if base_offset.shape!=(2,)or not np.isfinite(base_offset).all()or np.max(np.abs(base_offset))>.15 or(version!='HOME_ROOT_OBJECT_XY_V3'and np.any(base_offset)):raise ValueError('PREDECLARED_FINITE_ROOT_INITIAL_POSITION_ONLY')
        if version=='HOME_ROOT_OBJECT_XY_V3':m.key_qpos[home,:2]+=base_offset
        modified=r['scene_id']!='ORIGINAL'or bool(np.any(base_offset));path=a.output/(r['scene_id']+'.mjb')if modified else Path(s['original_model'])
        if modified:mujoco.mj_saveModel(m,str(path),None)
        m=mujoco.MjModel.from_binary_path(str(path));actual_fp,_=model_fingerprint(m,version)
        if actual_fp!=fp:raise ValueError('FROZEN_MODEL_CHANGE_OUTSIDE_OBJECT_INITIAL_XY')
        d=mujoco.MjData(m)
        with forbid_integration(mujoco)as calls:
            if reset is not None:
                if reset(m._address,d._address)!=0:raise ValueError('ORIGINAL_NATIVE_STARTUP_RESET_REJECTED')
            else:mujoco.mj_forward(m,d)
        bad=[]
        for c in d.contact:
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))for g in(c.geom1,c.geom2)]
            if not allowed(*names)or('task_cube_geom'in names and float(c.dist)<-.004):bad.append(dict(names=names,distance_m=float(c.dist)))
        if calls:raise RuntimeError('STATIC_SCENE_PREPARATION_MUST_NOT_INTEGRATE')
        if not np.allclose(d.xpos[body],initial+np.r_[offset,0.],atol=1e-10,rtol=0):raise ValueError('ACTUAL_OBJECT_LAYOUT_NOT_APPLIED')
        row=dict(**r,model_path=str(path),model_sha256=digest(path),robot_fingerprint=fp,actual_initial_object_position_m=d.xpos[body].tolist(),static_check='REJECTED_UNSAFE_INITIAL_CONTACT'if bad else'PASSED_NO_INTEGRATION',unsafe_initial_contacts=bad,physics_integrations=0)
        if version=='HOME_ROOT_OBJECT_XY_V3':
            if not np.allclose(d.qpos[:2],original.key_qpos[home,:2]+base_offset,atol=1e-10,rtol=0):raise ValueError('ACTUAL_ROOT_INITIAL_PLACEMENT_NOT_APPLIED')
            row.update(base_offset_xy_m=base_offset.tolist(),actual_initial_root_position_m=d.qpos[:3].tolist())
        scenes.append(row)
    result=dict(schema='WM02_NIGHT1_SCENE_ALLOWLIST_V1',fingerprint_version=version,robot_fingerprint=fp,original_model_sha256=s['original_model_sha256'],native_binary_sha256=s['native_binary_sha256'],scene_variation='ROOT_AND_OBJECT_HOME_XY_ONLY_V3'if version=='HOME_ROOT_OBJECT_XY_V3'else('TASK_CUBE_HOME_KEYFRAME_XY_ONLY'if version=='HOME_KEYFRAME_XY_V2'else'TASK_CUBE_INITIAL_QPOS0_XY_ONLY'),scenes=scenes,source_sha=s['source_sha'],physics_integrations=0,WM_updates=0,Actor_updates=0)
    write_json(a.output/'ROBOT_FINGERPRINT.private.json',document);write_json(a.output/'SCENE_ALLOWLIST.private.json',result);print(json.dumps(dict(scenes=len(scenes),static_pass=sum(x['static_check']=='PASSED_NO_INTEGRATION'for x in scenes),robot_fingerprint=fp,physics_integrations=0)));return 0
if __name__=='__main__':raise SystemExit(main())
