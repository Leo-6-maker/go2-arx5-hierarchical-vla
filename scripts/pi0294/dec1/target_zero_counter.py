"""Existing frozen full-body task asset counter; no policy/controller/physics.

A new current registered DEC1 child and both resource reservations are mandatory
BEFORE importing MuJoCo. Old R4 worker, its pilot scope and ledgers are never
called. Every constructor intent is retained; no automatic retry is provided.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
from . import target_zero_counter_contract as contract


def read(path):return contract.obj(Path(path).read_bytes())
def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def put(path,value):
    path=Path(path);raw=contract.encode(value)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return contract.sha(raw)


def validate(root):
    need=contract.need;root=Path(root);now=time.time()
    m=read(root/'RUN_MANIFEST.json');q=read(root/'REQUEST.json');b=read(root/'OUTPUT_BINDING.private.json')
    inputs=read(root/'INPUT_BINDINGS.private.json');code=root/m['code']['capsule_relative']
    need(root.is_absolute() and not root.is_symlink() and root.resolve()==Path(b['run_root']).resolve()
         and root.parent.name=='children' and root.parent.parent.name=='20261018T182956Z-localpc-pv1'
         and m['run_uuid']==root.name==q['child_run_uuid'],'REGISTERED_CHILD_ROOT_AND_RUN_UUID')
    need(not code.is_symlink() and code.resolve().is_relative_to(root.resolve())
         and Path(__file__).resolve()==(code/'scripts/pi0294/dec1/target_zero_counter.py').resolve(),
         'ACTUAL_IMPORTED_IMMUTABLE_CAPSULE')
    need(m['schema']=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1' and m['scope']==q['scope']==contract.ROLE
         and m['owner']=='EXISTING_LOCAL_PC' and m['authority']['protocol_ref']==contract.PARENT_AUTHORITY
         and m['authority']['amendments']==[contract.AUTHORITY,contract.VAP1_AUTHORITY],
         'CURRENT_EXPLICIT_ZERO_COUNTER_SCOPE_NOT_OLD_PILOT')
    e=m['execution'];need(e['profile']=='4080-simulation' and e['host']==os.uname().nodename
         and e['uid']==os.getuid()==b['uid'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
         and 'PBS_JOBID' not in os.environ and not os.environ.get('LD_PRELOAD') and os.environ.get('GR00T_WBC_PYTHON')==sys.executable==inputs['runtime_python']
         and os.environ.get('PV1_RUN_MANIFEST')==str(code.parent/'WORKER_RUN_MANIFEST.json')
         and read(code.parent/'WORKER_RUN_MANIFEST.json')==m,'ACTUAL_STANDARD_PROFILE_HOST_UID_AND_WORKER_MANIFEST')
    parent=root.parent.parent;adoption=read(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    for relative in ('authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json','authority/VAP1/VAP1_ADOPTION_AND_DELTA_BUDGET.json'):
        need((root/'plans'/Path(relative).name).read_bytes()==(parent/relative).read_bytes(),'ACTUAL_CURRENT_ADOPTION_BYTES')
    need(filehash(parent/'authority/VAP1/VAP1_ADOPTION_AND_DELTA_BUDGET.json')==contract.VAP1_SHA,'ADOPTED_VAP1_BYTES')
    need(q['schema']==contract.REQUEST_SCHEMA and q['authority_release']==contract.AUTHORITY
         and q['original_T0_epoch']==m['limits']['PV1_T0_epoch']==adoption['T0_epoch']==1792348196
         and m['limits']['PV1_hard_close_epoch']==adoption['hard_closeout_epoch']==contract.HARD_CLOSE
         and now<q['deadline_epoch']==m['limits']['operation_deadline_epoch']<=min(adoption['stop_new_physics_epoch'],1792866596),
         'ACTUAL_ORIGINAL_CLOCK_NO_REOPENED_WINDOW')
    contract.vector(q['requested'])
    need(q['requested']==contract.REQUESTED and m['limits']['engine_starts']==2
         and all(type(m['limits'][k]) is int and m['limits'][k]==0 for k in ('physics_ticks','policy_episodes'))
         and all(type(q[k]) is int and q[k]==0 for k in ('GPU_submissions','training_updates','GR00T_queries','policy_episodes','physics_ticks')),
         'ZERO_INTEGRATION_ONLY_NOT_A_PHYSICAL_DISPATCH')
    need(filehash(root/'OUTPUT_BINDING.private.json')==e['output_binding_sha256']
         and filehash(root/'INPUT_BINDINGS.private.json')==e['input_binding_sha256'],
         'EXACT_CURRENT_INPUT_OUTPUT_BINDINGS')
    role=contract.ROLE+'__'+root.name;need(b['role']==e['private_path_role']==role,'EXACT_NEW_REGISTERED_ROLE')
    lease_path=Path(b['lease']);lease=read(lease_path)
    need(lease['status']=='ACTIVE' and lease['run_uuid']==root.name and lease['scope']==contract.ROLE
         and lease['authority_release']==contract.PARENT_AUTHORITY and lease['deadline_epoch']==q['deadline_epoch'],
         'ACTUAL_CURRENT_CHILD_LEASE')
    rows=[r for r in read(b['registry'])['entries'] if r.get('role')==role]
    need(len(rows)==1 and rows[0]['run_uuid']==root.name and Path(rows[0]['path']).resolve()==root.resolve(),
         'EXISTING_CANONICAL_REGISTRY_SINGLE_CHILD')
    pb=read(parent/'OUTPUT_BINDING.private.json');pl=read(pb['lease'])
    need(pl['status']=='ACTIVE_CONTROL_PLANE_ONLY' and pl['authority_release']==contract.PARENT_AUTHORITY,
         'EXISTING_PARENT_CONTROL_OWNER')
    need(not any(p.exists() for p in (parent/'STOP.json',root/'STOP.json')),'NO_PARENT_OR_CHILD_STOP')
    seal=read(root/m['code']['source_seal_relative']);need(seal['source_sha']==m['code']['vla_commit']
         and filehash(root/m['code']['source_seal_relative'])==m['code']['source_seal_sha256'],'SOURCE_SEAL')
    for name,pin in seal['files'].items():
        path=code/name;need(not path.is_symlink() and path.resolve().is_relative_to(code.resolve())
            and not path.stat().st_mode&0o222 and filehash(path)==pin,'SEALED_SOURCE_BYTES_'+name)
    for name,pin in contract.FROZEN_SOURCES.items():need(seal['files'].get(name)==pin,'ORIGINAL_FROZEN_TASK_OVERLAY_'+name)
    for name,pin in e['sealed_inputs'].items():
        path=root/name;need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
            and not path.stat().st_mode&0o222 and filehash(path)==pin,'EXACT_SEALED_INPUT_'+name)
    ci=read(root/'plans/PUBLICATION_AND_CI.private.json');need(ci['source']==m['code']['vla_commit']
         and ci['CI_conclusion']=='success' and ci['all_12_steps_success'] is True
         and all(seal['files'].get(n)==h for n,h in ci['full_readback_file_sha256'].items() if n in seal['files']),
         'EXACT_PUBLISHED_COUNTER_CODE_CI')
    need(set(contract.FROZEN_SOURCES)|{'scripts/pi0294/dec1/target_zero_counter.py',
        'scripts/pi0294/dec1/target_zero_counter_contract.py'}<=set(seal['files']), 'FULL_COUNTER_AND_FROZEN_OVERLAY_CODE')
    need(all(ci['full_readback_file_sha256'].get(n)==seal['files'][n] for n in
        ('scripts/pi0294/dec1/target_zero_counter.py','scripts/pi0294/dec1/target_zero_counter_contract.py')),
        'EXACT_CI_AND_READBACK_OF_ACTUAL_NATIVE_CODE')
    population=(root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json').read_bytes();plan=contract.plan(population)
    need(read(root/'plans/TARGET_ZERO_COUNTER_PLAN.json')==plan
         and filehash(root/'plans/TARGET_ZERO_COUNTER_PLAN.json')==m['data']['manifest_sha256'],
         'EXACT_PREDECLARED_ZERO_TIME_TRAIN_GEOMETRY')
    need(filehash(e['entry_path'])==e['entry_sha256']=='0c7973e0a3bc0e585e7ab96cfcc9033532a98f56e9c9410882282a291bb7082f'
         and filehash(e['doctor_path'])==e['doctor_sha256']=='013e7cea41740b10fdd213e30f6d65b0fbcaee295cce44b624fc61e23fe6d861',
         'ACTUAL_EXISTING_STANDARD_ENTRY_AND_DOCTOR')
    current=read(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json');rid=q['request_id']
    need(current['unknown_effects']==[],'NO_UNKNOWN_CORE_METADATA')
    row=current['target_zero_counter_reservations'][rid]
    need(row['status']=='RESERVED' and row['schema']==contract.ROW_SCHEMA and row['scope']==contract.ROLE
         and row['manifest_sha256']==filehash(root/'RUN_MANIFEST.json')
         and row['request_sha256']==filehash(root/'REQUEST.json') and row['unknown_effects']==[]
         and row['remaining']==row['requested']==contract.REQUESTED,'ACTUAL_CORE_COMPLETE_RESOURCE_RESERVATION')
    confirm=read(root/'LOCAL_RESERVATION_CONFIRMED.private.json')
    need(confirm['request_id']==rid and confirm['manifest_sha256']==row['manifest_sha256']
         and confirm['reservation']==row and confirm['status']=='ACTUAL_LOCAL_PARENT_RESERVATION_CONFIRMED',
         'ACTUAL_SEALED_LOCAL_RESERVATION_READBACK')
    need(type(m['limits']['memory_bytes']) is int and 0<m['limits']['memory_bytes']<=8*1024**3,'EIGHT_GIB_MEMORY_BOUND')
    need(len(m['limits']['cpus']) in (1,2) and len(set(m['limits']['cpus']))==len(m['limits']['cpus'])
         and set(m['limits']['cpus'])<=os.sched_getaffinity(0),'FRESH_NATIVE_CPU_AFFINITY')
    assets=read(root/'plans/ASSET_IDENTITY.private.json')
    canonical=json.dumps(assets,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    need(contract.sha(canonical)=='edcd253c9f6a1ee09237bba9135957015e3e901d3fa25ff9a8cdd9c8643ec4a0',
         'ORIGINAL_26_GEOMETRY_ASSET_IDENTITIES')
    need(inputs['frozen_paths'].get(inputs['compiled_model'])=='168b1232df2a50a163709c1c1e57d13d39aa3efb92a4c98005e797924142376d',
         'EXACT_ORIGINAL_FULLBODY_COMPILED_PLANT')
    for name,pin in dict(assets['absolute_files'],**inputs['frozen_paths']).items():
        need(filehash(name)==pin,'ACTUAL_FROZEN_ASSET_BYTES')
    return m,q,inputs,plan


def run(root):
    root=Path(root);m,q,inputs,plan=validate(root);need=contract.need
    intent=root/'evidence/ZERO_COUNTER_INTENT.json'
    put(intent,dict(scope=contract.ROLE,request_id=q['request_id'],manifest_sha256=filehash(root/'RUN_MANIFEST.json'),
        pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],once=True,new_physics_permitted=0))
    os.sched_setaffinity(0,set(m['limits']['cpus']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_AS,(m['limits']['memory_bytes'],m['limits']['memory_bytes']))
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',
        GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('MUJOCO_EGL_DEVICE_ID',None)
    def expire(*_):raise TimeoutError('BOUNDED_ZERO_COUNTER_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expire);signal.alarm(300)
    started=time.process_time();attempts=[];blocked=[];renderer=None;result=None;originals={}
    def guard():
        need(time.time()<q['deadline_epoch'] and not(root/'STOP.json').exists()
             and not(root.parent.parent/'STOP.json').exists(),'ORIGINAL_DEADLINE_OR_STOP')
    try:
        import mujoco
        import numpy as np
        from scripts.pi0294.task_plant import xml_scene,check_frozen_robot
        from scripts.pi0294.task_contacts import allowed
        originals={name:getattr(mujoco,name) for name in ('mj_step','mj_step1','mj_step2')}
        def deny(*args,**kwargs):blocked.append('INTEGRATION_ATTEMPT');raise PermissionError('ZERO_COUNTER_HAS_NO_PHYSICS_AUTHORITY')
        for name in originals:setattr(mujoco,name,deny)
        def construct(label,fn):
            guard();need(len(attempts)<2,'TWO_NATIVE_CONSTRUCTOR_CAP')
            attempts.append(label);put(root/'evidence'/('CONSTRUCTOR_'+str(len(attempts))+'_INTENT.json'),dict(label=label,attempt=len(attempts)))
            model=fn();put(root/'evidence'/('CONSTRUCTOR_'+str(len(attempts))+'_RETURNED.json'),dict(label=label,nq=model.nq,nv=model.nv,nu=model.nu))
            return model
        original=construct('ORIGINAL_FROZEN_COMPILED_FULLBODY',lambda:mujoco.MjModel.from_binary_path(inputs['compiled_model']))
        text=xml_scene(plan['card'],inputs['asset_root'])
        model=construct('ORIGINAL_FROZEN_TASK_OVERLAY',lambda:mujoco.MjModel.from_xml_string(text))
        frozen=check_frozen_robot(original,model)
        data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
        for name in ('x5_joint2','x5_joint3'):
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);data.qpos[model.jnt_qposadr[j]]=.9
        mujoco.mj_forward(model,data)
        need(data.time==0. and all(allowed(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),
             mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)) for c in data.contact), 'ORIGINAL_ZERO_STEP_INITIAL_CONTACT_RULE')
        state=lambda:contract.sha(np.concatenate((data.qpos,data.qvel,data.ctrl,[data.time])).tobytes())
        before=state();renderer=mujoco.Renderer(model,height=256,width=256);images={}
        for camera in ('body_rgb','wrist_rgb'):
            guard();renderer.update_scene(data,camera=camera);pixels=renderer.render().copy()
            need(pixels.shape==(256,256,3) and pixels.dtype==np.uint8,'ORIGINAL_TWO_RGB_CAMERAS')
            raw=pixels.tobytes();path=root/'evidence'/(camera+'.qualification-only.rgb')
            with path.open('xb') as f:f.write(raw)
            images[camera]=dict(sha256=contract.sha(raw),shape=list(pixels.shape),bytes=len(raw))
        from OpenGL.GL import glGetString,GL_RENDERER
        driver=glGetString(GL_RENDERER).decode();need('llvmpipe' in driver.lower() or 'softpipe' in driver.lower(),'CPU_SOFTWARE_RENDERER_ONLY')
        need(state()==before and data.time==0. and not blocked,'ZERO_TIME_STATE_UNCHANGED_DURING_RENDER')
        need(not any(n==p or n.startswith(p+'.') for n in sys.modules for p in ('torch','rclpy')),'NO_NEURAL_OR_ROS_RUNTIME_IN_ZERO_COUNTER')
        result=dict(status='PASS_FROZEN_FULLBODY_TASK_ZERO_INTEGRATION_ASSET_COUNTER_ONLY',
            frozen_robot=frozen,nq=model.nq,nv=model.nv,nu=model.nu,MuJoCo_version=mujoco.__version__,
            zero_time_state_sha256=before,task_xml_sha256=contract.sha(text.encode()),images=images,renderer=driver)
    except BaseException as exc:
        result=dict(status='TARGET_ZERO_COUNTER_ENGINEERING_FAILURE_PRESERVED',error=type(exc).__name__+': '+str(exc))
    finally:
        signal.alarm(0)
        if renderer is not None:
            try:renderer.close()
            except Exception as exc:
                result['status']='TARGET_ZERO_COUNTER_ENGINEERING_FAILURE_PRESERVED';result['renderer_close_error']=repr(exc)
        if originals:
            for name,fn in originals.items():setattr(mujoco,name,fn)
        result.update(scope=contract.ROLE,request_id=q['request_id'],manifest_sha256=filehash(root/'RUN_MANIFEST.json'),
            constructor_attempts=len(attempts),constructor_labels=attempts,integration_attempts=blocked,
            physics_ticks=0,policy_episodes=0,WBC_or_leg_policy_forwards=0,GR00T_queries=0,
            target_task_native_qualified=False,current_policy_prefix_ACK_STOP_qualified=False,
            qualification_frames_not_training_data=True,actual_worker_CPU_s=time.process_time()-started)
        put(root/'evidence/ZERO_COUNTER_RESULT.json',result)
    print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    value=run(args.root)
    raise SystemExit(0 if value['status'].startswith('PASS_') else 1)
