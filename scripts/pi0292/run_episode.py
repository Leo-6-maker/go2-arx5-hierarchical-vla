"""One pre-registered actual episode; strict single owner and no retry."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image

from scripts.pi0291_arx5_data.config import PARAMETERS as P,DOMAIN,canonical_hash
AUTHORITY = "261bc5b37f20aaff1ad7d5742252d3dc1a91fa65"
from scripts.pi0291_arx5_data.engine import Engine
from scripts.pi0291_arx5_data.teacher import Teacher
from go2arx5_vla.task_data.io import write,jsonable,file_hash,manifest
from go2arx5_vla.task_data.codec import pose10
from .resource_guard import Budget
from scripts.pi0291_arx5_data.score import saved_score,BOOL_FIELDS
from go2arx5_vla.trajectory.so3 import matrix_to_quat


def run(root,identifier,source,mode,backend):
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:4]))
    root=Path(root);folder=root/'episodes'/identifier
    code_seal=__import__('json').loads((root/'CODE_SEAL.json').read_text())
    assert code_seal['source_sha']==source
    for name,digest in code_seal['files'].items():assert file_hash(root/'code'/name)==digest,name
    if backend=='mesa_cpu':
        assert os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')=='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        assert os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        assert 'MUJOCO_EGL_DEVICE_ID' not in os.environ
    else:
        gpu_check=__import__('subprocess').run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if gpu_check.stdout.strip():raise RuntimeError('GPU1 compute resource not idle: preserve other user work')
    if folder.exists():raise RuntimeError('Existing episode evidence: will not overwrite or retry')
    seal=__import__('json').loads((root/'ATTEMPT_PLAN.json').read_text())
    assert seal['parameter_sha256']==canonical_hash(P) and mode=='normal'
    card=seal['attempts'][identifier]['card']
    private=__import__('json').loads((root/'PRIVATE_ASSET_BINDING.json').read_text())
    card={**card,'asset_root':private['native_asset_root']}
    budget=Budget(root,gpu_service=backend=='nvidia_egl');budget.start(identifier,source)
    folder.mkdir();write(folder/'CARD.json',{**card,'mode':mode,'source_sha':source,'authority':AUTHORITY,'domain':DOMAIN})
    began=time.monotonic();engine=None;samples=[];physical=[];refs=[];highrate=[];status='ENGINEERING_INCOMPLETE';error=None;GPU_s=0.
    try:
        engine=Engine(card,mode);first=engine.snapshot();teacher=Teacher(card,first['object_pose'],mode)
        for step in range(641):
            s=engine.snapshot();assert abs(s['time_s']-step/20)<=1e-8
            if not s['safety_valid']:raise RuntimeError('REAL_SAFETY_FAIL')
            images,geometry=engine.render(s)
            if step==0:
                from OpenGL import GL
                gl_renderer=GL.glGetString(GL.GL_RENDERER).decode()
                gl_vendor=GL.glGetString(GL.GL_VENDOR).decode()
                if backend=='mesa_cpu':assert gl_vendor=='Mesa' and 'llvmpipe' in gl_renderer
                gpu=__import__('subprocess').run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
                own_rows=[x for x in gpu.stdout.splitlines() if x.strip().startswith(str(os.getpid())+',')]
                if backend=='mesa_cpu':assert not own_rows
                write(folder/'RENDER_PROCESS.json',dict(pid=os.getpid(),observed_own_rows=own_rows,backend=backend,GL_VENDOR=gl_vendor,GL_RENDERER=gl_renderer,requested_EGL_device=1 if backend=='nvidia_egl' else None))
            snapshot=f'{identifier}:{engine.steps}';obs=dict(instruction=card['instruction'],actual_tcp=s['tcp'],
                actual_gripper=s['gripper_fraction'],actual_gripper_speed=s['gripper_speed'],timestamp_s=s['time_s'],valid=True)
            for camera,image in images.items():
                path=folder/camera/f'{step:04d}.png';path.parent.mkdir(exist_ok=True);Image.fromarray(image).save(path)
                obs[camera]=dict(path=path.relative_to(folder).as_posix(),sha256=file_hash(path),frame_index=step,
                    timestamp_s=s['time_s'],snapshot_id=snapshot,source='actual_mujoco_renderer',valid=True)
            teacher.observe(s)
            supervisor=None
            if step<640:
                if step%8==0:
                    plan=teacher.plan(s);ref_id=f'{identifier}:ref{step//8:03d}'
                    ref=dict(ref_id=ref_id,decision_index=step//8,decision_time_s=s['time_s'],observation_snapshot=snapshot,
                        prior_committed_count=step,anchor_actual_tcp=s['tcp'],planned_H40=plan['poses'],gripper_H40=plan['opening'],
                        offsets_s=plan['offset_s'],rows10=np.stack([pose10(a,b) for a,b in zip(plan['poses'],plan['opening'])]),
                        raw_quaternion_xyzw=np.stack([matrix_to_quat(a[:3,:3]) for a in plan['poses']]),
                        privileged_teacher_stage=plan['stage'],source='ACTUAL_FEEDBACK_TEACHER',committed_prefix_count=0)
                    refs.append(ref)
                k=step%8;target=plan['poses'][k];opening=float(plan['opening'][k])
                supervisor=dict(ref_id=ref_id,sequence=step,point_index=k,command_time_s=s['time_s'],
                    desired=target,accepted=target.copy(),installed=target.copy(),committed=target.copy(),
                    desired_gripper=opening,committed_gripper=opening,valid=mode=='normal',actually_committed=True)
                refs[-1]['committed_prefix_count']+=1
            history=dict(episode_id=identifier,segment_id=0,previous_samples=step,previous_committed=step,reset=False,abort=False,takeover=False)
            physical.append({k:s[k] for k in ['time_s','object_pose','tcp','object_linear_velocity','object_angular_velocity',*BOOL_FIELDS]})
            samples.append(dict(snapshot_id=snapshot,policy_obs=obs,causal_history=history,supervision=supervisor,
                execution=dict(actual_tcp=s['tcp'],actual_finger_q=s['finger_q'],actual_finger_qd=s['finger_qd'],actual_gripper_gap_m=s['gripper_gap_m'],
                    actuator_force=s['actuator_force'],actual_joint_q=s['arm_q'],actual_joint_qd=s['arm_qd'],previous_finger_targets=engine.finger_targets.copy(),reference_pose_error=None if supervisor is None else float(np.linalg.norm(target[:3,3]-s['tcp'][:3,3]))),
                privileged=dict(object_pose=s['object_pose'],object_linear_velocity=s['object_linear_velocity'],object_angular_velocity=s['object_angular_velocity'],
                    current_grasped=s['native_grasp'],ever_lifted=teacher.lifted,ever_carried=teacher.carried,teacher_stage=teacher.stage,
                    contacts=s['contacts'],opposing_contact=s['opposing_contact'],relative_motion_valid=s['relative_motion_valid'],object_points=s['object_points']),geometry=geometry))
            budget.progress(identifier,engine.steps,len(samples));budget.guard()
            if step<640:
                for tick in range(25):
                    try:engine.step(target,opening)
                    except Exception:
                        highrate.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                            np.zeros(6),1.])
                        raise
                    highrate.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                        engine.data.xfrc_applied[engine.ids['tool']],float(bool(engine.safety))])
        status='VALID_COMPLETE'
        write(folder/'TEACHER_EVENTS.json',teacher.events)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        write(folder/'ERROR.json',dict(error=error,traceback=traceback.format_exc(),steps=0 if engine is None else engine.steps))
    finally:
        if engine is not None:engine.close()
        GPU_s=time.monotonic()-began if backend=='nvidia_egl' else 0.
        if physical:
            arrays={k:np.asarray([s[k] for s in physical]) for k in physical[0]}
            arrays['tcp_world']=arrays.pop('tcp');np.savez_compressed(folder/'physics.npz',**arrays)
        if highrate:np.savez_compressed(folder/'highrate.npz',rows=np.asarray(highrate),nq=engine.model.nq,nv=engine.model.nv,
            layout=np.asarray(['time','qpos','qvel','finite_joint_motor_ctrl','zero_external_wrench','safety_flag']))
        write(folder/'samples.json',samples);write(folder/'references.json',refs)
        if engine is not None:write(folder/'scene.xml.json',dict(xml=engine.xml,sha256=__import__('hashlib').sha256(engine.xml.encode()).hexdigest()))
        result=dict(id=identifier,status=status,error=error,source_sha=source,physics_steps=0 if engine is None else engine.steps,
            paired_samples=len(samples),GPU_service_s=GPU_s,render_backend=backend,domain=DOMAIN,selected='NONE',deployment='NONE')
        write(folder/'RESULT.json',result)
        budget.progress(identifier,result['physics_steps'],len(samples))
        try:
            if physical:
                score=saved_score(folder);result['score']=score
            write(folder/'RESULT.json',result);write(folder/'FILE_MANIFEST.json',manifest(folder))
        finally:
            budget.finish(identifier,status,GPU_s)
        print(__import__('json').dumps(jsonable(result)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--mode',choices=['normal','empty_grasp','single_side'],default='normal');a=p.parse_args()
    run(a.root,a.id,a.source_sha,a.mode,'mesa_cpu' if os.environ.get('LIBGL_ALWAYS_SOFTWARE')=='1' else 'nvidia_egl')

