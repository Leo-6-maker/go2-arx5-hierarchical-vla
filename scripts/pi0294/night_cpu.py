"""Finite saved-state admission and full-source local timing, zero physics."""
import hashlib
from pathlib import Path
import numpy as np
from .night_runtime import FOLDER,read,sha,reserve_static,finish_static
from .night_source import load,packages_for,local_schedule
from .night_client import Client,PlannerFailure
from scripts.pi0292.durable import write


def saved_input(root,card):
    import mujoco
    from .sdr_v2_cpu import saved_input as previous
    from .task_state import ActualState
    from .night_geometry import xml_scene
    manifest=read(Path(root)/'RUN_MANIFEST.json');old=Path(manifest['dependencies']['prior_run'])
    oldengine,state,base=previous(old,card['task'])
    model=mujoco.MjModel.from_xml_string(xml_scene(card,read(Path(root)/'INPUT_BINDINGS.private.json')['asset_root']))
    packet={k:state[k].copy() for k in ['qpos','qvel','ctrl']}
    packet['qpos'][29]+=card['table_top_m']-.44;packet['qpos'][25:27]=card['initial_finger_q_m']
    packet.update(time_s=state['time_s'],step=0,contacts=[],contact_time_s=state['time_s']-.002)
    engine=ActualState(model);preview=engine.update(packet)
    preview['CPU_view']='SAVED_ROOT_ARM_REGISTERED_TASK_INIT_PREVIEW_NOT_ACTUAL_OR_RGB'
    return engine,preview,base


def source_path(root,client,state,record,endpoint,stage_name):
    import mujoco
    from go2arx5_vla.trajectory.so3 import matrix_to_quat
    program=load(root,record);m,d=client.model,client.native_data;d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl'];d.qpos[client.arm_q]=endpoint
    jacp=np.zeros((3,m.nv));jacr=np.zeros_like(jacp);qs=[];iterations=0;status='INCOMPLETE'
    reserve_static(root,stage_name,5*len(program['poses']))
    try:
        for target in program['poses']:
            for iteration in range(5):
                iterations+=1;mujoco.mj_forward(m,d)
                ep=target[:3,3]-d.site_xpos[client.site];delta=matrix_to_quat(target[:3,:3]@d.site_xmat[client.site].reshape(3,3).T)
                if delta[3]<0:delta=-delta
                length=np.linalg.norm(delta[:3]);er=delta[:3]*2*np.arctan2(length,delta[3])/max(length,1e-15)
                if np.linalg.norm(ep)<=1e-5 and np.linalg.norm(er)<=1e-4:break
                mujoco.mj_jacSite(m,d,jacp,jacr,client.site)
                step=np.linalg.lstsq(np.vstack([jacp[:,client.arm_v],jacr[:,client.arm_v]]),np.r_[ep,er],rcond=1e-10)[0]
                if not np.isfinite(step).all():raise PlannerFailure('NIGHT_FULL_SOURCE_NONFINITE_IK',{})
                d.qpos[client.arm_q]+=step
            else:raise PlannerFailure('NIGHT_FULL_SOURCE_CONTINUATION_IK',dict(completed_knots=len(qs)))
            if np.any(d.qpos[client.arm_q]<m.jnt_range[client.arm_j,0]) or np.any(d.qpos[client.arm_q]>m.jnt_range[client.arm_j,1]):
                raise PlannerFailure('NIGHT_FULL_SOURCE_NATIVE_LIMIT',dict(completed_knots=len(qs)))
            qs.append(d.qpos[client.arm_q].copy())
        timing,proof=local_schedule(program['poses'],np.asarray(qs),program['time']);status='PASS_LOCAL_JOINT_TIME_NO_PHYSICS'
        return dict(program,time=timing),proof,np.asarray(qs)
    finally:finish_static(root,stage_name,iterations,status)


def qualify_source(root,manifest,card):
    p=Path(root)/FOLDER;engine,state,base=saved_input(root,card);client=Client(root,engine);packages=packages_for(root,card['task']);failures=[]
    for package in packages:
        name=card['teacher_version']+'_CPU_'+card['task']+'_'+str(package['index'])+card.get('CPU_attempt_suffix','')
        try:
            _,certificate,index=client.bundle(state,[package],'approach',name,0)
            record=next(r for r in read(p/'PROGRAMS.json')['rows'] if r['id']==package['source_id'])
            endpoint=np.asarray(client.last_route['position'])[-1]
            program,proof,q=source_path(root,client,state,record,endpoint,name+'_FULL_SOURCE')
            needed=certificate['duration_s']+float(program['time'][-1])+4
            duration=next((t for t in (32,64,96,128) if needed<=t),None)
            if duration is None:raise PlannerFailure('NIGHT_PREDECLARED_DOMAIN_TOO_SHORT',dict(required_s=needed))
            filename=card['teacher_version']+'-'+record['file'];dest=p/'programs'/filename
            np.savez_compressed(dest,**program)
            events=[]
            for e in record['events']:
                index=int(np.searchsorted(program['source_indices'],e['source_index']));events.append(dict(e,program_time_s=float(program['time'][index])))
            updated=dict(record,file=filename,sha256=sha(dest),active_duration_s=float(program['time'][-1]),events=events,original_V3_sha256=record['sha256'],local_timing=proof)
            rows=read(p/'PROGRAMS.json');rows['rows']=[updated if r['id']==record['id'] else r for r in rows['rows']];write(p/'PROGRAMS.json',rows)
            np.savez_compressed(p/(name+'-STATIC_JOINT_WITNESS.npz'),joint_q=q,time=program['time'])
            result=dict(status='PASS_SOURCE_ROUTE_FULL_STATIC_TIME',task=card['task'],source_id=record['id'],program_sha256=updated['sha256'],duration_s=duration,certificate=certificate,local_time=proof,CPU_view=state['CPU_view'],earlier_failures=failures,new_physics=0)
            write(p/(name+'-ADMISSION.json'),result);return result
        except (PlannerFailure,ValueError) as exc:
            value=dict(source_id=package['source_id'],status='NOT_ADMITTED',reason=str(exc),evidence=getattr(exc,'evidence',None));failures.append(value);write(p/(name+'-FAIL.json'),value)
    result=dict(status='NO_COMPLETE_FIXED_SOURCE_ROUTE',task=card['task'],failures=failures,new_physics=0);write(p/(card['teacher_version']+'-CPU-'+card['task']+'-RESULT.json'),result);return result
