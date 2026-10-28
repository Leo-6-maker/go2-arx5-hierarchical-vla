"""The two adopted source adaptations; actual states and WBC are untouched."""
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_v2_runtime import FOLDER,require_adoption
from .source_retime_v2_analysis import so3_angles
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,quat_slerp

def prepare(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);require_adoption(root,manifest);p=root/FOLDER;out=p/'PROGRAMS.json'
    if out.exists():return read(out)
    old=root/'source-demo-retarget-v1';gate=read(old/'RETARGET_TIME_GATE.json');source=read(old/'SOURCE_EXTRACTION.json');rows=[]
    folder=p/'programs';folder.mkdir(exist_ok=True)
    for row in gate['rows']:
        file=old/'sources'/row['retarget_file'];assert sha(file)==row['retarget_sha256']
        with np.load(file,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        record=next(r for r in source['records'] if r['id']==row['id']);lo=0
        if row['task']=='T2':
            distances=np.linalg.norm(np.diff(d['poses'][:,:3,3],axis=0),axis=1);angles=so3_angles(d['poses'][:-1,:3,:3],d['poses'][1:,:3,:3])
            minimum=np.maximum.reduce([np.full(len(distances),.05),distances/.06,angles/.3]);ticks=np.maximum(1,np.ceil(minimum/.05-1e-10).astype(np.int64))
            times=np.r_[0.,np.cumsum(ticks*.05)];assert np.all(distances/(ticks*.05)<=.06*(1+1e-10)) and np.all(angles/(ticks*.05)<=.3*(1+1e-10))
        else:
            event=next(e for e in record['events'] if e['stage']=='close');index=int(np.searchsorted(np.arange(640)*.05,event['time_s']))
            lo=int(np.searchsorted(d['source_indices'],index));assert index==30 and d['gripper'][lo]==1.
            times=(d['source_indices'][lo:]-index)*.05*row['slowdown']
        events=[]
        for event in record['events']:
            index=int(np.searchsorted(np.arange(640)*.05,event['time_s']));local=int(np.searchsorted(d['source_indices'][lo:],index))
            if index<int(d['source_indices'][lo]) or local>=len(times):continue
            events.append(dict(stage=event['stage'],source_index=index,program_time_s=float(times[local])))
        dst=folder/(row['id']+'.npz');assert not dst.exists()
        np.savez_compressed(dst,poses=d['poses'][lo:],gripper=d['gripper'][lo:],time=times,source_indices=d['source_indices'][lo:])
        rows.append(dict(id=row['id'],task=row['task'],file=dst.name,sha256=sha(dst),source_input_sha256=row['retarget_sha256'],
            cut_start_index=int(d['source_indices'][lo]),cut_end_index=int(d['source_indices'][-1]),active_duration_s=float(times[-1]),events=events,
            original_uniform_factor=row['slowdown'],new_pose_values=False,actual_state_modified=False))
    value=dict(status='SIX_FROZEN_V2_PROGRAMS',source_sha=manifest['code']['vla_commit'],rows=rows,new_physics=0,new_IK=0);write(out,value);return value

def load(root,record):
    file=Path(root)/FOLDER/'programs'/record['file'];assert sha(file)==record['sha256']
    with np.load(file,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def sample(program,at):
    at=np.asarray(at);times=program['time'];poses=program['poses'];indices=np.clip(np.searchsorted(times,at,side='right')-1,0,len(times)-1)
    result=[];opening=[]
    for value,i in zip(at,indices):
        j=min(i+1,len(times)-1);fraction=0. if i==j else float(np.clip((value-times[i])/(times[j]-times[i]),0,1))
        pose=np.eye(4);pose[:3,3]=(1-fraction)*poses[i,:3,3]+fraction*poses[j,:3,3]
        pose[:3,:3]=quat_to_matrix(quat_slerp(matrix_to_quat(poses[i,:3,:3]),matrix_to_quat(poses[j,:3,:3]),fraction))
        result.append(pose);opening.append(program['gripper'][i])
    return np.asarray(result),np.asarray(opening)

def with_entry(program,entry):
    duration=float(entry['time'][-1]);assert duration>0
    return dict(time=np.r_[entry['time'][:-1],duration+program['time']],
        poses=np.concatenate([entry['poses'][:-1],program['poses']]),
        gripper=np.r_[np.ones(len(entry['time'])-1),program['gripper']])
