"""Transform only adopted source geometry/slice; preserve actual and original programs."""
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_v3_runtime import FOLDER,require_adoption
from .sdr_v2_source import sample,load as old_load

def prepare(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);require_adoption(root,manifest);p=root/FOLDER;out=p/'PROGRAMS.json'
    if out.exists():return read(out)
    old=root/'source-demo-retime-join-v2';rows=[];(p/'programs').mkdir(exist_ok=True)
    for record in read(old/'PROGRAMS.json')['rows']:
        f=old/'programs'/record['file'];assert sha(f)==record['sha256']
        with np.load(f,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        if record['task']=='T1':lo=0;d['poses'][:,:3,3]+=np.array([0,0,.03])
        else:
            event=next(e for e in record['events'] if e['stage']=='contact');lo=int(np.searchsorted(d['source_indices'],event['source_index']));assert int(d['source_indices'][lo])==event['source_index'] and d['gripper'][lo]==0.
        offset=float(d['time'][lo]);dst=p/'programs'/record['file'];assert not dst.exists()
        np.savez_compressed(dst,poses=d['poses'][lo:],gripper=d['gripper'][lo:],time=d['time'][lo:]-offset,source_indices=d['source_indices'][lo:])
        events=[dict(e,program_time_s=e['program_time_s']-offset) for e in record['events'] if e['source_index']>=d['source_indices'][lo]]
        rows.append(dict(record,sha256=sha(dst),source_input_sha256=record['sha256'],cut_start_index=int(d['source_indices'][lo]),active_duration_s=float(d['time'][-1]-offset),events=events,
            geometry_delta_z_m=.03 if record['task']=='T1' else 0.,new_pose_values=record['task']=='T1',actual_state_modified=False))
    value=dict(status='SIX_FROZEN_V3_SOURCE_PROGRAMS',source_sha=manifest['code']['vla_commit'],rows=rows,new_physics=0,new_IK=0);write(out,value);return value

def load(root,record):
    f=Path(root)/FOLDER/'programs'/record['file'];assert sha(f)==record['sha256']
    with np.load(f,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def with_entry(program,entry):
    duration=float(entry['time'][-1]);assert duration>0
    return dict(time=np.r_[entry['time'][:-1],duration+program['time']],poses=np.concatenate([entry['poses'][:-1],program['poses']]),gripper=np.r_[np.full(len(entry['time'])-1,program['gripper'][0]),program['gripper']])
