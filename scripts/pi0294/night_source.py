"""Source spatial knots are immutable; only bounded local timing is revised."""
from pathlib import Path
import numpy as np
from .night_runtime import FOLDER,read,sha
from .sdr_v2_source import sample
from .sdr_v3_source import with_entry
from .source_retime_v2_analysis import so3_angles


def load(root,record):
    path=Path(root)/FOLDER/'programs'/record['file']
    if sha(path)!=record['sha256']:raise ValueError('NIGHT_SOURCE_PROGRAM_CHANGED')
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def packages_for(root,task):
    packages=[]
    for i,r in enumerate(x for x in read(Path(root)/FOLDER/'PROGRAMS.json')['rows'] if x['task']==task):
        d=load(root,r)
        packages.append(dict(index=i,task=task,approach=d['poses'][0],contact=d['poses'][0],direct_join=True,
            active_duration_s=r['active_duration_s'],source_id=r['id'],source_program_sha256=r['sha256']))
    return packages


def local_schedule(poses, q, original_time):
    """Lengthen only intervals implicated by original rate/acceleration limits."""
    dt=np.diff(np.asarray(original_time)).copy();before=dt.copy()
    distance=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
    angle=so3_angles(poses[:-1,:3,:3],poses[1:,:3,:3])
    dq=np.diff(q,axis=0)
    dt=np.maximum.reduce([dt,distance/.06,angle/.3,np.max(np.abs(dq),axis=1)/.7])
    for iteration in range(16):
        velocity=dq/dt[:,None]
        acceleration=2*np.diff(velocity,axis=0)/(dt[1:]+dt[:-1])[:,None]
        ratios=np.max(np.abs(acceleration),axis=1)
        bad=np.flatnonzero(ratios>1*(1+1e-10))
        if not len(bad):break
        factors=np.ones_like(dt)
        for i in bad:factors[i:i+2]=np.maximum(factors[i:i+2],np.sqrt(ratios[i])*(1+1e-6))
        dt*=factors
    else:raise ValueError('NIGHT_LOCAL_TIMING_NOT_CONVERGED')
    dt=np.maximum(before,np.ceil(dt/.05-1e-10)*.05)
    # Quantization is independently verified; never clip an acceleration value.
    velocity=dq/dt[:,None];acceleration=2*np.diff(velocity,axis=0)/(dt[1:]+dt[:-1])[:,None]
    if np.max(np.abs(velocity))>.7*(1+1e-10) or np.max(np.abs(acceleration))>1*(1+1e-10):
        raise ValueError('NIGHT_QUANTIZED_JOINT_TIME_REJECT')
    return np.r_[0.,np.cumsum(dt)],dict(changed_intervals=int(np.count_nonzero(dt>before+1e-10)),
        original_duration_s=float(before.sum()),duration_s=float(dt.sum()),
        max_joint_velocity=float(np.max(np.abs(velocity))),max_joint_acceleration=float(np.max(np.abs(acceleration))),
        spatial_knots_changed=False,gripper_values_changed=False,uniform_global_drag=False,iterations=iteration+1)
