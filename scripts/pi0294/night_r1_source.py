"""Source spatial knots are immutable; only bounded local timing is revised."""
from pathlib import Path
import numpy as np
from .night_r1_plan_runtime import FOLDER,read,sha
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
    # Quantizing one interval can expose a neighboring acceleration. Iterate on
    # that same quantized schedule; no new path, goal or library call is made.
    for quantized_iteration in range(24):
        dt=np.maximum(before,np.ceil(dt/.05-1e-10)*.05)
        velocity=dq/dt[:,None];acceleration=2*np.diff(velocity,axis=0)/(dt[1:]+dt[:-1])[:,None]
        ratios=np.max(np.abs(acceleration),axis=1);bad=np.flatnonzero(ratios>1*(1+1e-10))
        if not len(bad):break
        factors=np.ones_like(dt)
        for i in bad:factors[i:i+2]=np.maximum(factors[i:i+2],np.sqrt(ratios[i])*(1+1e-6))
        dt*=factors
    else:raise ValueError('NIGHT_QUANTIZED_JOINT_TIME_REJECT')
    changed=dt>before+1e-10;attempts=[];selected=None
    for factor in (1.,1.25,1.5,2.):
        candidate=np.ceil(np.where(changed,dt*factor,before)/.05-1e-10)*.05
        knots=np.r_[0.,np.cumsum(candidate)];grid=np.arange(round(knots[-1]/.05)+1)*.05
        witness=np.stack([np.interp(grid,knots,q[:,j]) for j in range(q.shape[1])],axis=1)
        vel=np.diff(witness,axis=0)/.05;acc=np.diff(vel,axis=0)/.05
        proof=dict(factor=factor,duration_s=float(knots[-1]),max_joint_velocity=float(np.abs(vel).max()),max_joint_acceleration=float(np.abs(acc).max()),
            status='PASS_SAVED_PREFIX_WITNESS_ONLY' if np.abs(vel).max()<=.7*(1+1e-10) and np.abs(acc).max()<=1*(1+1e-10) else 'QUANTIZED_RESAMPLE_RATE_REJECT')
        attempts.append(proof)
        if selected is None and proof['status']=='PASS_SAVED_PREFIX_WITNESS_ONLY':selected=(knots,proof)
    if selected is None:raise ValueError('R1_FOUR_FIXED_LOCAL_TIMING_CANDIDATES_REJECT:'+str(attempts))
    knots,chosen=selected
    return knots,dict(changed_intervals=int(changed.sum()),original_duration_s=float(before.sum()),duration_s=float(knots[-1]),
        max_joint_velocity=chosen['max_joint_velocity'],max_joint_acceleration=chosen['max_joint_acceleration'],
        fixed_factor_attempts=attempts,selected_factor=chosen['factor'],spatial_knots_changed=False,gripper_values_changed=False,
        uniform_global_drag=False,iterations=iteration+1,quantized_iterations=quantized_iteration+1)
