"""Frozen world-TCP task scoring, independently replayable without a simulator."""
import numpy as np
from scipy.spatial.transform import Rotation

POSITION = .035
ORIENTATION = .10
HOLD = .25
PATH_DISTANCE = .015


def so3(a, b):
    return np.linalg.norm(Rotation.from_matrix(np.asarray(a) @ np.swapaxes(b,-1,-2)).as_rotvec(),axis=-1)


def score(times, position, rotation, installed_p, installed_r, spec, safety=False):
    t=np.asarray(times);p=np.asarray(position);r=np.asarray(rotation)
    assert t.ndim==1 and len(t)>0 and p.shape==(len(t),3) and r.shape==(len(t),3,3)
    assert np.isfinite(p).all() and np.isfinite(r).all() and np.isfinite(t).all()
    assert np.all(np.diff(t)>0) and t[0]>=0 and t[-1]<=20+1e-6
    goal=np.asarray(spec['goal']);goal_r=np.asarray(spec['goal_rotation'])
    xyz=np.linalg.norm(p-goal,axis=-1);angle=so3(r,goal_r)
    bins=[];start_admitted=spec['task']=='M1';reverse=False;gap=False;previous=0.;progress=[];path_ready=[]
    if spec['task']=='M3':
        center=np.asarray(spec['center']);radius=spec['radius'];direction=1 if spec['command']==0 else -1
        start_angle=-direction*3*np.pi/4;entry=center+[radius*np.cos(start_angle),radius*np.sin(start_angle),0]
        for i,point in enumerate(p):
            if not start_admitted:
                if np.linalg.norm(point-entry)<=POSITION and so3(r[i],goal_r)<=ORIENTATION:start_admitted=True
                else:progress.append(None);path_ready.append(False);continue
            delta=direction*(np.arctan2(point[1]-center[1],point[0]-center[0])-start_angle)
            theta=float(delta%(2*np.pi))
            if theta>2*np.pi-1e-6:theta=0.
            on_plane=abs(point[2]-center[2])
            radial=abs(np.linalg.norm(point[:2]-center[:2])-radius)
            near=np.hypot(on_plane,radial)<=PATH_DISTANCE
            # Credit only the next ordered bin. Reverse/gap evidence is never repaired.
            if near:
                if theta>1.5*np.pi+1e-6:gap=True
                if theta<previous-1e-6:reverse=True
                b=min(11,int(theta/(1.5*np.pi)*12))
                if 0<=theta<=1.5*np.pi+1e-6 and b==len(bins):bins.append(b)
                previous=theta
            progress.append(theta)
            path_ready.append(start_admitted and len(bins)==12 and not reverse and not gap)
    good=(xyz<=POSITION)&(angle<=ORIENTATION)
    since=None;hit=None
    for i,value in enumerate(good):
        full_path=spec['task']=='M1' or path_ready[i]
        since=(t[i] if since is None else since) if value and full_path else None
        if since is not None and t[i]-since>=HOLD-1e-9:hit=float(t[i]);break
    tracking=np.linalg.norm(p-np.asarray(installed_p),axis=-1);tracking_rot=so3(r,np.asarray(installed_r))
    return dict(success=hit is not None and not safety,time_to_success_s=hit,safety_failure=bool(safety),
        final_xyz_m=float(xyz[-1]),final_so3_rad=float(angle[-1]),min_xyz_m=float(xyz.min()),
        tracking_xyz_rms_m=float(np.sqrt(np.mean(tracking**2))),tracking_xyz_p95_m=float(np.quantile(tracking,.95)),
        tracking_so3_rms_rad=float(np.sqrt(np.mean(tracking_rot**2))),start_admitted=start_admitted,
        ordered_bins=bins,reverse=reverse,gap_shortcut=gap,progress_rad=progress)


def check_clock(now, before, dt, frame_time, post_reset=False):
    if abs(now-before-dt)>1e-6 or abs(now-frame_time)>1e-6 or post_reset:
        raise ValueError('CLOCK_FRAME_OR_POST_RESET_TELEMETRY')


def self_check():
    t=np.arange(0,20.0001,.02);r=np.repeat(np.eye(3)[None],len(t),axis=0)
    goal=np.array([.3,.1,.6]);p=np.repeat(goal[None],len(t),axis=0)
    spec=dict(task='M1',goal=goal,goal_rotation=np.eye(3))
    assert score(t,p,r,p,r,spec)['success']
    assert not score(t,p+[0,.075,0],r,p,r,spec)['success']
    hit=p.copy();hit[:10]+=.1;hit[11:]+=.1
    assert not score(t,hit,r,p,r,spec)['success']
    center=np.array([.3,0,.6]);rad=.05;a=-3*np.pi/4+np.clip((t-4)/12,0,1)*1.5*np.pi
    arc=center+np.column_stack((rad*np.cos(a),rad*np.sin(a),np.zeros(len(t))))
    spec=dict(task='M3',command=0,center=center,radius=rad,goal=arc[-1],goal_rotation=np.eye(3))
    assert score(t,arc,r,arc,r,spec)['success']
    shortcut=np.repeat(arc[-1][None],len(t),axis=0);shortcut[:50]=arc[0]
    assert not score(t,shortcut,r,arc,r,spec)['success']
    assert not score(t,arc[::-1],r,arc,r,spec)['success']
    assert not score(t,arc,r,arc,r,spec,True)['success']
    for values in ((.04,0,.02,.04,False),(.02,0,.02,0,False),(.02,0,.02,.02,True)):
        try:check_clock(*values)
        except ValueError:pass
        else:raise AssertionError('invalid clock/frame accepted')
    return dict(status='PASS_FROZEN_TASK_AND_TIME_NEGATIVES',tests=['wrong_target','brief_minimum',
        'endpoint_shortcut','reverse_arc','unsafe_hit','clock_mismatch','stale_frame','post_reset_pose'])


if __name__=='__main__':
    print(self_check())
