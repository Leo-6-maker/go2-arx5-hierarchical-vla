"""Prospectively fixed physical path V2; historical PI0265 score is untouched."""
import numpy as np
from pi0265_metrics import so3, score as v1


def held(t, good, duration):
    start = None
    for i, okay in enumerate(good):
        start = (t[i] if start is None else start) if okay else None
        if start is not None and t[i]-start >= duration-1e-9:
            return i
    return None


def score(t, p, r, ip, ir, spec, safety=False):
    t,p,r,ip,ir = map(np.asarray, (t,p,r,ip,ir))
    assert t.ndim == 1 and len(t) and p.shape == ip.shape == (len(t),3)
    assert r.shape == ir.shape == (len(t),3,3)
    assert all(np.isfinite(x).all() for x in (t,p,r,ip,ir))
    assert np.all(np.diff(t)>0) and t[0]>=0 and t[-1]<=40+1e-6
    xyz=np.linalg.norm(p-spec['goal'],axis=-1);rot=so3(r,spec['goal_rotation'])
    bins=[];reverse=False;gap=False;entry_i=None;finish_i=None
    progress=np.full(len(t),np.nan);path_ready=np.ones(len(t),bool)
    negative=0.;drawdown=0.;path_p95=None;path_max=None
    if spec['task']=='M3':
        center=np.asarray(spec['center']);radius=spec['radius'];direction=1 if spec['command']==0 else -1
        entry_angle=-direction*3*np.pi/4
        entry=center+[radius*np.cos(entry_angle),radius*np.sin(entry_angle),0]
        signed=direction*np.arctan2(p[:,1]-center[1],p[:,0]-center[0])
        entry_delta=np.arctan2(np.sin(signed-direction*entry_angle),np.cos(signed-direction*entry_angle))
        good=(np.linalg.norm(p-entry,axis=-1)<=.015)&(so3(r,spec['goal_rotation'])<=.10)&(abs(entry_delta)*radius<=.005)
        entry_i=held(t,good,.10);path_ready[:]=False
        if entry_i is not None:
            # Unwrap the actual directed angle; keep physical entry as zero, not the sampled pose.
            progress[entry_i:]=entry_delta[entry_i]+np.unwrap(signed[entry_i:])-signed[entry_i]
            arc_distance=np.hypot(np.linalg.norm(p[:,:2]-center[:2],axis=-1)-radius,p[:,2]-center[2])
            greatest=max(0.,progress[entry_i]);previous=progress[entry_i];dwell=None
            for i in range(entry_i,len(t)):
                theta=progress[i]
                negative+=max(0.,previous-theta)*radius
                greatest=max(greatest,theta);drawdown=max(drawdown,(greatest-theta)*radius)
                reverse |= (greatest-theta)*radius>.001+1e-12
                gap |= theta*radius<-.001-1e-12 or (theta-1.5*np.pi)*radius>.001+1e-12
                previous=theta
                if len(bins)<12:
                    # Deadband affects small boundary signs; it cannot invent samples in a skipped bin.
                    b=min(11,max(0,int(max(theta,0)/(1.5*np.pi)*12)))
                    okay=arc_distance[i]<=.015 and b==len(bins) and not gap
                    dwell=(t[i] if dwell is None else dwell) if okay else None
                    if dwell is not None and t[i]-dwell>=.04-1e-9:
                        bins.append(b);dwell=None
                        if len(bins)==12: finish_i=i
                if finish_i is not None:
                    traversed=arc_distance[entry_i:i+1]
                    path_p95=float(np.quantile(traversed,.95));path_max=float(traversed.max())
                    path_ready[i]=not reverse and not gap and path_p95<=.015 and path_max<=.035
    hit=held(t,(xyz<=.035)&(rot<=.10)&path_ready,.25)
    end=hit+1 if hit is not None else len(t)
    error=np.linalg.norm(p[:end]-ip[:end],axis=-1);angle=so3(r[:end],ir[:end])
    # Movement and terminal hold are diagnostic partitions, never denominators substituted into qualification.
    active=np.zeros(end,bool)
    if end>1: active[1:]=(np.linalg.norm(np.diff(ip[:end],axis=0),axis=-1)>1e-7)|(so3(ir[1:end],ir[:end-1])>1e-6)
    def summary(mask):
        return None if not mask.any() else dict(rows=int(mask.sum()),xyz_rms_m=float(np.sqrt(np.mean(error[mask]**2))),
            xyz_p95_m=float(np.quantile(error[mask],.95)),so3_rms_rad=float(np.sqrt(np.mean(angle[mask]**2))))
    prefix=t<=20+1e-6
    old=v1(t[prefix],p[prefix],r[prefix],ip[prefix],ir[prefix],spec,safety)
    return dict(schema='PI0266_PATH_V2',success=hit is not None and not safety,time_to_success_s=None if hit is None else float(t[hit]),
        safety_failure=bool(safety),entry_time_s=None if entry_i is None else float(t[entry_i]),ordered_bins=bins,
        reverse=bool(reverse),gap_shortcut=bool(gap),total_negative_variation_m=float(negative),max_backward_excursion_m=float(drawdown),
        path_distance_p95_m=path_p95,path_distance_max_m=path_max,progress_rad=[None if np.isnan(x) else float(x) for x in progress],
        final_xyz_m=float(xyz[-1]),final_so3_rad=float(rot[-1]),qualification_tracking=summary(np.ones(end,bool)),
        movement_tracking=summary(active),terminal_hold_tracking=summary(~active),legacy_v1_first20s=old)


def self_check():
    checks=[];t=np.arange(0,20.0001,.02);r=np.repeat(np.eye(3)[None],len(t),axis=0);center=np.array([.3,0,.6]);radius=.05
    for command in (0,1):
        direction=1 if command==0 else -1;a=-direction*3*np.pi/4
        theta=np.clip((t-1)/16,0,1)*1.5*np.pi
        def path(th):return center+np.column_stack((radius*np.cos(a+direction*th),radius*np.sin(a+direction*th),np.zeros(len(t))))
        p=path(theta);spec=dict(task='M3',command=command,center=center,radius=radius,goal=p[-1],goal_rotation=np.eye(3))
        assert score(t,p,r,p,r,spec)['success'];checks.append('noiseless_direction%d'%command)
        q=theta.copy();q[0]=-1e-4
        assert score(t,path(q),r,p,r,spec)['success'];checks.append('five_micrometre_entry%d'%command)
        q=theta+np.where(np.arange(len(t))%2, .00025/radius,-.00025/radius)
        q[t<1]=0;q[t>=17]=1.5*np.pi
        assert score(t,path(q),r,p,r,spec)['success'];checks.append('half_mm_jitter%d'%command)
        q=theta.copy();q[400:430]-=.005/radius
        assert not score(t,path(q),r,p,r,spec)['success'];checks.append('five_mm_reverse%d'%command)
        width=1.5*np.pi/12;q=theta.copy();q[(q>=4*width)&(q<5*width)]=5.1*width
        assert not score(t,path(q),r,p,r,spec)['success'];checks.append('missing_bin%d'%command)
        q=np.where(t<1,0.,1.5*np.pi)
        assert not score(t,path(q),r,p,r,spec)['success'];checks.append('endpoint_shortcut%d'%command)
        assert not score(t,p,r,p,r,spec,True)['success'];checks.append('unsafe_hit%d'%command)
        assert not score(t,p,r,p,r,dict(spec,goal=center+[.2,.2,.2]))['success'];checks.append('wrong_target%d'%command)
    try: score(t+41,p,r,p,r,spec)
    except AssertionError:checks.append('false_post_reset_time')
    else:raise AssertionError('false timestamp accepted')
    return dict(status='PASS_PROSPECTIVE_PATH_V2_POSITIVES_AND_NEGATIVES',checks=checks)


if __name__=='__main__':
    import json
    print(json.dumps(self_check()))
