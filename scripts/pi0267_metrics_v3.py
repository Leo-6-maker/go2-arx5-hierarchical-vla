"""PI0267-R1 observable entry acquisition; original V2 stays immutable."""
import numpy as np
from pi0266_metrics import held, score as v2
from pi0265_metrics import so3


def entry(t, near, arc):
    start = None
    for i, okay in enumerate(near):
        start = (t[i] if start is None else start) if okay else None
        if start is not None and t[i]-start >= .10-1e-9 and abs(arc[i]) <= .001+1e-12:
            return i
    return None


def score(t, p, r, ip, ir, spec, safety=False):
    t,p,r,ip,ir = map(np.asarray, (t,p,r,ip,ir))
    assert t.ndim == 1 and len(t) and p.shape == ip.shape == (len(t),3)
    assert r.shape == ir.shape == (len(t),3,3)
    assert all(np.isfinite(x).all() for x in (t,p,r,ip,ir))
    assert np.all(np.diff(t)>0) and t[0]>=0 and t[-1]<=40+1e-6
    assert len(t)==1 or np.diff(t).max()<=.0201, 'INPUT_INTEGRITY_SAMPLED_GAP'
    old=v2(t,p,r,ip,ir,spec,safety)
    if spec['task']=='M1':
        return dict(old,schema='PI0267_PATH_V3_OBSERVABLE_ENTRY',legacy_v2=old)
    assert spec['task']=='M3'
    xyz=np.linalg.norm(p-spec['goal'],axis=-1);rot=so3(r,spec['goal_rotation'])
    center=np.asarray(spec['center']);radius=spec['radius'];direction=1 if spec['command']==0 else -1
    entry_angle=-direction*3*np.pi/4
    target=center+[radius*np.cos(entry_angle),radius*np.sin(entry_angle),0]
    signed=direction*np.arctan2(p[:,1]-center[1],p[:,0]-center[0])
    delta=np.arctan2(np.sin(signed-direction*entry_angle),np.cos(signed-direction*entry_angle))
    arc=delta*radius
    near=(np.linalg.norm(p-target,axis=-1)<=.015)&(so3(r,spec['goal_rotation'])<=.10)&(abs(arc)<=.005)
    ready=held(t,near,.10);entry_i=entry(t,near,arc)
    bins=[];reverse=False;gap=False;finish_i=None
    progress=np.full(len(t),np.nan);path_ready=np.zeros(len(t),bool)
    negative=0.;drawdown=0.;path_p95=None;path_max=None
    if entry_i is not None:
        # Physical entry is zero. Preserve acquired delta; no rearming or failure reset.
        progress[entry_i:]=delta[entry_i]+np.unwrap(signed[entry_i:])-signed[entry_i]
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
    active=np.zeros(end,bool)
    if end>1:active[1:]=(np.linalg.norm(np.diff(ip[:end],axis=0),axis=-1)>1e-7)|(so3(ir[1:end],ir[:end-1])>1e-6)
    def summary(mask):
        return None if not mask.any() else dict(rows=int(mask.sum()),xyz_rms_m=float(np.sqrt(np.mean(error[mask]**2))),
            xyz_p95_m=float(np.quantile(error[mask],.95)),so3_rms_rad=float(np.sqrt(np.mean(angle[mask]**2))))
    before=len(t) if entry_i is None else entry_i+1
    return dict(schema='PI0267_PATH_V3_OBSERVABLE_ENTRY',success=hit is not None and not safety,
        reason='SAFETY' if safety else 'MISSING_ENTRY' if entry_i is None else None if hit is not None else 'TASK_CRITERIA_FAIL',
        time_to_success_s=None if hit is None else float(t[hit]),safety_failure=bool(safety),
        earliest_broad_readiness_s=None if ready is None else float(t[ready]),
        broad_readiness_arc_m=None if ready is None else float(arc[ready]),
        entry_time_s=None if entry_i is None else float(t[entry_i]),entry_index=entry_i,
        entry_arc_m=None if entry_i is None else float(arc[entry_i]),
        pre_acquisition_motion=dict(rows=before,path_length_m=float(np.linalg.norm(np.diff(p[:before],axis=0),axis=-1).sum()),
            displacement_m=float(np.linalg.norm(p[before-1]-p[0]))),
        ordered_bins=bins,reverse=bool(reverse),gap_shortcut=bool(gap),total_negative_variation_m=float(negative),
        max_backward_excursion_m=float(drawdown),path_distance_p95_m=path_p95,path_distance_max_m=path_max,
        progress_rad=[None if np.isnan(x) else float(x) for x in progress],final_xyz_m=float(xyz[-1]),final_so3_rad=float(rot[-1]),
        qualification_tracking=summary(np.ones(end,bool)),movement_tracking=summary(active),terminal_hold_tracking=summary(~active),
        legacy_v2=old)
