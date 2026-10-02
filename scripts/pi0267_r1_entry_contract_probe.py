"""PI0267-R1 constructed geometry tests, NOT archived-array or WBC replay.

V2 trace logic is transcribed from scripts/pi0266_metrics.py at
99f9c1c8e37743e6d82fb80c50a253700373fa31. V3 changes ONLY entry acquisition:
0.10 s continuous broad proximity, then measured angular capture in the
existing +/-1 mm path deadband. No source phase/reference is used.
This script is a test specification, not an installed production scorer.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EPS = 0.001

def held(t, good, duration):
    start = None
    for i, okay in enumerate(good):
        start = (t[i] if start is None else start) if okay else None
        if start is not None and t[i] - start >= duration - 1e-9:
            return i
    return None

def acquire_v3(t, near, tangential):
    start = None
    for i, okay in enumerate(near):
        start = (t[i] if start is None else start) if okay else None
        if (start is not None and t[i] - start >= .10 - 1e-9
                and abs(tangential[i]) <= EPS + 1e-12):
            return i
    return None

def acquire_independent(t, near, tangential):
    indices = np.arange(len(t))
    run_starts = np.maximum.accumulate(np.where(~near, indices, -1)) + 1
    elapsed = t - t[np.minimum(run_starts, len(t)-1)]
    candidates = np.flatnonzero(near & (elapsed >= .10 - 1e-9)
                               & (np.abs(tangential) <= EPS + 1e-12))
    return None if len(candidates) == 0 else int(candidates[0])

def score(t, p, radius, command, version, *, safety=False, wrong_goal=False):
    center = np.array([.30, .00, .60]); direction = 1 if command == 0 else -1
    a = -direction * 3 * np.pi / 4
    entry = center + [radius*np.cos(a), radius*np.sin(a), 0.]
    end_angle = a + direction * 1.5*np.pi
    goal = center + [radius*np.cos(end_angle), radius*np.sin(end_angle), 0.]
    if wrong_goal: goal = goal + [.10, .10, .10]
    signed = direction * np.arctan2(p[:,1]-center[1], p[:,0]-center[0])
    delta = np.arctan2(np.sin(signed-direction*a), np.cos(signed-direction*a))
    # All constructed poses have correct orientation; production keeps SO(3) <= .10.
    near = (np.linalg.norm(p-entry, axis=1) <= .015) & (abs(delta)*radius <= .005)
    ready = held(t, near, .10)
    entry_i = ready if version == 2 else acquire_v3(t, near, delta*radius)
    assert acquire_v3(t, near, delta*radius) == acquire_independent(t, near, delta*radius)
    bins=[]; gap=False; reverse=False; drawdown=0.; finish=None
    path_ready=np.zeros(len(t), dtype=bool)
    if entry_i is not None:
        progress=delta[entry_i]+np.unwrap(signed[entry_i:])-signed[entry_i]
        dist=np.hypot(np.linalg.norm(p[:,:2]-center[:2],axis=1)-radius,p[:,2]-center[2])
        greatest=max(0., progress[0]); dwell=None
        for off, theta in enumerate(progress):
            i=entry_i+off; greatest=max(greatest,theta)
            drawdown=max(drawdown,(greatest-theta)*radius)
            reverse |= (greatest-theta)*radius > EPS+1e-12
            gap |= theta*radius < -EPS-1e-12 or (theta-1.5*np.pi)*radius > EPS+1e-12
            if len(bins)<12:
                b=min(11,max(0,int(max(theta,0)/(1.5*np.pi)*12)))
                okay=dist[i]<=.015 and b==len(bins) and not gap
                dwell=(t[i] if dwell is None else dwell) if okay else None
                if dwell is not None and t[i]-dwell>=.04-1e-9:
                    bins.append(b);dwell=None
                    if len(bins)==12:finish=i
            if finish is not None:
                ds=dist[entry_i:i+1]
                path_ready[i]=not reverse and not gap and np.quantile(ds,.95)<=.015 and ds.max()<=.035
    hit=held(t, (np.linalg.norm(p-goal,axis=1)<=.035) & path_ready, .25)
    return dict(success=bool(hit is not None and not safety), bins=len(bins),
                ready_s=None if ready is None else float(t[ready]),
                entry_s=None if entry_i is None else float(t[entry_i]),
                entry_arc_m=None if entry_i is None else float(delta[entry_i]*radius),
                reverse=bool(reverse), gap=bool(gap), max_drawdown_m=float(drawdown))

def path(t, radius, command, *, side=0, kind='valid'):
    center=np.array([.30, .00, .60]); direction=1 if command==0 else -1
    a=-direction*3*np.pi/4; radial=np.array([np.cos(a),np.sin(a),0.])
    tangent=direction*np.array([-np.sin(a),np.cos(a),0.]); entry=center+radius*radial
    theta=np.clip((t-4.20)/12.,0.,1.)*1.5*np.pi
    if kind=='reverse':theta[(t>=8.)&(t<8.30)]-=.005/radius
    if kind=='wrong_direction':theta=-theta
    if kind=='missing_bin':
        width=1.5*np.pi/12;theta[(theta>=4*width)&(theta<5*width)]=5.1*width
    if kind=='shortcut':theta=np.where(t<4.20,0.,1.5*np.pi)
    if kind=='missing_entry':theta=np.maximum(theta,.012/radius)
    if kind=='jitter':
        theta=theta+np.where(np.arange(len(t))%2,.00025/radius,-.00025/radius)
        theta[t<4.20]=0.;theta[t>=16.20]=1.5*np.pi
    p=center+np.column_stack((radius*np.cos(a+direction*theta),radius*np.sin(a+direction*theta),np.zeros(len(t))))
    if side:
        p0=entry+.070*radial+side*.030*tangent+[0.,0.,.020]
        u=np.clip(t/4.,0,1);q=10*u**3-15*u**4+6*u**5
        mask=t<4.;p[mask]=p0+(entry-p0)*q[mask,None]
    if kind=='late_reverse_recover':
        # Capture occurs before .1 s in the static start fixture. The later
        # 3 mm negative excursion must remain a failure even after recovery.
        th=theta.copy();th[(t>=1.)&(t<1.3)]=-.003/radius
        p=center+np.column_stack((radius*np.cos(a+direction*th),radius*np.sin(a+direction*th),np.zeros(len(t))))
    return p

def run():
    t=np.arange(0,20.0001,.02);rows=[]
    for command in (0,1):
        for radius in (.04,.05):
            for side in (-1,1):
                p=path(t,radius,command,side=side)
                v2=score(t,p,radius,command,2);v3=score(t,p,radius,command,3)
                assert v3['success'],(command,radius,side,v3)
                rows.append(dict(case=f'approach_c{command}_r{radius}_side{side}',expected_v3=True,v2=v2,v3=v3))
        for kind in ('valid','jitter','reverse','wrong_direction','missing_bin','shortcut','missing_entry','late_reverse_recover','unsafe','wrong_goal'):
            actual='valid' if kind in ('unsafe','wrong_goal') else kind
            p=path(t,.05,command,kind=actual)
            kw=dict(safety=kind=='unsafe',wrong_goal=kind=='wrong_goal')
            v2=score(t,p,.05,command,2,**kw);v3=score(t,p,.05,command,3,**kw)
            expected=kind in ('valid','jitter')
            assert v3['success']==expected,(kind,command,v3)
            rows.append(dict(case=f'{kind}_c{command}',expected_v3=expected,v2=v2,v3=v3))
    return dict(status='CONSTRUCTED_FIXTURES_PASS',source='pi0266_metrics.py@99f9c1c8e37743e6d82fb80c50a253700373fa31',
        evidence_scope='Constructed paths only. Not original 64-array replay, not teacher integration, not WBC or GR00T.',
        design='V3 separates broad entry readiness from narrow physical acquisition; no direction threshold increase.',
        fixture_count=len(rows),expected_positive=sum(x['expected_v3'] for x in rows),
        expected_negative=sum(not x['expected_v3'] for x in rows),
        approach_old_fails=sum(not x['v2']['success'] for x in rows if x['case'].startswith('approach')),
        approach_new_passes=sum(x['v3']['success'] for x in rows if x['case'].startswith('approach')),
        independent_entry_selector_agrees=True,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=run();result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
