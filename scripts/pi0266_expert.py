"""The single authorized observable correction, without elapsed time or phase memory."""
import ast
import copy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from pi0266_common import OLD
from pi0255_motion3 import scene
from pi0265_metrics import so3


text=(OLD/'code/pi0259_kinematic.py').read_text()
nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('quintic','reference')]
scope=dict(np=np,Rotation=Rotation,scene=scene)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_pi0259_reference','exec'),scope)
reference=scope['reference'];quintic=scope['quintic']


class ObservableCorrection:
    def __init__(self, reg, family, task, command):
        self.reg,self.family,self.task,self.command=copy.deepcopy(reg),family,task,command
        self.end=6. if task=='M1' else 16.
        self.grid=np.arange(round(self.end/.01)+1)*.01
        self.p,self.r=self.curve(self.grid)

    def curve(self, t):
        return reference(self.reg,self.family,self.task,self.command,np.clip(np.asarray(t),0,self.end))

    def label(self, p, r):
        p,r=np.asarray(p),np.asarray(r)
        cost=np.sum((self.p-p)**2,axis=-1)+(.05*so3(r,self.r))**2
        # Stable tie: earliest grid phase among numerical equal minima, then bounded local refinement.
        i=int(np.flatnonzero(cost<=cost.min()+1e-15)[0])
        def objective(s):
            cp,cr=self.curve([s]);return float(np.sum((cp[0]-p)**2)+(.05*so3(r,cr[0]))**2)
        lo,hi=self.grid[max(0,i-1)],self.grid[min(len(self.grid)-1,i+1)]
        fitted=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options=dict(xatol=1e-10,maxiter=100))
        choices=[(objective(s),s) for s in (float(lo),float(self.grid[i]),float(fitted.x),float(hi))]
        minimum=min(c for c,s in choices);s=min(s for c,s in choices if c<=minimum+1e-15)
        cp,cr=self.curve([s]);h=np.arange(40)*.05;future_p,future_r=self.curve(s+h)
        weight=1-quintic(h/1.)
        out_p=future_p+weight[:,None]*(p-cp[0])
        residual=Rotation.from_matrix(r@cr[0].T).as_rotvec()
        out_r=Rotation.from_rotvec(weight[:,None]*residual).as_matrix()@future_r
        assert np.linalg.norm(out_p[0]-p)<=1e-12 and so3(out_r[0],r)<=1e-10
        # Separate minima at codec resolution are disclosed; different-future aliases are excluded.
        minima=np.flatnonzero((cost<=np.r_[np.inf,cost[:-1]])&(cost<=np.r_[cost[1:],np.inf]))
        aliases=[]
        for j in minima:
            if abs(self.grid[j]-s)<=.4 or cost[j]>minimum+1e-10:continue
            ap,ar=self.curve(self.grid[j]+h)
            if np.max(np.linalg.norm(ap-future_p,axis=-1))>1e-5 or np.max(so3(ar,future_r))>1e-4:
                aliases.append(dict(phase=float(self.grid[j]),objective=float(cost[j])))
        return out_p,out_r,dict(phase_s=float(s),quartile=min(3,int(s/self.end*4)),terminal_hold=s>=self.end-.01,
            projection_objective=float(minimum),phase_aliases=aliases,valid_label=not aliases,
            label_rule='OBSERVABLE_CORRECTION_V1',time_or_record_id_used=False)


def self_check(reg, families):
    results=[]
    initial=np.array(reg['initial_TCP_world']);reg=copy.deepcopy(reg)
    initial[:3,:3]=Rotation.from_euler('xyz',[.31,-.47,.83]).as_matrix();reg['initial_TCP_world']=initial.tolist()
    for family in families:
        for task in ('M1','M3'):
            for command in (0,1):
                expert=ObservableCorrection(reg,family,task,command)
                for phase in (0.,.5,expert.end/2,expert.end-1.,expert.end):
                    cp,cr=expert.curve([phase]);p=cp[0]+[.002,-.003,.001]
                    r=Rotation.from_rotvec([.03,-.02,.01]).as_matrix()@cr[0]
                    a,b,meta=expert.label(p,r);a2,b2,meta2=expert.label(p.copy(),r.copy())
                    assert np.array_equal(a,a2) and np.array_equal(b,b2) and meta==meta2
                    assert np.linalg.norm(a[0]-p)<=1e-12 and so3(b[0],r)<=1e-10
                    target_p,target_r=expert.curve(meta['phase_s']+np.arange(40)*.05)
                    assert np.max(abs(a[20:]-target_p[20:]))<=1e-12 and np.max(so3(b[20:],target_r[20:]))<=1e-10
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    assert np.max(abs(b@b.transpose(0,2,1)-np.eye(3)))<1e-10
                    results.append(dict(family=family['family_id'],task=task,command=command,nominal_phase=phase,**meta))
    return dict(status='PASS_OBSERVABLE_CURRENT_ANCHOR_LEFT_SO3_REPEAT_IDENTITY',rows=results,
        checks=['nonidentity_noncommuting_rotation','lagging_off_curve','both_commands','entry_exit_hold','identical_input_different_wall_time'],
        new_physics=0,model_queries=0)
