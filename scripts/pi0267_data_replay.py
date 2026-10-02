"""Independent yaw-curve projection and matrix-log label replay of real TRAIN shards."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from pi0267_data_admission import BASE,ROOT,R1,RAW,read,sha,write
from pi0266_expert import reference
from pi0265_metrics import so3


def exp(v):
    theta=np.linalg.norm(v);K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if theta<1e-12:return np.eye(3)+K
    return np.eye(3)+np.sin(theta)/theta*K+(1-np.cos(theta))/theta**2*(K@K)


def log(r):
    theta=np.arccos(np.clip((np.trace(r)-1)/2,-1,1));v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return v if theta<1e-10 else v*theta/np.sin(theta)


def rz(yaw):
    yaw=np.atleast_1d(yaw);c=np.cos(yaw);s=np.sin(yaw);r=np.zeros((len(yaw),3,3));r[:,0,0]=r[:,1,1]=c
    r[:,0,1]=-s;r[:,1,0]=s;r[:,2,2]=1;return r


def main():
    card=read(ROOT/'design/SCENE_CARD.json');admission=read(R1/'design/DATA_ADMISSION.json');checks=[]
    for f in card['families']:
        if f['family_id'] not in admission['families_admitted']:continue
        folder=RAW/f['family_id'];rows=read(folder/'rows.json');z=np.load(folder/'proposals.npz');frames=read(folder/'frame_index.json')
        worst_xyz=0.;worst_so3=0.;worst_ell=0.;n=0
        for task in ('M1','M3'):
            for command in (0,1):
                end=6 if task=='M1' else 16;cp,cr=reference(card['registry'],f,task,command,np.arange(end*100+1)*.01)
                yaw=np.arctan2(cr[:,1,0],cr[:,0,0]);lengths=np.r_[0,np.cumsum(np.sqrt(np.sum(np.diff(cp,axis=0)**2,axis=1)+(.05*np.diff(yaw))**2))]
                keep=np.r_[True,np.diff(lengths)>0];cp,yaw,lengths=cp[keep],yaw[keep],lengths[keep]
                def at(ell):
                    ell=np.clip(np.atleast_1d(ell),0,lengths[-1])
                    return np.column_stack([np.interp(ell,lengths,cp[:,j]) for j in range(3)]),rz(np.interp(ell,lengths,yaw))
                for row in rows:
                    if (row['task'],row['command'])!=(task,command):continue
                    i=row['index'];p,r=z['position_world'][i],z['rotation_world'][i]
                    # Separate scalar objective using trace, rather than producer geodesic helper.
                    def cost(ell):
                        pp,rr=at(ell);angles=np.arccos(np.clip((np.trace(rr@r.T,axis1=1,axis2=2)-1)/2,-1,1))
                        return np.sum((pp-p)**2,axis=1)+(.05*angles)**2
                    gridcost=cost(lengths);minima=np.flatnonzero((gridcost<=np.r_[np.inf,gridcost[:-1]])&(gridcost<=np.r_[gridcost[1:],np.inf]))
                    candidates=[]
                    for j in minima:
                        lo,hi=lengths[max(0,j-1)],lengths[min(len(lengths)-1,j+1)]
                        fitted=minimize_scalar(lambda ell:float(cost(ell)[0]),bounds=(lo,hi),method='bounded',options=dict(xatol=1e-12))
                        candidates.extend((float(cost(s)[0]),float(s)) for s in (lo,lengths[j],fitted.x,hi))
                    best=min(c for c,s in candidates);ell=min(s for c,s in candidates if c<=best+1e-15)
                    bp,br=at(ell+.02*np.arange(40)*.05);decay=np.exp(-2*np.arange(40)*.05)
                    expected_p=bp+decay[:,None]*(p-bp[0]);residual=log(r@br[0].T)
                    expected_r=np.stack([exp(w*residual) for w in decay])@br
                    xyz=float(np.max(np.linalg.norm(expected_p-z['label_position_world'][i],axis=1)))
                    angle=float(np.max(so3(expected_r,z['label_rotation_world'][i])))
                    assert xyz<=1e-5 and angle<=1e-4,(f['family_id'],i,xyz,angle)
                    info=frames[row['frame_index']]
                    assert np.array_equal(np.array(info['TCP_position_world']),p) and np.array_equal(np.array(info['TCP_rotation_world']),r)
                    expected_state=np.r_[p-np.array(card['policy_anchor_world_m']),r[:2].reshape(6),.5].astype(np.float32)
                    assert np.array_equal(expected_state,z['state_v1'][i]) and info['timestamp']==0 and info['physics_steps']==0
                    worst_xyz=max(worst_xyz,xyz);worst_so3=max(worst_so3,angle);worst_ell=max(worst_ell,abs(ell-row['label']['ell_m']));n+=1
        assert n==448
        checks.append(dict(family=f['family_id'],proposals=n,independent_label_XYZ_max_m=worst_xyz,independent_label_SO3_max_rad=worst_so3,
            independent_projection_ell_max_m=worst_ell,raw_sha256=sha(folder/'proposals.npz'),exact_frame_state_timestamp_identity=True))
        print(f['family_id']+' REPLAY_PASS',flush=True)
    write(R1/'evidence/DATA_CPU_REPLAY.json',dict(status='PASS_10752_TRAIN_SHARDS_INDEPENDENT_PROJECTION_AND_LABEL_FORMULA',rows=checks,
        proposals=sum(c['proposals'] for c in checks),source_curve_shared=True,projection='separate trace-angle scalar objective; no ArclengthCorrection import',
        rotations='separate matrix logarithm/Rodrigues residual decay; source yaw-only, perturbed states general SO3',
        renderer='saved raster/pose correspondence; no rerendering',model_queries=0,physics_steps=0,EVAL_opened=False))


if __name__=='__main__':main()
