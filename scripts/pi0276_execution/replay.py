"""Reconstruct object predicates from recorded physics, with no simulator/model calls."""
import json
from pathlib import Path
import numpy as np
from run import ROOT, sha, write

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')

def reconstruct(z):
    d=z['cubeA_pose'][:,:3]-z['cubeB_pose'][:,:3]
    on=(np.linalg.norm(d[:,:2],axis=1)<=np.linalg.norm(np.array([.02,.02],np.float32))+.005)&(np.abs(d[:,2]-.04)<=.005)
    static=(np.linalg.norm(z['cubeA_velocity'],axis=1)<=.01)&(np.linalg.norm(z['cubeA_angular'],axis=1)<=.5)
    grasp=np.ones(len(d),bool)
    for side in ('left','right'):
        f=z[side+'_force'];v=z[side+'_direction'];n=np.linalg.norm(f,axis=1)
        cosine=np.sum(f*v,axis=1)/np.maximum(n*np.linalg.norm(v,axis=1),1e-12)
        grasp &= (n>=.5)&(np.degrees(np.arccos(np.clip(cosine,-1,1)))<=85)
    return dict(is_cubeA_on_cubeB=on,is_cubeA_static=static,is_cubeA_grasped=grasp,success=on&static&~grasp)

def main():
    rows=[];mismatches=[]
    for p in sorted((RAW/'episodes').glob('*/attempt-*/RESULT.json')):
        record=json.loads(p.read_text())
        row=dict(slot=p.parent.parent.name,attempt=p.parent.name,status=record['status'],result_sha256=sha(p))
        for name,digest in record.get('files',{}).items():assert sha(p.parent/name)==digest
        if (p.parent/'physics.npz').exists():
            with np.load(p.parent/'physics.npz',allow_pickle=False) as z:
                actual=reconstruct(z);bad={k:int(np.count_nonzero(a!=z[k])) for k,a in actual.items()}
                success=actual['success'][::5];indices=np.flatnonzero(success[:601])
                row.update(samples=len(actual['success']),predicate_discrepancies=bad,
                    first_success_s=float(indices[0]*.05) if len(indices) else None,
                    persistent_success=bool(len(success)>=601 and success[560:601].all()))
                assert row['first_success_s']==record['first_success_s'] and row['persistent_success']==record['persistent_success']
                if any(bad.values()):mismatches.append(row)
                if (p.parent/'observations.npz').exists():
                    with np.load(p.parent/'observations.npz') as obs:
                        count=len(obs['state_v1']);tcp=z['tcp'][::5][:count];q=z['qpos'][::5][:count]
                        expected=np.c_[tcp[:,:3,3],tcp[:,:2,:3].reshape(count,6),(q[:,-2:].sum(1)-.00004)/.08].astype(np.float32)
                        assert np.max(np.abs(expected-obs['state_v1']))<=1e-6
                        assert np.max(np.abs(obs['time_s']-np.arange(count)*.05))<=1e-9
        rows.append(row)
    write(ROOT/'evidence/PHYSICS_CPU_REPLAY.json',dict(status='PASS_SAVED_PHYSICAL_PREDICATES' if not mismatches else 'PREDICATE_CONTRADICTION',
        rows=rows,mismatches=mismatches,shared_math='Pinned native geometry/static/grasp definitions; independent NumPy aggregation',
        fresh_physics_steps=0,model_queries=0))
    assert not mismatches
    print('PASS_SAVED_PHYSICAL_PREDICATES',len(rows),flush=True)

def check():
    z=dict(cubeA_pose=np.array([[0,0,.06,1,0,0,0]],np.float32),cubeB_pose=np.array([[0,0,.02,1,0,0,0]],np.float32),
        cubeA_velocity=np.zeros((1,3)),cubeA_angular=np.zeros((1,3)))
    for side in ('left','right'):
        z[side+'_force']=np.zeros((1,3));z[side+'_direction']=np.array([[0,1,0]])
    assert reconstruct(z)['success'][0]
    z['left_force'][:]=z['right_force'][:]=[0,1,0]
    assert not reconstruct(z)['success'][0]

if __name__=='__main__':check();main()
