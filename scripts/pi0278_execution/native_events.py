"""Describe saved native outcomes and reference/install/achieved errors; no new calls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from evaluation_card import ROOT,RAW,sha,freeze

def spans(values,times):
    indices=np.flatnonzero(values)
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if indices.size else []
    return [[float(times[g[0]]),float(times[g[-1]])] for g in groups]

def report(point,lane):
    rows=[]
    for result in sorted((RAW/'episodes').glob(f'{lane}-*-{point}/attempt-*/RESULT.json')):
        v=json.loads(result.read_text())
        if not v['status'].startswith('VALID'):continue
        folder=result.parent
        for name in ('physics.npz','commands.npz','installed.npz'):assert sha(folder/name)==v['files'][name]
        with np.load(folder/'physics.npz') as z:
            times=z['time_s'];tcp=z['tcp'];events={k:spans(z[k],times) for k in ('success','is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static')}
        with np.load(folder/'commands.npz') as z:command=z['action'];hold=z['ik_hold']
        with np.load(folder/'installed.npz') as z:installed=z['installed_target_fk']
        assert len(command)==len(installed)==v['control_calls']
        # Existing source contract: one command per five .01s native steps.
        endpoints=np.arange(1,len(command)+1,dtype=int)*5
        assert not len(endpoints) or (len(times)>endpoints[-1] and np.allclose(times[endpoints],np.arange(1,len(command)+1)*.05,atol=1e-8))
        metrics={
            'command_to_installed_FK_XYZ_m':np.linalg.norm(command[:,:3]-installed[:,:3,3],axis=1),
            'installed_FK_to_achieved_at_control_end_XYZ_m':np.linalg.norm(installed[:,:3,3]-tcp[endpoints,:3,3],axis=1),
            'command_to_achieved_at_control_end_XYZ_m':np.linalg.norm(command[:,:3]-tcp[endpoints,:3,3],axis=1)} if len(command) else {k:np.empty(0) for k in ('command_to_installed_FK_XYZ_m','installed_FK_to_achieved_at_control_end_XYZ_m','command_to_achieved_at_control_end_XYZ_m')}
        rows.append(dict(episode=folder.parent.name,status=v['status'],persistent_success=v['persistent_success'],first_success_s=v['first_success_s'],events_at_native_100Hz=events,IK_hold_control_times_s=(np.flatnonzero(hold)*.05).tolist(),
            errors={k:dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None for k,a in metrics.items()},result_sha256=sha(result)))
    assert rows
    value=dict(point=point,lane=lane,rows=rows,gate='UNCHANGED_ORIGINAL_NATIVE_CONTROL_SAMPLE_GATE',limits=['Intervals are descriptive native100Hz observations, not a new success gate','Control-end errors include one .05s physical response interval','No causal attribution from these unmatched checkpoint comparisons'],new_model_queries=0,new_physics_steps=0)
    freeze(ROOT/'evidence'/f'NATIVE_EVENTS_{point}_{lane}.json',value);print(json.dumps(value),flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--check']:
        assert spans(np.array([False,True,True,False,True]),np.arange(5))==[[1.,2.],[4.,4.]]
        assert spans(np.zeros(1,dtype=bool),np.zeros(1))==[]
        assert np.arange(1,1,dtype=int).size==0
        print('PASS_EXACT_INTERVALS_AND_ZERO_EXECUTION_BOUNDARY');sys.exit()
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=['TRAIN_PHYSICS','TEST','CONFIRM']);a=p.parse_args();report(a.point,a.lane)
