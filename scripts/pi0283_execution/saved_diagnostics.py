"""CPU-only prefix/tail/event details from the fixed saved DIAG48 outputs."""
import argparse,json,os
from pathlib import Path
import numpy as np
from closed_loop_audit import ROOT,sha
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,geodesic_angle

def run(point):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=ROOT/'source/diagnostics'/point
    original=json.loads((root/'CPU_RECOMPUTE.json').read_text());plan=[json.loads(x) for x in (ROOT/'data/DIAG48.jsonl').read_text().splitlines()];rows=[]
    for previous,p in zip(original['rows'],plan):
        assert previous['uid']==p['uid']
        folder=root/f"{previous['index']:02d}"
        with np.load(folder/'output.npz') as f:pred=f['decoded'].copy();raw=f['raw'][0,:,:10].copy()
        with np.load(folder/'target.npz') as f:target=f['commands'].copy()
        assert pred.shape==target.shape==(40,10)
        pieces={}
        for label,a,b in [('prefix8',0,8),('tail32',8,40),('fullH40',0,40)]:
            x,y=pred[a:b],target[a:b];v={}
            try:
                assert np.isfinite(x).all(),'NONFINITE_DECODE'
                v=dict(XYZ_RMS_m=float(np.sqrt(np.mean(np.sum((x[:,:3]-y[:,:3])**2,1)))),SO3_RMS_rad=float(np.sqrt(np.mean(geodesic_angle(rot6d_rows_to_matrix(x[:,3:9]),rot6d_rows_to_matrix(y[:,3:9]))**2))),gripper_MAE=float(np.mean(abs(x[:,-1]-y[:,-1]))))
            except (ValueError,AssertionError) as e:v=dict(error=str(e))
            v['raw_OOB_by_group']={k:dict(outside=int((abs(raw[a:b,c:d])>1).sum()),denominator=(b-a)*(d-c)) for k,c,d in [('XYZ',0,3),('rotation',3,9),('gripper',9,10)]}
            pieces[label]=v
        if previous['error']=='':
            for k in ('XYZ_RMS_m','SO3_RMS_rad','gripper_MAE'):assert np.isclose(pieces['prefix8'][k],previous[k],rtol=1e-10,atol=1e-12)
        command=np.load(ROOT/'data/episodes'/f"{p['split']}-{p['seed']}"/'action.npy',mmap_mode='r')[:,9]
        a=p['anchor'];prev=command[a-1] if a else command[0];delta=np.diff(target[:,9],prepend=prev)
        events={name:np.flatnonzero(delta*sign>.1).tolist() for name,sign in [('open',1),('close',-1)]}
        rows.append(dict(index=previous['index'],uid=p['uid'],split=p['split'],phase=p['phase'],anchor=a,first8_command_events={k:[i for i in v if i<8] for k,v in events.items()},H40_command_events=events,metrics=pieces,fit_pass=previous['fit_pass']))
    assert len(rows)==48
    report=dict(status='PASS_SAVED_DIAG48_FULL_H40_EVENT_CPU_DETAILS',point=point,rows=rows,original_prefix_receipt_sha256=sha(root/'CPU_RECOMPUTE.json'),full_H40_is_descriptive_not_qualification=True,new_model_queries=0,new_physics=0)
    (root/'H40_EVENT_CPU.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],point=point,rows=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');run(p.parse_args().point)
