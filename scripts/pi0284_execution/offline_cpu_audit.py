"""Independent saved-array recomputation; never loads or queries a neural model."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_core import sha,write

def rotation(x):
    a=x[:,3:6].astype(float);a/=np.linalg.norm(a,axis=1)[:,None]
    b=x[:,6:9].astype(float);b-=a*(a*b).sum(1)[:,None];b/=np.linalg.norm(b,axis=1)[:,None]
    return np.stack((a,b,np.cross(a,b)),axis=1)

def metrics(x,y):
    assert np.isfinite(x).all()
    angles=Rotation.from_matrix(rotation(x)@rotation(y).transpose(0,2,1)).magnitude()
    return dict(XYZ_RMS=float(np.sqrt(np.mean(((x[:,:3]-y[:,:3])**2).sum(1)))),SO3_RMS=float(np.sqrt(np.mean(angles**2))),gripper_MAE=float(np.mean(abs(x[:,9]-y[:,9]))))

def run(root,point):
    root=Path(root);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    source=root/'endpoint_diagnostics'/point/'RESULT.json';saved=json.loads(source.read_text());rows=[]
    assert saved['rows']==96 and len(saved['records'])==96
    plan=json.loads((root/'design/PRE_SCORE_INPUTS.json').read_text())['rows']
    for row in saved['records']:
        assert sha(row['path'])==row['sha256'] and sha(row['target'])==row['target_sha256']
        with np.load(row['path']) as f:x=f['decoded'].copy();raw=f['raw'][0,:,:10].copy();state=f['actual_state'].copy();noise=f['noise_uint16'].copy()
        with np.load(row['target']) as f:y=f['commands'].copy()
        assert np.array_equal(noise,np.load(Path(plan[row['index']]['input']).parent/f"noise-{row['packet']}.npy"))
        if row['error']:
            rows.append(dict(row,recomputed='RETAINED_INVALID_OUTPUT'));continue
        prefix,full=metrics(x[:8],y[:8]),metrics(x,y)
        for actual,expected in ((prefix,row['prefix']),(full,row['fullH40'])):
            for key in actual:assert np.isclose(actual[key],expected[key],rtol=1e-5,atol=1e-7),(point,row['index'],key)
        from score import gate
        assert gate(x,state)==row['gate']
        assert ((raw[:8]<-1)|(raw[:8]>1)).sum(0).tolist()==row['raw_OOB_prefix']
        assert ((raw<-1)|(raw>1)).sum(0).tolist()==row['raw_OOB_H40']
        fit=prefix['XYZ_RMS']<=.01 and prefix['SO3_RMS']<=.1 and prefix['gripper_MAE']<=.1
        rows.append(dict(row,prefix=prefix,fullH40=full,joint_fit=fit,recomputed='PASS'))
    groups=[]
    for split in ('TRAIN','VAL'):
        for packet in (0,1):
            group=[r for r in rows if r['split']==split and r['packet']==packet]
            groups.append(dict(split=split,packet=packet,total=len(group),fit=sum(r.get('joint_fit',False) for r in group),REJECT=sum(r['gate']['decision']=='REJECT' for r in group),invalid=sum(bool(r['error']) for r in group),phases={str(p):dict(total=sum(r['phase']==p for r in group),fit=sum(r.get('joint_fit',False) and r['phase']==p for r in group)) for p in range(4)}))
    result=dict(status='PASS_INDEPENDENT_CPU_FIXED96_NOISE_METRICS_GATE_AND_RAW_OOB',point=point,source_sha256=sha(source),groups=groups,rows=rows,new_model_queries=0,new_physics=0,selected='NONE',deployment='NONE')
    write(root/'audit'/f'{point}_DIAG_CPU.json',result);print(json.dumps(dict(status=result['status'],point=point,groups=groups)))

def self_check():
    x=np.tile([0.,0.,0.,1.,0.,0.,0.,1.,0.,0.],(40,1));y=x.copy();y[:,0]=.01;y[:,9]=.2
    v=metrics(x,y);assert np.isclose(v['XYZ_RMS'],.01) and v['SO3_RMS']==0 and np.isclose(v['gripper_MAE'],.2)

if __name__=='__main__':
    self_check();p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',required=True);a=p.parse_args();run(a.root,a.point)
