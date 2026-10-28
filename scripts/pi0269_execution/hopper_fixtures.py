"""Hopper physical inverse and cross-host fixture identity, without model calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import json
import numpy as np
import torch
torch.set_num_threads(4)
from train import ROOT,read,sha
import pi0260_r2_codec as codec
from scipy.spatial.transform import Rotation

def matrices(x):
    u=x[...,:3].astype(float);u/=np.linalg.norm(u,axis=-1,keepdims=True)
    v=x[...,3:].astype(float);v-=np.sum(u*v,axis=-1,keepdims=True)*u;v/=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.stack((u,v,np.cross(u,v)),axis=-2)

def main():
    def forbidden(*a,**kw):raise RuntimeError('CPU_FIXTURE_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    rows=read(ROOT/'design/FIT112_FIRST2048.json')['rows'];checks=[]
    for arm in ('ROT',):
        stats=ROOT/'contracts'/arm/'source/dataset/statistics.json';proc=codec.processor(read(stats))
        for row in rows:
            path=ROOT/'design/fit_fixtures'/row['fixture_file'];assert sha(path)==row['fixture_sha256']
            with np.load(path) as z:s=z['state_v1'];a=z['raw_action']
            state,action,mask=codec.encode(proc,s,a);out=codec.decode(proc,action,s)
            xyz=float(np.linalg.norm(out[:,:3]-a[:,:3],axis=-1).max())
            rot=float(Rotation.from_matrix(matrices(out[:,3:9])@matrices(a[:,3:9]).swapaxes(-1,-2)).magnitude().max())
            assert xyz<1e-5 and rot<1e-4
            checks.append(dict(arm=arm,input_id=row['input_id'],xyz_m=xyz,so3_rad=rot))
    from temporal import self_check
    temporal=self_check()
    import pi0260_r3_physical as physical
    physical.install()
    from unclamped_fixture import check as decode_check
    decode_receipt=decode_check()
    target=ROOT/'evidence/HOPPER_PHYSICAL_FIXTURES.json';assert not target.exists()
    target.write_text(json.dumps(dict(status='PASS_ROT_CONTRACT_PHYSICAL_AND_WEIGHTED_FIXTURES',checks=checks,temporal=temporal,unclamped_noncommuting=decode_receipt,model_queries=0,script_sha256=sha(__file__)),indent=2)+'\n')
    print('PASS_HOPPER_PHYSICAL_FIXTURES',len(checks),flush=True)

if __name__=='__main__':main()
