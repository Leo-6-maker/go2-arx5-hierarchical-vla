"""Independent CPU comparison of immutable raw labels and both original encoders."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0267');R1=ROOT/'R1'
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/S2')
sys.path[:0]=[str(DATA/'R3'),str(DATA/'runtime'),str(DATA/'qa'),'/home/liuyu-nfs/Projects/isaac-gr00t']
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import pi0260_r2_codec as codec
from pi0260_r3_inputs import identity


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(decoded,raw):
    def matrices(x):
        x=x.astype(float);u=x[:,:3];u=u/np.linalg.norm(u,axis=1)[:,None]
        v=x[:,3:]-np.sum(u*x[:,3:],axis=1)[:,None]*u;v=v/np.linalg.norm(v,axis=1)[:,None]
        return np.stack([u,v,np.cross(u,v)],axis=1)
    a,b=matrices(decoded[:,3:9]),matrices(raw[:,3:9])
    return float(np.linalg.norm(decoded[:,:3]-raw[:,:3],axis=1).max()),float(Rotation.from_matrix(a@b.transpose(0,2,1)).magnitude().max())


def main():
    path=R1/'evidence/R_RUNTIME_LABEL_REPLAY.json';hop=read(path);arrays=path.with_suffix('.npz')
    assert hop['status']=='PASS_ACTUAL_R_LABEL_CPU_REPLAY_PENDING_CROSS_HOST_RECONCILIATION' and sha(arrays)==hop['arrays_sha256']
    plan_path=R1/'evidence/FULL_TRAIN_INPUT_PLAN.json';plan=read(plan_path)
    assert sha(plan_path)==hop['frozen_corelab_plan_sha256']
    expected=[r for r in plan['rows'] if r['arm']=='R'];assert len(expected)==8192
    assert sha(R1/'design/statistics.json')==hop['statistics_sha256']
    proc=codec.processor(read(R1/'design/statistics.json'));z=np.load(arrays);family={};checks={};results=[]
    first=np.load(R1/'evidence/R_LABEL_REPLAY_FIRST_corelab.npz')
    _,a,m=codec.encode(proc,first['state'],first['raw_action'])
    assert np.array_equal(a[None],first['action']) and np.array_equal(m[None].astype(first['mask'].dtype),first['mask'])
    for k,key in enumerate(z['keys'].tolist()):
        fid,idx=key.rsplit(':',1);idx=int(idx)
        if fid not in family:
            p=RAW/fid/'proposals.npz';assert sha(p)==hop['source_proposal_hashes'][fid]
            source=np.load(p);family[fid]=(source['state_v1'],source['raw_action'])
        state,raw=family[fid][0][idx],family[fid][1][idx]
        assert np.array_equal(state,z['state'][k]) and np.array_equal(raw,z['raw_action'][k])
        _,core,mask=codec.encode(proc,state,raw);assert not core[:,10:].any()
        hopper=np.pad(z['encoded_active'][k],((0,0),(0,122)))
        label_hash=identity(dict(action=torch.from_numpy(core[None]),action_mask=torch.from_numpy(mask[None].astype(first['mask'].dtype))))
        hp,cp=codec.decode(proc,hopper,state),codec.decode(proc,core,state)
        hm,cm,delta=metrics(hp,raw),metrics(cp,raw),metrics(hp,cp)
        assert all(x<=1e-5 and a<=1e-4 for x,a in (hm,cm,delta)),('ORIGINAL_PHYSICAL_CODEC_TOLERANCE',key,hm,cm,delta)
        checks[(fid,idx)]=label_hash
        results.append(dict(key=key,corelab_label_sha256=label_hash,raw_and_state_byte_identical=True,
            encoded_bytes_equal=bool(np.array_equal(core,hopper)),normalized_max_abs=float(abs(core-hopper).max()),
            hopper_via_corelab_XYZ_m=hm[0],hopper_via_corelab_SO3_rad=hm[1],corelab_XYZ_m=cm[0],corelab_SO3_rad=cm[1],
            decoded_cross_host_XYZ_m=delta[0],decoded_cross_host_SO3_rad=delta[1]))
    for e,h in zip(expected,hop['rows']):
        assert (e['unit'],e['micro'],e['family'],e['index'])==(h['unit'],h['micro'],h['family'],h['index'])
        assert checks[e['family'],e['index']]==e['labels_masks_sha256']==h['corelab_plan_label_sha256']
    out=dict(status='PASS_RAW_LABEL_IDENTITY_BOTH_ORIGINAL_RUNTIME_REPLAYS_AND_ORIGINAL_PHYSICAL_BOUNDS',
        consumed_rows=8192,unique_source_labels=len(results),results=results,statistics_sha256=hop['statistics_sha256'],
        source_proposal_hashes=hop['source_proposal_hashes'],hopper_replay_receipt_sha256=sha(path),
        frozen_corelab_plan_sha256=sha(plan_path),actual_sample_audit_sha256=hop['actual_sample_audit_sha256'],
        cross_host_label_byte_identity=False,cross_host_byte_mismatches=hop['cross_host_byte_mismatches'],
        original_failed_audit_retained=True,training_and_processor_unchanged=True,
        physical_XYZ_tolerance_m=1e-5,physical_SO3_tolerance_rad=1e-4,model_queries=0,gradient_queries=0,
        diagnostic_source_sha256=sha(__file__),evaluation_admission=False)
    target=R1/'evidence/R_CROSS_HOST_LABEL_RECONCILIATION.json';target.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('results','source_proposal_hashes')}),flush=True)


if __name__=='__main__':main()
