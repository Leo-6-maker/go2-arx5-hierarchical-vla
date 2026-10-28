"""Read-only replay of consumed R labels; retains the failed cross-host byte check."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0267')
GROOT=Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t')
sys.path[:0]=[str(ROOT.with_name('PI0260')/'R3/execution'),str(GROOT)]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import pi0260_r2_codec as codec
from pi0260_r3_inputs import identity


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    statistics=ROOT/'source/dataset/statistics.json'
    assert digest(statistics)=='4439e24348bc43f5c93d380c9ae3c25b4057dbd8111e1e9bf9563b868c53bc73'
    proc=codec.processor(read(statistics))
    actual=[json.loads(s) for s in (ROOT/'R/main/training/sample_audit.jsonl').read_text().splitlines()]
    expected=[x for x in read(ROOT/'FULL_TRAIN_INPUT_PLAN.json')['rows'] if x['arm']=='R']
    assert len(actual)==len(expected)==8192
    first=np.load(ROOT/'evidence/R_LABEL_REPLAY_FIRST_hopper.npz')
    _,packed,mask=codec.encode(proc,first['state'],first['raw_action'])
    assert np.array_equal(packed[None],first['action'])
    assert np.array_equal(mask[None].astype(first['mask'].dtype),first['mask'])
    mask_dtype=first['mask'].dtype
    cache={};sources={};rows=[];families={};differences=[]
    for a,e in zip(actual,expected):
        assert {k:v for k,v in a.items() if k!='labels_masks_sha256'}=={k:v for k,v in e.items() if k!='labels_masks_sha256'}
        fid=a['family'];index=a['index'];key=(fid,index)
        if fid not in families:
            folder=ROOT/'source/S2'/fid;complete=read(folder/'COMPLETE.json')
            assert digest(folder/'proposals.npz')==complete['hashes']['proposals.npz']
            z=np.load(folder/'proposals.npz');families[fid]=(z['state_v1'],z['raw_action'])
            sources[fid]=complete['hashes']['proposals.npz']
        if key not in cache:
            states,actions=families[fid];state=states[index];raw=actions[index]
            _,encoded,mask=codec.encode(proc,state,raw)
            assert encoded.shape==(40,132) and np.isfinite(encoded).all() and not encoded[:,10:].any()
            value=identity(dict(action=torch.from_numpy(encoded[None]),action_mask=torch.from_numpy(mask[None].astype(mask_dtype))))
            decoded=codec.decode(proc,encoded,state)
            def matrices(x):
                x=x.astype(np.float64);u=x[:,:3];u=u/np.linalg.norm(u,axis=1)[:,None]
                v=x[:,3:]-np.sum(u*x[:,3:],axis=1)[:,None]*u;v=v/np.linalg.norm(v,axis=1)[:,None]
                return np.stack([u,v,np.cross(u,v)],axis=1)
            xyz=float(np.linalg.norm(decoded[:,:3]-raw[:,:3],axis=1).max())
            dr,rr=matrices(decoded[:,3:9]),matrices(raw[:,3:9])
            angle=float(Rotation.from_matrix(dr@rr.transpose(0,2,1)).magnitude().max())
            assert xyz<=1e-5 and angle<=1e-4,('ORIGINAL_PHYSICAL_CODEC_TOLERANCE',key,xyz,angle)
            cache[key]=dict(identity=value,encoded=encoded[:,:10],raw=raw,state=state,XYZ_max_m=xyz,SO3_max_rad=angle)
        v=cache[key]
        assert v['identity']==a['labels_masks_sha256'],('ACTUAL_RUNTIME_LABEL_NOT_REPRODUCED',a['unit'],a['micro'])
        row=dict(unit=a['unit'],micro=a['micro'],family=fid,index=index,actual_label_sha256=a['labels_masks_sha256'],
                 corelab_plan_label_sha256=e['labels_masks_sha256'],runtime_replay_exact=True)
        rows.append(row)
        if a['labels_masks_sha256']!=e['labels_masks_sha256']:differences.append(row)
    keys=sorted(cache)
    arrays=ROOT/'evidence/R_RUNTIME_LABEL_REPLAY.npz'
    np.savez_compressed(arrays,keys=np.array([f'{f}:{i}' for f,i in keys]),
        encoded_active=np.stack([cache[k]['encoded'] for k in keys]),raw_action=np.stack([cache[k]['raw'] for k in keys]),
        state=np.stack([cache[k]['state'] for k in keys]))
    result=dict(status='PASS_ACTUAL_R_LABEL_CPU_REPLAY_PENDING_CROSS_HOST_RECONCILIATION',consumed_rows=len(rows),
        unique_source_labels=len(keys),rows=rows,source_proposal_hashes=sources,statistics_sha256=digest(statistics),
        actual_sample_audit_sha256=digest(ROOT/'R/main/training/sample_audit.jsonl'),
        frozen_corelab_plan_sha256=digest(ROOT/'FULL_TRAIN_INPUT_PLAN.json'),cross_host_byte_mismatches=len(differences),
        max_roundtrip_XYZ_m=max(v['XYZ_max_m'] for v in cache.values()),max_roundtrip_SO3_rad=max(v['SO3_max_rad'] for v in cache.values()),
        original_XYZ_tolerance_m=1e-5,original_SO3_tolerance_rad=1e-4,arrays_sha256=digest(arrays),
        diagnostic_source_sha256=digest(__file__),model_queries=0,gradient_queries=0,training_updates=0,
        original_failed_audit_retained=True,changed_training_or_codec=False,evaluation_admission=False)
    (ROOT/'evidence/R_RUNTIME_LABEL_REPLAY.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_proposal_hashes')}),flush=True)


if __name__=='__main__':main()
