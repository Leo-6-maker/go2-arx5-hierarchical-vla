"""Metadata/actual-noise seals only. No new held-out RGB, label or model score."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(BASE/'PI0264/code'),str(BASE/'PI0263/code'),str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0264_plan import read,sha,write


def seal(path,value):
    if path.exists():assert read(path)==value
    else:write(path,value)


def main():
    card=read(R1/'design/EVALUATION_CARD.json');scene=read(ROOT/'design/SCENE_CARD.json');old=read(BASE/'PI0264/UNION_EVAL_CARD.json')
    legacy=[r for r in old['rows'] if r['kind']=='legacy' and r['source_row']['source']=='R3_ORIGINAL'];assert len(legacy)==440
    for r in legacy:
        cell=BASE/'PI0264/eval/L2048/cells'/r['uid'];record=read(cell/'record.json')
        assert record['status']=='VALID' and record['row']==r and record['tensor_sha256']==sha(cell/'tensors.npz')
        r['PI0267_donor_cell']=str(cell);r['PI0267_donor_record_sha256']=sha(cell/'record.json');r['PI0267_donor_tensor_sha256']=sha(cell/'tensors.npz')
    noise=list(card['noise'])
    old_noise=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2/R3/noise')
    for name in ('B0','B1'):
        source=old_noise/(name+'.npz');target=RAW/'noise'/('legacy-'+name+'.npz')
        if not target.exists():shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        with np.load(target) as z:value=z['float32'];assert value.shape==(1,40,132)
        noise.append(dict(file=str(target),sha256=sha(target),keys=[dict(name='float32',noise_sha256=hashlib.sha256(value.tobytes()).hexdigest())],original_file=str(source)))
    specs=[];recovery_ordinal=0
    for f in scene['families']:
        if f['split']!='NEW_SIM_EVAL':continue
        for task in ('M1','M3'):
            for command in (0,1):
                for j,ell in enumerate((.125,.375,.625,.875)):
                    for recovery in (False,True):
                        variant=1+recovery_ordinal%6 if recovery else 0
                        if recovery:recovery_ordinal+=1
                        specs.append(dict(uid=f'o-{len(specs):03d}',family=f['family_id'],task=task,command=command,ell_fraction=ell,variant=variant,M1_template='T0',
                            noise_file=str(RAW/'noise'/(f['family_id']+'.npz')),noise_name=f'{task}_E0_{2*j+int(recovery):03d}'))
    assert len(specs)==256 and recovery_ordinal==128
    value=dict(schema='PI0267_FIXED_OFFLINE_METADATA_AND_EXISTING_NOISE_V1',legacy_rows=legacy,new_rows=specs,noise=noise,
        donor_legacy_cache_reuse=440,new_legacy_queries=880,new_domain_queries=768,
        noise_rule='New offline: reuse prospectively stored VIRTUAL E0 stream positions0..7, shared across commands/model points. Legacy: original actual B0/B1.',
        old_contract_card_sha256=sha(BASE/'PI0264/UNION_EVAL_CARD.json'),
        old_statistics_sha256=sha(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2/dataset/statistics.json')),statistics_sha256=card['statistics_sha256'],
        EVAL_payload_opened=False,model_queries=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    seal(R1/'design/FIXED_OFFLINE_CARD.json',value)
    # The bounded secondary lane shares the same fixed family/task/E0/replan key rule, without new seeds.
    import torch
    families=[f for f in scene['families'] if f['split']=='FEEDBACK_TRAIN'][:2];secondary_noise=[];cases=[]
    for f in families:
        path=RAW/'noise'/('WBC-'+f['family_id']+'.npz');values={};keys=[]
        for task in ('M1','M3'):
            for replan in range(100):
                key=f"PI0267:VIRTUAL:26741:{f['family_id']}:{task}:E0:{replan}";seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
                value=torch.randn((1,40,132),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).numpy()
                name=f'{task}_E0_{replan:03d}';values[name]=value;keys.append(dict(name=name,key=key,seed=seed,noise_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
            for c in (0,1):cases.append(dict(family=f['family_id'],task=task,command=c))
        if path.exists():
            with np.load(path) as z:assert all(np.array_equal(z[k],v) for k,v in values.items())
        else:np.savez_compressed(path,**values)
        secondary_noise.append(dict(file=str(path),sha256=sha(path),keys=keys))
    seal(R1/'design/SECONDARY_CARD.json',dict(schema='PI0267_FIXED_SECONDARY_METADATA_AND_ACTUAL_NOISE_V1',cases=cases,expert=8,model=24,noise=secondary_noise,M1_template='T0',
        scope='First two fixed TRAIN families / WBC transfer diagnostic, not held-out physical generalization',model_queries=0,physics_steps=0))
    print('PASS_FIXED_440_LEGACY_256_OFFLINE_8_SECONDARY_METADATA_ACTUAL_NOISE',flush=True)


if __name__=='__main__':main()
