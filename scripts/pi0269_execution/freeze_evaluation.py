"""Reuse closed inputs/noise; freeze new diagnostic poses/packets without renders."""
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
import torch
torch.set_num_threads(4)
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0269';OLD=BASE/'PI0268/R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
sys.path[:0]=[str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0266_expert import reference
from pi0255_motion3 import scene

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(v,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text()==content
    else:p.write_text(content)

def main():
    assert not (ROOT/'evidence/model_attempts.jsonl').exists()
    points=['ROT2048','UNIFORM4096','PREFIX4096'];statsha=sha(OLD/'contracts/ROT/source/dataset/statistics.json')
    for relative in ('contracts/ROT/source/dataset/statistics.json','design/PAIRING_PLAN.json'):
        source=OLD/relative;target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:target.write_bytes(source.read_bytes())
    offline=read(OLD/'design/FIXED_OFFLINE_CARD.json')
    for row in offline['fit_rows']+offline['factor_rows']:assert sha(row['input'])==row['input_sha256']
    for row in offline['noise']:assert sha(row['file'])==row['sha256']
    offline.update(schema='PI0269_FIXED_OFFLINE_V1',release='e77ad092406555645fb8c68ba34cc8594ef18ee6',model_points=points,
        contracts={'ROT':statsha},logical_rows=2904,donor_reused_rows=968,source_card_sha256=sha(OLD/'design/FIXED_OFFLINE_CARD.json'))
    write(ROOT/'design/FIXED_OFFLINE_CARD.json',offline)
    virtual=read(OLD/'design/EVALUATION_CARD.json')
    for row in virtual['noise']:assert sha(row['file'])==row['sha256']
    virtual.update(schema='PI0269_FIXED_VIRTUAL_V1',model_points=points,primary_episodes=192,contracts={'ROT':statsha},
        donor_reused_episodes=64,new_planned_episodes=128,source_card_sha256=sha(OLD/'design/EVALUATION_CARD.json'),
        caps=dict(inference_attempts=18432,inference_heads=73728,new_virtual_starts=240,total_new_frame_pairs=17000,
                  planned_main_queries=12800,planned_component_queries=1920,planned_offline_queries=1936))
    write(ROOT/'design/EVALUATION_CARD.json',virtual)
    source=read(BASE/'PI0267/design/SCENE_CARD.json');reg=source['registry'];families=[f for f in source['families'] if f['split']=='FEEDBACK_TRAIN'][:2]
    starts=[];noise=[]
    for family in families:
     data=scene(reg,family)
     for command in (0,1):
      for backend in ('E0','E1'):
       for kind,task,t in [('M1_TERMINAL','M1',6.),('M3_ENTRY','M3',4.),('M3_MIDARC','M3',10.),('M3_TERMINAL','M3',16.)]:
        p,r=reference(reg,family,task,command,[t]);key=f"{family['family_id']}-{kind}-c{command}-{backend}"
        if task=='M3':
            initial=(-3 if command==0 else 3)*np.pi/4
            angle=initial+(1 if command==0 else -1)*1.5*np.pi*((t-4)/12)
            expected=np.array(data['center'])+[data['radius']*np.cos(angle),data['radius']*np.sin(angle),0]
            assert np.linalg.norm(p[0]-expected)<1e-12
        starts.append(dict(case=key,family=family['family_id'],task=task,command=command,backend=backend,start_type=kind,
                           initial_position_world=p[0].tolist(),initial_rotation_world=r[0].tolist(),horizon_s=8.,max_queries=20,
                           initialization='VIRTUAL_DIAGNOSTIC_PLACEMENT_NOT_POLICY_ACHIEVEMENT',new_training_labels=0,phase_input=False))
        path=RAW/'noise'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);values={};keys=[]
        for replan in range(20):
            name=f'q{replan:03d}';namespace=f'PI0269:COMPONENT:26741:{key}:{replan}'
            seed=int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8],'big')
            generator=torch.Generator(device='cpu').manual_seed(seed)
            value=torch.randn((1,40,132),generator=generator,dtype=torch.float32).numpy();values[name]=value
            keys.append(dict(name=name,key=namespace,seed=seed,noise_sha256=hashlib.sha256(value.tobytes()).hexdigest(),shape=list(value.shape),dtype=str(value.dtype)))
        if path.exists():
            with np.load(path) as z:assert z.files==list(values) and all(np.array_equal(z[n],v) for n,v in values.items())
        else:np.savez_compressed(path,**values)
        noise.append(dict(file=str(path),sha256=sha(path),keys=keys))
    assert len(starts)==32
    write(ROOT/'design/COMPONENT_CARD.json',dict(schema='PI0269_DIAGNOSTIC_START_V1',points=points,rows=starts,noise=noise,
        proposals=96,source_scene_sha256=sha(BASE/'PI0267/design/SCENE_CARD.json'),qualification_authority=False,
        task_or_phase_in_policy=False,first_valid_failures_retained=True,new_WBC_physics=0))
    inventory=[]
    priorraw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1')
    for domain,rows in [('fit',offline['fit_rows']),('factor',offline['factor_rows']),('legacy',offline['legacy_rows'])]:
      for row in rows:
        p=priorraw/'offline/ROT2048'/domain/row['uid']/'record.json';v=read(p)
        assert v['status']=='VALID' and v['row']==row and v['point']=='ROT2048'
        assert sha(v['tensor'])==v['tensor_sha256'];inventory.append(dict(path=str(p),sha256=sha(p),tensor_sha256=v['tensor_sha256']))
    for family in [f for f in source['families'] if f['split']=='NEW_SIM_EVAL']:
      for task in ('M1','M3'):
       for c in (0,1):
        for backend in ('E0','E1'):
         p=priorraw/'virtual/ROT2048'/f"{family['family_id']}-{task}-c{c}-{backend}"/'result.json';v=read(p)
         assert v['validity']=='VALID' and v['point']=='ROT2048' and sha(p.parent/'trajectory.npz')==v['arrays_sha256']
         inventory.append(dict(path=str(p),sha256=sha(p),arrays_sha256=v['arrays_sha256']))
    assert len(inventory)==1032
    write(ROOT/'evidence/DONOR_EXACT_REUSE.json',dict(status='PASS_EXISTING_968_AND_64_EXACT_REUSE',members=inventory,
        binding=read(OLD/'model_bindings/ROT2048.json'),offline_card_sha256=sha(OLD/'design/FIXED_OFFLINE_CARD.json'),
        virtual_card_sha256=sha(OLD/'design/EVALUATION_CARD.json'),no_new_queries=True))
    print('FROZEN_OFFLINE_2904_MAIN192_COMPONENT96_EXACT_DONOR1032',flush=True)

if __name__=='__main__':main()
