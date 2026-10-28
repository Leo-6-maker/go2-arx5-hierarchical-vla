"""Frozen paired visitation and pre-query actual noise, plus finite real-input checks."""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from pi0267_data_admission import BASE,ROOT,R1,RAW,DATA,read,sha,write,codec,build,batch,identity
sys.path.append(str(BASE/'PI0264/code'))
from pi0264_plan import TEMPLATES
from pi0258_kinematic import instructions


def seed_for(unit):return int.from_bytes(hashlib.sha256(f'PI0267:TRAIN:26741:{unit}'.encode()).digest()[:8],'big')%2**31


def freeze():
    admission=read(R1/'design/DATA_ADMISSION.json');assert admission['status']=='PASS_CONTINUE_ACTUAL_INPUT_AND_TRAINING'
    card=read(ROOT/'design/SCENE_CARD.json');families=[f for f in card['families'] if f['family_id'] in admission['families_admitted']]
    assert len(families)>=16
    bank=[]
    for anchor in range(16):
        for f in families:
            for task in ('M1','M3'):
                rows=read(RAW/f['family_id']/'rows.json');valid=set(read(RAW/f['family_id']/'codec_admission.json')['eligible_indices'])
                groups=[[r for r in rows if r['task']==task and r['command']==c and r['anchor_index']==anchor and r['index'] in valid] for c in (0,1)]
                if not all(any(r['nominal'] for r in g) for g in groups):continue
                recovery=[v for v in range(1,7) if all(any(r['variant']==v for r in g) for g in groups)]
                if not recovery:continue
                bank.append(dict(family=f['family_id'],task=task,anchor=anchor,nominal=[next(r['index'] for r in g if r['nominal']) for g in groups],
                    recovery={str(v):[next(r['index'] for r in g if r['variant']==v) for g in groups] for v in recovery}))
    pools={t:[r for r in bank if r['task']==t] for t in ('M1','M3')};units=[]
    for i in range(4096):
        task=('M1','M3')[i%2];slot=i//2;visit,bank_index=divmod(slot//2,len(pools[task]));source=pools[task][bank_index]
        common=slot%2==0;variants=sorted(map(int,source['recovery']))
        # Balance the six signed axes across source visit order, including each family's stages.
        variant=variants[(visit+source['anchor']+bank_index%len(families))%len(variants)]
        template=f'T{(visit+bank_index)%8}' if task=='M1' else 'ORIGINAL_M3'
        units.append(dict(unit=i+1,task=task,family=source['family'],anchor=source['anchor'],visit=visit,bank_index=bank_index,
            slot='COMMON_NOMINAL' if common else 'RECOVERY_INTERVENTION',order=[0,1] if (i//4)%2==0 else [1,0],template=template,
            N_indices=source['nominal'],R_indices=source['nominal'] if common else source['recovery'][str(variant)],
            recovery_variant=0 if common else variant,seed=seed_for(i+1)))
    assert Counter(u['task'] for u in units)=={'M1':2048,'M3':2048}
    assert Counter(u['slot'] for u in units)=={'COMMON_NOMINAL':2048,'RECOVERY_INTERVENTION':2048}
    assert {u['recovery_variant'] for u in units if u['slot']=='RECOVERY_INTERVENTION'}==set(range(1,7))
    for t in ('M1','M3'):
        assert {u['family'] for u in units if u['task']==t}==set(admission['families_admitted'])
        assert {u['anchor'] for u in units if u['task']==t}==set(range(16))
    plan=dict(schema='PI0267_N_R_PAIRED_VIRTUAL_RECOVERY_PLAN_V1',seed=26741,units=units,source_bank=bank,
        admission_sha256=sha(R1/'design/DATA_ADMISSION.json'),statistics_sha256=admission['statistics_sha256'],
        training_updates_per_arm=4096,microbatches_per_update=2,templates=TEMPLATES,
        source_family_colors={f['family_id']:f['factors']['colors'] for f in families})
    path=R1/'design/PAIRING_PLAN.json'
    if path.exists():assert read(path)==plan
    else:write(path,plan)
    # Metadata-only evaluator/noise freeze; no held-out observation, label or score accessed.
    noise_dir=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/noise');noise_dir.mkdir(exist_ok=True)
    noise_rows=[]
    for f in card['families']:
        if f['split']!='NEW_SIM_EVAL':continue
        path=noise_dir/(f['family_id']+'.npz');values={};keys=[]
        for task in ('M1','M3'):
            for executor in ('E0','E1'):
                for replan in range(100):
                    key=f"PI0267:VIRTUAL:26741:{f['family_id']}:{task}:{executor}:{replan}"
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
                    value=torch.randn((1,40,132),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).numpy()
                    name=f'{task}_{executor}_{replan:03d}';values[name]=value
                    keys.append(dict(name=name,key=key,seed=seed,noise_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
        if path.exists():
            with np.load(path) as z:assert all(np.array_equal(z[k],v) for k,v in values.items())
        else:np.savez_compressed(path,**values)
        noise_rows.append(dict(family=f['family_id'],file=str(path),sha256=sha(path),keys=keys))
    evaluator=dict(schema='PI0267_FROZEN_VIRTUAL_EVALUATION_V3',primary_episodes=192,secondary_expert=8,secondary_model=24,
        model_points=['donor','N4096','R4096'],scorer_sha256=sha(ROOT/'code/pi0267_metrics_v3.py'),legacy_scorer='PI0266_V2',
        source_scene_sha256=sha(ROOT/'design/SCENE_CARD.json'),tool_card_sha256=sha(R1/'design/VIRTUAL_TOOL_CARD.json'),
        statistics_sha256=admission['statistics_sha256'],EVAL_payload_opened=False,max_queries_per_episode=100,replan_s=.4,dt_s=.02,
        executors=dict(E0='interpolated installed reference',E1='left-SO3 first order lag tau=.15'),noise=noise_rows,
        generator=dict(library='torch',version=torch.__version__,dtype='float32',device='cpu',shape=[1,40,132]),
        caps=dict(queries=24576,inference_heads=98304,official_updates=8192,discarded_canary_updates=32,raw_update_cap=8480,raw_training_heads=210000,frame_pairs=40000))
    ep=R1/'design/EVALUATION_CARD.json'
    if ep.exists():assert read(ep)==evaluator
    else:write(ep,evaluator)
    return plan


def text_for(plan,unit,command):
    if unit['task']=='M1':return plan['templates'][unit['template']].format(color=plan['source_family_colors'][unit['family']][command])
    return instructions('M3',command,plan['source_family_colors'][unit['family']][command])[0]


def input_checks():
    plan=read(R1/'design/PAIRING_PLAN.json');proc=build(read(R1/'design/statistics.json'),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    direct=codec.processor(read(R1/'design/statistics.json'));rows=[];seen=set();byfamily={};checks=[]
    for unit in plan['units']:
        if unit['anchor'] not in (0,4,8,12,15):continue
        for arm in ('N','R'):
            for command in (0,1):
                index=unit[f'{arm}_indices'][command];text=text_for(plan,unit,command);key=(unit['family'],index,text)
                if key in seen:continue
                seen.add(key)
                if unit['family'] not in byfamily:
                    folder=RAW/unit['family'];z=np.load(folder/'proposals.npz');fr=np.load(folder/'policy_frames.npz')
                    byfamily[unit['family']]=(z,fr,read(folder/'rows.json'))
                z,fr,meta=byfamily[unit['family']];j=meta[index]['frame_index']
                packet=dict(body_rgb=fr['body_rgb'][j],wrist_rgb=fr['wrist_rgb'][j],state_v1=z['state_v1'][index],language=text)
                before=(repr(random.getstate()),repr(np.random.get_state()),torch.get_rng_state().clone())
                result=batch(proc,packet,z['raw_action'][index]);again=batch(proc,packet,z['raw_action'][index])
                assert before[:2]==(repr(random.getstate()),repr(np.random.get_state())) and torch.equal(before[2],torch.get_rng_state())
                assert identity(result)==identity(again)
                ps,pa,mask=codec.encode(direct,packet['state_v1'],z['raw_action'][index]);inputs=result['inputs']
                assert np.max(abs(inputs['state'].numpy().reshape(1,132)-ps))<=1e-7
                assert np.max(abs(inputs['action'].numpy().reshape(40,132)-pa))<=1e-7
                assert np.array_equal(inputs['action_mask'].numpy().reshape(40,132),mask)
                tokens=inputs['attention_mask'].numpy().astype(bool);visible=inputs['input_ids'].numpy()[tokens]
                decoded=proc.collator.processor.tokenizer.decode(visible,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                import re
                expected=re.sub(r'[^\w\s]','',text.lower()) if proc.formalize_language else text
                assert expected in decoded,('LANGUAGE_CLAUSE_TRUNCATED',text)
                rows.append(dict(family=unit['family'],task=unit['task'],command=command,index=index,arm=arm,template=unit['template'],
                    language=text,input_sha256=identity(inputs),frame_key=meta[index]['frame_key'],RNG_unchanged=True))
                if not checks:
                    for extra in ('elapsed_time','target_coordinates','phase','teacher','source_id','depth','mask','executor'):
                        try:batch(proc,dict(packet,**{extra:0}),z['raw_action'][index])
                        except ValueError:checks.append(extra)
                        else:raise AssertionError(('INPUT_LEAK_ACCEPTED',extra))
    assert {r['family'] for r in rows}==set(plan['source_family_colors'])
    assert {r['template'] for r in rows if r['task']=='M1'}==set(TEMPLATES)
    out=dict(status='PASS_PI0267_REAL_INPUT_COLOR_TEMPLATE_CODEC_RNG_BOUNDARY',checked_inputs=len(rows),rows=rows,negative_fields_rejected=checks,
        pairing_plan_sha256=sha(R1/'design/PAIRING_PLAN.json'),evaluation_card_sha256=sha(R1/'design/EVALUATION_CARD.json'),
        statistics_sha256=sha(R1/'design/statistics.json'),EVAL_payload_opened=False,model_queries=0)
    write(R1/'design/PROCESSED_PLAN_AUDIT.json',out);print(out['status'],len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','inputs'));args=parser.parse_args()
    freeze() if args.mode=='freeze' else input_checks()
