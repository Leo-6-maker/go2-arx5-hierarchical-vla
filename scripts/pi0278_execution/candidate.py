"""Freeze one qualifying development point; open confirmation only after its proofs."""
import argparse,hashlib,json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from evaluation_card import COMMON,ROOT,RAW,RELEASE,read,sha,freeze,write

def candidate(point):
    assert point in ('COVER931024','COVER932048','COVER934096')
    proofs={lane:read(COMMON/'R3/evidence'/f'FAST_{point}_{lane}.json') for lane in ('TRAIN_PHYSICS','TEST')}
    for lane,v in proofs.items():
        assert v['point']==point and v['lane']==lane and v['governing_release']==RELEASE and all(v['checks'].values())
        assert v['binding_sha256']==sha(ROOT/'model_bindings'/f'{point}.json') and v['card_sha256']==sha(ROOT/'design/EVALUATION_CARD.json')
        for row in v['result_files']:assert sha(row['path'])==row['sha256']
    assert proofs['TRAIN_PHYSICS']['task']['denominator']==4 and proofs['TRAIN_PHYSICS']['task']['persistent_stacks']>=3
    assert proofs['TEST']['task']['denominator']==8 and proofs['TEST']['task']['persistent_stacks']>=6
    freeze(ROOT/'evidence/FROZEN_CANDIDATE.json',dict(point=point,governing_release=RELEASE,development_checkpoint=point,model_binding_sha256=sha(ROOT/'model_bindings'/f'{point}.json'),
        proofs={lane:sha(COMMON/'R3/evidence'/f'FAST_{point}_{lane}.json') for lane in proofs},formal_selected_checkpoint='NONE',deployment='NONE',confirmation='NOT_YET_EXECUTED'))
    print('FROZEN_ONE_DEVELOPMENT_CANDIDATE',point)

def confirmation(point):
    frozen=read(ROOT/'evidence/FROZEN_CANDIDATE.json');assert frozen['point']==point
    for lane in ('TRAIN_PHYSICS','TEST'):
        value=read(ROOT/'evidence'/f'CPU_REPLAY_{point}_{lane}.json');assert value['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
    assert read(ROOT/'evidence'/f'FROZEN_BYTES_{point}.json')['status']=='PASS_559_FROZEN_TENSOR_PAYLOAD_BYTES_IDENTICAL'
    permit=ROOT/'evidence/CONFIRMATION_ADMISSION.json'
    if permit.exists():assert read(permit)['point']==point;print('REUSED_ONE_CONFIRMATION_ADMISSION',point);return
    seeds=read(ROOT/'design/EVALUATION_CARD.json')['CONFIRM16'];assert seeds==list(range(30032,30048))
    collisions=[];sources=[];preparation=ROOT/'evidence/CONFIRMATION_PREPARATION.json'
    if preparation.exists():assert read(preparation)['point']==point
    # Inspect actual records/packets, not prospective declarations of reserved seeds.
    for name in ('PI0276','PI0277','PI0278'):
        base=RAW.parent.with_name(name)
        for sub in ('episodes','R2/episodes'):
            for p in (base/sub).glob('*'):
                if any(re.search(rf'(?<![A-Za-z0-9]){seed}(?![A-Za-z0-9])',p.name) for seed in seeds):collisions.append(str(p))
        for p in (base/'noise').glob('*.npz'):
            if any(re.search(rf'(?<![A-Za-z0-9]){seed}(?![A-Za-z0-9])',p.name) for seed in seeds):
                if not (name=='PI0278' and preparation.exists() and p.name in [f'CONFIRM-{seed}.npz' for seed in seeds]):collisions.append(str(p))
        for p in (COMMON.with_name(name)/'evidence').glob('model_attempts.jsonl'):
            sources.append(dict(path=str(p),sha256=sha(p)))
            for line in p.read_text().splitlines():
                v=json.loads(line);request=v.get('request',{})
                if any(re.search(rf'(?<![A-Za-z0-9]){seed}(?![A-Za-z0-9])',request.get(k,'')) for seed in seeds for k in ('input','output','noise_file')):collisions.append(f'{p}:attempt{v.get("ordinal")}')
        for p in [base/'evidence/budget.jsonl']:
            if not p.exists():continue
            sources.append(dict(path=str(p),sha256=sha(p)))
            for line in p.read_text().splitlines():
                v=json.loads(line)
                if v.get('kind')=='starts' and any(re.search(rf'(?<![A-Za-z0-9]){seed}(?![A-Za-z0-9])',v.get('slot','')) for seed in seeds):collisions.append(f'{p}:{v["slot"]}')
    if not preparation.exists():freeze(ROOT/'evidence/CONFIRMATION_UNTOUCHED.json',dict(seeds=seeds,collisions=collisions,actual_journal_sources=sources,packets_not_created_at_check=True))
    assert not collisions,'Original confirmation namespace already accessed; no seed substitution'
    before=ROOT/'evidence/EVALUATION_NOISE_BEFORE_CONFIRM.json'
    original=read(before) if before.exists() else read(ROOT/'design/EVALUATION_NOISE.json');freeze(before,original)
    recipe=read(COMMON/'design/CONFIRM_NOISE_RECIPE.json');rows=[]
    freeze(preparation,dict(point=point,recipe_sha256=sha(COMMON/'design/CONFIRM_NOISE_RECIPE.json'),untouched_sha256=sha(ROOT/'evidence/CONFIRMATION_UNTOUCHED.json')))
    for seed in seeds:
        rng_seed=int.from_bytes(hashlib.sha256(recipe['namespace'].format(seed=seed).encode()).digest()[:8],'big')%2**31
        rng=np.random.Generator(np.random.PCG64(rng_seed));values={f'q{i:03d}':rng.standard_normal((1,40,132),dtype=np.float32) for i in range(80)}
        p=RAW.parent/'noise'/f'CONFIRM-{seed}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:assert set(z.files)==set(values) and all(np.array_equal(z[k],v) for k,v in values.items())
        else:
            temp=p.with_suffix('.partial.npz');np.savez(temp,**values);temp.replace(p)
        rows.extend(dict(lane='CONFIRM',condition=seed,index=i,file=str(p),name=k,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for i,(k,v) in enumerate(values.items()))
    value=dict(original,rows=original['rows']+rows,confirmation='ONE_FROZEN_CANDIDATE_ONLY',confirmation_recipe_sha256=sha(COMMON/'design/CONFIRM_NOISE_RECIPE.json'))
    write(ROOT/'design/EVALUATION_NOISE.json',value)
    freeze(permit,dict(point=point,governing_release=RELEASE,seeds=seeds,frozen_candidate_sha256=sha(ROOT/'evidence/FROZEN_CANDIDATE.json'),untouched_sha256=sha(ROOT/'evidence/CONFIRMATION_UNTOUCHED.json'),noise_sha256=sha(ROOT/'design/EVALUATION_NOISE.json'),formal_selected_checkpoint='NONE',deployment='NONE'))
    print('PASS_ONE_UNTOUCHED_CONFIRMATION16_ADMISSION',point)

def check():
    seed=30032;pattern=rf'(?<![A-Za-z0-9]){seed}(?![A-Za-z0-9])'
    assert re.search(pattern,'CONFIRM-30032-COVER934096') and not re.search(pattern,'TRAIN-130032-COVER934096') and not re.search(pattern,'ab30032cd')
    a=np.random.Generator(np.random.PCG64(27803)).standard_normal((1,40,132),dtype=np.float32)
    b=np.random.Generator(np.random.PCG64(27803)).standard_normal((1,40,132),dtype=np.float32)
    assert a.dtype==np.float32 and np.array_equal(a,b) and np.isfinite(a).all()
    print('PASS_CONFIRMATION_NAMESPACE_BOUNDARY_AND_PACKET_BYTE_REPEATABILITY')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','confirmation','check'));p.add_argument('point',nargs='?');a=p.parse_args()
    if a.stage=='check':check()
    else:assert a.point;candidate(a.point) if a.stage=='freeze' else confirmation(a.point)
