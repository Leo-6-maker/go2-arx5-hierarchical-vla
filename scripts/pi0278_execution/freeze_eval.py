"""Freeze the fixed diagnostic/physical inputs, copying actual old noise bytes."""
import hashlib,json,os,shutil
from pathlib import Path
import numpy as np
from evaluation_card import COMMON,ROOT,RAW,OLD,OLD_RAW,RELEASE,read,sha,freeze,packet

def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(source)==sha(dest),dest
    else:
        try:os.link(source,dest)
        except OSError:shutil.copyfile(source,dest)

def main():
    prior=COMMON.with_name('PI0277')/'R2';old_noise=read(prior/'design/EVALUATION_NOISE.json')
    fit=read(COMMON/'design/DIAGNOSTIC48.json')['rows'];assert len(fit)==48 and len({r['uid'] for r in fit})==48
    points=['FOCUS4096']+[f'COVER93{n}' for n in (512,1024,2048,4096)]
    card=dict(release=RELEASE,canonical=6027913340,fit=fit,TRAIN4=list(range(10000,10004)),DEV8=list(range(30000,30008)),CONFIRM16=list(range(30032,30048)),
        points=points,physics_limits=dict(starts=52,physics=170000,control=34000,rgb=34500,teacher=0),noise_index=0,max_screens=4,max_DEV_candidates=2,confirm_candidates=1,native_episode_s=32)
    freeze(ROOT/'design/FIT_CARD.json',dict(rows=fit));freeze(ROOT/'design/EVALUATION_CARD.json',card)
    rows=[];noise_root=RAW.parent/'noise';noise_root.mkdir(parents=True,exist_ok=True)
    for row in old_noise['rows']:
        if row['lane'] not in ('TRAIN_PHYSICS','TEST'):continue
        source=Path(row['file']);dest=noise_root/source.name;link(source,dest)
        with np.load(dest,allow_pickle=False) as z:assert hashlib.sha256(z[row['name']].tobytes()).hexdigest()==row['sha256']
        rows.append(dict(row,file=str(dest)))
    assert len(rows)==12*80
    old_fit={r['condition']:r for r in old_noise['rows'] if r['lane']=='FIT' and r['index']==0};reuse=[]
    binding=read(prior/'model_bindings/FOCUS4096.json')
    for item in fit:
        uid=item['uid'];dest=noise_root/f'FIT-{uid}.npz'
        if uid in old_fit:
            row=old_fit[uid];link(Path(row['file']),dest)
            with np.load(dest,allow_pickle=False) as z:noise=z[row['name']].copy()
            assert hashlib.sha256(noise.tobytes()).hexdigest()==row['sha256']
            noise_name=row['name']
        else:
            seed=int.from_bytes(hashlib.sha256(f'PI0278:DIAGNOSTIC48:27803:{uid}'.encode()).digest()[:8],'big')%2**31
            noise=np.random.RandomState(seed).normal(size=(1,40,132)).astype(np.float32);noise_name='n0'
            if dest.exists():
                with np.load(dest,allow_pickle=False) as z:assert np.array_equal(z[noise_name],noise)
            else:np.savez(dest,**{noise_name:noise})
        assert noise.shape==(1,40,132) and noise.dtype==np.float32 and np.isfinite(noise).all()
        digest=hashlib.sha256(noise.tobytes()).hexdigest()
        rows.append(dict(lane='FIT',condition=uid,index=0,file=str(dest),name=noise_name,sha256=digest))
        src=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2/quality/FOCUS4096/FIT')/uid/'n0'
        if uid in old_fit and (src/'output.npz').is_file():
            side=read(src/'output.receipt.json');assert side['binding']==binding and side['input_sha256']==item['condition_sha256'] and side['noise_sha256']==digest
            assert sha(src/'output.npz')==side['output_sha256']
            value=packet(item)
            with np.load(src/'input.npz',allow_pickle=False) as z:assert set(z.files)==set(value) and all(np.array_equal(z[k],v) for k,v in value.items())
            target=RAW/'quality/FOCUS4096/FIT'/uid/'n0'
            for name in ('input.npz','output.npz','output.receipt.json'):link(src/name,target/name)
            reuse.append(dict(uid=uid,source=str(src),destination=str(target),input_sha256=item['condition_sha256'],noise_sha256=digest,output_sha256=side['output_sha256']))
    freeze(ROOT/'design/EVALUATION_NOISE.json',dict(release=RELEASE,rows=rows,old_card_sha256=sha(prior/'design/EVALUATION_NOISE.json'),confirmation='LOCKED_UNTIL_ONE_FROZEN_CANDIDATE'))
    freeze(ROOT/'evidence/EXACT_PARENT_OUTPUT_REUSE.json',dict(point='FOCUS4096',rows=reuse,actual_reused=len(reuse),remaining_parent_queries=48-len(reuse),no_parent_physics_repeated=True))
    stats=ROOT/'contracts/TASK/source/dataset/statistics.json';stats.parent.mkdir(parents=True,exist_ok=True);link(prior/'contracts/TASK/source/dataset/statistics.json',stats)
    for name in ('ManiSkill','project-src'):
        p=COMMON/name
        if not p.exists():p.symlink_to(OLD/name,target_is_directory=True)
    p=COMMON/'design/SOURCE_FILESET.json';link(OLD/'design/SOURCE_FILESET.json',p)
    (COMMON/'runtime').mkdir(exist_ok=True);(RAW/'evidence').mkdir(parents=True,exist_ok=True)
    shared=RAW.parent/'evidence/budget.jsonl';shared.parent.mkdir(parents=True,exist_ok=True);shared.touch(exist_ok=True)
    ledger=RAW/'evidence/budget.jsonl'
    if not ledger.exists():ledger.symlink_to(shared)
    assert ledger.resolve()==shared.resolve() and ledger.resolve()!=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/evidence/budget.jsonl').resolve()
    paths=list((ROOT/'code').glob('*.py'))+[ROOT/'design/EVALUATION_CARD.json',ROOT/'design/FIT_CARD.json',stats,
        *[prior/'code'/n for n in ('evaluation_card.py','panel.py','client.py','replay_panel.py','contexts.py')],
        COMMON.with_name('PI0277')/'R3/code/fast_decision.py',*[OLD/'code'/n for n in ('source_worker.py','evaluate.py','replay.py')]]
    seal=dict(release=RELEASE,canonical=6027913340,files=[dict(path=str(p),sha256=sha(p)) for p in paths],algorithm='UNCHANGED_S4_BF16_FP32_DECODE_NATIVE_PHYSICS')
    freeze(ROOT/'design/EVALUATION_FILESET.json',seal);freeze(COMMON/'R3/EXECUTION_FILESET.json',seal)
    freeze(ROOT/'evidence/PANEL_FREEZE.json',dict(status='PASS_FIXED_DIAGNOSTIC48_TRAIN4_DEV8',card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),noise_sha256=sha(ROOT/'design/EVALUATION_NOISE.json'),
        exact_parent_reused=len(reuse),new_model_queries=0,new_physics_steps=0,confirmation='NOT_MATERIALIZED',selected_checkpoint='NONE',deployment='NONE'))
    print(json.dumps(dict(status='PASS_FIXED_DIAGNOSTIC48_TRAIN4_DEV8',parent_reused=len(reuse),remaining_parent_queries=48-len(reuse))))

if __name__=='__main__':main()
