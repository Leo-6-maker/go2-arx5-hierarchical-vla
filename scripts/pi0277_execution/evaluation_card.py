"""Freeze registered evaluation inputs/noise without importing a model or simulator."""
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0277')
OLD=ROOT.with_name('PI0276')
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277')
OLD_RAW=RAW.with_name('PI0276')
RELEASE='63151a4548243ab0fb4ba172eea6f7d92e02c576'

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    with tmp.open('w') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def freeze(p,value):
    if Path(p).exists():assert read(p)==value,p
    else:write(p,value)

def packet(row):
    folder=Path(row['episode']);index=row['anchor']
    with np.load(folder/'observations.npz',allow_pickle=False) as z:
        value=dict(body_rgb=z['base_camera_96'][index],wrist_rgb=z['hand_camera_96'][index],state_v1=z['state_v1'][index])
    assert value['state_v1'].shape==(10,)
    assert all(value[k].shape==(96,96,3) and value[k].dtype==np.uint8 for k in ('body_rgb','wrist_rgb'))
    return value

def main():
    prior=read(OLD/'design/EVALUATION_NOISE.json');fit=read(ROOT/'design/FIT_CARD.json')
    plan=read(ROOT/'design/TRAIN_PLAN.json')['stream'];seen={r['uid'] for r in plan}
    assert len(fit['rows'])==64 and all(r['uid'] in seen for r in fit['rows'])
    seeds=prior['TRAIN_diagnostic_seeds'];assert sorted({r['seed'] for r in fit['rows']})==seeds and len(seeds)==16
    for seed in seeds:assert sorted(r['group'] for r in fit['rows'] if r['seed']==seed)==[0,1,2,3]
    # Existing OFFLINE/TEST/TRAIN noise is read from the original files, never regenerated.
    noises=[];verified={}
    for row in prior['rows']:
        p=Path(row['file']);assert p.resolve().is_relative_to((OLD_RAW/'noise').resolve())
        if str(p) not in verified:
            with np.load(p,allow_pickle=False) as z:verified[str(p)]={k:hashlib.sha256(z[k].tobytes()).hexdigest() for k in z.files}
        assert verified[str(p)][row['name']]==row['sha256']
        noises.append(row)
    rng=np.random.Generator(np.random.PCG64(27711))
    for row in fit['rows']:
        p=RAW/'noise'/f"FIT-{row['uid']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
        values={f'n{n}':rng.standard_normal((1,40,132),dtype=np.float32) for n in (0,1)}
        if p.exists():
            with np.load(p,allow_pickle=False) as z:assert set(z.files)==set(values) and all(np.array_equal(z[k],v) for k,v in values.items())
        else:np.savez(p,**values)
        noises.extend(dict(lane='FIT',condition=row['uid'],index=n,file=str(p),name=f'n{n}',sha256=hashlib.sha256(values[f'n{n}'].tobytes()).hexdigest()) for n in (0,1))
    offline=read(OLD/'design/OFFLINE_CARD.json')['rows'];assert len(offline)==192
    reuse=[]
    for seed in range(30000,30032):
        for point in ('joint','eef','TASK8192'):
            paths=[p for p in (OLD_RAW/'episodes'/f'TEST-{seed}-{point}').glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')]
            assert len(paths)==1
            reuse.append(dict(historical_lane='TEST',current_lane='REPAIR_DEV32',seed=seed,point=point,
                result=str(paths[0]),result_sha256=sha(paths[0]),initial_sha256=sha(paths[0].with_name('initial.npz'))))
    for seed in seeds:
        paths=[p for p in (OLD_RAW/'episodes'/f'TRAIN_PHYSICS-{seed}-TASK8192').glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')]
        assert len(paths)==1;reuse.append(dict(historical_lane='TRAIN_PHYSICS',current_lane='TRAIN_PHYSICS',seed=seed,point='TASK8192',result=str(paths[0]),result_sha256=sha(paths[0])))
    fit_inputs=[]
    for row in fit['rows']:
        folder=Path(row['episode']);assert sha(folder/'commands.npz')==row['command_sha256']
        inp=packet(row);dest=RAW/'fixed_inputs/FIT'/row['uid']/'input.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:assert set(z.files)==set(inp) and all(np.array_equal(z[k],v) for k,v in inp.items())
        else:np.savez(dest,**inp)
        with np.load(folder/'commands.npz',allow_pickle=False) as z:target=z['action'][row['anchor']:row['anchor']+40]
        assert target.shape==(40,10)
        fit_inputs.append(dict(uid=row['uid'],file=str(dest),file_sha256=sha(dest),actual_consumed=True,
            physical_label_sha256=hashlib.sha256(target.tobytes()).hexdigest(),condition_sha256=row['condition_sha256'],labels_masks_sha256=row['labels_masks_sha256']))
    freeze(ROOT/'design/EVALUATION_NOISE.json',dict(release=RELEASE,rows=noises,fit_seed=27711,fit_namespace='PI0277_FIT64_TWO_PACKETS',
        old_noise_card_sha256=sha(OLD/'design/EVALUATION_NOISE.json'),TRAIN_diagnostic_seeds=seeds,confirmation_packets='LOCKED_UNTIL_BOTH_FINALS_AND_BASIC_MILESTONE'))
    freeze(ROOT/'design/EVALUATION_CARD.json',dict(release=RELEASE,offline=offline,fit=fit['rows'],fit_inputs=fit_inputs,
        policies=['CONTROL4096','GRIP4096'],fit_policies=['TASK8192','CONTROL4096','GRIP4096'],
        physical={'TRAIN_PHYSICS':seeds,'TEST':list(range(30000,30032))},lane_labels={'TEST':'REPAIR_DEV32','TRAIN_PHYSICS':'TRAIN_PHYSICS'},
        confirmation_reserved_seeds=list(range(30032,30064)),confirmation_instantiated=False,old_results=reuse,
        quality_attempt_cap=14336,quality_head_cap=57344,physics_limits=dict(starts=234,control=150000,physics=750000,rgb=155000,teacher=48000),
        technical_physical_replacements=8,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    freeze(ROOT/'evidence/EVALUATION_CARD_FREEZE.json',dict(status='PASS_PREQUERY_FIXED_INPUT_NOISE_AND_REUSE_CARD',
        card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),noise_card_sha256=sha(ROOT/'design/EVALUATION_NOISE.json'),
        fit_conditions=64,fit_packets=128,reused_offline_conditions=192,old_physical_results=len(reuse),new_model_queries=0,new_physics_steps=0))
    print('EVALUATION_CARD_FROZEN',flush=True)

if __name__=='__main__':main()
