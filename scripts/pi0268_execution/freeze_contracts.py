"""Freeze the authorized two-contract study using existing TRAIN bytes only."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
    OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1')
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0268/R1'; PREP=BASE/'PI0268/preparation'; OLD=BASE/'PI0267'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
sys.path[:0]=[str(DATA/'qa'),str(DATA/'R3'),str(DATA/'runtime'),str(BASE.parent/'isaac-gr00t')]
RELEASE='e5ec6d52100f6bb2e32be5b9661c756b0adb61f4'
STAT_SHA='4439e24348bc43f5c93d380c9ae3c25b4057dbd8111e1e9bf9563b868c53bc73'
PLAN_SHA='fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    if p.exists(): assert p.read_bytes()==content, ('IMMUTABLE_OUTPUT_CHANGED',str(p))
    else: p.write_bytes(content)

def rotation_contract(stat):
    result=copy.deepcopy(stat)
    for block in ('state','relative_action'):
        for field,val in (('min',-1.),('q01',-1.),('max',1.),('q99',1.)):
            rows=[result['new_embodiment'][block]['eef_9d'][field]] if block=='state' else result['new_embodiment'][block]['eef_9d'][field]
            assert len(rows)==(1 if block=='state' else 40) and all(len(r)==9 for r in rows)
            for row in rows: row[3:9]=[val]*6
    return result

def allowed_diff(old,new):
    restored=copy.deepcopy(new); changes=[]
    for block in ('state','relative_action'):
        for field,val in (('min',-1.),('q01',-1.),('max',1.),('q99',1.)):
            before=old['new_embodiment'][block]['eef_9d'][field]
            after=new['new_embodiment'][block]['eef_9d'][field]
            a,b=([before],[after]) if block=='state' else (before,after)
            for h,(x,y) in enumerate(zip(a,b)):
                assert y[:3]==x[:3] and y[3:9]==[val]*6
                for d in range(3,9):
                    if x[d]!=y[d]: changes.append(dict(block=block,field=field,horizon=h,feature=d,old=x[d],new=y[d]))
            restored['new_embodiment'][block]['eef_9d'][field]=copy.deepcopy(before)
    assert restored==old and old is not new
    return changes

def select_fit(consumed):
    first={}
    for row in sorted(consumed,key=lambda r:(r['unit'],r['micro'],r['condition_sha256'])):
        if row['arm']=='R' and row['unit']<=2048: first.setdefault(row['condition_sha256'],row)
    fit=[]; coverage=[]
    for task in ('M1','M3'):
        for cmd in (0,1):
            for variant in range(7):
                rows=sorted((r for r in first.values() if (r['task'],r['command'],r['variant'])==(task,cmd,variant)),
                    key=lambda r:(r['quartile'],r['unit'],r['micro'],r['condition_sha256']))
                chosen=[]; missing=[]
                for q in range(4):
                    own=[r for r in rows if r['quartile']==q]
                    if own: chosen.append(own[0])
                    else: missing.append(q)
                for row in rows:
                    if len(chosen)==4: break
                    if row not in chosen: chosen.append(row)
                coverage.append(dict(task=task,command=cmd,variant=variant,selected=len(chosen),missing_quartiles=missing))
                for row in chosen: fit.append(dict(row,input_id=f'fit-first2048-{len(fit):03d}'))
    assert len(fit)<=112 and len({r['condition_sha256'] for r in fit})==len(fit)
    assert all(r['eligible'] and r['split']=='TRAIN' and r['unit']<=2048 for r in fit)
    return fit,coverage

def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30)); os.nice(19)
    if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    import numpy as np
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    original_call=torch.nn.Module.__call__
    def no_network_module(self,*args,**kw):
        if type(self).__module__.startswith('torchvision.transforms'): return original_call(self,*args,**kw)
        raise RuntimeError('NO_NEURAL_QUERY_IN_P1')
    torch.nn.Module.__call__=no_network_module
    def no_optimizer(*args,**kw): raise RuntimeError('NO_OPTIMIZER_IN_P1')
    torch.optim.Optimizer.__init__=no_optimizer
    import pi0260_r2_codec as codec
    from pi0260_r3_inputs import build,batch,identity
    source=PREP/'source_metadata/actual_statistics.json'; assert sha(source)==STAT_SHA
    stat=read(source); frozen=copy.deepcopy(stat); rot=rotation_contract(stat)
    changes=allowed_diff(stat,rot); assert stat==frozen
    for arm,value in [('OLD',stat),('ROT',rot)]:
        target=ROOT/'contracts'/arm/'source/dataset/statistics.json'; target.parent.mkdir(parents=True,exist_ok=True)
        if arm=='OLD':
            if target.exists(): assert target.read_bytes()==source.read_bytes()
            else: target.write_bytes(source.read_bytes())
        else: write(target,value)
    planpath=OLD/'R1/design/PAIRING_PLAN.json'; assert sha(planpath)==PLAN_SHA
    plan=read(planpath); units=plan['units'][:2048]; assert len(units)==2048
    # The original executed seed uses big endian. P0's prose typo is not a seed change.
    seed=lambda u:int.from_bytes(hashlib.sha256(f'PI0267:TRAIN:26741:{u}'.encode()).digest()[:8],'big')%2**31
    consumed=read(PREP/'output/CONSUMED_CONDITION_INDEX.json')
    original=[r for r in consumed if r['arm']=='R' and r['unit']<=2048]
    assert len(original)==4096
    for row in original:
        u=units[row['unit']-1]
        assert u['unit']==row['unit'] and u['order'][row['micro']]==row['command']
        assert u['R_indices'][row['command']]==row['index'] and u['family']==row['family']
        if 'seed' in u: assert u['seed']==seed(u['unit'])
    fit,coverage=select_fit(consumed)
    meta=read(PREP/'output/TRAIN_NUMERIC_INDEX.json'); lookup={(r['family'],r['index']):i for i,r in enumerate(meta)}
    with np.load(PREP/'cache/train_numeric.npz') as z:
        states=z['state']; actions=z['action']; old_s=z['normalized_state']; old_a=z['normalized_action']
    indices=sorted({lookup[r['family'],r['index']] for r in original})
    assert len(indices)==3584 and len({r['condition_sha256'] for r in original})==3840
    from scipy.spatial.transform import Rotation
    def matrices(x):
        u=x[...,:3].astype(float); u/=np.linalg.norm(u,axis=-1,keepdims=True)
        v=x[...,3:].astype(float); v-=np.sum(u*v,axis=-1,keepdims=True)*u; v/=np.linalg.norm(v,axis=-1,keepdims=True)
        return np.stack((u,v,np.cross(u,v)),axis=-2)
    numerical={}
    for arm,value in [('OLD',stat),('ROT',rot)]:
        proc=codec.processor(value); ss=[]; aa=[]; xyz=[]; so3=[]
        for i in indices:
            s,a,mask=codec.encode(proc,states[i],actions[i]); assert s.shape==(1,132) and a.shape==(40,132)
            assert s.dtype==a.dtype==np.float32 and np.isfinite(s).all() and np.isfinite(a).all()
            assert not s[:,10:].any() and not a[:,10:].any() and not mask[:,10:].any() and (mask[:,:10]==1).all()
            if arm=='OLD': assert np.array_equal(s[0,:10],old_s[i]) and np.array_equal(a[:,:10],old_a[i])
            else:
                assert np.array_equal(s[0,[0,1,2,9]],old_s[i,[0,1,2,9]])
                assert np.array_equal(a[:,[0,1,2,9]],old_a[i][:,[0,1,2,9]])
                assert max(abs(s[:,3:9]).max(),abs(a[:,3:9]).max())<=1+2e-6
            decoded=codec.decode(proc,a,states[i]); xyz.append(float(np.linalg.norm(decoded[:,:3]-actions[i,:,:3],axis=-1).max()))
            so3.append(float(Rotation.from_matrix(matrices(decoded[:,3:9])@matrices(actions[i,:,3:9]).swapaxes(-1,-2)).magnitude().max()))
            ss.append(s[0,:10]); aa.append(a[:,:10])
        assert max(xyz)<1e-5 and max(so3)<1e-4
        numerical[arm]=dict(sources=len(indices),xyz_roundtrip_max_m=max(xyz),so3_roundtrip_max_rad=max(so3),
            ranges={name:dict(min=np.min(x,axis=0).tolist(),max=np.max(x,axis=0).tolist(),abs_p99=float(np.quantile(abs(np.asarray(x)),.99))) for name,x in [('state',ss),('relative_action',aa)]})
    print('PASS_NUMERIC_BOTH',len(indices),flush=True)
    processors={arm:build(value,BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot') for arm,value in [('OLD',stat),('ROT',rot)]}
    fixtures=ROOT/'design/fit_fixtures'; fixtures.mkdir(parents=True,exist_ok=True)
    family_cache={}
    for r in fit:
        fid=r['family']; i=r['index']; j=r['frame_index']
        if fid not in family_cache:
            folder=RAW/'S2'/fid
            assert sha(folder/'proposals.npz')==r['proposal_sha256'] and sha(folder/'policy_frames.npz')==r['frame_file_sha256']
            family_cache[fid]=(np.load(folder/'proposals.npz'),np.load(folder/'policy_frames.npz'))
        z,f=family_cache[fid]; packet=dict(body_rgb=f['body_rgb'][j],wrist_rgb=f['wrist_rgb'][j],state_v1=z['state_v1'][i],language=r['language'])
        values={arm:batch(p,packet,z['raw_action'][i])['inputs'] for arm,p in processors.items()}
        assert identity({k:v for k,v in values['OLD'].items() if k not in ('action','action_mask')})==r['condition_sha256']
        assert values['OLD'].keys()==values['ROT'].keys()
        for key in values['OLD']:
            if key not in ('state','action'): assert identity(values['OLD'][key])==identity(values['ROT'][key]),key
        r['processed']={arm:dict(condition_sha256=identity({k:v for k,v in value.items() if k not in ('action','action_mask')}),
            labels_masks_sha256=identity({k:value[k] for k in ('action','action_mask')}),complete_input_sha256=identity(value)) for arm,value in values.items()}
        dst=fixtures/(r['input_id']+'.npz')
        if not dst.exists(): np.savez_compressed(dst,**{k:v for k,v in packet.items() if k!='language'},raw_action=z['raw_action'][i])
        r['fixture_file']=dst.name; r['fixture_sha256']=sha(dst)
    for z,f in family_cache.values(): z.close(); f.close()
    noises=read(PREP/'output/SEEN_TRAIN_112_MANIFEST.json')['existing_noise']
    for n in noises: assert sha(n['path'])==n['file_sha256']
    write(ROOT/'design/FIT112_FIRST2048.json',dict(status='SEALED_BEFORE_NEW_QUALITY',rows=fit,coverage=coverage,
        actual_new_consumption='PENDING_TRAINING_REPLAY',noise=noises,selection='quartile then first actual unit/micro/condition hash; fill same cell',source_plan_sha256=PLAN_SHA))
    write(ROOT/'design/EXACT_STATISTICS_DIFF.json',dict(allowed_only=True,changes=changes,source_unmodified=stat==frozen))
    write(ROOT/'evidence/CORELAB_CPU_ADMISSION.json',dict(status='PASS_ACTUAL_PROCESSOR_NUMERIC_AND_INPUT_BOUNDARIES',
        numerical=numerical,fixture_count=len(fit),original_conditions=3840,original_raw_sources=3584,microbatches=4096,
        source_statistics_sha256=STAT_SHA,plan_sha256=PLAN_SHA,model_queries=0,optimizer_updates=0,
        cross_host_labels='Physical bounds checked; Hopper actual byte replay remains mandatory',script_sha256=sha(__file__)))
    write(ROOT/'design/APPROVED_RUN_CARD.json',dict(authority='PI0268_R1_NORMALIZATION_AND_RELIABLE_CLOSED_LOOP_GO_V1',
        release=RELEASE,canonical=5957403226,launch_allowed=True,launch_preconditions=['HOPPER_CPU_ADMISSION','both real canaries PASS'],
        arms=['OLD','ROT'],units_per_arm=2048,microbatches_per_unit=2,source_plan_sha256=PLAN_SHA,source_indices='R_indices',
        seed=26741,unit_seed='sha256(PI0267:TRAIN:26741:{unit}) first8 big endian mod 2**31',
        contracts={arm:sha(ROOT/'contracts'/arm/'source/dataset/statistics.json') for arm in ('OLD','ROT')},
        donor=dict(repo='leoyuer/pi0246-gate-pass-campaign',revision='e4290e6ca387abd2de4337b5f12b9c72a5f85c99',prefix='arms/PI0264/L/checkpoint-2048',tree_sha256='2f61d8ed8c8fe8444b7ef23164cb4df2bcacd8a73c93f19be77fcdf459ef6fae'),
        optimizer=dict(name='AdamW',fresh=True,lr=1e-5,warmup=10,cosine=2048,weight_decay=1e-5,gradclip=1.),
        save=[1024,2048],quality=[0,2048],loss='native + .4 position + .1 increment + .1 rotation + .1 speed-margin',
        completed_update_cap=4384,training_head_cap=90000,inference_attempt_cap=24576,inference_head_cap=98304,
        new_virtual_starts_cap=208,new_virtual_queries_cap=19200,new_frame_pairs_cap=20000,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_CORELAB_CONTRACTS_AND_FIT',len(fit),flush=True)

if __name__=='__main__': main()
