"""Frozen old/recovery schedule and genuine suffix codec proofs; CPU only."""
import argparse,ast,collections,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from pilot import ROOT,RAW,RELEASE,read,sha,write,freeze
OLD=ROOT.with_name('PI0276')
STATS='b3da3d500d514ce3d1ce83001de30846d9ab76d699dc48831db8e4367ce98572'


def schedule(rows):
    old={(s,g):sorted([r for r in rows if r['pool']=='OLD' and r['seed']==s and r['group']==g],key=lambda r:r['anchor']) for s in range(10000,10016) for g in range(4)}
    recovery={(s,f,c):sorted([r for r in rows if r['pool']=='RECOVERY' and (r['seed'],r['fork'],r['category'])==(s,f,c)],key=lambda r:hashlib.sha256(f"PI0279|27903|{s}|{f}|{c}|{r['uid']}".encode()).digest()) for s in range(10000,10004) for f in sorted({r['fork'] for r in rows if r['pool']=='RECOVERY' and r['seed']==s}) for c in range(3)}
    assert all(old.values()) and all(recovery.values())
    from collections import Counter
    visits=Counter();stream=[]
    for i in range(8192):
        rank=i//2
        if i%2==0:
            g=rank%4;s=10000+(rank//4)%16;rs=old[s,g];j=visits[s,g,'OLD'];visits[s,g,'OLD']+=1
            if j<2:r=rs[0 if j==0 else -1]
            else:
                bins=np.array_split(np.arange(len(rs)),min(8,len(rs)));b=(j-2)%len(bins);cycle=(j-2)//len(bins)
                ranked=sorted((rs[k] for k in bins[b]),key=lambda r:hashlib.sha256(f"PI0279|27903|{s}|OLD|{g}|{r['uid']}".encode()).digest());r=ranked[cycle%len(ranked)]
        else:
            s=10000+rank%4;j=rank//4;forks=sorted({f for seed,f,c in recovery if seed==s});z=j%(3*len(forks));c=z%3;f=forks[(z//3+z%3)%len(forks)]
            rs=recovery[s,f,c];n=visits[s,f,c,'RECOVERY'];visits[s,f,c,'RECOVERY']+=1;r=rs[n%len(rs)]
        stream.append(dict(unit=i//2+1,micro=i%2,micro_index=i,uid=r['uid'],group=r['group'],pool=r['pool']))
    mapping={r['uid']:r for r in rows};exposure=[mapping[v['uid']] for v in stream]
    assert Counter(r['group'] for r in exposure if r['pool']=='OLD')=={g:1024 for g in range(4)}
    assert Counter(r['seed'] for r in exposure if r['pool']=='RECOVERY')=={s:1024 for s in range(10000,10004)}
    for s in range(10000,10004):
        own=[r for r in exposure if r['pool']=='RECOVERY' and r['seed']==s]
        for field in ('fork','category'):
            counts=Counter(r[field] for r in own);assert max(counts.values())-min(counts.values())<=1
        assert any(r['anchor']>=560 for r in own),'FROZEN_LATE_HOLD_EXPOSURE_MISSING'
    return stream


def build():
    assert read(ROOT/'evidence/PILOT_VERDICT.json')['launch_allowed']
    prior=read(OLD/'design/DATASET_CARD.json');proof=read(OLD/'design/PROCESSED_INPUTS.json')
    processor_identity=sha(OLD/'evidence/DATA_ADMISSION.json')
    for r in read(OLD/'evidence/DATA_ADMISSION.json')['modules']:assert sha(r['path'])==r['sha256']
    rows=[dict(r,pool='OLD') for r in prior['rows'] if r['split']=='TRAIN' and r['seed'] in range(10000,10016) and r['eligible']]
    episodes=[dict(e,pool='OLD') for e in prior['episodes'] if e['split']=='TRAIN' and e['seed'] in range(10000,10016)];assert len(episodes)==16 and all(e['admitted'] for e in episodes)
    excluded=[]
    for q in read(ROOT/'evidence/PILOT_VERDICT.json')['rows']:
        qual=q['branches']['OPEN_CLEAR'];folder=Path(qual['path']);rec=read(folder/'RESULT.json')
        for n,h in rec['files'].items():assert sha(folder/n)==h,(folder,n)
        if not qual['positive']:excluded.append(qual);continue
        with np.load(folder/'commands.npz') as z:a=z['action'];assert a.shape==(640,10) and np.isfinite(a).all()
        with np.load(folder/'observations.npz') as z:
            assert z['state_v1'].shape==(640,10) and np.isfinite(z['state_v1']).all()
            for k in ('base_camera_96','hand_camera_96'):assert z[k].shape==(640,96,96,3) and z[k].dtype==np.uint8
        assert all(np.isfinite(z).all() for z in [a])
        identity=hashlib.sha256((rec['files']['commands.npz']+rec['files']['observations.npz']+STATS+processor_identity).encode()).hexdigest()[:20]
        seed=q['seed'];fork=q['fork'];episodes.append(dict(seed=seed,fork=fork,split='TRAIN',path=str(folder),pool='RECOVERY',admitted=True,result_sha256=sha(folder/'RESULT.json')))
        for anchor in range(fork,601):
            j=anchor-fork;category=0 if j<8 else 1 if j<32 else 2
            rows.append(dict(uid=f'PI0279-{seed}-f{fork}-a{anchor}-{identity}',seed=seed,fork=fork,category=category,group=3,split='TRAIN',pool='RECOVERY',episode=str(folder),anchor=anchor,eligible=True,reasons=[],command_sha256=rec['files']['commands.npz'],observation_sha256=rec['files']['observations.npz']))
    stream=schedule(rows);assert len({r['uid'] for r in rows})==len(rows)
    freeze(ROOT/'design/DATASET_CARD.json',dict(episodes=episodes,rows=rows,excluded=excluded,statistics_sha256=STATS,processor_identity=processor_identity,positive_prefix_or_recorded_labels=False))
    freeze(ROOT/'design/TRAIN_PLAN.json',dict(release=RELEASE,seed=27903,updates=4096,noise_namespace='PI0279:CLEAR:27903:{unit}',stream=stream))
    panel=[]
    for seed in range(10000,10004):
        chosen=set()
        for group,target in [(0,None),(1,None),(2,None),(3,None),(3,400),(3,580)]:
            rs=[r for r in rows if r['pool']=='OLD' and r['seed']==seed and r['group']==group and r['uid'] not in chosen];assert rs
            r=min(rs,key=lambda r:(abs(r['anchor']-target),r['anchor'])) if target is not None else min(rs,key=lambda r:r['anchor']);chosen.add(r['uid']);panel.append(r)
        forks=sorted({r['fork'] for r in rows if r['pool']=='RECOVERY' and r['seed']==seed});chosen=set()
        for j in range(6):
            category=j%3;fork=forks[(j//3+j%3)%len(forks)]
            rs=[r for r in rows if r['pool']=='RECOVERY' and (r['seed'],r['fork'],r['category'])==(seed,fork,category) and r['uid'] not in chosen];assert rs
            r=min(rs,key=lambda r:hashlib.sha256(('PI0279|DIAGNOSTIC48|'+r['uid']).encode()).digest());chosen.add(r['uid']);panel.append(r)
    assert len(panel)==len({r['uid'] for r in panel})==48
    freeze(ROOT/'design/DIAGNOSTIC48.json',dict(rows=panel,uses_scores=False,new_queries=0))
    freeze(ROOT/'design/INHERITED_PROOFS.json',dict(rows=[r for r in proof['rows'] if r['uid'] in {r['uid'] for r in rows if r['pool']=='OLD'}],statistics_sha256=STATS))
    print('PASS_FROZEN_GENUINE_RECOVERY_AND_8192_VISITS',len(rows),flush=True)


def codec():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    base=ROOT.parent;oldraw=RAW.parents[1]/'PI0260/R2'
    sys.path[:0]=[str(oldraw/'R3'),str(oldraw/'runtime'),str(oldraw/'qa'),str(base/'PI0250/runtime/reference-317a/src'),str(base/'PI0250/runtime/reference-317a/scripts'),'/home/liuyu-nfs/Projects/isaac-gr00t',str(ROOT/'project-src')]
    import torch,pi0260_r2_codec as c
    from pi0260_r3_inputs import build as processor,batch,identity
    from scipy.spatial.transform import Rotation
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available()
    # No neural inference is admissible in the CPU data path.
    stock=torch.nn.Module.__call__
    def no_model(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return stock(self,*a,**kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    torch.nn.Module.__call__=no_model
    stats=OLD/'contracts/TASK/source/dataset/statistics.json';assert sha(stats)==STATS
    p=ROOT/'contracts/TASK/source/dataset/statistics.json';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.write_bytes(stats.read_bytes())
    assert sha(p)==STATS;proc=processor(read(stats),base/'PI0250/inputs/frozen_v1/cosmos_snapshot');raw=c.processor(read(stats))
    node=next(n for n in ast.parse((OLD/'code/source_worker.py').read_text()).body if getattr(n,'name',None)=='pose_gate');scope=dict(ROOT=ROOT,np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'qualified-pose-gate','exec'),scope);gate=scope['pose_gate']
    card=read(ROOT/'design/DATASET_CARD.json');proof=read(ROOT/'design/INHERITED_PROOFS.json')['rows'];duplicates=collections.defaultdict(set)
    for r in proof:duplicates[r['condition_sha256']].add(r['labels_masks_sha256'])
    ledger=ROOT/'runtime/cpu_items.jsonl';count=sum(1 for _ in ledger.open()) if ledger.exists() else 0
    for e in card['episodes']:
        if e['pool']!='RECOVERY':continue
        folder=Path(e['path']);rows=[r for r in card['rows'] if r['pool']=='RECOVERY' and r['episode']==str(folder)]
        with np.load(folder/'commands.npz') as z:commands=z['action'].astype(np.float32)
        with np.load(folder/'observations.npz') as z:obs={k:z[k].copy() for k in ('state_v1','base_camera_96','hand_camera_96')}
        for r in rows:
            j=r['anchor'];state=obs['state_v1'][j];action=commands[j:j+40];assert action.shape==(40,10)
            s,a,m=c.encode(raw,state,action);decoded=c.decode(raw,a,state)
            xyz=float(np.linalg.norm(decoded[:,:3]-action[:,:3],axis=-1).max());grip=float(np.abs(decoded[:,9]-action[:,9]).max())
            angle=float(Rotation.from_matrix(rot6d_rows_to_matrix(decoded[:,3:9])@rot6d_rows_to_matrix(action[:,3:9]).transpose(0,2,1)).magnitude().max())
            assert xyz<=1e-5 and angle<=1e-4 and grip<=1e-6
            assert s.shape==(1,132) and a.shape==m.shape==(40,132) and not s[:,10:].any() and not a[:,10:].any() and np.all(m[:,:10]==1) and not m[:,10:].any()
            packet=dict(body_rgb=obs['base_camera_96'][j],wrist_rgb=obs['hand_camera_96'][j],state_v1=state,language='Stack the red cube on the green cube and release it.')
            for purpose in ('SUPERVISED_CODEC','NO_LABEL_INPUT_IDENTITY'):
                count+=1;assert count<=65536
                with ledger.open('a') as f:f.write(json.dumps(dict(ordinal=count,index=j,uid=r['uid'],route='REF_W0',purpose=purpose,pid=os.getpid()))+'\n');f.flush();os.fsync(f.fileno())
            v=batch(proc,packet,action)['inputs'];condition={k:v for k,v in v.items() if k not in ('action','action_mask')};assert identity(condition)==identity(batch(proc,packet)['inputs'])
            assert v['state'].dtype==v['action'].dtype==torch.float32
            before=gate(action,state,j*.05,r['uid']);after=gate(decoded,state,j*.05,r['uid']);assert before['decision']==after['decision']!='REJECT'
            item=dict(uid=r['uid'],condition_sha256=identity(condition),labels_masks_sha256=identity({k:v[k] for k in ('action','action_mask')}),XYZ_m=xyz,SO3_rad=angle,gripper=grip,normalized_state_abs_max=float(np.abs(s).max()),normalized_action_abs_max=float(np.abs(a).max()),gate=before['decision'])
            proof.append(item);duplicates[item['condition_sha256']].add(item['labels_masks_sha256'])
        print('ACTUAL_RECOVERY_CODEC',e['seed'],e['fork'],len(rows),flush=True)
    freeze(ROOT/'design/PROCESSED_INPUTS.json',dict(rows=proof,statistics_sha256=STATS))
    freeze(ROOT/'evidence/DATA_ADMISSION.json',dict(status='PASS_ACTUAL_FIXED_RECOVERY_INPUTS_AND_UNCHANGED_CODEC',launch_allowed=True,rows=len(proof),processor_calls=count,statistics_sha256=STATS,input_fields=['body_rgb','wrist_rgb','state_v1','language'],privileged_fields_in_inputs=False,identical_conditions_with_multiple_labels=sum(len(v)>1 for v in duplicates.values()),multimodality_not_automatically_codec_failure=True,new_models=0,new_physics=0))


def check():
    rows=[dict(pool='OLD',seed=s,group=g,anchor=i,uid=f'old-{s}-{g}-{i}') for s in range(10000,10016) for g in range(4) for i in range(16)]
    rows += [dict(pool='RECOVERY',seed=s,fork=f,category=c,group=3,anchor=560+i,uid=f'new-{s}-{f}-{c}-{i}') for s in range(10000,10004) for f in ([160,320] if s==10001 else [160,320,480]) for c in range(3) for i in range(8)]
    a=schedule(rows);assert a==schedule(list(reversed(rows))) and len(a)==8192
    assert sum(r['pool']=='OLD' for r in a)==4096
    print('PASS_FIXED_BALANCED_ALL_FORK_CATEGORY_PAIRS_AND_ORDER')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','codec','check']);a=p.parse_args();globals()[a.stage]()
