"""TRAIN-only full-support schedule and saved-array adoption; no neural work."""
import argparse, collections, hashlib, json
from pathlib import Path

RELEASE='83af9099f3bab9a4ac08eaa85380c2cfd6351f34'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==v,p
    else:p.write_bytes((json.dumps(v,sort_keys=True)+'\n').encode())

def strata(rows):
    n=min(8,len(rows));q,r=divmod(len(rows),n);out=[];start=0
    for k in range(n):
        end=start+q+(k<r);out.append(rows[start:end]);start=end
    assert start==len(rows) and all(out)
    return out

def schedule(rows):
    seeds=sorted({r['seed'] for r in rows});assert len(seeds)==93
    pools={'F':list(range(10000,10016)),'B':[s for s in seeds if s not in range(10000,10016)]}
    assert len(pools['B'])==77
    groups={(s,g):sorted([r for r in rows if r['seed']==s and r['group']==g],key=lambda r:(r['anchor'],r['uid'])) for s in seeds for g in range(4)}
    assert all(len(v)>=2 for v in groups.values())
    ranked={k:[sorted(v,key=lambda r:hashlib.sha256(f"PI0278|27803|{k[0]}|{k[1]}|{r['uid']}".encode()).digest()) for v in strata(rs)] for k,rs in groups.items()}
    counts=collections.Counter();visits=collections.Counter();stream=[]
    for i in range(8192):
        g=i%4;pool='F' if (i//4)%2==0 else 'B';slot=counts[pool,g];counts[pool,g]+=1
        s=pools[pool][slot%len(pools[pool])];j=visits[s,g];visits[s,g]+=1;rs=groups[s,g]
        if j<2:r=rs[0 if j==0 else -1]
        else:
            bins=ranked[s,g];k=(j-2)%len(bins);cycle=(j-2)//len(bins);r=bins[k][cycle%len(bins[k])]
        stream.append(dict(unit=i//2+1,micro=i%2,micro_index=i,uid=r['uid'],group=g,pool=pool))
    assert counts=={(pool,g):1024 for pool in pools for g in range(4)}
    uidkey={r['uid']:k for k,rs in groups.items() for r in rs};bykey=collections.defaultdict(list)
    for v in stream:bykey[uidkey[v['uid']]].append(v['uid'])
    for (s,g),rs in groups.items():
        used=bykey[s,g]
        assert len(used)==(64 if s in pools['F'] else visits[s,g]) and len(used)>=13
        assert rs[0]['uid'] in used and rs[-1]['uid'] in used
        assert all(set(used)&{r['uid'] for r in b} for b in strata(rs))
    return stream,groups,pools

def exposure(stream,groups,proof):
    byuid={r['uid']:r for rs in groups.values() for r in rs};counts=collections.Counter(v['uid'] for v in stream);out=[]
    assert all(uid in byuid for uid in counts)
    for (s,g),rs in groups.items():
        used=[r for r in rs if counts[r['uid']]];anchors=[r['anchor'] for r in used]
        out.append(dict(seed=s,group=g,eligible=len(rs),visits=sum(counts[r['uid']] for r in used),unique_anchors=len(used),
            first=min(anchors) if anchors else None,last=max(anchors) if anchors else None,
            eligible_first=rs[0]['anchor'],eligible_last=rs[-1]['anchor'],full_H40_label_span=[min(anchors),max(anchors)+39] if anchors else None,
            stratum_unique=[sum(r['uid'] in counts for r in b) for b in strata(rs)],
            observed_20_28s=sum(counts[r['uid']] for r in rs if 400<=r['anchor']<560),observed_28_30s=sum(counts[r['uid']] for r in rs if 560<=r['anchor']<=600),
            distinct_condition_hashes=len({proof[r['uid']]['condition_sha256'] for r in used}),distinct_label_hashes=len({proof[r['uid']]['labels_masks_sha256'] for r in used})))
    return out

def build(old,out,journal):
    cardpath=old/'design/DATASET_CARD.json';pp=old/'design/PROCESSED_INPUTS.json';card=read(cardpath);proof={r['uid']:r for r in read(pp)['rows']}
    admitted=sorted(e['seed'] for e in card['episodes'] if e['split']=='TRAIN' and e['admitted']);assert len(admitted)==93
    rows=[r for r in card['rows'] if r['eligible'] and r['split']=='TRAIN' and r['seed'] in admitted];assert len(rows)==55893 and len({r['uid'] for r in rows})==len(rows)
    assert all(r['uid'] in proof for r in rows)
    stream,groups,pools=schedule(rows)
    membership=[dict(r,condition_sha256=proof[r['uid']]['condition_sha256'],labels_masks_sha256=proof[r['uid']]['labels_masks_sha256']) for r in sorted(rows,key=lambda r:r['uid'])]
    write(out/'ELIGIBLE_MEMBERSHIP.json',dict(card_sha256=sha(cardpath),proof_sha256=sha(pp),F=pools['F'],B=pools['B'],rows=membership))
    write(out/'TRAIN_PLAN.json',dict(release=RELEASE,seed=27803,updates=4096,noise_namespace='PI0278:COVER93:27803:{unit}',stream=stream,membership_sha256=sha(out/'ELIGIBLE_MEMBERSHIP.json')))
    panel=[]
    for s in pools['F'][:4]+pools['B'][:4]:
        selected=set()
        for g,target in [(0,None),(1,None),(2,None),(3,None),(3,400),(3,580)]:
            rs=[r for r in groups[s,g] if r['uid'] not in selected];assert rs
            r=min(rs,key=lambda r:(abs(r['anchor']-target),r['anchor'])) if target is not None else rs[0];selected.add(r['uid'])
            panel.append(dict(r,requested_anchor=target,condition_sha256=proof[r['uid']]['condition_sha256'],labels_masks_sha256=proof[r['uid']]['labels_masks_sha256']))
    assert len(panel)==len({r['uid'] for r in panel})==48
    write(out/'DIAGNOSTIC48.json',dict(rows=panel,TRAIN_only=True,new_model_queries=0,selection_uses_results=False))
    consumed=[json.loads(x) for x in journal.read_text().splitlines()];assert len(consumed)==8192 and [r['micro_index'] for r in consumed]==list(range(8192))
    for r in consumed:
        assert proof[r['uid']]['condition_sha256']==r['condition_sha256'] and proof[r['uid']]['labels_masks_sha256']==r['labels_masks_sha256']
    # Match the actual old chronological sampler, rather than assuming code was consumed.
    focus=pools['F'];oldgroups={k:v for k,v in groups.items() if k[0] in focus}
    for i,r in enumerate(consumed):
        g=i%4;cycle=i//4;s=focus[cycle%16];rs=groups[s,g];assert r['uid']==rs[(cycle//16)%len(rs)]['uid']
    write(out/'PREDECESSOR_ACTUAL_EXPOSURE.json',dict(status='PASS_ACTUAL_8192_CONSUMED_UIDS_AND_REVIEWED_CHRONOLOGICAL_RULE',journal_sha256=sha(journal),rows=exposure(consumed,groups,proof),prefetch_is_not_consumption=True,new_queries=0))
    write(out/'PROSPECTIVE_COVERAGE.json',dict(status='PASS_FROZEN_93x4_FULL_ENDPOINT_STRATUM_SUPPORT',rows=exposure(stream,groups,proof),actual_consumed=False,phase_pool_counts={f'{p}:{g}':sum(r['pool']==p and r['group']==g for r in stream) for p in pools for g in range(4)}))
    print(json.dumps(dict(status='PASS_ACTUAL_PREDECESSOR_AND_FROZEN_COVER93',stream=8192,panel=48,eligible=55893)),flush=True)

def saved_audit(old,raw,out):
    import numpy as np
    card=read(old/'design/DATASET_CARD.json');motion=[]
    for e in card['episodes']:
        if e['split']!='TRAIN' or not e['admitted']:continue
        f=Path(e['path']);obs=f/'observations.npz'
        with np.load(obs,allow_pickle=False) as z:
            state=z['state_v1'][400:601];rgb=z['base_camera_96'][400:601].astype(np.int16);wrist=z['hand_camera_96'][400:601].astype(np.int16)
            motion.append(dict(seed=e['seed'],observation_sha256=sha(obs),state_shape=list(state.shape),late_TCP_xyz_span_m=np.ptp(state[:,:3],axis=0).tolist(),late_opening_span=float(np.ptp(state[:,9])),
                base_RGB_adjacent_mean_abs=float(np.abs(np.diff(rgb,axis=0)).mean()),wrist_RGB_adjacent_mean_abs=float(np.abs(np.diff(wrist,axis=0)).mean()),
                duplicate_late_raw_states=len(state)-len({x.tobytes() for x in state}),not_a_substantive_state_diversity_proof=True))
    write(out/'LATE_SAVED_OBSERVATION_DIVERSITY.json',dict(rows=motion,new_queries=0,new_physics=0,definition='Actual EEF StateV1 xyz/opening range and unembedded adjacent pixel differences over20..30s; hashes alone do not imply distinct physical situations'))
    events=[]
    for seed in range(30000,30008):
        f=raw/'episodes'/f'TEST-{seed}-FOCUS4096'/'attempt-000';result=read(f/'RESULT.json')
        with np.load(f/'physics.npz') as z,np.load(f/'commands.npz') as c,np.load(f/'installed.npz') as inst:
            def first(a,dt):return float(np.flatnonzero(a)[0]*dt) if np.any(a) else None
            on=z['is_cubeA_on_cubeB'];static=z['is_cubeA_static'];grasp=z['is_cubeA_grasped'];success=z['success'];assert np.array_equal(success,on&static&~grasp)
            window=slice(2800,3001,5);parts={k:int((~v[window]).sum()) for k,v in [('on_top',on),('static',static),('not_grasped',~grasp)]}
            assert len(success[window])==41 and bool(success[window].all())==result['persistent_success']
            losses=np.flatnonzero(on[:-1]&~on[1:]);holds=c['ik_hold'];cmd=c['action'];tcp=z['tcp'][::5][:640];target=inst['installed_target_fk'];opening=z['qpos'][::5][:640,-2:].sum(axis=-1)
            events.append(dict(seed=seed,persistent=result['persistent_success'],qualification_subpredicate_false_counts=parts,first_native_success_s=first(success,.01),first_IK_hold_s=first(holds,.05),IK_holds=int(holds.sum()),on_top_loss_times_s=((losses+1)*.01).tolist(),
                raw_to_installed_xyz_mean_m=float(np.linalg.norm(cmd[:,:3]-target[:,:3,3],axis=-1).mean()),installed_to_actual_xyz_mean_m=float(np.linalg.norm(target[:,:3,3]-tcp[:,:3,3],axis=-1).mean()),
                raw_gripper_range=[float(cmd[:,9].min()),float(cmd[:,9].max())],measured_opening_range_m=[float(opening.min()),float(opening.max())],first_drop_after_lift_s=first((z['cubeA_pose'][:-1,2]-z['cubeA_pose'][0,2]>=.02)&(z['cubeA_pose'][1:,2]-z['cubeA_pose'][0,2]<.02),.01),result_sha256=sha(f/'RESULT.json'),physics_sha256=sha(f/'physics.npz'),causality='Chronological precedence only; no unique contact/IK cause inferred'))
    write(out/'PREDECESSOR_DEV_EVENT_AUDIT.json',dict(status='PASS_SAVED_NATIVE_SUBPREDICATES_AND_ACTUAL_INSTALLATION',rows=events,new_queries=0,new_physics=0))
    print('PASS_SAVED93_LATE_OBSERVATION_AND_DEV8_EVENT_AUDIT',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--journal',type=Path);p.add_argument('--raw',type=Path);a=p.parse_args()
    if a.journal:build(a.old,a.out,a.journal)
    if a.raw:saved_audit(a.old,a.raw,a.out)
