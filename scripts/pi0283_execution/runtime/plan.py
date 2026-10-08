"""Frozen MORE/EVENT visits; original UID membership and labels stay unchanged."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def key(uid):return hashlib.sha256(('PI0283-28203-'+uid).encode()).hexdigest()


def build(rows,prior,commands):
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='TRAIN':groups[r['seed'],r['group']].append(r)
    seeds=sorted({s for s,p in groups});assert len(seeds)==93 and len(groups)==372
    offsets=Counter((r['seed'],r['phase']) for r in prior)
    assert len(prior)==16384 and set(offsets)==set(groups)
    streams={};events={};audit=[]
    for pair,g in sorted(groups.items()):
        g.sort(key=lambda r:(r['anchor'],r['uid']))
        strata=[list(x) for x in np.array_split(np.arange(len(g)),min(8,len(g))) if len(x)]
        orders=[sorted(x,key=lambda i:key(g[i]['uid'])) for x in strata]
        streams[pair]=(g,orders)
        command=commands[pair[0]]
        transitions=np.flatnonzero(np.abs(np.diff(command[:,9]))>.10)+1
        centers=sorted(set(transitions.tolist()+[g[0]['anchor'],g[-1]['anchor'],0]))
        events[pair]=sorted([r for r in g if any(abs(r['anchor']-c)<=7 for c in centers)],key=lambda r:(r['anchor'],key(r['uid'])))
        audit.append(dict(seed=pair[0],phase=pair[1],continued_visit_offset=offsets[pair],event_anchors=len(events[pair]),transitions=transitions.tolist(),boundaries=centers))
    def ordinary(pair,visit):
        g,orders=streams[pair];slot=visit%44
        if slot<2:return dict(g[0 if slot==0 else -1],stratum=0 if slot==0 else len(orders)-1)
        sid=(slot-2)%len(orders);i=orders[sid][((visit-2)//len(orders))%len(orders[sid])]
        return dict(g[i],stratum=sid)
    plans={}
    for stage,draws in [('MORE',16384),('EVENT',8192)]:
        counts=Counter();event_counts=Counter();plan=[]
        for cycle in range(draws//4):
            seed=seeds[cycle%93]
            for phase in range(4):
                pair=seed,phase;visit=offsets[pair]+counts[pair]+(more_counts[pair] if stage=='EVENT' else 0)
                use_event=stage=='EVENT' and (counts[pair]+seeds.index(seed)+phase)%2==0
                pool=events[pair]
                row=dict(pool[event_counts[pair]%len(pool)],stratum='EVENT') if use_event and pool else ordinary(pair,visit)
                row.update(index=len(plan),phase=phase,stage_namespace='PI0283-'+stage,continued_visit=visit,event_draw=bool(use_event and pool),event_pool_empty=bool(use_event and not pool))
                plan.append(row);counts[pair]+=1
                if use_event:event_counts[pair]+=1
        assert len(plan)==draws and Counter(r['phase'] for r in plan)==dict.fromkeys(range(4),draws//4)
        assert max(counts.values())-min(counts.values())<=1
        if stage=='MORE':
            more_counts=counts.copy()
            for pair,(g,orders) in streams.items():
                used=[r for r in plan if (r['seed'],r['phase'])==pair]
                assert {g[0]['uid'],g[-1]['uid']}<={r['uid'] for r in used}
                assert set(r['stratum'] for r in used)==set(range(len(orders)))
        else:assert sum(event_counts.values())==4096
        plans[stage]=plan
    return plans,audit


def freeze(root,card):
    root=Path(root);old=root.parent/'PI0282';data=root/'data';data.mkdir(parents=True,exist_ok=True)
    card=json.loads(Path(card).read_text());rows=[r for r in card['rows'] if r['eligible']]
    prior=[json.loads(x) for x in (old/'data/PLAN16384.jsonl').read_text().splitlines()]
    commands={s:np.load(old/f'data/episodes/TRAIN-{s}/action.npy',mmap_mode='r') for s in sorted({r['seed'] for r in prior})}
    plans,audit=build(rows,prior,commands)
    bindings={}
    for name,plan in plans.items():
        path=data/(name+'_PLAN.jsonl');payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in plan)
        if path.exists():assert path.read_text()==payload
        else:path.write_text(payload)
        bindings[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),draws=len(plan),unique_uids=len({r['uid'] for r in plan}),event_draws=sum(r['event_draw'] for r in plan))
    value=dict(status='FROZEN_SAME_93_TRAIN_LABELS_PHASES_CONTINUED_OFFSETS',plans=bindings,groups=audit,new_model_queries=0,new_physics=0)
    (data/'PLAN_AUDIT.json').write_text(json.dumps(value,indent=2)+'\n');return value


def self_check():
    rows=[dict(split='TRAIN',seed=s,group=p,anchor=p*100+a,uid=f'{s}-{p}-{a}') for s in range(10000,10093) for p in range(4) for a in range(30)]
    prior=[dict(rows[(cycle%93)*120+phase*30],phase=phase) for cycle in range(4096) for phase in range(4)]
    commands={s:np.zeros((640,10)) for s in range(10000,10093)}
    one,a=build(rows,prior,commands);two,b=build(rows,prior,commands);assert one==two and a==b
    assert one['MORE'][0]['continued_visit']==45 and len(one['EVENT'])==8192
    print('PASS_CONTINUED_VISITS_PHASE_FIRST_LAST_STRATA_EVENT_50_PERCENT; model/optimizer/physics=0')


if __name__=='__main__':
    import sys
    if sys.argv[1]=='self-check':self_check()
    else:print(json.dumps(freeze(sys.argv[1],sys.argv[2])['plans']))
