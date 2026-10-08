"""Continue the sealed EVENT streams once; both arms consume these exact visits."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from prepare_core import write,sha

def freeze(root,old,dataset_card):
    root,old=Path(root),Path(old);target=root/'data/PAIRED_PLAN16384.jsonl'
    assert not target.exists(),'Reuse frozen common plan'
    rows=[r for r in json.loads(Path(dataset_card).read_text())['rows'] if r['eligible'] and r['split']=='TRAIN']
    groups=defaultdict(list)
    for row in rows:groups[row['seed'],row['group']].append(row)
    prior=[json.loads(x) for x in (old/'data/EVENT_PLAN.jsonl').read_text().splitlines()]
    counts_prior=Counter((r['seed'],r['phase']) for r in prior)
    event_prior=Counter((r['seed'],r['phase']) for r in prior if r['event_draw'])
    offsets={p:max(r['continued_visit'] for r in prior if (r['seed'],r['phase'])==p)+1 for p in groups}
    seeds=sorted({s for s,p in groups});assert len(seeds)==93 and len(groups)==372 and len(prior)==8192
    key=lambda uid:hashlib.sha256(('PI0283-28203-'+uid).encode()).hexdigest()
    streams={};events={}
    for pair,g in groups.items():
        g.sort(key=lambda r:(r['anchor'],r['uid']))
        strata=[list(x) for x in np.array_split(np.arange(len(g)),min(8,len(g))) if len(x)]
        orders=[sorted(x,key=lambda i:key(g[i]['uid'])) for x in strata];streams[pair]=(g,orders)
        command=np.load(old/f'data/episodes/TRAIN-{pair[0]}/action.npy',mmap_mode='r')
        transitions=np.flatnonzero(abs(np.diff(command[:,9]))>.10)+1
        centers=sorted(set(transitions.tolist()+[g[0]['anchor'],g[-1]['anchor'],0]))
        events[pair]=sorted([r for r in g if any(abs(r['anchor']-c)<=7 for c in centers)],key=lambda r:(r['anchor'],key(r['uid'])))
    counts=Counter();event_counts=Counter();plan=[]
    for cycle in range(4096):
        seed=seeds[cycle%93]
        for phase in range(4):
            pair=seed,phase;visit=offsets[pair]+counts[pair]
            use_event=(counts_prior[pair]+counts[pair]+seeds.index(seed)+phase)%2==0
            pool=events[pair]
            if use_event and pool:row=dict(pool[(event_prior[pair]+event_counts[pair])%len(pool)],stratum='EVENT')
            else:
                g,orders=streams[pair];slot=visit%44
                if slot<2:row=dict(g[0 if slot==0 else -1],stratum=0 if slot==0 else len(orders)-1)
                else:
                    sid=(slot-2)%len(orders);i=orders[sid][((visit-2)//len(orders))%len(orders[sid])];row=dict(g[i],stratum=sid)
            row.update(index=len(plan),phase=phase,stage_namespace='PI0284_SHARED',continued_visit=visit,event_draw=bool(use_event and pool),event_pool_empty=bool(use_event and not pool))
            plan.append(row);counts[pair]+=1
            if use_event:event_counts[pair]+=1
    assert len(plan)==16384 and Counter(r['phase'] for r in plan)==dict.fromkeys(range(4),4096)
    assert sum(r['event_draw'] for r in plan)==8192 and max(counts.values())-min(counts.values())<=1
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in plan))
    audit=dict(status='FROZEN_ONE_EXTENDED_EVENT_FAMILY_PLAN_FOR_BOTH_ARMS',path=str(target),sha256=sha(target),parent_EVENT_plan_sha256=sha(old/'data/EVENT_PLAN.jsonl'),dataset_card_sha256=sha(dataset_card),draws=16384,formal_draws_two_arms=32768,unique_UIDs=len({r['uid'] for r in plan}),TRAIN_episodes=93,phase_counts=dict(Counter(r['phase'] for r in plan)),critical_event_draws=8192,ordinary_draws=8192,continued_group_offsets={f'{s}:{p}':v for (s,p),v in sorted(offsets.items())},labels_statistics_unchanged=True,new_native_forwards=0,new_physics=0)
    write(root/'data/PAIRED_PLAN_AUDIT.json',audit);return audit

if __name__=='__main__':
    import sys
    print(json.dumps(freeze(*sys.argv[1:])))
