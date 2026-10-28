"""Only the authorized F sampler and exact full-parent continuation."""
import hashlib,json
from collections import defaultdict,Counter
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def focus_plan(rows):
    from data import order_key
    import numpy as np
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='TRAIN':groups[r['seed'],r['group']].append(r)
    seeds=sorted({s for s,p in groups});assert len(seeds)==93 and len(groups)==372
    focus=seeds[:16];others=seeds[16:];assert set(range(10000,10004))<=set(focus)
    streams={}
    for key,g in groups.items():
        g.sort(key=lambda r:(r['anchor'],r['uid']))
        strata=[list(x) for x in np.array_split(np.arange(len(g)),min(8,len(g))) if len(x)]
        orders=[sorted(x,key=lambda i:order_key(g[i]['uid'])) for x in strata]
        stream=[dict(g[0],stratum=0),dict(g[-1],stratum=len(strata)-1)]
        total=192 if key[0] in focus else 14
        for i in range(total-2):
            sid=i%len(orders);idx=orders[sid][(i//len(orders))%len(orders[sid])]
            stream.append(dict(g[idx],stratum=sid))
        streams[key]=stream
    visits=Counter();plan=[];counters=[0,0]
    for visit in range(4096):
        concentrated=visit%4!=3;pool=focus if concentrated else others;slot=0 if concentrated else 1
        seed=pool[counters[slot]%len(pool)];counters[slot]+=1
        for phase in range(4):
            key=seed,phase;row=dict(streams[key][visits[key]])
            row.update(index=len(plan),phase=phase);plan.append(row);visits[key]+=1
    assert len(plan)==16384 and sum(r['seed'] in focus for r in plan)==12288
    assert Counter(r['phase'] for r in plan)==dict.fromkeys(range(4),4096)
    assert all(visits[s,p]==192 for s in focus for p in range(4))
    assert all(13<=visits[s,p]<=14 for s in others for p in range(4))
    for key,g in groups.items():
        used=[r for r in plan if (r['seed'],r['phase'])==key]
        assert {g[0]['uid'],g[-1]['uid']}<={r['uid'] for r in used}
        assert len({r['stratum'] for r in used})==min(8,len(g))
    return plan


def validate(root,path):
    v=json.loads(path.read_text());assert v['branch']=='F' and v['launch_allowed'] is True
    assert v['authority_release']=='fdcde0e29f7eb689d0eea0f4dd20586309e77326'
    assert v['parent_step'] in (1024,2048) and v['formal_updates_before']>=2048
    assert v['full_baseline_development_qualified'] is False
    for p,h in v['source_bindings'].items():assert sha(p)==h,p
    panel=json.loads(Path(v['baseline_train_path']).read_text())
    assert len(panel['rows'])==4 and all(r['status']!='ENGINEERING_INCOMPLETE' for r in panel['rows'])
    if panel['persistent_count']>=3:
        dev=json.loads(Path(v['baseline_dev_path']).read_text())
        assert len(dev['rows'])==8 and dev['persistent_count']<6
    ledger=[json.loads(x) for x in (Path(root)/'learning/completed_updates.jsonl').read_text().splitlines()]
    assert sum(r['work_kind']=='formal' for r in ledger)==v['formal_updates_before']
    assert sum(r['arm']=='ACTION2048' and r['phase']=='main' for r in ledger)==2048
    assert not any(r['arm']=='ACTION_F2048' for r in ledger)
    fit=json.loads(Path(v['fit_path']).read_text())
    assert len(fit['rows'])==48 and fit['TRAIN_denominator']==24 and fit['fit_adequate'] is False
    assert sha(v['plan_path'])==v['plan_sha256']
    parent=Path(v['parent']);assert parent.is_relative_to(Path(root)/'learning/ACTION2048/main')
    assert json.loads((parent/'PI0282_RECOVERY.json').read_text())['cursor']==v['parent_step']*8
    for name,h in v['parent_files'].items():assert sha(parent/name)==h,name
    assert not (Path(root)/'learning/ACTION_F2048/main/PROCESS_resume0_stop2048.json').exists()
    return v


def restore_model(model,v):
    import torch
    from safetensors import safe_open
    from scope import digest
    expected=json.loads(Path(v['parent_export']).read_text())['full_FP32_master_state'];seen=set()
    parameters=dict(model.named_parameters())
    with torch.no_grad():
        for shard in sorted(Path(v['parent']).glob('*.safetensors')):
            with safe_open(shard,framework='pt',device='cpu') as f:
                for n in f.keys():
                    x=f.get_tensor(n);p=parameters[n]
                    assert x.dtype==p.dtype and x.shape==p.shape
                    assert dict(dtype=str(x.dtype),shape=list(x.shape),sha256=digest(x))==expected[n]
                    p.copy_(x);assert digest(p)==expected[n]['sha256'];seen.add(n)
    assert seen==set(parameters)==set(expected)


def restore_optimizer(optimizer,v,output):
    import torch
    from learning import tensor_tree
    saved=torch.load(Path(v['parent'])/'optimizer.pt',map_location='cpu',weights_only=False)
    expected=tensor_tree(saved['state']);optimizer.load_state_dict(saved)
    assert tensor_tree(optimizer.state_dict()['state'])==expected,'PARENT_ADAM_MOMENTS_OR_STEPS_CHANGED'
    assert all(int(s['step'])==v['parent_step'] for s in optimizer.state.values())
    for i,g in enumerate(optimizer.param_groups):g['lr']=g['initial_lr']=(5e-5 if i==0 else 5e-6)
    Path(output).write_text(json.dumps(dict(status='PASS_EXACT_PARENT_MODEL_ADAM_MOMENTS_STEPS_CARRY',parent=v['parent'],parent_files=v['parent_files'],new_warmup=32,new_cosine_steps=2048,only_scheduler_reset=True),indent=2))


def self_check():
    rows=[dict(split='TRAIN',seed=s,group=p,anchor=a,uid=f'{s}-{p}-{a}') for s in range(10000,10093) for p in range(4) for a in range(17)]
    one=focus_plan(rows);assert one==focus_plan(rows)
    assert len({r['uid'] for r in one})>1000
    print('PASS_F_PLAN_75_25_PHASE_FIRST_LAST_STRATA_AND_DETERMINISM; model/optimizer/physics=0')


if __name__=='__main__':self_check()
