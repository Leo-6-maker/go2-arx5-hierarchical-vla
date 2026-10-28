"""Freeze only the authorized contingency's existing TRAIN membership and visits."""
import argparse,collections,hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
NAMESPACE='PI0277:FOCUS:R2:27703:{unit}'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def freeze(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert json.loads(path.read_text())==value
    else:
        tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,sort_keys=True)+'\n');tmp.replace(path)

def build(root,destination):
    binding=json.loads((root/'design/INPUT_BINDING.json').read_text());old=Path(binding['old_root'])
    cardpath=old/'design/DATASET_CARD.json';proofpath=old/'design/PROCESSED_INPUTS.json'
    card=json.loads(cardpath.read_text());proof={r['uid']:r for r in json.loads(proofpath.read_text())['rows']}
    seeds=sorted(e['seed'] for e in card['episodes'] if e['split']=='TRAIN' and e['admitted'])[:16]
    assert seeds==list(range(10000,10016))
    eligible=[r for r in card['rows'] if r['split']=='TRAIN' and r['eligible'] and r['seed'] in seeds]
    groups={(s,g):sorted((r for r in eligible if r['seed']==s and r['group']==g),key=lambda r:(r['anchor'],r['uid'])) for s in seeds for g in range(4)}
    assert all(groups.values()) and len({r['uid'] for r in eligible})==len(eligible)
    assert all(r['uid'] in proof for r in eligible)
    stream=[]
    for i in range(8192):
        group=i%4;cycle=i//4;seed=seeds[cycle%16];anchors=groups[seed,group];row=anchors[(cycle//16)%len(anchors)]
        stream.append(dict(unit=i//2+1,micro=i%2,micro_index=i,group=group,uid=row['uid']))
    counts=collections.Counter(r['group'] for r in stream);assert counts==dict.fromkeys(range(4),2048)
    byuid={r['uid']:r for r in eligible}
    assert all(collections.Counter(byuid[r['uid']]['seed'] for r in stream if r['group']==g)==dict.fromkeys(seeds,128) for g in range(4))
    membership=[dict(uid=r['uid'],seed=r['seed'],anchor=r['anchor'],group=r['group'],condition_sha256=proof[r['uid']]['condition_sha256'],labels_masks_sha256=proof[r['uid']]['labels_masks_sha256'],command_sha256=r['command_sha256'],observation_sha256=r['observation_sha256']) for r in sorted(eligible,key=lambda r:r['uid'])]
    freeze(destination/'ELIGIBLE_MEMBERSHIP.json',dict(seeds=seeds,rows=membership,card_sha256=sha(cardpath),processed_inputs_sha256=sha(proofpath),new_data=0,new_processor_fits=0))
    freeze(destination/'TRAIN_PLAN.json',dict(release='1a20200ff066ef5a003650110cd20106f9a4bf64',arm_order=['FOCUS'],seed=27703,noise_namespace=NAMESPACE,
        updates=4096,stream=stream,source_plan_sha256=sha(root/'design/TRAIN_PLAN.json'),actual_noise_byte_comparison='REQUIRED_IN_CONTINUOUS_SPLIT_CANARY',membership_sha256=sha(destination/'ELIGIBLE_MEMBERSHIP.json')))
    freeze(destination/'PREPARATION.json',dict(status='CONTINGENCY_PREPARED_NOT_TRIGGERED',launch_allowed=False,training_queries=0,quality_queries=0,
        CPU_processor_calls=0,teacher_calls=0,physics_steps=0,groups=dict(counts),seeds=seeds,eligible_members=len(membership),used_members=len({r['uid'] for r in stream}),
        plan_sha256=sha(destination/'TRAIN_PLAN.json'),trigger='Original1024/2048 no-improvement tuple or primary4096 not development-qualified; unmeasured1024 cannot prove the earlier trigger'))
    print(json.dumps(dict(status='CONTINGENCY_PREPARED_NOT_TRIGGERED',launch_allowed=False,eligible_members=len(membership),used_members=len({r['uid'] for r in stream}),groups=dict(counts))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root,a.out)
