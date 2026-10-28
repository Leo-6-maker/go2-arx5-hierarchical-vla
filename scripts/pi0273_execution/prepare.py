"""Freeze the complete historical stream and the sole supplemental-source contrast."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0273'
RELEASE='b6e4e436dced30445a014721e2c0df268989244f'
STATS_SHA='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
SAMPLE_SHA='938d1514a372b48a506ab538d7aa206689bc73c7f9258af3c66b5274af9e5cbc'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    text=json.dumps(value,indent=2,sort_keys=True)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_text()==text,('FROZEN_FILE_CHANGED',str(p))
    else:p.write_text(text)
def seed_for(unit):
    assert 1<=unit<=5120
    return int.from_bytes(hashlib.sha256(f'PI0273:TRAIN:27303:{unit}'.encode()).digest()[:8],'big')%2**31
def key(row):return tuple(row[k] for k in ('family','task','command','language'))


def schedules(base,target):
    assert len(base)==8192 and len(target)==296 and len(set(target))==296
    pools=defaultdict(list)
    for i,r in enumerate(base):pools[key(r)].append(i)
    n=Counter();T=[];C=[]
    for j in range(2048):
        t=target[j%296];k=key(base[t]);pool=pools[k]
        c=pool[n[k]%len(pool)];n[k]+=1;T.append(t);C.append(c)
        assert key(base[c])==key(base[t])
    streams={}
    for arm,supp in (('BROAD',C),('TARGET',T)):
        rows=[]
        for b in range(1024):
            for slot,indices in [('BASE',range(8*b,8*b+8)),('SUPPLEMENT',supp[2*b:2*b+2])]:
                for i in indices:
                    ordinal=len(rows);unit=ordinal//2+1
                    rows.append(dict(ordinal=ordinal,unit=unit,micro=ordinal%2,source_index=i,stream=slot,seed=seed_for(unit)))
        assert len(rows)==10240 and [r['source_index'] for r in rows if r['stream']=='BASE']==list(range(8192))
        streams[arm]=rows
    assert Counter(Counter(T).values())=={7:272,6:24}
    return streams,T,C,pools


def main():
    previous=BASE/'PI0272/runtime/final-publication/FINAL_CLOSEOUT_RECEIPT.json';closed=read(previous)
    assert closed['execution']=='COMPLETE' and not closed['missing_scientific_requirements'] and closed['canonical']['id']==5981967270
    p0=BASE/'PI0270';audit=p0/'evidence/training_hopper/AGG/main/training/sample_audit.jsonl'
    assert sha(audit)==SAMPLE_SHA
    base=[json.loads(s) for s in audit.read_text().splitlines()]
    assert len(base)==8192 and [(r['unit'],r['micro']) for r in base]==[(i//2+1,i%2) for i in range(8192)]
    identities={'PAIRING_PLAN.json':'fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef',
        'RECOVERY_MAPPING.json':'53fe049e406f079e079d03af71e25c0ffd339b587fab7d8e5dc80bebe35fd633',
        'NEW_FIT192.json':'553d254cea5ad5c32d0d8acf615f2abfe222459ac10e9d0bb0e09b786884d226'}
    for name,digest in identities.items():assert sha(p0/'design'/name)==digest
    plan=read(p0/'design/PAIRING_PLAN.json');mapping={(r['unit'],r['micro']):r for r in read(p0/'design/RECOVERY_MAPPING.json')['rows']}
    for row in base:
        u=plan['units'][row['unit']-1];assert row['command']==u['order'][row['micro']]
        assert all(row[k]==u[k] for k in ('task','family','slot','template'))
        if row['slot']=='RECOVERY_INTERVENTION':
            r=mapping[row['unit'],row['micro']]
            assert row['condition_sha256']==r['condition_sha256'] and row['labels_masks_sha256']==r['labels_masks_sha256']
            assert row['actual_source_uid']==r['source_uid'] and row['source_sample_sha256']==r['sample_sha256']
            row['source_bucket']=r['bucket']
        else:assert row['slot']=='COMMON_NOMINAL';row['source_bucket']='COMMON_NOMINAL'
    late=BASE/'PI0272/design/FITTING_CARD.json';assert sha(late)=='02a77e4f30a7ddb6c2d4219bdc30418ff603b52b047568d1e0e48ff8161b6880'
    fit=read(late);target=[]
    for r in sorted(fit['rows'],key=lambda r:(r['source_mapping']['unit'],r['source_mapping']['micro'])):
        m=r['source_mapping'];i=2*(m['unit']-1)+m['micro'];b=base[i]
        assert b['condition_sha256']==r['condition_sha256'] and b['labels_masks_sha256']==r['labels_masks_sha256']
        assert key(b)==key(m);target.append(i)
    streams,T,C,pools=schedules(base,target)
    exposure={}
    for arm,stream in streams.items():
        counts=Counter((v['stream'],*key(base[v['source_index']]),base[v['source_index']]['source_bucket']) for v in stream)
        exposure[arm]=[dict(zip(('stream','family','task','command','language','bucket'),k),microbatches=n) for k,n in sorted(counts.items())]
    overlap=dict(supplement_slots=2048,identical_original_occurrence=sum(t==c for t,c in zip(T,C)),
        identical_processed_input_and_label=sum((base[t]['condition_sha256'],base[t]['labels_masks_sha256'])==(base[c]['condition_sha256'],base[c]['labels_masks_sha256']) for t,c in zip(T,C)),
        original_base_unique_input_label_pairs=len({(r['condition_sha256'],r['labels_masks_sha256']) for r in base}),
        target_unique_input_label_pairs=len({(base[i]['condition_sha256'],base[i]['labels_masks_sha256']) for i in T}),
        broad_unique_input_label_pairs=len({(base[i]['condition_sha256'],base[i]['labels_masks_sha256']) for i in C}))
    write(ROOT/'design/FITTING_CARD.json',dict(schema='PI0273_FULL_BASE_CONTEXT_MATCHED_REPLAY_V1',release=RELEASE,
        base=base,base_actual_audit_sha256=sha(audit),target_source_indices=target,supplemental_indices=dict(BROAD=C,TARGET=T),
        streams=streams,exposure=exposure,overlap=overlap,training_seed=27303,updates_per_arm=5120,
        context_pools=[dict(context=list(k),source_indices=v) for k,v in sorted(pools.items())],
        statistics_sha256=STATS_SHA,model_queries=0,new_teacher_labels=0,actual_training_not_yet_observed=True))
    cards={}
    for src,name in [('EVALUATION_CARD.json','MAIN_CARD.json'),('COMPONENT_CARD.json','COMPONENT_CARD.json'),('FIXED_OFFLINE_CARD.json','OFFLINE_CARD.json')]:
        path=p0/'design'/src;value=read(path)
        write(ROOT/'design'/name,dict(source_path=str(path),source_sha256=sha(path),frozen_original=value,
            new_points=['BROAD5120','TARGET5120'],release=RELEASE))
        cards[name]=sha(ROOT/'design'/name)
    support=[dict(r,noise=[n for n in r['noise'] if n['j'] in (0,1)]) for r in fit['rows']]
    assert len(support)==296 and all(len(r['noise'])==2 for r in support)
    early=read(p0/'design/NEW_FIT192.json')
    write(ROOT/'design/SUPPORT_CARD.json',dict(late=support,early=early['rows'],source_late_sha256=sha(late),source_early_sha256=identities['NEW_FIT192.json']))
    cards['SUPPORT_CARD.json']=sha(ROOT/'design/SUPPORT_CARD.json')
    write(ROOT/'design/EVALUATION_CARD.json',dict(schema='PI0273_FROZEN_ALL_ENDPOINT_PANELS_V1',release=RELEASE,cards=cards,
        points=['BROAD5120','TARGET5120'],per_point=dict(offline=968,early=192,late=592,main=64,component=32),
        caps=dict(quality_attempts=18432,quality_heads=73728,virtual_starts=208,frame_pairs=15360),
        planned_max_queries=17584,planned_virtual_starts=192,planned_frame_pairs=14080,
        main_duration_s=40,component_duration_s=8,max_main_queries=100,max_component_queries=20,
        final_only=True,old_valid_results_reused=True,new_teacher_labels=0,new_WBC_physics=0))
    authority=read(ROOT/'authority/CANONICAL.json');assert authority['id']==5982104756 and RELEASE in authority['body']
    old_resolution=BASE/'PI0272/evidence/PRECISION_CONTRACT_RESOLUTION.json'
    resolution=read(old_resolution);assert resolution['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY'
    write(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json',dict(resolution,continuation_release=RELEASE,
        continuation_canonical=5982104756,predecessor_resolution_sha256=sha(old_resolution)))
    original_mode=BASE/'PI0272/design/S4FIT_READOUT_R1.json'
    for arm in ('BROAD','TARGET'):
        mode=read(original_mode);assert mode['generated_nodes']==4
        mode.update(schema='PI0273_NATIVE_S4_READOUT_PHASE_CONTRACT_V1',release=RELEASE,arm=arm,mode='S4',
            inherited_readout_sha256=sha(original_mode),resolution_sha256=sha(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
        write(ROOT/'design'/f'{arm}_READOUT_R1.json',mode)
    write(ROOT/'evidence/A0_SOURCE_AND_PLAN.json',dict(status='PASS_CLOSED_PREDECESSOR_FULL_B_T_C_AND_FIXED_EVALUATION_PLAN',release=RELEASE,
        predecessor_closeout_sha256=sha(previous),actual_AGG_audit_sha256=sha(audit),fitting_card_sha256=sha(ROOT/'design/FITTING_CARD.json'),
        evaluation_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),overlap=overlap,base_occurrences_per_arm=8192,
        microbatches_per_arm=10240,actual_Hopper_encoding_and_canaries_still_required=True,model_queries=0,teacher_labels=0))
    print('PASS_FROZEN_COMPLETE_BASE_AND_CONTEXT_MATCHED_REPLAY',json.dumps(overlap),flush=True)


def self_check():
    base=[dict(family=i%24,task='M1' if i%2 else 'M3',command=i%2,language='exact '+str(i%5)) for i in range(8192)]
    streams,T,C,pools=schedules(base,list(range(296)))
    assert [r['stream'] for r in streams['TARGET'][:10]]==['BASE']*8+['SUPPLEMENT']*2
    assert all(key(base[t])==key(base[c]) for t,c in zip(T,C)) and T[:296]==list(range(296))
    assert streams['BROAD'][-1]['unit']==5120 and len(C)==2048
    print('PASS_CONSTRUCTED_FULL_BASE_CONTEXT_POOLS_SCHEDULE_AND_SEED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true')
    self_check() if p.parse_args().self_check else main()
