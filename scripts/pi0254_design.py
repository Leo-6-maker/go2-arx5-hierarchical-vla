"""Materialize PI0254 cohorts and schedules before observing model outputs."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json

AUTH = '7e5a4aae69912ba8e593d50d313f288abc16a914'


def digest(text):
    return hashlib.sha256(text.encode('ascii')).hexdigest()


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered(values, namespace):
    return sorted(values, key=lambda value: (digest(namespace+'|'+value),value))


def validate(cohorts, schedules):
    ids = {k:{r['source_episode_id'] for r in v} for k,v in cohorts.items()}
    rollouts = {k:{r['source_episode_id'].split(':obs-')[0] for r in v} for k,v in cohorts.items()}
    assert {k:len(v) for k,v in ids.items()} == {'O64':64,'P64':64,'N512':512,'E64':64}
    assert not (rollouts['O64']&rollouts['P64'] or rollouts['O64']&rollouts['N512'] or rollouts['P64']&rollouts['N512'])
    assert len(rollouts['P64'])==64 and len(rollouts['N512'])==128 and len(rollouts['E64'])==64
    assert ids['E64']<=ids['N512']
    for name in ('P64','E64'):
        assert Counter((r['target_family'],r['phase']) for r in cohorts[name]) == Counter({(a,p):4 for a in 'abcd' for p in range(4)})
    assert Counter((r['target_family'],r['phase']) for r in cohorts['N512']) == Counter({(a,p):32 for a in 'abcd' for p in range(4)})
    for arm,rows in schedules.items():
        assert len(rows)==4608 and [r['optimizer_step'] for r in rows]==list(range(1,4609))
        counts=Counter(r['source_episode_id'] for r in rows)
        assert set(counts)==ids['O64']|ids['N512']
        assert all(counts[k]==(8 if arm=='U576' else 24) for k in ids['O64'])
        assert all(counts[k]==(8 if arm=='U576' else 6) for k in ids['N512'])
        for source in counts:
            assert [r['visit'] for r in rows if r['source_episode_id']==source]==list(range(1,counts[source]+1))
        if arm=='R576':assert all(r['pool']==('O64' if i%3==0 else 'N512') for i,r in enumerate(rows))
    u={(r['source_episode_id'],r['visit']):r for r in schedules['U576']}
    shared=0
    for r in schedules['R576']:
        match=u.get((r['source_episode_id'],r['visit']))
        if match:
            assert match['rng_key']==r['rng_key'] and match['rng_seed']==r['rng_seed'];shared+=1
    assert shared==3584


def build(full, old, old16):
    by_rollout=defaultdict(list)
    for row in full:
        assert row['split']=='TRAIN' and row['target_family'] in 'abcd' and row['phase'] in range(4)
        assert row['source_rollout_id']==row['source_episode_id'].split(':obs-')[0]
        by_rollout[row['source_rollout_id']].append(row)
    assert len(full)==9792 and len(by_rollout)==612
    by_id={r['source_episode_id']:r for r in full}
    assert len(by_id)==9792
    cohorts={'O64':[dict(by_id[r['source_episode_id']],old16=r['source_episode_id'] in old16) for r in old]}
    assert len(old16)==16 and old16<=set(r['source_episode_id'] for r in old)
    excluded={r['source_rollout_id'] for r in cohorts['O64']}
    candidates={}

    def pick_window(rollout,phase):
        rows=[r for r in by_rollout[rollout] if r['phase']==phase]
        assert rows, (rollout,phase)
        times={r['source_episode_id']:Fraction(r['observation_timestamp']) for r in rows}
        twice_midpoint=min(times.values())+max(times.values())
        row=min(rows,key=lambda r:(abs(2*times[r['source_episode_id']]-twice_midpoint),r['source_episode_id']))
        return dict(row,selection_phase_midpoint_s=float(twice_midpoint/2),selection_window_start_s=row['observation_timestamp'],old16=False)

    for name,n in [('P64',16),('N512',32),('E64',16)]:
        selected=[]
        for task in 'abcd':
            eligible=[k for k,v in by_rollout.items() if v[0]['target_family']==task and (k in {r['source_rollout_id'] for r in cohorts['N512']} if name=='E64' else k not in excluded)]
            assert all({r['phase'] for r in by_rollout[k]}==set(range(4)) for k in eligible)
            order=ordered(eligible,f'PI0254_COHORT_V1|{name}|{task}')
            candidates[name+'|'+task]=[dict(source_rollout_id=k,sha256_order_key=digest(f'PI0254_COHORT_V1|{name}|{task}|{k}')) for k in order]
            assert len(order)>=n,(name,task,len(order))
            for i,k in enumerate(order[:n]):
                selected.extend(pick_window(k,p) for p in (range(4) if name=='N512' else [i%4]))
        cohorts[name]=selected
        if name!='E64':excluded.update(r['source_rollout_id'] for r in selected)
    support=sorted(cohorts['O64']+cohorts['N512'],key=lambda r:r['source_episode_id'])
    index={r['source_episode_id']:i for i,r in enumerate(support)}
    schedules={}
    for arm in ('U576','R576'):
        def cycles(pool,count):
            ids=[r['source_episode_id'] for r in (support if pool=='ALL576' else cohorts[pool])]
            return [source for cycle in range(1,count+1) for source in ordered(ids,f'PI0254_SCHEDULE_V1|25402|{arm}|{pool}|cycle={cycle}')]
        if arm=='U576':sequence=cycles('ALL576',8)
        else:
            a,b=cycles('O64',24),cycles('N512',6)
            sequence=[source for i in range(1536) for source in (a[i],b[2*i],b[2*i+1])]
        visits=Counter(); rows=[];old_ids={r['source_episode_id'] for r in cohorts['O64']}
        for step,source in enumerate(sequence,1):
            visits[source]+=1;key=f'PI0254|25402|stage=SUPPORT576|source={source}|visit={visits[source]}'
            seed=int.from_bytes(hashlib.sha256(key.encode('ascii')).digest()[:8],'big')&((1<<63)-1)
            rows.append(dict(by_id[source],optimizer_step=step,fixed_index=index[source],pool='O64' if source in old_ids else 'N512',visit=visits[source],rng_key=key,rng_seed=seed))
        schedules[arm]=rows
    validate(cohorts,schedules)
    return cohorts,support,schedules,candidates


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    full=json.loads((a.inputs/'full_cohort.json').read_text())['rows'];old=json.loads((a.inputs/'O64_manifest.json').read_text())['files']
    times=json.loads((a.inputs/'original_observation_times.json').read_text())['rows']
    by_source={r['source_episode_id']:r for r in times}
    assert len(by_source)==len(full)==9792
    for row in full:
        actual=by_source[row['source_episode_id']]
        assert actual['episode_index']==row['episode_index']
        row['observation_timestamp']=actual['observation_timestamp']
    old16=set(json.loads((a.inputs/'G_score_card.json').read_text())['support']['old16_source_ids'])
    cohorts,support,schedules,candidates=build(full,old,old16)
    a.output.mkdir(parents=True,exist_ok=True)
    def write(name,payload):
        data=json.dumps(payload,indent=2,sort_keys=True)+'\n';target=a.output/name
        if target.exists():assert target.read_text()==data, target
        else:target.write_text(data)
    for name,rows in cohorts.items():write(name+'.json',dict(schema='PI0254_COHORT_V1',rows=rows,role=name,authority_commit=AUTH))
    write('support576.json',dict(schema='PI0254_TRAIN_SUPPORT_V1',rows=[dict(r,fixed_index=i) for i,r in enumerate(support)]))
    write('selection_candidates.json',candidates)
    for arm,rows in schedules.items():write(arm+'_schedule.json',dict(schema='PI0254_CONSUMED_UPDATE_SCHEDULE_V1',arm=arm,rows=rows))
    # CPU poison checks exercise the actual disjointness and schedule assertions.
    for kind in ('rollout_overlap','extra_update','wrong_visit'):
        cc=json.loads(json.dumps(cohorts));ss=json.loads(json.dumps(schedules))
        if kind=='rollout_overlap':cc['P64'][0]=cc['O64'][0]
        elif kind=='extra_update':ss['U576'].append(ss['U576'][-1])
        else:ss['R576'][0]['visit']=2
        try:validate(cc,ss)
        except AssertionError:pass
        else:raise AssertionError('Poison accepted: '+kind)
    report=dict(status='PASS_DESIGN_CPU_CARDINALITY_DISJOINTNESS_EXPOSURE_AND_POISON',authority_commit=AUTH,
        input_hashes={n:hash_file(a.inputs/n) for n in ('full_cohort.json','O64_manifest.json','G_score_card.json','original_observation_times.json')},
        cohort_sizes={k:len(v) for k,v in cohorts.items()},rollout_sizes={k:len({r['source_rollout_id'] for r in v}) for k,v in cohorts.items()},
        jitter_coverage={k:dict(Counter(r['source_rollout_id'].split('_j',1)[1].split('_s',1)[0] for r in v)) for k,v in cohorts.items()},
        shared_source_visit_draws=3584,training_updates_per_arm=4608,planned_queries=1856,query_cap=1920,inference_head_cap=7680,
        new_model_queries=0,new_optimizer_updates=0,deployment_selected_checkpoint='NONE')
    write('design_cpu_receipt.json',report);print(json.dumps(report))


if __name__=='__main__':main()
