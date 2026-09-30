"""Independent computation-path CPU replay of all fixed PI0263 saved arrays."""
import csv
import argparse
import itertools
import json
from collections import defaultdict

import numpy as np
import pi0263_eval as evaluation
from pi0263_plan import read,write,sha
from pi0260_r3_inputs import batch,identity
from pi0260_r3_eval_cpu import metrics

ROOT=evaluation.ROOT
OUT=ROOT
r3=evaluation.r3


def rmse(a,b):
    return float(np.sqrt(np.mean(np.sum((a[:,:3].astype(np.float64)-b[:,:3].astype(np.float64))**2,axis=1))))


def save_csv(name,rows):
    path=OUT/name
    assert rows and not path.exists()
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def numeric(values):
    xyz=[r['xyz_rmse_m'] for r in values];so3=[r['so3_rmse_rad'] for r in values]
    out=dict(n=len(values),median_xyz_m=float(np.median(xyz)),p95_xyz_m=float(np.percentile(xyz,95)),
        median_so3_rad=float(np.median(so3)),reject=sum(r['reject'] for r in values))
    out['numeric_pass']=out['median_xyz_m']<=.020 and out['p95_xyz_m']<=.035 and out['median_so3_rad']<=.050 and out['reject']==0
    return out


def replay(donor_only=False):
    global OUT
    import pi0242_train_infer_bridge as bridge
    import pi0228_train_calib_checkpoint_selection as selector
    from gr00t.data.embodiment_tags import EmbodimentTag
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    card=read(ROOT/'UNION_EVAL_CARD.json');assert len(card['rows'])==1152 and len(card['aliases'])==1320
    points=['P2000'] if donor_only else card['points']
    OUT=ROOT/'donor_cpu' if donor_only else ROOT
    OUT.mkdir(parents=True,exist_ok=True)
    proc=evaluation.crossed.processor()
    checked,conditions,source_cache={}, {}, {}
    provenance=[];fresh_queries=set();cached_rows=0
    for point in points:
        binding=card['donor_binding'] if point=='P2000' else read(ROOT/'model_bindings'/f'{point}.json')
        for row in card['rows']:
            uid=row['uid'];cell=ROOT/'eval'/point/'cells'/uid;rec=read(cell/'record.json')
            assert rec['status']=='VALID' and rec['row']==row and rec['point']==point
            assert rec['model_binding']==binding and rec['eval_card_sha256']==sha(ROOT/'UNION_EVAL_CARD.json')
            assert sha(cell/'tensors.npz')==rec['tensor_sha256']
            if point!='P2000':
                if rec.get('forward_reused'):
                    reused=rec['forward_reused'];other=ROOT/'eval'/point/'cells'/reused['uid']
                    prior=read(other/'record.json')
                    assert sha(other/'record.json')==reused['record_sha256'] and sha(other/'tensors.npz')==reused['tensor_sha256']
                    assert prior['model_binding']==binding
                    assert all(prior['row']['identities'][k]==row['identities'][k] for k in ('condition_sha256','noise_sha256','state_sha256'))
                    cached_rows+=1
                else:
                    fresh_queries.add((point,uid))
            if uid not in source_cache:
                source_cache[uid]=evaluation.source(row)
            packet,expert=source_cache[uid]
            key=row['identities']['condition_sha256']
            if key not in conditions:
                conditions[key]=identity(batch(proc,packet)['inputs'])
            assert conditions[key]==key
            with np.load(cell/'tensors.npz',allow_pickle=False) as z:
                assert z['raw'].shape==(1,40,132) and z['decoded'].shape==(1,40,10)
                assert z['trace_x'].shape==z['trace_v'].shape==(4,1,40,132)
                assert np.isfinite(z['raw']).all() and np.isfinite(z['decoded']).all()
                assert np.array_equal(z['expert'],expert) and np.array_equal(z['state'],packet['state_v1'])
                assert evaluation.digest(z['noise'])==row['identities']['noise_sha256']
                decoded=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,z['raw'],packet['state_v1'])
                assert np.max(np.abs(decoded-z['decoded']))<=1e-7
                out=z['decoded'][0].copy()
                if point!='P2000' and rec.get('forward_reused'):
                    with np.load(other/'tensors.npz',allow_pickle=False) as old:
                        assert all(np.array_equal(old[k],z[k]) for k in ('raw','decoded','trace_x','trace_v','trace_t'))
                if point=='P2000':
                    with np.load(evaluation.Path(row['donor_cell'])/'tensors.npz',allow_pickle=False) as old:
                        assert np.array_equal(old['raw'],z['raw']) and np.array_equal(old['decoded'],z['decoded'])
            observed=metrics(out,expert)
            assert all(abs(observed[k]-rec['metrics'][k])<=1e-9 for k in observed)
            state=packet['state_v1']
            gate=TrajectorySafetyGateV1().evaluate(selector._build_chunk(out[None],chunk_id=f'cpu-{point}-{uid}'),now=0.,
                achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0])
            assert gate.decision.value==rec['gate']['decision']
            assert [x.code for x in gate.issues]==[x['code'] for x in rec['gate']['issues']]
            checked[point,uid]=(out,expert,dict(**observed,reject=gate.decision.value=='REJECT',gate=gate.decision.value))
            provenance.append(dict(point=point,uid=uid,record_sha256=sha(cell/'record.json'),tensor_sha256=rec['tensor_sha256']))
    assert len(checked)==1152*len(points)

    legacy_rows,cross_rows,predictions,references=[],[],{},{}
    motion=read(r3.R3/'motion2_eval_cases.json')['cases']
    for point in points:
        for alias in card['aliases']:
            out,expert,observed=checked[point,alias['uid']]
            src=alias['source_row'];noise=src['noise']
            if alias['panel']=='crossed':
                command=src['command'];family=src['family']
                rec,_,labels=r3.sequence(f'{family}-M1-c{1-command}')
                j=next(j for j,r in enumerate(rec['rows']) if r['index']==0)
                own,opposite=rmse(out,expert),rmse(out,labels[j]);margin=opposite-own
                cross_rows.append(dict(point=point,uid=alias['uid'],panel_index=alias['index'],split=src['split'],family=family,
                    command=command,color=src['color'],template=src['template'],noise=noise,own_m=own,opposite_m=opposite,
                    margin_m=margin,correct=margin>0,accepted_correct=margin>0 and not observed['reject'],**observed))
                predictions[point,family,src['template'],noise,command]=out
                references[family,command]=expert
            else:
                case=src if src['cohort']=='TRAIN_ENTRY' else motion[src['case_index']] if src['cohort']=='Motion2' else {}
                task=case.get('task','O64');kind=case.get('kind','TRAIN_ENTRY' if src['cohort']=='TRAIN_ENTRY' else 'O64')
                family=case.get('family',case.get('sequence','').rsplit(f'-{task}-c',1)[0]);command=case.get('command',-1)
                own,opposite,margin=None,None,None
                if kind in ('SEMANTIC','PARAPHRASE','TRAIN_ENTRY'):
                    rec,_,labels=r3.sequence(f'{family}-{task}-c{1-command}')
                    j=next(j for j,r in enumerate(rec['rows']) if r['index']==0)
                    own,opposite=rmse(out,expert),rmse(out,labels[j]);margin=opposite-own
                legacy_rows.append(dict(point=point,uid=alias['uid'],panel_index=alias['index'],task=task,kind=kind,
                    family=family,command=command,noise=noise,own_m=own,opposite_m=opposite,margin_m=margin,
                    correct=margin>0 if margin is not None else None,**observed))
    assert len(legacy_rows)==680*len(points) and len(cross_rows)==640*len(points)
    save_csv('LEGACY_PANEL_ROWS.csv',legacy_rows);save_csv('CROSSED_PANEL_ROWS.csv',cross_rows)
    legacy_groups,cross_groups,qualification=[],[],[]
    for point in points:
        numerics=[];semantics=[]
        for task in ('M1','M3'):
            for noise in ('B0','B1'):
                vals=[r for r in legacy_rows if (r['point'],r['task'],r['kind'],r['noise'])==(point,task,'KIN-VAL',noise)]
                assert len(vals)==16
                group=dict(point=point,task=task,kind='KIN-VAL',noise=noise,**numeric(vals))
                numerics.append(group['numeric_pass']);legacy_groups.append(group)
                for kind in ('SEMANTIC','PARAPHRASE'):
                    vals=[r for r in legacy_rows if (r['point'],r['task'],r['kind'],r['noise'])==(point,task,kind,noise)]
                    assert len(vals)==12
                    group=dict(point=point,task=task,kind=kind,noise=noise,n=12,
                        correct=sum(r['correct'] for r in vals),reject=sum(r['reject'] for r in vals),
                        c0=sum(r['correct'] for r in vals if r['command']==0),c1=sum(r['correct'] for r in vals if r['command']==1))
                    group['semantic_pass']=group['correct']>=10
                    semantics.append(group['semantic_pass']);legacy_groups.append(group)
        passes=[]
        for split in ('TRAIN','KIN-VAL'):
            for template in ('T0','T1','V0','V1'):
                for noise in ('B0','B1'):
                    vals=[r for r in cross_rows if (r['point'],r['split'],r['template'],r['noise'])==(point,split,template,noise)]
                    assert len(vals)==(60 if split=='TRAIN' else 20)
                    families=defaultdict(list)
                    for r in vals:families[r['family']].append(r)
                    group=dict(point=point,split=split,template=template,noise=noise,**numeric(vals),
                        correct=sum(r['correct'] for r in vals),c0=sum(r['correct'] for r in vals if r['command']==0),
                        c1=sum(r['correct'] for r in vals if r['command']==1),
                        both_commands_correct=sum(all(r['correct'] for r in v) for v in families.values()),
                        ties=sum(r['margin_m']==0 for r in vals))
                    if split=='KIN-VAL':
                        group['robustness_pass']=group['correct']>=17 and group['numeric_pass'];passes.append(group['robustness_pass'])
                    cross_groups.append(group)
        qualification.append(dict(point=point,legacy_full_quality_pass=all(numerics) and all(semantics),
            crossed_M1_robustness_pass=all(passes),combined_pass=all(numerics) and all(semantics) and all(passes)))
    write(OUT/'QUALIFICATION.json',dict(points=qualification,legacy_groups=legacy_groups,crossed_groups=cross_groups,
        synthetic_research_candidate='NONE' if donor_only else next((p for p in ('F2000','X2000') if next(q for q in qualification if q['point']==p)['combined_pass']),'NONE'),
        historical_PI0261='FULL_QUALIFICATION_FAIL_UNCHANGED',historical_PI0262='COMPLETE_MIXED_UNCHANGED',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    groups=defaultdict(list)
    for r in cross_rows:groups[r['point'],r['split'],r['template'],r['color'],r['command'],r['noise']].append(r)
    save_csv('COLOR_COMMAND_GROUPS.csv',[dict(point=p,split=s,template=t,color=c,command=k,noise=n,
        count=len(v),correct=sum(r['correct'] for r in v),reject=sum(r['reject'] for r in v)) for (p,s,t,c,k,n),v in sorted(groups.items())])
    save_csv('LEGACY_REGRESSION_GROUPS.csv',[dict(point=p,task=t,kind=k,noise=n,**numeric(v))
        for (p,t,k,n),v in sorted(_group(legacy_rows,('point','task','kind','noise')).items())])
    flips=[]
    indexed={(r['point'],r['family'],r['command'],r['template'],r['noise']):r for r in cross_rows}
    for point,family,command,noise in sorted({(r['point'],r['family'],r['command'],r['noise']) for r in cross_rows}):
        for a,b in itertools.combinations(('T0','T1','V0','V1'),2):
            left,right=indexed[point,family,command,a,noise],indexed[point,family,command,b,noise]
            flips.append(dict(point=point,split=left['split'],family=family,command=command,noise=noise,
                from_template=a,to_template=b,from_correct=left['correct'],to_correct=right['correct'],
                correct_to_wrong=left['correct'] and not right['correct'],wrong_to_correct=not left['correct'] and right['correct']))
    save_csv('WORDING_FLIPS.csv',flips)
    decomposition=[]
    for point,family,template,noise in sorted({k[:4] for k in predictions}):
        refs=[references[family,c] for c in (0,1)];d=rmse(*refs);assert d>0
        errors=[[rmse(predictions[point,family,template,noise,c],a) for a in refs] for c in (0,1)]
        s0,s1=[(v[0]**2-v[1]**2)/d**2 for v in errors];g,b=(s1-s0)/2,(s0+s1)/2
        both=s0<0<s1;assert both==(g>abs(b))
        decomposition.append(dict(point=point,split=indexed[point,family,0,template,noise]['split'],family=family,
            template=template,noise=noise,reference_separation_m=d,s0=s0,s1=s1,g=g,b=b,
            both_commands_correct=both,category='both_correct' if both else 'nonpositive_ordering' if g<=0 else 'positive_ordering_bias_dominates'))
    save_csv('TARGET_CHOICE_DECOMPOSITION.csv',decomposition)
    noises=_group(decomposition,('point','split','family','template'));noise_flips=[]
    for (p,s,f,t),vals in sorted(noises.items()):
        vals={r['noise']:r for r in vals};a,b=vals['B0'],vals['B1']
        noise_flips.append(dict(point=p,split=s,family=f,template=t,B0_category=a['category'],B1_category=b['category'],
            c0_boundary_crossing=(a['s0']<0)!=(b['s0']<0),c1_boundary_crossing=(a['s1']>0)!=(b['s1']>0)))
    save_csv('NOISE_BOUNDARY_CROSSINGS.csv',noise_flips)
    attempts=[] if donor_only else [json.loads(r) for r in (ROOT/'attempts.jsonl').read_text().splitlines()]
    heads=[] if donor_only else [json.loads(r) for r in (ROOT/'head_forwards.jsonl').read_text().splitlines()]
    assert len(attempts)<=2432 and len(heads)<=9728
    assert [r['ordinal'] for r in attempts]==list(range(1,len(attempts)+1))
    assert [r['ordinal'] for r in heads]==list(range(1,len(heads)+1))
    assert all(r['point'] in ('F2000','X2000') for r in attempts+heads)
    assert {(r['point'],r['uid']) for r in attempts}==fresh_queries
    reserve=len(attempts)-len(fresh_queries);assert 0<=reserve<=128
    write(OUT/'CPU_REPLAY.json',dict(status='PASS_INDEPENDENT_COMPUTATION_PATH_CPU_DONOR_1152_REPLAY' if donor_only else 'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_3456_REPLAY',
        reviewer='same sole producer; independent computational path, not external reviewer',rows=len(checked),
        reused_donor_rows=1152,processed_conditions=len(conditions),new_attempts=len(attempts),heads=len(heads),
        invalid_only_reserve_used=reserve,new_endpoint_exact_forward_reuses=cached_rows,
        provenance=provenance,code_sha256=sha(__file__),
        eval_card_sha256=sha(ROOT/'UNION_EVAL_CARD.json'),selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status='PASS_PI0263_CPU_REPLAY',rows=len(checked),qualification=qualification)))


def _group(rows,keys):
    result=defaultdict(list)
    for r in rows:result[tuple(r[k] for k in keys)].append(r)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor-only',action='store_true');replay(p.parse_args().donor_only)
