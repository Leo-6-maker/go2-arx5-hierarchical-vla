"""Matched cases and local fields; no averages substitute for the original gates."""
import csv,json
from collections import Counter
from quality_gate import ROOT,RAW,read,sha
from quality_worker import write
from reuse import table

def saved_table(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            elif v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=json.loads(v)
                except ValueError:pass
    return rows

def cases(rows,before,after,label):
    key=lambda r:(r['family'],r['task'],r['command'],r['backend'])
    a={key(r):bool(r[label]) for r in rows if r['point']==before};b={key(r):bool(r[label]) for r in rows if r['point']==after}
    assert a.keys()==b.keys() and len(a)==64
    return [dict(family=k[0],task=k[1],command=k[2],backend=k[3],before=before,after=after,label=label,
        before_pass=a[k],after_pass=b[k],change='RETAINED' if a[k] and b[k] else 'LOST' if a[k] else 'GAINED' if b[k] else 'BOTH_FAIL') for k in a]

def main():
    out=RAW/'delivery';cpu=read(out/'CPU_REPLAY.json');assert cpu['status']=='PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES'
    rows=saved_table(out/'VIRTUAL192.csv');assert sha(out/'VIRTUAL192.csv')==cpu['tables']['VIRTUAL192.csv']['sha256']
    changes=[r for before in ('AGG4096','BROAD5120') for label in ('success','reliable') for r in cases(rows,before,'EQUIL5120',label)]
    strata=[dict(before=before,label=label,task=task,backend=backend,counts=dict(Counter(r['change'] for r in changes if (r['before'],r['label'],r['task'],r['backend'])==(before,label,task,backend))))
        for before in ('AGG4096','BROAD5120') for label in ('success','reliable') for task in ('M1','M3') for backend in ('E0','E1')]
    field=[];field_sources=[];matrices=[]
    for point in ('BROAD5120','EQUIL5120'):
        proof=read(out/f'FIELD_{point}.json');assert proof['observed_rows']==proof['registered_rows']==2496
        path=out/f'FIELD_{point}.csv';assert sha(path)==proof['table_sha256'];field+=saved_table(path);matrices+=proof['matrices']
        field_sources.append(dict(point=point,source_sha256=sha(out/f'FIELD_{point}.json'),summary=proof['summary']))
    index={(r['point'],r['pose_id'],r['j']):r for r in field};pairs=[]
    keys={(r['pose_id'],r['j']) for r in field if r['point']=='BROAD5120'};assert len(keys)==2496 and len(index)==4992
    metrics=('output_XYZ_m','output_SO3_rad','signed_rotation_increment_rad','rotation_perpendicular_increment_rad','coupled_position_increment_m',
        'input_to_next_scaled_error_ratio','first_XYZ_fit_m','first_SO3_fit_rad','prefix_XYZ_RMSE_m','prefix_SO3_RMSE_rad','full_XYZ_RMSE_m','full_SO3_RMSE_rad')
    for pid,j in sorted(keys):
        a,b=[index[p,pid,j] for p in ('BROAD5120','EQUIL5120')];assert a['input_z']==b['input_z'] and a['teacher_z_0p4']==b['teacher_z_0p4']
        row=dict(pose_id=pid,j=j,**{k:a[k] for k in ('family','task','command','variant')},before_gate=a['gate'],after_gate=b['gate'],
            before_actual_seen=a['actual_full_training_occurrences']>0,after_actual_seen=b['actual_full_training_occurrences']>0)
        for k in metrics:
            x,y=a.get(k),b.get(k);row['before_'+k]=x;row['after_'+k]=y;row['delta_'+k]=None if x is None or y is None else y-x
        pairs.append(row)
    old_onset=read(ROOT/'evidence/A0_ONSET_ANALYSIS.json');assert old_onset['status']=='PASS_ALL128_SAVED_ONSETS_AND_OBSERVED_VISIBILITY'
    new_onset=read(ROOT/'evidence/EQUIL_ONSET_ANALYSIS.json');assert new_onset['episodes']==64
    original_onsets=saved_table(out/'A0_ONSET128.csv');new_onsets=saved_table(out/'EQUIL_ONSET64.csv');assert len(original_onsets)==128 and len(new_onsets)==64
    for proof in (old_onset,new_onset):
        for name,digest in proof['tables'].items():assert sha(out/name)==digest
    old_diagnosis=[r for r in table('PI0273','FAILURE_AND_POST_SUCCESS_DIAGNOSIS128.csv') if r['point']=='BROAD5120']
    diagnoses=old_diagnosis+read(out/'TRANSITION_ANALYSIS.json')['episodes'];assert len(diagnoses)==128
    failure_groups=[]
    for point in ('BROAD5120','EQUIL5120'):
        for task in ('M1','M3'):
            own=[r for r in diagnoses if r['point']==point and r['task']==task];assert len(own)==32
            groups=Counter()
            for r in own:
                xyz='FINAL_WINDOW_XYZ_EXCEEDED' in r['flags'];rot='FINAL_WINDOW_SO3_EXCEEDED' in r['flags']
                label='BOTH' if xyz and rot else 'XYZ_ONLY' if xyz else 'SO3_ONLY' if rot else 'NEITHER';groups[label]+=1
            failure_groups.append(dict(point=point,task=task,counts=dict(groups),incomplete_window=sum(not r['final_window_complete'] for r in own),
                exit_orders=dict(Counter(r['exit_order'] for r in original_onsets+new_onsets if r['point']==point and r['task']==task))))
    for name,data in (('MATCHED_CASE_GAIN_LOSS256.csv',changes),('PAIRED_FIELD2496.csv',pairs)):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in data for k in r}));w.writeheader();w.writerows(data)
    write(out/'INTERVENTION_COMPARISON.json',dict(status='COMPLETE_FIXED_LOCAL_FIELD_AND_ORIGINAL_START_COMPARISON',matched_case_strata=strata,
        field_sources=field_sources,final_window_and_onset=failure_groups,tables={n:sha(out/n) for n in ('MATCHED_CASE_GAIN_LOSS256.csv','PAIRED_FIELD2496.csv')},
        local_expansion_not_stability_certificate=True,local_improvement_not_full_task_qualification=True,draws_not_independent_scenes=True,
        shared_original_gate_scorer_executor=True,model_queries=0,teacher_calls=0,renders=0,physics=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('COMPLETE_FIXED_LOCAL_AND_CASE_COMPARISON',flush=True)
def self_check():
    rows=[dict(point=p,family=str(i),task=t,command=c,backend=b,reliable=i%4 in allowed)
        for p,allowed in (('BROAD5120',(0,1)),('EQUIL5120',(0,2))) for t in ('M1','M3') for c in (0,1) for b in ('E0','E1') for i in range(8)]
    assert Counter(r['change'] for r in cases(rows,'BROAD5120','EQUIL5120','reliable'))==dict(RETAINED=16,LOST=16,GAINED=16,BOTH_FAIL=16)
    print('PASS_MATCHED_CASE_GAIN_LOSS_RETENTION_DENOMINATORS',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');self_check() if p.parse_args().self_check else main()
