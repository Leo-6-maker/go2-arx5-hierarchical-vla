"""All fixed saved-output panels, native S4 CPU replay and unchanged qualifications."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from quality_worker import BASE,ROOT,read,sha,write,admitted
from reuse import table

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')
POINTS=('AGG4096','BROAD5120','TARGET5120')
METRICS=('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad',
    'suffix_xyz_rmse_m','suffix_so3_rmse_rad','prefix_endpoint_error_m','first_actual_position_bias_m',
    'prefix_xyz_mse_m2','first_xyz_error_m','first_so3_error_rad','prefix_endpoint_so3_error_rad','direction_cosine','signed_teacher_direction_progress_m')

def helpers():
    sys.path.insert(0,str(BASE/'PI0270/code'))
    path=BASE/'PI0270/code/quality_report.py'
    assert sha(path)=='f7b2760b11d25ba6173b2b00b880f615f35c4b777cc869a87d7f6ab7a15c071b'
    spec=importlib.util.spec_from_file_location('pi0270_report_helper',path)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    prior.ROOT=ROOT;prior.RAW=RAW
    scope=prior.helpers()
    scope['read']=lambda p:read(ROOT/'design/RUNTIME_COMPONENT_CARD.json' if Path(p)==ROOT/'design/COMPONENT_CARD.json' else p)
    sys.path.insert(0,str(BASE/'PI0271/code'))
    spec=importlib.util.spec_from_file_location('pi0271_measure_helper',BASE/'PI0271/code/report.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    import pi0243_staged_learning_to_sim as stage43
    scope['screen_metrics']=stage43._screen_metrics
    scope['raw_gate']=support.metric_helpers()['raw_gate']
    sys.path.insert(0,str(ROOT/'code'))
    return prior,support,scope

def paired(rows,keys,before,after):
    groups={p:{tuple(r[k] for k in keys):r for r in rows if r['point']==p} for p in (before,after)}
    assert groups[before].keys()==groups[after].keys()
    result=[]
    for key,a in groups[before].items():
        b=groups[after][key]
        if 'condition_sha256' in a:assert a['condition_sha256']==b['condition_sha256']
        row=dict(before=before,after=after,**dict(zip(keys,key)),**{k:a.get(k) for k in ('family','task','command','bucket')},
            before_gate=a.get('gate'),after_gate=b.get('gate'))
        for metric in METRICS:
            x,y=a.get(metric),b.get(metric)
            row['delta_'+metric]=float(y-x) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y) else None
        result.append(row)
    return result

def condition_means(rows):
    result=[]
    for point,uid in sorted({(r['point'],r['input_id']) for r in rows}):
        own=[r for r in rows if (r['point'],r['input_id'])==(point,uid)]
        assert len(own)==2 and {r['j'] for r in own}=={0,1}
        row={k:own[0][k] for k in ('point','input_id','condition_sha256','family','task','command','bucket')}
        row.update(draws=2,independent_conditions=1,REJECT=sum(r['gate']=='REJECT' for r in own))
        for metric in METRICS:
            values=[r.get(metric) for r in own]
            row[metric]=float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None
        result.append(row)
    return result

def main():
    admitted();assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    prior,support,scope=helpers()
    import torch
    import pi0260_r2_codec as codec
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('PI0273_SAVED_ARRAYS_NO_LEARNED_CALL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    proc=codec.processor(read(ROOT/'contracts/ROT/source/dataset/statistics.json'))
    scope.update(processors={'ROT':proc},torch=torch,codec=codec)
    from trajectory_replay import main as trajectory
    for point in POINTS[1:]:
        for lane in ('virtual','component'):trajectory(lane,point)
    offline=[r for r in table('PI0270','OFFLINE2904.csv') if r['point']=='AGG4096']
    virtual=[r for r in table('PI0270','VIRTUAL192.csv') if r['point']=='AGG4096']
    components=[r for r in table('PI0270','COMPONENT96.csv') if r['point']=='AGG4096']
    old_support=table('PI0272','ALL_POINTS3504.csv')
    early=[r for r in old_support if r['point'] in ('AGG4096','S4FIT1024') and r['panel']=='early_guard']
    late=[r for r in old_support if r['point'] in ('AGG4096','S4FIT1024') and r['panel']=='support' and r['j'] in (0,1)]
    predictions=[];metric_checks=[];offline_card=read(ROOT/'design/RUNTIME_OFFLINE_CARD.json')
    fitting=read(ROOT/'design/FITTING_CARD.json')
    consumed={a:{(fitting['base'][v['source_index']]['condition_sha256'],fitting['base'][v['source_index']]['labels_masks_sha256']) for v in fitting['streams'][a]} for a in ('BROAD','TARGET')}
    assert read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')['actual_inputs_match_CPU_plan']
    for point in POINTS[1:]:
        for domain in ('fit','factor','legacy'):
            for row in offline_card[domain+'_rows']:
                rec=read(RAW/'offline'/point/domain/row['uid']/'record.json')
                assert rec['row']==row and rec['point']==point and rec['card_sha256']==sha(ROOT/'design/RUNTIME_OFFLINE_CARD.json')
                if rec.get('metrics') is None and rec['status']=='VALID':
                    assert rec['gate']['decision']=='REJECT' and sha(rec['tensor'])==rec['tensor_sha256']
                    flat=dict(point=point,domain=domain,uid=row['uid'],source_row=json.dumps(row,sort_keys=True),gate='REJECT',numerical_failure=True,
                        **{k:row.get(k) for k in ('family','task','command','variant','anchor')})
                    check=None
                else:flat,check=prior.flatten(rec,domain,scope)
                if domain=='fit':
                    ident=row['processed']['ROT'];flat['actual_current_arm_seen']=(ident['condition_sha256'],ident['labels_masks_sha256']) in consumed[point.removesuffix('5120')]
                offline.append(flat)
                if check:metric_checks.append(check)
                if rec.get('tensor'):predictions.append(scope['replay'](Path(rec['tensor']),proc,torch,codec))
        card=read(ROOT/'design/SUPPORT_CARD.json')
        for panel,sources in (('early_guard',card['early']),('support',card['late'])):
            for row in sources:
                assert sha(row['input'])==row['input_sha256']
                with np.load(row['input'],allow_pickle=False) as z:expert=z['raw_action'].copy()
                for j in (('ORIGINAL_NEWFIT',) if panel=='early_guard' else (0,1)):
                    folder=RAW/'newfit'/point/row['input_id'] if panel=='early_guard' else RAW/'late'/point/row['input_id']/f'j{j}'
                    rec=read(folder/'record.json');path=Path(rec['tensor']);assert sha(path)==rec['tensor_sha256']
                    values,decoded,valid=support.measure(path,row['input'],expert,scope)
                    assert values['gate']==rec['gate']['decision']
                    mapping=row['source_mapping']
                    flat=dict(point=point,panel=panel,input_id=row['input_id'],j=j,condition_sha256=row['condition_sha256'],
                        **{k:mapping[k] for k in ('task','command','family','bucket')},**values)
                    flat['prefix_xyz_mse_m2']=float(np.mean(np.sum((decoded[:9,:3]-expert[:9,:3])**2,axis=1))) if valid else None
                    (early if panel=='early_guard' else late).append(flat)
                    predictions.append(scope['replay'](path,proc,torch,codec))
        for path in sorted((RAW/'virtual'/point).glob('*/result.json')):
            rec=read(path);check=read(ROOT/'evidence/replay/virtual'/point/(path.parent.name+'.trajectory.json'))
            assert sha(path)==check['source_result_sha256'];retention=check['reliability'];score=rec['scorer']
            with np.load(path.parent/'trajectory.npz') as z:
                spec=json.loads(str(z['spec_json']));installed_angle=float(scope['so3'](z['installed_r'][-1],np.asarray(spec['goal_rotation'])))
            virtual.append(dict(point=point,family=rec['family'],task=rec['task'],command=rec['command'],backend=rec['backend'],
                success=score['success'],reliable=retention['reliable'],gate_refusals=retention['gate_refusals'],dangerous_collision=bool(rec['collision_events']),
                execution_failure=rec['stop_reason'] is not None,queries=len(rec['queries']),elapsed_s=rec['actual_elapsed_s'],reason=rec['stop_reason'],
                V2_success=score['legacy_v2']['success'],bins=len(score['ordered_bins']),entry_time_s=score.get('entry_time_s'),time_to_success_s=score['time_to_success_s'],
                actual_final_xyz_m=score['final_xyz_m'],actual_final_so3_rad=score['final_so3_rad'],installed_final_xyz_m=check['reference_final_xyz_m'],
                installed_final_so3_rad=installed_angle,same_tick_tracking_xyz_rms_m=check['same_tick_tracking_xyz_rms_m'],
                same_tick_tracking_so3_rms_rad=check['same_tick_tracking_so3_rms_rad'],retention=json.dumps(retention,sort_keys=True)))
            for query in rec['queries']:predictions.append(scope['replay'](Path(query['query']['output']),proc,torch,codec))
        diag,checks=scope['component_rows'](point);components+=diag;predictions+=checks
    gate_checks=[]
    for lane in ('virtual','component'):
        for point in POINTS[1:]:
            for path in sorted((RAW/lane/point).glob('*/result.json')):
                rec=read(path);gates={r['replan']:r['result'] for r in rec.get('gates',[])}
                for q in rec['queries']:
                    value,_,valid=support.measure(Path(q['query']['output']),q['query']['input'],None,scope,visited=True)
                    saved=gates.get(q['replan'])
                    if saved is None:
                        assert not valid and value['gate']=='REJECT' and rec['stop_reason'] in ('NONFINITE_POLICY_OUTPUT','INVALID_ROT6D_POLICY_OUTPUT')
                    else:
                        rebuilt=json.loads(value['gate_detail'])
                        assert saved['decision']==value['gate'] and saved['metrics']==rebuilt['metrics']
                    gate_checks.append(dict(point=point,lane=lane,case=path.parent.name,replan=q['replan'],decision=value['gate'],original_gate_exact=True))
    primary_early=[r for r in early if r['point'] in POINTS]
    assert (len(offline),len(primary_early),len(late),len(virtual),len(components))==(2904,576,2368,192,96)
    original={p:scope['qualify']([r for r in virtual if r['point']==p],'success') for p in POINTS}
    reliability={p:scope['qualify']([r for r in virtual if r['point']==p],'reliable') for p in POINTS}
    panels={p:{d:scope['summary']([r for r in offline if r['point']==p and r['domain']==d]) for d in ('fit','factor','legacy')} for p in POINTS}
    baseline=scope['summary']([r for r in table('PI0270','OFFLINE2904.csv') if r['point']=='ROT2048' and r['domain']=='legacy'])
    legacy={p:panels[p]['legacy']['REJECT']<=baseline['REJECT'] and all(panels[p]['legacy'][k] is not None and panels[p]['legacy'][k]<=baseline[k] for k in ('median_xyz_rmse_m','median_so3_rmse_rad')) for p in POINTS}
    safe={p:all(not r['gate_refusals'] and not r['dangerous_collision'] and not r['execution_failure'] for r in virtual if r['point']==p) for p in POINTS}
    candidates=[p for p in POINTS[1:] if original[p]['PASS'] and reliability[p]['PASS'] and safe[p] and legacy[p]]
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    sources={r['input_id']:r for r in read(ROOT/'design/SUPPORT_CARD.json')['late']}
    for row in late:
        with np.load(row['tensor'],allow_pickle=False) as z:decoded=z['decoded'][0].astype(float)
        with np.load(sources[row['input_id']]['input'],allow_pickle=False) as z:expert=z['raw_action'].astype(float)
        if row['numeric_status']=='FINITE_VALID_ROTATION':
            angles=scope['so3'](rot6d_rows_to_matrix(decoded[:,3:9]),rot6d_rows_to_matrix(expert[:,3:9]))
            row['first_so3_error_rad']=float(angles[0]);row['prefix_endpoint_so3_error_rad']=float(angles[8])
        else:row['first_so3_error_rad']=row['prefix_endpoint_so3_error_rad']=None
    means=condition_means(late);comparisons=[];early_changes=[];offline_changes=[]
    for before,after in (('AGG4096','BROAD5120'),('AGG4096','TARGET5120'),('BROAD5120','TARGET5120')):
        comparisons+=paired(means,('input_id',),before,after)
        early_changes+=paired(primary_early,('input_id',),before,after)
        offline_changes+=prior.paired(offline,before,after)
    out=RAW/'delivery';out.mkdir(parents=True,exist_ok=True)
    tables={'OFFLINE2904.csv':offline,'NEW_FIT576.csv':primary_early,'EARLY_CONTEXT768.csv':early,'LATE_CONTEXT2368.csv':late,
        'VIRTUAL192.csv':virtual,'COMPONENT96.csv':components,'LATE_PER_CONDITION1184.csv':means,
        'PAIRED_LATE_CONDITION_CHANGES.csv':comparisons,'PAIRED_EARLY_CHANGES.csv':early_changes,'PAIRED_OFFLINE_CHANGES.csv':offline_changes,
        'SUPPORT_STRATA.csv':support.summarize(early+late,('point','panel','task','command','bucket'),METRICS),
        'SUPPORT_FAMILIES.csv':support.summarize(early+late,('point','panel','family','task','command','bucket'),METRICS)}
    for name,rows in tables.items():scope['csv_write'](out/name,rows)
    source_predictions={str(p) for lane in ('offline','newfit','late','virtual','component') for p in (RAW/lane).rglob('prediction.npz')}
    assert len(predictions)==len(source_predictions) and {r['file'] for r in predictions}==source_predictions
    write(out/'CPU_REPLAY.json',dict(status='PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES',predictions=predictions,metric_checks=metric_checks,actual_feedback_gate_checks=gate_checks,
        tables={n:dict(rows=len(v),sha256=sha(out/n)) for n,v in tables.items()},old_CPU_results_reused=True,old_neural_queries_repeated=0,
        shared_V3_gate_executor_disclosed=True,model_queries=0,teacher_calls=0,renders=0,physics=0))
    write(out/'PRIMARY_VERDICT.json',dict(status='PRIMARY_EVALUATION_COMPLETE',ORIGINAL_VIRTUAL_QUALIFICATION=original,
        RELIABLE_VIRTUAL_QUALIFICATION=reliability,full_point_safe=safe,panels=panels,original_ROT2048_legacy_baseline=baseline,
        legacy_retention_vs_original_ROT2048=legacy,reliable_virtual_research_candidates=candidates,offline_rows=2904,early_rows=576,
        late_context_rows=2368,virtual_episodes=192,component_proposals=96,not_WBC_not_physical_not_blind_TEST=True,
        transition_analysis_video_HF_and_terminal_still_required=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_PI0273_ALL_FIXED_TABLES_CPU_REPLAY_AND_QUALIFICATIONS',len(predictions),flush=True)

def self_check():
    prior,support,scope=helpers()
    rows=[dict(task=t,backend=b,command=c,success=i<7,reliable=i<7,gate_refusals=0,dangerous_collision=False,execution_failure=False)
        for t in ('M1','M3') for b in ('E0','E1') for c in (0,1) for i in range(8)]
    assert scope['qualify'](rows,'reliable')['PASS']
    rows[0]['reliable']=rows[1]['reliable']=False;assert not scope['qualify'](rows,'reliable')['PASS']
    prototype=dict(input_id='x',condition_sha256='same',family='f',task='M3',command=0,bucket='M3_TERMINAL',gate='ACCEPT')
    records=[dict(prototype,point=p,j=j,prefix_xyz_mse_m2=value) for p,value in (('BROAD5120',.0001),('TARGET5120',.0002)) for j in (0,1)]
    means=condition_means(records);assert len(means)==2
    assert paired(means,('input_id',),'BROAD5120','TARGET5120')[0]['delta_prefix_xyz_mse_m2']==.0001
    records[0]['prefix_xyz_mse_m2']=None
    assert paired(condition_means(records),('input_id',),'BROAD5120','TARGET5120')[0]['delta_prefix_xyz_mse_m2'] is None
    source=read(BASE/'PI0270/design/NEW_FIT192.json')['rows'][0]
    rec=read(RAW.parent/'PI0270/newfit/AGG4096'/source['input_id']/'record.json')
    with np.load(source['input'],allow_pickle=False) as z:expert=z['raw_action'].copy()
    measured,_,valid=support.measure(Path(rec['tensor']),source['input'],expert,scope)
    assert valid and measured['gate']==rec['gate']['decision']
    old,check=prior.flatten(rec,'newfit',scope);assert check['independent_metrics_max_abs']<=1e-6
    assert abs(old['prefix_xyz_rmse_m']-measured['prefix_xyz_rmse_m'])<=1e-12
    print('PASS_CONSTRUCTED_ORIGINAL_QUALIFICATION_AND_PAIRED_MSE_DENOMINATORS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true')
    self_check() if p.parse_args().self_check else main()
