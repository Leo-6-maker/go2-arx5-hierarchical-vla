"""Saved-output diagnosis; no model, teacher, renderer or new task qualification."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import numpy as np
from support import BASE, ROOT, OLD, RAW, read, sha, write
from error_bank import decompose, displacement


def metric_helpers():
    import quality_worker  # Reuse qualified helper paths without loading its model.
    from quality_report import helpers
    scope=helpers()
    import pi0243_staged_learning_to_sim as stage43
    from scipy.spatial.transform import Rotation
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    path=BASE/'PI0267/code/pi0267_virtual_data.py'
    sealed=read(OLD/'design/QUALITY_EXECUTION_FILESET.json')['files']
    matches=[r for r in sealed if Path(r['path']).resolve()==path.resolve()]
    assert len(matches)==1 and sha(path)==matches[0]['sha256']
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='raw_gate')
    gate_scope=dict(np=np,Rotation=Rotation,TrajectoryChunkV1=TrajectoryChunkV1,TrajectorySafetyGateV1=TrajectorySafetyGateV1)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),gate_scope)
    scope.update(raw_gate=gate_scope['raw_gate'],screen_metrics=stage43._screen_metrics)
    return scope


def measure(path, observation, expert, scope, visited=False):
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
    with np.load(path,allow_pickle=False) as z: decoded=z['decoded'][0].astype(np.float64); sv=z['state'].copy()
    with np.load(observation,allow_pickle=False) as z:
        assert np.array_equal(sv,z['state_v1'])
        p=z['position_world'].copy() if 'position_world' in z else None
        r=z['rotation_world'].copy() if 'rotation_world' in z else None
        now=float(z['timestamp']) if 'timestamp' in z else 0.
    row=dict(tensor=str(path),tensor_sha256=sha(path),numeric_status='NONFINITE',teacher_metrics_available=False)
    if not np.isfinite(decoded).all():
        row.update(gate='REJECT',gate_detail=json.dumps(dict(reason='NONFINITE_POLICY_OUTPUT')));return row,decoded,False
    try: rotation=rot6d_rows_to_matrix(decoded[:,3:9])
    except ValueError:
        row.update(numeric_status='INVALID_ROTATION',gate='REJECT',gate_detail=json.dumps(dict(reason='INVALID_ROT6D_POLICY_OUTPUT')));return row,decoded,False
    if visited:
        anchor=np.asarray(read(BASE/'PI0267/design/SCENE_CARD.json')['policy_anchor_world_m'])
        assert p is not None and r is not None
        gate=scope['raw_gate'](decoded[:,:3]+anchor,rotation,p,r,anchor,now=now,source='PI0267_NATIVE_S4_POLICY',gripper=decoded[:,9])
        row['first_actual_position_bias_m']=float(np.linalg.norm(decoded[0,:3]+anchor-p))
    else:
        provider=GR00TTrajectoryProvider(decoded[None,:,:],config=GR00TProviderConfig(
            source_frame='PI0267_STATIC_POLICY_ANCHOR_WORLD_AXES',tcp_semantic_id='PI0267_VIRTUAL_TOOL_TCP_V1',
            action_sample_period_s=.05,timestamp_provenance='PI0267_FIXED_OFFLINE'),
            observation_timestamp=0.,chunk_id='PI0271-SUPPORT')
        gate=TrajectorySafetyGateV1().evaluate(provider.next_chunk(0.),now=0.,achieved_position=sv[:3],
            achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(sv[None,3:9]))[0]).as_dict()
        row['first_actual_position_bias_m']=float(np.linalg.norm(decoded[0,:3]-sv[:3]))
    row.update(numeric_status='FINITE_VALID_ROTATION',gate=gate['decision'],gate_detail=json.dumps(gate,sort_keys=True))
    if expert is not None:
        expert=np.asarray(expert,dtype=np.float64)
        original=scope['screen_metrics'](decoded,expert,None); independent=scope['independent_metrics'](decoded,expert)
        maximum=max(abs(independent[k]-original[k]) for k in independent);assert maximum<=1e-6
        row.update(original);row.update(scope['prefix_metrics'](decoded,expert,sv))
        row['legacy_direction_cosine']=row.pop('direction_cosine')
        row.update(displacement(sv[:3],expert[8,:3],decoded[8,:3]))
        row.update(teacher_metrics_available=True,independent_metrics_max_abs=maximum)
    return row,decoded,True


def paired(rows,keys,metrics):
    index={(r['point'],*(r[k] for k in keys)):r for r in rows}; assert len(index)==len(rows)
    output=[]
    for key in sorted({tuple(r[k] for k in keys) for r in rows},key=str):
        a,b=[index[(point,*key)] for point in ('PREFIX4096','AGG4096')]
        row=dict(zip(keys,key));row.update({k:a.get(k) for k in ('family','task','command','bucket','panel')})
        row.update(before='PREFIX4096',after='AGG4096',before_gate=a.get('gate'),after_gate=b.get('gate'))
        for k in metrics:
            x,y=a.get(k),b.get(k)
            row['delta_'+k]=float(y-x) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y) else None
        output.append(row)
    return output


def summarize(rows,keys,metrics):
    output=[]
    for key in sorted({tuple(r.get(k) for k in keys) for r in rows},key=str):
        own=[r for r in rows if tuple(r.get(k) for k in keys)==key]
        group=dict(zip(keys,key));group.update(rows=len(own),REJECT=sum(r.get('gate')=='REJECT' for r in own),
            numeric_invalid=sum(r.get('numeric_status') not in (None,'FINITE_VALID_ROTATION') for r in own))
        for name in metrics:
            values=[r.get(name) for r in own];finite=[float(v) for v in values if v is not None and np.isfinite(v)]
            group['defined_'+name]=len(finite);group['undefined_'+name]=len(values)-len(finite)
            group['median_defined_'+name]=float(np.median(finite)) if finite else None
            group['p25_defined_'+name]=float(np.quantile(finite,.25)) if finite else None
            group['p75_defined_'+name]=float(np.quantile(finite,.75)) if finite else None
            group['complete_denominator_'+name]=len(finite)==len(values)
            if name in ('direction_cosine','signed_teacher_direction_progress_m'):
                group['negative_'+name]=sum(v<0 for v in finite)
                group['zero_'+name]=sum(v==0 for v in finite)
                group['positive_'+name]=sum(v>0 for v in finite)
        output.append(group)
    return output


def main():
    from worker import admitted
    admitted()
    complete=read(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json');assert complete['status']=='ALL_FIXED_PI0271_OUTPUTS_COMPLETE'
    labels_complete=read(ROOT/'evidence/VISITED_LABELS_COMPLETE.json');assert len(labels_complete['records'])==64
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    scope=metric_helpers()
    import torch
    import pi0260_r2_codec as codec
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('SAVED_DIAGNOSIS_NO_NEURAL_OR_OPTIMIZER_CALL')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    proc=codec.processor(read(OLD/'contracts/ROT/source/dataset/statistics.json'))
    plan=read(ROOT/'design/QUERY_PLAN.json');support=read(ROOT/'design/LATE_SUPPORT_CARD.json');visited=read(ROOT/'design/VISITED_STATE_CARD_V2.json')
    source_by_id={r['input_id']:r for r in support['rows']};support_rows=[];visited_rows=[];bank_arrays={};cpu=[]
    outputs={r['path']:r for r in complete['rows']};assert len(outputs)==len(plan['rows'])
    for query in plan['rows']:
        path=Path(query['request']['output']);proof=outputs[str(path)];assert sha(path)==proof['sha256']
        cpu.append(scope['replay'](path,proc,torch,codec))
        if query['panel']=='A3':continue
        source=source_by_id[query['input_id']];mapping=source['source_mapping']
        assert sha(source['input'])==source['input_sha256']
        with np.load(source['input'],allow_pickle=False) as z:expert=z['raw_action'].copy()
        metric,decoded,valid=measure(path,source['input'],expert,scope)
        row=dict(point=query['point'],input_id=query['input_id'],j=query['j'],panel=query['panel'],
            **{k:mapping[k] for k in ('family','task','command','bucket')},actual_AGG_seen=True,
            PREFIX_consumption=source['PREFIX_consumption'],source_timestamp_s=source['source_timestamp_s'],
            source_lane=source['source_lane'],component_placement_is_not_original_task_success=mapping['bucket'].startswith('M'),**metric)
        support_rows.append(row)
        if mapping['bucket']=='ORIGINAL_Q3':bank_arrays.setdefault((query['point'],query['input_id']),{})[query['j']]=(decoded,valid,expert)
    assert len(support_rows)==2336 and len(bank_arrays)==192
    banks=[];bank_regions=[]
    for (point,uid),draws in bank_arrays.items():
        assert set(draws)==set(range(8));target=draws[0][2]
        assert all(np.array_equal(v[2],target) for v in draws.values())
        decomposition=decompose(np.stack([draws[j][0][:,:3] for j in range(8)]),target[:,:3],np.array([draws[j][1] for j in range(8)]))
        mapping=source_by_id[uid]['source_mapping'];base=dict(point=point,input_id=uid,**{k:mapping[k] for k in ('family','task','command','bucket')})
        banks.append(dict(**base,**decomposition))
        for region in ('first','endpoint_0p4','prefix','suffix','full'):
            bank_regions.append(dict(**base,region=region,status=decomposition['status'],K=8,
                **decomposition.get('regions',{}).get(region,{})))
    with (ROOT/'evidence/predecessor-analysis/SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv').open() as f:
        transitions={(r['case'],int(r['query_index'])):r for r in csv.DictReader(f)}
    for source in visited['rows']:
        label_path=RAW/'labels-visited'/source['input_id']/'record.json';label=read(label_path)
        assert label['row']==source
        expert=None
        if label.get('eligible'):
            assert sha(label['sample'])==label['sample_sha256']
            with np.load(label['sample'],allow_pickle=False) as z:expert=z['raw_action'].copy()
        for point in ('PREFIX4096','AGG4096'):
            row=dict(point=point,input_id=source['input_id'],j='EXACT_AGG_NOISE',panel='A3',
                **{k:source[k] for k in ('family','task','command','backend','query_index')},
                label_status=label['status'],label_eligible=label.get('eligible',False),DIAGNOSIS_ONLY=True)
            if source['status']=='MISSING_AFTER_STOP':
                row.update(numeric_status='MISSING_AFTER_STOP',gate=None,teacher_metrics_available=False);visited_rows.append(row);continue
            path=Path(source['AGG_prediction']) if point=='AGG4096' else RAW/'visited/PREFIX4096'/source['input_id']/'prediction.npz'
            if point=='AGG4096':assert sha(path)==source['AGG_prediction_sha256']
            metric,decoded,valid=measure(path,source['request']['input'],expert,scope,visited=True);row.update(metric)
            result=read(source['source_result']);gate=next((g['result'] for g in result['gates'] if g['replan']==source['query_index']),None)
            if point=='AGG4096' and valid:
                assert gate is not None and gate['decision']==row['gate']
                # Time and geometric scalar metrics are independently rebuilt by the same original gate.
                regenerated=json.loads(row['gate_detail']);assert regenerated['metrics']==gate['metrics']
            transition=transitions[(source['case'],source['query_index'])]
            row.update(actual_AGG_prefix_executed=transition['actual_prefix_executed'],
                actual_AGG_achieved_target_error_change_m=transition.get('achieved_target_error_change_m'),
                actual_AGG_same_tick_tracking_xyz_m=transition.get('same_tick_tracking_xyz_m'),
                actual_AGG_installed_target_xyz_m=transition.get('installed_reference_same_tick_target_xyz_m'),
                counterfactual_PREFIX_rollout=False,predicted_endpoint_is_not_achieved=True)
            if valid and label.get('eligible'):
                anchor=np.asarray(read(BASE/'PI0267/design/SCENE_CARD.json')['policy_anchor_world_m'])
                with np.load(source['request']['input'],allow_pickle=False) as z:current=z['position_world'].copy()
                with np.load(label['sample'],allow_pickle=False) as z:teacher_end=z['label_position_world'][8].copy()
                goal=np.asarray(label['terminal_position_world']);before=float(np.linalg.norm(current-goal))
                row.update(current_terminal_distance_m=before,model_endpoint_terminal_change_m=float(np.linalg.norm(decoded[8,:3]+anchor-goal))-before,
                    teacher_endpoint_terminal_change_m=float(np.linalg.norm(teacher_end-goal))-before,M3_distance_is_not_ordered_path_qualification=True)
            visited_rows.append(row)
    assert len(visited_rows)==128 and len(cpu)==len(plan['rows'])
    metrics=('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad','suffix_xyz_rmse_m','suffix_so3_rmse_rad',
        'prefix_endpoint_error_m','first_actual_position_bias_m','direction_cosine','signed_teacher_direction_progress_m')
    support_pairs=paired(support_rows,('input_id','j'),metrics);visited_pairs=paired(visited_rows,('input_id','j'),metrics)
    bank_metrics=('MSE_m2','empirical_mean_error_squared_m2','within_bank_dispersion_m2')
    bank_pairs=paired(bank_regions,('input_id','region'),bank_metrics)
    out=RAW/'delivery';out.mkdir(parents=True,exist_ok=True)
    tables={'SUPPORT_ALL_NOISES2336.csv':support_rows,'A1_SUPPORT1184.csv':[r for r in support_rows if r['panel']=='A1'],
        'PAIRED_SUPPORT1168.csv':support_pairs,'A1_PAIRED592.csv':[r for r in support_pairs if r['panel']=='A1'],
        'K8_REGIONS960.csv':bank_regions,'PAIRED_K8_REGIONS480.csv':bank_pairs,
        'VISITED128.csv':visited_rows,'PAIRED_VISITED64.csv':visited_pairs,
        'SUPPORT_STRATA.csv':summarize(support_rows,('point','panel','task','command','bucket'),metrics),
        'SUPPORT_FAMILIES.csv':summarize(support_rows,('point','panel','family','task','command','bucket'),metrics),
        'PAIRED_SUPPORT_FAMILIES.csv':summarize(support_pairs,('panel','family','task','command','bucket'),tuple('delta_'+k for k in metrics)),
        'PAIRED_K8_FAMILIES.csv':summarize(bank_pairs,('family','region'),tuple('delta_'+k for k in bank_metrics)),
        'VISITED_STRATA.csv':summarize(visited_rows,('point','family','task','command','query_index'),metrics)}
    for name,rows in tables.items():scope['csv_write'](out/name,rows)
    write(out/'K8_BANKS192.json',dict(banks=banks,K=8,primary_scene_units=24,no_significance_or_population_bias_claim=True))
    sources=[dict(path=str(p),sha256=sha(p)) for p in (ROOT/'design/QUERY_PLAN.json',ROOT/'design/LATE_SUPPORT_CARD.json',
        ROOT/'design/VISITED_STATE_CARD_V2.json',ROOT/'evidence/A0_DECISION.json',ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json',
        ROOT/'evidence/VISITED_LABELS_COMPLETE.json',Path(__file__),ROOT/'code/error_bank.py')]
    write(out/'CPU_REPLAY.json',dict(status='PASS_ALL_NEW_S4_DECODE_METRICS_AND_FIXED_K8_REPLAY',predictions=cpu,
        tables={n:dict(rows=len(r),sha256=sha(out/n)) for n,r in tables.items()},source_files=sources,
        reused_AGG_A3_predictions=64,replayed_old_AGG_predictions=0,shared_original_gate=True,
        invalid_bank_count=sum(b['status']!='FINITE_FIXED_K8_DECOMPOSITION' for b in banks),
        diagnostic_only=True,model_queries=0,teacher_calls=0,episodes=0,renders=0,
        original_task_qualification_unchanged=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_ALL_FIXED_DIAGNOSTIC_TABLES_AND_NEW_S4_CPU_REPLAY',len(cpu),flush=True)


def check_only():
    scope=metric_helpers()
    card=read(OLD/'design/NEW_FIT192.json');source=card['rows'][0]
    record=read(RAW.with_name('PI0270')/'newfit/AGG4096'/source['input_id']/'record.json')
    with np.load(source['input'],allow_pickle=False) as z:expert=z['raw_action'].copy()
    metric,_,valid=measure(Path(record['tensor']),source['input'],expert,scope)
    assert valid and metric['gate']==record['gate']['decision']
    for key,value in record['metrics'].items():assert abs(metric[key]-value)<=1e-12,(key,metric[key],value)
    visited=read(ROOT/'design/VISITED_STATE_CARD_V2.json')['rows'][0]
    assert sha(visited['AGG_prediction'])==visited['AGG_prediction_sha256'] and sha(visited['request']['input'])==visited['observation_sha256']
    actual,_,valid=measure(Path(visited['AGG_prediction']),visited['request']['input'],None,scope,visited=True)
    original=next(g['result'] for g in read(visited['source_result'])['gates'] if g['replan']==visited['query_index'])
    assert valid and json.loads(actual['gate_detail'])==original
    example=[dict(point=p,input_id='x',j=0,family='f',task='M1',command=0,bucket='Q3',value=v) for p,v in [('PREFIX4096',1.),('AGG4096',None)]]
    assert paired(example,('input_id','j'),('value',))[0]['delta_value'] is None
    assert summarize(example,('family',),('value',))[0]['undefined_value']==1
    write(ROOT/'evidence/REPORT_CPU_CHECK_V3.json',dict(status='PASS_EXISTING_METRIC_AND_ACTUAL_GATE_FIXTURES_AND_UNDEFINED_PAIR_DENOMINATOR',
        source_sha256=sha(__file__),reused_prediction_sha256=sha(record['tensor']),model_queries=0,teacher_calls=0,
        reused_actual_AGG_gate_prediction_sha256=sha(visited['AGG_prediction']),actual_saved_gate_exact=True,
        new_diagnostic_predictions_not_yet_evaluated=True))
    print('PASS_EXISTING_CLOSED_METRIC_FIXTURE_NO_MODEL_QUERY')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    if parser.parse_args().check_only:check_only()
    else:main()
