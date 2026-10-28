"""Final-only saved-output CPU return; reuse closed control tables, never query a model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import argparse
import ast
import csv
import json
import sys
from pathlib import Path
import numpy as np
from admission import BASE, ROOT, read, sha, write

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')
CONTROL = RAW.with_name('PI0269')
POINTS = ('ROT2048', 'PREFIX4096', 'AGG4096')
TABLES = {'OFFLINE2904.csv': 'dc8b9cd52bf22fe5951ded63132549dad69b9a619c80a6aea586ae1c24e766a8',
          'VIRTUAL192.csv': '36bafd7e7df2b99d28e2b288e71ebee238463d6f5e20c7505906b29cba674b97',
          'COMPONENT96.csv': '89dbb5cfb5c3cd37c96e9930fc702d03a1a2b27ae16f4f6990c5a9055c763db6'}


def value(text):
    if text == '': return None
    if text in ('True', 'False'): return text == 'True'
    try: return json.loads(text)
    except ValueError:
        try: return float(text)
        except ValueError: return text


def closed_table(name):
    path = CONTROL/'delivery'/name; assert sha(path) == TABLES[name]
    with path.open() as stream:
        return [{k: v if k == 'source_row' else value(v) for k,v in r.items()}
                for r in csv.DictReader(stream) if r['point'] in POINTS[:2]]


def helpers():
    sys.path[:0] = [str(BASE/'PI0267/code'), str(BASE/'PI0264/code'), str(BASE/'PI0266/code'),
                    str(BASE/'PI0268/R1/code'), str(BASE/'PI0250/runtime/reference-317a/src')]
    from pi0267_eval_report import independent_metrics, csv_write, score, independent_entry, so3
    scope = dict(np=np, Path=Path, json=json, read=read, sha=sha, ROOT=ROOT, RAW=RAW,
                 independent_metrics=independent_metrics, csv_write=csv_write, score=score,
                 independent_entry=independent_entry, so3=so3)
    paths = [BASE/'PI0268/R1/code/report.py', BASE/'PI0269/code/report.py']
    expected = ['1db82278af12f085144990a644f43c183b24bb072c149662ff04c5758c6b3e73',
                'b933b78aeb4b420a3c74cbc83dd97e1bed041bdc198e1dd8f607756b29a4436a']
    for path,digest,names in zip(paths,expected,[('qualify','summary','replay'),('prefix_metrics',)]):
        assert sha(path) == digest
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    # ponytail: reuse the qualified component arithmetic as one function, rather than copy its formulas.
    old = ast.parse(paths[-1].read_text()); main = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop = next(n for n in main.body if isinstance(n,ast.For) and any(isinstance(x,ast.For) and
                isinstance(x.target,ast.Name) and x.target.id=='proposal' for x in n.body))
    wrapper = ast.parse("def component_rows(point):\n diagnostics=[]\n predictions=[]\n component=read(ROOT/'design/COMPONENT_CARD.json')\n return diagnostics,predictions\n")
    wrapper.body[0].body[3:3] = loop.body
    exec(compile(ast.fix_missing_locations(wrapper),str(paths[-1]),'exec'),scope)
    return scope


def flatten(rec, domain, scope):
    row = rec['row']; uid = row.get('uid', row.get('input_id')); assert uid
    result = dict(point=rec['point'],domain=domain,uid=uid,gate=rec.get('gate',{}).get('decision','REJECT'),
                  numerical_failure=rec.get('metrics') is None,source_row=json.dumps(row,sort_keys=True),
                  **{k:row.get(k) for k in ('task','command','family','variant','anchor')})
    if domain == 'newfit':
        result.update({k:row['source_mapping'][k] for k in ('task','command','family')})
        result.update(actual_AGG_seen=True,TRAIN_support=True,blind_TEST=False)
    transform = row.get('transform',{})
    result.update(factor_kind=transform.get('kind'),factor_axis=transform.get('axis'),factor_sign=transform.get('sign'),
                  noise=row.get('noise') or (Path(row['noise_file']).name+':'+row.get('noise_name','float32')))
    if rec['status'] == 'PREQUERY_EXCLUDED_KEEP_DENOMINATOR':
        assert domain == 'factor' and not row['admitted'] and not rec.get('tensor')
        result.update(prequery_excluded=True,reasons=rec['reasons']); return result,None
    assert rec['status']=='VALID' and sha(rec['tensor'])==rec['tensor_sha256']
    with np.load(rec['tensor']) as z: decoded=z['decoded'][0].astype(float); state=z['state'].copy()
    if domain == 'legacy':
        from pi0264_freeze_eval import source
        _,expert=source(row)
    else:
        assert sha(row['input'])==row['input_sha256']
        with np.load(row['input']) as z: expert=z['raw_action'].copy()
    difference=None
    if rec['metrics'] is not None:
        actual=scope['independent_metrics'](decoded,expert.astype(float))
        difference=max(abs(actual[k]-rec['metrics'][k]) for k in actual)
        assert difference<=1e-6,(domain,uid,difference)
    result.update(rec['metrics'] or {})
    result.update(scope['prefix_metrics'](decoded,expert.astype(float),state))
    return result,dict(point=rec['point'],domain=domain,uid=uid,
                       independent_metrics_max_abs=difference,tensor_sha256=rec['tensor_sha256'])


def paired(rows,before,after):
    a={r['uid']:r for r in rows if r['point']==before}; b={r['uid']:r for r in rows if r['point']==after}
    assert a.keys()==b.keys()
    output=[]
    for uid,x in a.items():
        y=b[uid]; assert x['source_row']==y['source_row']
        row=dict(before=before,after=after,uid=uid,**{k:x.get(k) for k in ('domain','family','task','command','noise')},
                 before_gate=x['gate'],after_gate=y['gate'])
        for k in ('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad','suffix_xyz_rmse_m','suffix_so3_rmse_rad','prefix_endpoint_error_m'):
            aa,bb=x.get(k),y.get(k)
            row['delta_'+k]=bb-aa if aa is not None and bb is not None and np.isfinite(aa) and np.isfinite(bb) else None
        output.append(row)
    return output


def main():
    from quality_gate import admitted
    admitted(); assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    scope=helpers()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('SAVED_ARRAY_REPORT_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    sys.path.insert(0,str(RAW.parent/'PI0260/R2/qa'))
    import pi0260_r2_codec as codec
    proc=codec.processor(read(ROOT/'contracts/ROT/source/dataset/statistics.json'))
    scope.update(processors={'ROT':proc},torch=torch,codec=codec)
    from trajectory_replay import main as trajectory
    trajectory('virtual');trajectory('component')
    old_cpu=CONTROL/'delivery/CPU_REPLAY.json'
    assert sha(old_cpu)=='5a1b5f1ba6aa8acb779f8bc79cb0137d2bfbc5ea55b14ccd1c30225f5d434ef4'
    reuse=read(ROOT/'evidence/EVALUATION_EXACT_REUSE.json')
    for r in reuse['records']:
        assert sha(r['record']['path'])==r['record']['sha256']
        if r['payload']:assert sha(r['payload']['path'])==r['payload']['sha256']
    offline=closed_table('OFFLINE2904.csv');virtual=closed_table('VIRTUAL192.csv');components=closed_table('COMPONENT96.csv')
    assert (len(offline),len(virtual),len(components))==(1936,128,64)
    metrics=[];predictions=[]
    card_path=ROOT/'design/FIXED_OFFLINE_CARD.json';card=read(card_path)
    for domain,rows in [('fit',card['fit_rows']),('factor',card['factor_rows']),('legacy',card['legacy_rows'])]:
        for row in rows:
            rec=read(RAW/'offline/AGG4096'/domain/row['uid']/'record.json')
            assert rec['point']=='AGG4096' and rec['row']==row and rec['card_sha256']==sha(card_path)
            flat,check=flatten(rec,domain,scope);offline.append(flat)
            if check:metrics.append(check);predictions.append(scope['replay'](Path(rec['tensor']),proc,torch,codec))
    samples=ROOT/'evidence/training_hopper/AGG/main/training/sample_audit.jsonl'
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');assert sha(samples)==audit['sample_audit_sha256']
    actual=[json.loads(x) for x in samples.read_text().splitlines()];assert len(actual)==8192
    consumed={(r['condition_sha256'],r['labels_masks_sha256']) for r in actual}
    seen={r['uid']:(r['processed']['ROT']['condition_sha256'],r['processed']['ROT']['labels_masks_sha256']) in consumed for r in card['fit_rows']}
    for row in offline:
        if row['domain']=='fit':row['actual_AGG_seen']=seen[row['uid']]
    newfit=[];fit_path=ROOT/'design/NEW_FIT192.json';fit=read(fit_path)
    for point in POINTS[1:]:
        for row in fit['rows']:
            rec=read(RAW/'newfit'/point/row['input_id']/'record.json')
            assert rec['point']==point and rec['row']==row and rec['card_sha256']==sha(fit_path) and rec['actual_AGG_seen']
            assert (row['condition_sha256'],row['labels_masks_sha256']) in consumed
            flat,check=flatten(rec,'newfit',scope);newfit.append(flat);metrics.append(check)
            predictions.append(scope['replay'](Path(rec['tensor']),proc,torch,codec))
    for path in sorted((RAW/'virtual/AGG4096').glob('*/result.json')):
        rec=read(path); check=read(ROOT/'evidence/replay/virtual'/(path.parent.name+'.trajectory.json'))
        assert sha(path)==check['source_result_sha256'];retention=check['reliability'];result=rec['scorer']
        with np.load(path.parent/'trajectory.npz') as z:
            spec=json.loads(str(z['spec_json']))
            installed_angle=float(scope['so3'](z['installed_r'][-1],np.asarray(spec['goal_rotation'])))
        virtual.append(dict(point='AGG4096',family=rec['family'],task=rec['task'],command=rec['command'],backend=rec['backend'],
            success=result['success'],reliable=retention['reliable'],gate_refusals=retention['gate_refusals'],
            dangerous_collision=bool(rec['collision_events']),execution_failure=rec['stop_reason'] is not None,queries=len(rec['queries']),
            elapsed_s=rec['actual_elapsed_s'],reason=rec['stop_reason'],V2_success=result['legacy_v2']['success'],
            bins=len(result['ordered_bins']),entry_time_s=result.get('entry_time_s'),time_to_success_s=result['time_to_success_s'],
            actual_final_xyz_m=result['final_xyz_m'],actual_final_so3_rad=result['final_so3_rad'],
            installed_final_xyz_m=check['reference_final_xyz_m'],installed_final_so3_rad=installed_angle,
            same_tick_tracking_xyz_rms_m=check['same_tick_tracking_xyz_rms_m'],
            same_tick_tracking_so3_rms_rad=check['same_tick_tracking_so3_rms_rad'],retention=json.dumps(retention,sort_keys=True)))
        for query in rec['queries']:predictions.append(scope['replay'](Path(query['query']['output']),proc,torch,codec))
    diag,checks=scope['component_rows']('AGG4096');components+=diag;predictions+=checks
    assert (len(offline),len(newfit),len(virtual),len(components))==(2904,384,192,96)
    original={p:scope['qualify']([r for r in virtual if r['point']==p],'success') for p in POINTS}
    reliability={p:scope['qualify']([r for r in virtual if r['point']==p],'reliable') for p in POINTS}
    panels={p:{d:scope['summary']([r for r in offline if r['point']==p and r['domain']==d]) for d in ('fit','factor','legacy')} for p in POINTS}
    donor,agg=panels['ROT2048']['legacy'],panels['AGG4096']['legacy']
    legacy_ok=agg['REJECT']<=donor['REJECT'] and all(donor[k] is not None and agg[k] is not None and agg[k]<=donor[k] for k in ('median_xyz_rmse_m','median_so3_rmse_rad'))
    qualified=original['AGG4096']['PASS'] and reliability['AGG4096']['PASS'] and legacy_ok
    out=RAW/'delivery';out.mkdir(exist_ok=True)
    for name,rows in [('OFFLINE2904.csv',offline),('NEW_FIT384.csv',newfit),('VIRTUAL192.csv',virtual),('COMPONENT96.csv',components),
                      ('PAIRED_OFFLINE_CHANGES.csv',paired(offline,'PREFIX4096','AGG4096')),
                      ('PAIRED_NEW_FIT_CHANGES.csv',paired(newfit,'PREFIX4096','AGG4096'))]:scope['csv_write'](out/name,rows)
    subgroups=[]
    for point in POINTS:
        for domain,axes in [('fit',('family','task','command','variant','noise','actual_AGG_seen')),
                            ('factor',('family','task','command','factor_kind','factor_axis','factor_sign')),
                            ('legacy',('noise',)),('newfit',('family','task','command'))]:
            own=[r for r in offline+newfit if r['point']==point and r['domain']==domain]
            for key in sorted({tuple(r.get(k) for k in axes) for r in own},key=str):
                group=[r for r in own if tuple(r.get(k) for k in axes)==key]
                prefix=float(np.median([r.get('prefix_xyz_rmse_m',float('inf')) for r in group]))
                subgroups.append(dict(point=point,domain=domain,**dict(zip(axes,key)),**scope['summary'](group),
                                     median_prefix_xyz_rmse_m=prefix if np.isfinite(prefix) else None))
    scope['csv_write'](out/'OFFLINE_SUBGROUPS.csv',subgroups)
    changes=paired(newfit,'PREFIX4096','AGG4096');cell_changes=[]
    for key in sorted({(r['family'],r['task'],r['command']) for r in changes}):
        own=[r for r in changes if (r['family'],r['task'],r['command'])==key];assert len(own)==2
        cell=dict(family=key[0],task=key[1],command=key[2],paired_rows=2)
        for metric in ('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad','prefix_endpoint_error_m'):
            values=[r['delta_'+metric] for r in own]
            cell['undefined_'+metric]=sum(v is None for v in values)
            cell['paired_median_delta_'+metric]=float(np.median(values)) if all(v is not None for v in values) else None
        cell_changes.append(cell)
    scope['csv_write'](out/'PAIRED_NEW_FIT_CELLS.csv',cell_changes)
    family_changes=[]
    for family,task,backend in sorted({(r['family'],r['task'],r['backend']) for r in virtual}):
        groups={p:[r for r in virtual if (r['point'],r['family'],r['task'],r['backend'])==(p,family,task,backend)] for p in POINTS[1:]}
        assert all(len(rows)==2 for rows in groups.values())
        row=dict(family=family,task=task,backend=backend,paired_commands=2,independent_scene_count=1)
        for metric in ('success','reliable'):
            a,b=[sum(bool(r[metric]) for r in groups[p]) for p in POINTS[1:]]
            row.update({f'PREFIX_{metric}':a,f'AGG_{metric}':b,f'delta_{metric}':b-a})
        family_changes.append(row)
    scope['csv_write'](out/'PAIRED_MAIN_FAMILY_CHANGES.csv',family_changes)
    write(out/'CPU_REPLAY.json',dict(status='PASS_SAVED_ARRAY_REPLAY',predictions=predictions,metrics=metrics,
        reused_closed_control_donor_CPU_sha256=sha(old_cpu),old_CPU_predictions_replayed=0,scorer_shared=True,
        trajectory_receipts=[dict(path=str(ROOT/'evidence'/n),sha256=sha(ROOT/'evidence'/n)) for n in ('TRAJECTORY_CPU_virtual_64.json','TRAJECTORY_CPU_component_32.json')],
        source_sha256=sha(__file__),model_queries=0,gradient_queries=0,teacher_calls=0,renderer_calls=0,physics_steps=0))
    write(out/'PRIMARY_VERDICT.json',dict(status='PRIMARY_EVALUATION_COMPLETE',ORIGINAL_VIRTUAL_QUALIFICATION=original,
        RELIABLE_VIRTUAL_QUALIFICATION=reliability,panels=panels,legacy_retention_vs_original_ROT2048=legacy_ok,
        reliable_virtual_research_candidates=['AGG4096'] if qualified else [],offline_rows=2904,newfit_rows=384,
        virtual_episodes=192,component_proposals=96,old_FIT_actual_AGG_consumption=seen,
        interpretation='RELIABLE_VIRTUAL_RESEARCH_CANDIDATE' if qualified else ('MAIN_RELIABILITY_IMPROVED_LEGACY_FAIL' if reliability['AGG4096']['PASS'] and not legacy_ok else 'VALID_QUALITY_FAIL'),
        not_WBC_not_physical_not_blind_TEST=True,additional_transition_analysis_video_HF_and_terminal_still_required=True,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('ACTUAL_FIXED_2904_384_192_96_CPU_TABLES_COMPLETE',len(predictions),flush=True)


def self_check():
    s=helpers()
    assert value('False') is False and value('0')==0 and value('') is None and np.isinf(value('inf'))
    rows=[dict(task=t,backend=b,command=c,success=i<7,reliable=i<7,gate_refusals=0,dangerous_collision=False,execution_failure=False)
          for t in ('M1','M3') for b in ('E0','E1') for c in (0,1) for i in range(8)]
    assert s['qualify'](rows,'reliable')['PASS']
    rows[0]['reliable']=False;rows[1]['reliable']=False;assert not s['qualify'](rows,'reliable')['PASS']
    assert s['summary']([dict(gate='REJECT',numerical_failure=True)])['median_xyz_rmse_m'] is None
    a=dict(point='PREFIX4096',uid='x',domain='newfit',source_row='same',gate='ACCEPT',prefix_xyz_rmse_m=.002)
    b=dict(a,point='AGG4096',prefix_xyz_rmse_m=.001);assert paired([a,b],a['point'],b['point'])[0]['delta_prefix_xyz_rmse_m']==-.001
    old=read(CONTROL/'delivery/PRIMARY_VERDICT.json');virtual=closed_table('VIRTUAL192.csv')
    assert len(closed_table('OFFLINE2904.csv'))==1936 and len(virtual)==128 and len(closed_table('COMPONENT96.csv'))==64
    for point in POINTS[:2]:
        own=[r for r in virtual if r['point']==point]
        assert s['qualify'](own,'success')==old['ORIGINAL_VIRTUAL_QUALIFICATION'][point]
        assert s['qualify'](own,'reliable')==old['RELIABLE_VIRTUAL_QUALIFICATION'][point]
    print('PASS_CPU_TYPED_REUSE_ORIGINAL_AND_GATE_DENOMINATOR_PAIRED_DELTA_NO_QUERY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true')
    if p.parse_args().self_check:self_check()
    else:main()
