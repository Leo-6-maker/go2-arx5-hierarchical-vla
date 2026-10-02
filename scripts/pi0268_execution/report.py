"""Saved-array CPU replay and three separate prospective verdicts; no policy calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from infer_worker import BASE,ROOT,read,sha,write,admitted
sys.path[:0]=[str(BASE/'PI0267/code'),str(BASE/'PI0264/code')]
from pi0267_eval_report import independent_metrics,csv_write,score,independent_entry,so3
from reliability import reliable
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1')
OLD_RAW=RAW.parents[1]/'PI0267/R1'
POINTS=('OLD0','ROT0','OLD2048','ROT2048')


def qualify(rows,label):
    assert len(rows)==64
    safe=all(not (r['gate_refusals'] or r['dangerous_collision'] or r['execution_failure']) for r in rows)
    groups=[]
    for task in ('M1','M3'):
        for backend in ('E0','E1'):
            group=[r for r in rows if (r['task'],r['backend'])==(task,backend)];assert len(group)==16
            commands=[]
            for c in (0,1):
                own=[r for r in group if r['command']==c];assert len(own)==8
                commands.append(sum(bool(r[label]) for r in own))
            total=sum(commands)
            groups.append(dict(task=task,backend=backend,success=total,total=16,command_success=commands,PASS=total>=13 and min(commands)>=6 and safe))
    return dict(PASS=all(r['PASS'] for r in groups),full64_safe=safe,groups=groups)


def summary(rows):
    assert rows
    result=dict(total=len(rows),REJECT=sum(r['gate']=='REJECT' for r in rows),nonfinite=sum(r['numerical_failure'] for r in rows))
    for name in ('xyz_rmse_m','so3_rmse_rad'):
        value=float(np.median([r.get(name,float('inf')) for r in rows]))
        result['median_'+name]=value if np.isfinite(value) else None
    return result


def replay(path,processor,torch,codec):
    with np.load(path) as z:
        raw=z['raw'];tx,tv=z['trace_x'],z['trace_v']
        assert tx.shape==tv.shape==(4,1,40,132)
        assert np.array_equal(z['actual_initial_noise'],torch.from_numpy(z['noise']).to(torch.bfloat16).float().numpy())
        for i in range(4):
            x=torch.from_numpy(tx[i]).to(torch.bfloat16);v=torch.from_numpy(tv[i]).to(torch.bfloat16)
            expected=(x+(.25*v)*torch.ones_like(x)).float().numpy()
            assert np.array_equal(expected,tx[i+1] if i<3 else raw,equal_nan=True),(path,i)
        finite=bool(np.isfinite(raw).all() and np.isfinite(z['decoded']).all())
        difference=None
        if finite:
            decoded=codec.decode(processor,raw[0],z['state']).astype(np.float32)
            difference=float(np.max(abs(decoded-z['decoded'][0])));assert difference<=1e-7,(path,difference)
    return dict(file=str(path),sha256=sha(path),native_four_step_Euler_exact=True,codec_max_abs=difference)


def paired_tables(out,offline,virtual):
    index={(r['point'],r['domain'],r['uid']):r for r in offline};pairs=[]
    for before,after,meaning in [('OLD0','OLD2048','OLD_WITHIN_CONTRACT_LEARNING'),('ROT0','ROT2048','ROT_WITHIN_CONTRACT_LEARNING'),
                                  ('OLD0','ROT0','SAME_WEIGHTS_CONTRACT_CHANGE'),('OLD2048','ROT2048','MATCHED_FINAL_COMBINED_TREATMENT')]:
        keys={(r['domain'],r['uid']) for r in offline if r['point']==before};assert len(keys)==968
        for domain,uid in sorted(keys):
            a,b=index[before,domain,uid],index[after,domain,uid]
            assert a['source_row']==b['source_row']
            row=dict(contrast=meaning,before=before,after=after,domain=domain,uid=uid,task=a['task'],command=a['command'],
                family=a['family'],variant=a['variant'],factor_kind=a['factor_kind'],factor_axis=a['factor_axis'],noise=a['noise'],
                before_gate=a['gate'],after_gate=b['gate'],before_numerical_failure=a['numerical_failure'],after_numerical_failure=b['numerical_failure'])
            for metric in ('xyz_rmse_m','so3_rmse_rad','first_xyz_error_m','endpoint_xyz_error_m'):
                row['delta_'+metric]=b[metric]-a[metric] if metric in a and metric in b else None
            pairs.append(row)
    csv_write(out/'PAIRED_OFFLINE_CHANGES.csv',pairs)
    subgroups=[]
    for point in POINTS:
        for domain,axes in [('fit',('task','command','variant','noise')),('factor',('task','command','factor_kind','factor_axis','factor_sign')),('legacy',('noise',))]:
            own=[r for r in offline if r['point']==point and r['domain']==domain]
            for key in sorted({tuple(r[k] for k in axes) for r in own},key=str):
                group=[r for r in own if tuple(r[k] for k in axes)==key]
                subgroups.append(dict(point=point,domain=domain,**dict(zip(axes,key)),**summary(group)))
    csv_write(out/'OFFLINE_SUBGROUPS.csv',subgroups)
    lines=['# PI0268-R1 fixed normalization study','',
        'All four points and all fixed conditions are retained. Formal selected/deployment: NONE.',
        'OLD0 and ROT0 use identical weights with different processors. Their difference is not fine-tuning gain.',
        'Native losses use different numerical scales; physical errors and gate outcomes are compared below.','',
        '## Original task and independent final-five-second retention','',
        '| Point | Task/backend | Original /16 | Reliable /16 | Reliable command0/1 /8 | Collisions | Gate refusals |',
        '|---|---|---:|---:|---|---:|---:|']
    for point in POINTS:
        for task in ('M1','M3'):
            for backend in ('E0','E1'):
                g=[r for r in virtual if (r['point'],r['task'],r['backend'])==(point,task,backend)]
                commands=[sum(r['reliable'] for r in g if r['command']==c) for c in (0,1)]
                lines.append(f"| {point} | {task}/{backend} | {sum(r['success'] for r in g)} | {sum(r['reliable'] for r in g)} | {commands[0]}, {commands[1]} | {sum(r['dangerous_collision'] for r in g)} | {sum(r['gate_refusals'] for r in g)} |")
    lines+=['','## Physical fitting and original retention panel','',
        '| Point | Panel | Rows | REJECT | Nonfinite | Median XYZ RMSE (m) | Median SO3 RMSE (rad) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for point in POINTS:
        for domain in ('fit','factor','legacy'):
            s=summary([r for r in offline if r['point']==point and r['domain']==domain])
            lines.append(f"| {point} | {domain} | {s['total']} | {s['REJECT']} | {s['nonfinite']} | {s['median_xyz_rmse_m']} | {s['median_so3_rmse_rad']} |")
    lines+=['','These aggregates do not replace subgroup qualification. OFFLINE_SUBGROUPS.csv retains each fixed task/command/variant/noise or factor condition. OFFLINE3872.csv keeps family, source identity, all metrics and all failures.',
        'PAIRED_OFFLINE_CHANGES.csv separately reports each contract\'s own-step0 learning, same-weight contract change and final treatment comparison. Negative error deltas are improvements; undefined changes remain blank rather than converting failures into numerical scores.',
        'VIRTUAL256.csv separates installed-reference endpoint, achieved endpoint, same-tick tracking, original V2/V3, gate/collision, first-success exit and every-tick final-five-second retention. The policy did not use those scores to stop or correct its output.',
        'A nonfinite physical metric is ranked as positive infinity in fixed-denominator median summaries; an infinite median is reported null. Finite-only averages do not silently remove those rows.',
        'CPU replay recomputes saved native BF16 Euler arithmetic and decoding, independent physical error metrics and independent entry geometry. Original task scoring shares the frozen V3 code; that limitation remains explicit.',
        'These are related development conditions from eight existing scene families and fixed noise streams. E0/E1 are mathematical executors, not robots. No blind TEST, physical transfer, real-time or population reliability claim is made.','']
    (out/'PRIMARY_REPORT.md').write_text('\n'.join(lines))


def main():
    admitted()
    import torch
    import pi0260_r2_codec as codec
    from pi0264_freeze_eval import source
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('SAVED_ARRAY_REPORT_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    processors={a:codec.processor(read(ROOT/'contracts'/a/'source/dataset/statistics.json')) for a in ('OLD','ROT')}
    virtual=[];entry_checks=[];predictions=[];offline=[];metric_checks=[]
    for point in POINTS:
        arm='ROT' if point.startswith('ROT') else 'OLD'
        folder=OLD_RAW/'virtual/donor' if point=='OLD0' else RAW/'virtual'/point
        cases=sorted(folder.glob('*/result.json'));assert len(cases)==64
        for path in cases:
            record=read(path);assert record['validity']=='VALID' and record['point']==('donor' if point=='OLD0' else point)
            assert sha(path.parent/'trajectory.npz')==record['arrays_sha256']
            with np.load(path.parent/'trajectory.npz') as z:
                spec=json.loads(str(z['spec_json']));t,p,r=z['time'],z['actual_p'],z['actual_r']
                result=score(t,p,r,z['installed_p'],z['installed_r'],spec,record['scorer']['safety_failure'])
                assert result==record['scorer']
                if spec['task']=='M3':
                    broad,entry,arc=independent_entry(t,p,r,spec)
                    assert entry==result['entry_index'] and (None if broad is None else float(t[broad]))==result['earliest_broad_readiness_s']
                refusals=sum(g['result']['decision']=='REJECT' for g in record['gates'])
                collisions=len(record['collision_events'])
                failure=record['stop_reason'] is not None
                retention=reliable(t,p,r,spec['goal'],spec['goal_rotation'],result,gate_refusals=refusals,collisions=collisions,execution_failure=failure)
                virtual.append(dict(point=point,family=record['family'],task=record['task'],command=record['command'],backend=record['backend'],
                    success=result['success'],reliable=retention['reliable'],gate_refusals=refusals,dangerous_collision=bool(collisions),execution_failure=failure,
                    queries=len(record['queries']),reason=record['stop_reason'] or result.get('reason'),elapsed_s=float(t[-1]),
                    bins=len(result['ordered_bins']),entry_time_s=result.get('entry_time_s'),V2_success=result['legacy_v2']['success'],
                    actual_final_xyz_m=result['final_xyz_m'],actual_final_so3_rad=result['final_so3_rad'],
                    installed_final_xyz_m=float(np.linalg.norm(z['installed_p'][-1]-spec['goal'])),
                    installed_final_so3_rad=float(so3(z['installed_r'][-1],np.asarray(spec['goal_rotation']))),
                    same_tick_tracking_xyz_rms_m=float(np.sqrt(np.mean(np.sum((p-z['installed_p'])**2,axis=1)))),
                    retention=json.dumps(retention,sort_keys=True)))
                entry_checks.append(dict(point=point,case=path.parent.name,arrays_sha256=record['arrays_sha256'],shared_V3_exact=True,independent_entry=True,reliability=retention))
        for domain,n in (('fit',224),('factor',304),('legacy',440)):
            records=sorted((RAW/'offline'/point/domain).glob('*/record.json'));assert len(records)==n
            for path in records:
                rec=read(path);row=rec['row'];assert rec['status']=='VALID' and sha(rec['tensor'])==rec['tensor_sha256']
                with np.load(rec['tensor']) as z:decoded=z['decoded'][0].astype(float)
                if domain=='legacy':_,expert=source(row)
                else:
                    with np.load(row['input']) as z:expert=z['raw_action'].copy()
                metrics=rec['metrics'];difference=None
                if metrics is not None:
                    independent=independent_metrics(decoded,expert.astype(float))
                    difference=max(abs(independent[k]-metrics[k]) for k in independent);assert difference<=1e-6,(point,domain,row['uid'],difference)
                metric_checks.append(dict(point=point,domain=domain,uid=row['uid'],independent_metrics_max_abs=difference,tensor_sha256=rec['tensor_sha256']))
                transform=row.get('transform',{})
                offline.append(dict(point=point,domain=domain,uid=row['uid'],gate=rec['gate']['decision'],numerical_failure=metrics is None,
                    task=row.get('task'),command=row.get('command'),family=row.get('family'),variant=row.get('variant'),anchor=row.get('anchor'),
                    factor_kind=transform.get('kind'),factor_axis=transform.get('axis'),factor_sign=transform.get('sign'),
                    noise=row.get('noise') or (Path(row['noise_file']).name+':'+row['noise_name']),source_row=json.dumps(row,sort_keys=True),**(metrics or {})))
        # OLD0 cached traces already have their exact full CPU replay; only new outputs need replay.
        for lane in ('offline','virtual'):
            for path in sorted((RAW/lane/point).rglob('prediction.npz')):predictions.append(replay(path,processors[arm],torch,codec))
    original={p:qualify([r for r in virtual if r['point']==p],'success') for p in POINTS}
    reliable_verdict={p:qualify([r for r in virtual if r['point']==p],'reliable') for p in POINTS}
    contrast=[]
    for task in ('M1','M3'):
        own={p:summary([r for r in offline if r['point']==p and r['domain']=='factor' and r['task']==task and r['factor_kind']=='combined' and r['factor_axis'] in (0,1)]) for p in POINTS}
        old,rot=own['OLD2048'],own['ROT2048']
        passed=rot['REJECT']<=old['REJECT']
        passed=passed and rot['median_xyz_rmse_m'] is not None and old['median_xyz_rmse_m'] is not None and rot['median_xyz_rmse_m']<old['median_xyz_rmse_m']
        contrast.append(dict(task=task,points=own,PASS=passed))
    legacy={p:summary([r for r in offline if r['point']==p and r['domain']=='legacy']) for p in POINTS}
    a,b=legacy['OLD2048'],legacy['ROT2048']
    no_regression=b['REJECT']<=a['REJECT'] and all(b[k] is not None and a[k] is not None and b[k]<=a[k] for k in ('median_xyz_rmse_m','median_so3_rmse_rad'))
    rejection_improved=any(r['points']['ROT2048']['REJECT']<r['points']['OLD2048']['REJECT'] for r in contrast)
    rejection_reduction_met=rejection_improved or all(r['points']['OLD2048']['REJECT']==0 for r in contrast)
    numerical=all(r['PASS'] for r in contrast) and no_regression and rejection_reduction_met
    candidates=[p for p in ('OLD2048','ROT2048') if original[p]['PASS'] and reliable_verdict[p]['PASS']]
    out=RAW/'delivery';out.mkdir(exist_ok=True)
    csv_write(out/'VIRTUAL256.csv',virtual);csv_write(out/'OFFLINE3872.csv',offline)
    paired_tables(out,offline,virtual)
    write(out/'CPU_REPLAY.json',dict(status='PASS_SAVED_ARRAY_REPLAY',episodes=entry_checks,predictions=predictions,metrics=metric_checks,
        old_cached_CPU_replay_sha256=sha(OLD_RAW/'delivery/CPU_REPLAY.json'),model_queries=0,gradient_queries=0,physics_steps=0,scorer_shared=True,entry_and_metric_paths_independent=True))
    write(out/'PRIMARY_VERDICT.json',dict(status='PRIMARY_COMPLETE',NUMERICAL_TREATMENT_EFFECT=dict(PASS=numerical,XY_coupled_by_task=contrast,legacy=legacy,no_legacy_regression=no_regression,rejection_reduction_met=rejection_reduction_met),
        ORIGINAL_VIRTUAL_QUALIFICATION=original,RELIABLE_VIRTUAL_QUALIFICATION=reliable_verdict,reliable_virtual_research_candidates=candidates,
        interpretation='RELIABLE_VIRTUAL_RESEARCH_CANDIDATE' if candidates else ('NUMERICAL_GAIN_ONLY' if numerical else 'VALID_QUALITY_FAIL'),
        offline_rows=3872,virtual_episodes=256,exact_OLD0_offline_reuse=440,exact_OLD0_virtual_reuse=64,
        same_weights_different_processors_are_different_initial_policies=True,not_WBC_not_physical_not_blind_TEST=True,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PRIMARY_COMPLETE',len(offline),len(virtual),'CANDIDATES',candidates,flush=True)


def self_check():
    rows=[dict(task=t,backend=b,command=c,success=True,reliable=True,gate_refusals=0,dangerous_collision=False,execution_failure=False) for t in ('M1','M3') for b in ('E0','E1') for c in (0,1) for _ in range(8)]
    assert qualify(rows,'reliable')['PASS']
    rows[0]['gate_refusals']=1;assert not qualify(rows,'reliable')['PASS'];rows[0]['gate_refusals']=0
    for i in range(3):rows[i]['reliable']=False
    assert not qualify(rows,'reliable')['PASS'] and qualify(rows,'success')['PASS']
    rows[2]['reliable']=True;assert qualify(rows,'reliable')['PASS']
    import tempfile
    offline=[]
    for point in POINTS:
        for domain,n in [('fit',224),('factor',304),('legacy',440)]:
            for i in range(n):
                offline.append(dict(point=point,domain=domain,uid=str(i),task='M1',command=0,family='fixture',variant=0,
                    factor_kind='nominal',factor_axis=None,factor_sign=0,noise='B0',source_row=str(i),
                    gate='REJECT' if i==0 else 'PASS',numerical_failure=i==0,
                    **({} if i==0 else dict(xyz_rmse_m=.02,so3_rmse_rad=.03,first_xyz_error_m=.02,endpoint_xyz_error_m=.02))))
    virtual=[dict(r,point=point) for point in POINTS for r in rows]
    with tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='cpu-report-fixture-') as folder:
        out=Path(folder);paired_tables(out,offline,virtual)
        import csv
        with (out/'PAIRED_OFFLINE_CHANGES.csv').open() as stream:paired=list(csv.DictReader(stream))
        assert len(paired)==3872 and paired[0]['delta_xyz_rmse_m']==''
        assert all(r['delta_xyz_rmse_m'] in ('','0.0') for r in paired)
    print('PASS_PROSPECTIVE_GROUP_AND_FULL_EPISODE_SAFETY_BOUNDARIES',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    if parser.parse_args().self_check:self_check()
    else:main()
