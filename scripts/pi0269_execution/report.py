"""Saved-output replay plus prespecified prefix, retention and task verdicts."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json
import sys
import importlib.util
from pathlib import Path
import numpy as np
from infer_worker import BASE,ROOT,read,sha,write,admitted
sys.path[:0]=[str(BASE/'PI0267/code'),str(BASE/'PI0264/code')]
sys.path.append(str(BASE/'PI0268/R1/code'))
from pi0267_eval_report import independent_metrics,csv_write,score,independent_entry,so3
spec=importlib.util.spec_from_file_location('pi0268_closed_report',BASE/'PI0268/R1/code/report.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
qualify,summary,replay=helpers.qualify,helpers.summary,helpers.replay
from reliability import reliable
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
OLD_RAW=RAW.parent/'PI0268/R1'
POINTS=('ROT2048','UNIFORM4096','PREFIX4096')


def prefix_metrics(pred,expert,state):
    if not np.isfinite(pred).all():return dict(prefix_xyz_rmse_m=float('inf'),prefix_so3_rmse_rad=float('inf'),suffix_xyz_rmse_m=float('inf'),suffix_so3_rmse_rad=float('inf'),prefix_endpoint_error_m=float('inf'),teacher_hold=None,direction_cosine=None,signed_alignment_m=None)
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    error=np.linalg.norm(pred[:,:3]-expert[:,:3],axis=1)
    angle=so3(rot6d_rows_to_matrix(pred[:,3:9]),rot6d_rows_to_matrix(expert[:,3:9]))
    target=expert[8,:3]-state[:3];generated=pred[8,:3]-state[:3];length=float(np.linalg.norm(target));hold=length<=1e-12
    return dict(prefix_xyz_rmse_m=float(np.sqrt(np.mean(error[:9]**2))),prefix_so3_rmse_rad=float(np.sqrt(np.mean(angle[:9]**2))),
        suffix_xyz_rmse_m=float(np.sqrt(np.mean(error[9:]**2))),suffix_so3_rmse_rad=float(np.sqrt(np.mean(angle[9:]**2))),
        prefix_endpoint_error_m=float(error[8]),teacher_hold=hold,target_displacement_m=length,
        predicted_residual_motion_m=float(np.linalg.norm(generated)),
        direction_cosine=None if hold or np.linalg.norm(generated)<=1e-12 else float(np.dot(generated,target)/np.linalg.norm(generated)/length),
        signed_alignment_m=None if hold else float(np.dot(generated,target)/length))


def main():
    admitted()
    import torch
    import pi0260_r2_codec as codec
    from pi0264_freeze_eval import source
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('SAVED_ARRAY_REPORT_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    processors={'ROT':codec.processor(read(ROOT/'contracts/ROT/source/dataset/statistics.json'))}
    virtual=[];entry_checks=[];predictions=[];offline=[];metric_checks=[]
    for point in POINTS:
        arm='ROT'
        pointroot=OLD_RAW if point=='ROT2048' else RAW
        folder=pointroot/'virtual'/point
        cases=sorted(folder.glob('*/result.json'));assert len(cases)==64
        for path in cases:
            record=read(path);assert record['validity']=='VALID' and record['point']==point
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
            records=sorted((pointroot/'offline'/point/domain).glob('*/record.json'));assert len(records)==n
            for path in records:
                rec=read(path);row=rec['row'];assert rec['status']=='VALID' and sha(rec['tensor'])==rec['tensor_sha256']
                with np.load(rec['tensor']) as z:decoded=z['decoded'][0].astype(float);state=z['state'].copy()
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
                    noise=row.get('noise') or (Path(row['noise_file']).name+':'+row['noise_name']),source_row=json.dumps(row,sort_keys=True),**(metrics or {}),**prefix_metrics(decoded,expert.astype(float),state)))
        # Reuse the closed donor native replay; only new outputs need new arithmetic replay.
        for lane in (() if point=='ROT2048' else ('offline','virtual')):
            for path in sorted((RAW/lane/point).rglob('prediction.npz')):predictions.append(replay(path,processors[arm],torch,codec))
    original={p:qualify([r for r in virtual if r['point']==p],'success') for p in POINTS}
    reliable_verdict={p:qualify([r for r in virtual if r['point']==p],'reliable') for p in POINTS}
    diagnostics=[]
    component=read(ROOT/'design/COMPONENT_CARD.json')
    for point in POINTS:
        for proposal in component['rows']:
            path=RAW/'component'/point/proposal['case']/'result.json';record=read(path)
            assert record['component_start']==proposal and record['point']==point
            row=dict(point=point,**proposal,validity=record['validity'],source_sha256=sha(path),qualification_authority=False)
            if record['validity']=='PREQUERY_EXCLUDED':
                assert not record['queries'];row.update(reasons=record['reasons'],queries=0)
            else:
                assert record['validity']=='VALID' and sha(path.parent/'trajectory.npz')==record['arrays_sha256']
                with np.load(path.parent/'trajectory.npz') as z:
                    t,p,r=z['time'],z['actual_p'],z['actual_r'];spec=json.loads(str(z['spec_json']))
                    result=score(t,p,r,z['installed_p'],z['installed_r'],spec,record['scorer']['safety_failure']);assert result==record['scorer']
                    goal_error=np.linalg.norm(p-np.array(spec['goal']),axis=1);goal_angle=so3(r,np.array(spec['goal_rotation']))
                    pose=(goal_error<=.035)&(goal_angle<=.10)
                    exits=np.flatnonzero(~pose)
                    angle=np.unwrap(np.arctan2(p[:,1]-spec['center'][1],p[:,0]-spec['center'][0]))
                    progress=(1 if proposal['command']==0 else -1)*(angle-angle[0])*spec['radius']
                    remaining={'M3_ENTRY':12,'M3_MIDARC':6,'M3_TERMINAL':0}.get(proposal['start_type'],0)
                    ordered=[];skipped_bin=False
                    for value in progress:
                        current_bin=int(np.floor(value/(np.pi/8*spec['radius'])+1e-10))
                        if current_bin>len(ordered)+1 and len(ordered)<remaining:skipped_bin=True
                        if not skipped_bin and len(ordered)<remaining and current_bin==len(ordered)+1:
                            ordered.append(len(ordered)+1)
                    row.update(queries=len(record['queries']),elapsed_s=float(t[-1]),original_V3=result,
                        full8s=abs(float(t[-1])-8.)<1e-6,original_V2=result['legacy_v2'],
                        terminal_start=proposal['start_type'].endswith('TERMINAL'),
                        time_in_goal_pose_s=float(pose[1:].sum()*.02),first_pose_exit_s=float(t[exits[0]]) if len(exits) else None,
                        final_goal_xyz_m=float(goal_error[-1]),final_goal_so3_rad=float(goal_angle[-1]),
                        signed_local_arc_motion_m=float(progress[-1]) if proposal['task']=='M3' else None,
                        max_reversal_m=float(np.max(np.maximum.accumulate(progress)-progress)) if proposal['task']=='M3' else None,
                        diagnostic_angular_bins=ordered,diagnostic_remaining_bins=remaining,
                        diagnostic_skipped_angular_bin=skipped_bin,
                        diagnostic_bins_are_angle_only_not_original_task_pass=True,
                        max_radial_departure_m=float(abs(np.linalg.norm(p[:,:2]-np.array(spec['center'])[:2],axis=1)-spec['radius']).max()) if proposal['task']=='M3' else None,
                        gate_refusals=sum(g['result']['decision']=='REJECT' for g in record['gates']),
                        collision_events=record['collision_events'],stop_reason=record['stop_reason'],
                        arrays_sha256=record['arrays_sha256'],not_complete_original_task_from_midarc=True)
                for query in record['queries']:
                    predictions.append(replay(Path(query['query']['output']),processors['ROT'],torch,codec))
            diagnostics.append(row)
    assert len(offline)==2904 and len(virtual)==192 and len(diagnostics)==96
    index={(r['point'],r['domain'],r['uid']):r for r in offline};paired=[]
    for before,after,name in [('ROT2048','UNIFORM4096','UNIFORM_CONTINUATION'),('ROT2048','PREFIX4096','PREFIX_CONTINUATION'),('UNIFORM4096','PREFIX4096','TEMPORAL_TREATMENT')]:
        for a in [r for r in offline if r['point']==before]:
            b=index[after,a['domain'],a['uid']];assert a['source_row']==b['source_row']
            row=dict(contrast=name,before=before,after=after,domain=a['domain'],uid=a['uid'],task=a['task'],command=a['command'],family=a['family'],noise=a['noise'],variant=a['variant'],before_gate=a['gate'],after_gate=b['gate'])
            for metric in ('xyz_rmse_m','so3_rmse_rad','prefix_xyz_rmse_m','prefix_so3_rmse_rad','suffix_xyz_rmse_m','suffix_so3_rmse_rad','prefix_endpoint_error_m'):
                aa,bb=a.get(metric),b.get(metric)
                row['delta_'+metric]=bb-aa if aa is not None and bb is not None and np.isfinite(aa) and np.isfinite(bb) else None
            paired.append(row)
    contrast=[]
    for task in ('M1','M3'):
        rows=[r for r in paired if r['contrast']=='TEMPORAL_TREATMENT' and r['domain']=='fit' and r['task']==task]
        values=[r['delta_prefix_xyz_rmse_m'] for r in rows]
        delta=float(np.median(values)) if all(v is not None for v in values) else None
        contrast.append(dict(task=task,rows=len(rows),paired_median_delta_prefix_xyz_rmse_m=delta,PASS=delta is not None and delta<0))
    panels={point:{domain:summary([r for r in offline if r['point']==point and r['domain']==domain]) for domain in ('fit','factor','legacy')} for point in POINTS}
    reject_ok=all(panels['PREFIX4096'][d]['REJECT']<=panels['UNIFORM4096'][d]['REJECT'] for d in ('factor','legacy'))
    legacy_ok={}
    for point in POINTS[1:]:
        a,b=panels['ROT2048']['legacy'],panels[point]['legacy']
        legacy_ok[point]=b['REJECT']<=a['REJECT'] and all(a[k] is not None and b[k] is not None and b[k]<=a[k] for k in ('median_xyz_rmse_m','median_so3_rmse_rad'))
    candidates=[p for p in POINTS[1:] if original[p]['PASS'] and reliable_verdict[p]['PASS'] and legacy_ok[p]]
    out=RAW/'delivery';out.mkdir(exist_ok=True)
    csv_write(out/'OFFLINE2904.csv',offline);csv_write(out/'VIRTUAL192.csv',virtual);csv_write(out/'COMPONENT96.csv',diagnostics)
    csv_write(out/'PAIRED_OFFLINE_CHANGES.csv',paired)
    subgroups=[]
    for point in POINTS:
        for domain,axes in [('fit',('task','command','variant','noise')),('factor',('task','command','factor_kind','factor_axis','factor_sign')),('legacy',('noise',))]:
            own=[r for r in offline if r['point']==point and r['domain']==domain]
            for key in sorted({tuple(r[k] for k in axes) for r in own},key=str):
                group=[r for r in own if tuple(r[k] for k in axes)==key]
                subgroups.append(dict(point=point,domain=domain,**dict(zip(axes,key)),**summary(group),
                    median_prefix_xyz_rmse_m=float(np.median([r['prefix_xyz_rmse_m'] for r in group]))))
    csv_write(out/'OFFLINE_SUBGROUPS.csv',subgroups)
    write(out/'CPU_REPLAY.json',dict(status='PASS_SAVED_ARRAY_REPLAY',episodes=entry_checks,predictions=predictions,metrics=metric_checks,
        closed_donor_CPU_replay_sha256=sha(OLD_RAW/'delivery/CPU_REPLAY.json'),model_queries=0,gradient_queries=0,physics_steps=0,
        scorer_shared=True,entry_and_metric_paths_independent=True,diagnostic_scoring_has_no_qualification_authority=True))
    write(out/'PRIMARY_VERDICT.json',dict(status='PRIMARY_EVALUATION_COMPLETE',TEMPORAL_TREATMENT_CONTRAST=dict(PASS=all(r['PASS'] for r in contrast) and reject_ok,tasks=contrast,no_factor_legacy_refusal_increase=reject_ok),
        ORIGINAL_VIRTUAL_QUALIFICATION=original,RELIABLE_VIRTUAL_QUALIFICATION=reliable_verdict,panels=panels,legacy_retention_vs_donor=legacy_ok,
        reliable_virtual_research_candidates=candidates,offline_rows=2904,virtual_episodes=192,component_proposals=96,
        interpretation='RELIABLE_VIRTUAL_RESEARCH_CANDIDATE' if candidates else 'VALID_QUALITY_FAIL',
        not_WBC_not_physical_not_blind_TEST=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PRIMARY_EVALUATION_COMPLETE',len(offline),len(virtual),len(diagnostics),'CANDIDATES',candidates,flush=True)

def self_check():
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix();rot=np.tile(r[:2].reshape(6),(40,1))
    label=np.c_[np.zeros((40,3)),rot,np.full(40,.5)];state=np.r_[np.zeros(3),r[:2].reshape(6),.5]
    assert prefix_metrics(label,label,state)['teacher_hold']
    assert prefix_metrics(label,label,state)['direction_cosine'] is None
    pred=label.copy();pred[:9,0]=-.008;label[8,0]=.008
    metric=prefix_metrics(pred,label,state);assert metric['direction_cosine']==-1 and metric['prefix_endpoint_error_m']==.016
    assert np.isinf(prefix_metrics(pred*np.nan,label,state)['prefix_xyz_rmse_m'])
    (ROOT/'runtime').mkdir(parents=True,exist_ok=True)
    helpers.self_check()
    print('PASS_NONLEARNED_PREFIX_HOLD_DIRECTION_FAILURE_AND_ORIGINAL_QUALIFICATION_CPU')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true')
    if p.parse_args().self_check:self_check()
    else:main()
