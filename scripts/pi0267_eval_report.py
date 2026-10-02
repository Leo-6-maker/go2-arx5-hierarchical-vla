"""Read actual episode/output arrays; independent Euler/codec/numerical replay and fixed tables."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
sys.path[:0]=[str(ROOT/'code'),str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src'),str(DATA/'qa')]
from pi0255_motion3 import write,sha
from pi0267_metrics_v3 import score
from pi0267_r1_cpu import independent_entry


def read(p):return json.loads(Path(p).read_text())


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def independent_metrics(decoded,expert):
    from scipy.spatial.transform import Rotation
    def matrices(x):
        a=x[:,:3]/np.linalg.norm(x[:,:3],axis=1)[:,None]
        b=x[:,3:]-np.sum(a*x[:,3:],axis=1)[:,None]*a;b=b/np.linalg.norm(b,axis=1)[:,None]
        return np.stack([a,b,np.cross(a,b)],axis=1)
    r,e=matrices(decoded[:,3:9]),matrices(expert[:,3:9]);angles=Rotation.from_matrix(r@e.transpose(0,2,1)).magnitude()
    xyz=np.linalg.norm(decoded[:,:3]-expert[:,:3],axis=1)
    return dict(xyz_rmse_m=float(np.sqrt(np.mean(xyz**2))),so3_rmse_rad=float(np.sqrt(np.mean(angles**2))),
        gripper_mae=float(np.mean(abs(decoded[:,9]-expert[:,9]))),first_xyz_error_m=float(xyz[0]),endpoint_xyz_error_m=float(xyz[-1]),
        max_linear_speed_mps=float(np.linalg.norm(np.diff(decoded[:,:3],axis=0),axis=1).max()/.05),
        max_angular_speed_radps=float(Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).magnitude().max()/.05))


def primary():
    assert read(R1/'evidence/MAIN_TRAINING_AUDIT.json')['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY'
    import torch
    import pi0260_r2_codec as codec
    processor=codec.processor(read(R1/'design/statistics.json'));points=('donor','N4096','R4096');rows=[];checks=[]
    for point in points:
        cases=sorted((RAW/'virtual'/point).glob('*/result.json'));assert len(cases)==64
        for path in cases:
            record=read(path);assert record['validity']=='VALID' and record['point']==point and sha(path.parent/'trajectory.npz')==record['arrays_sha256']
            z=np.load(path.parent/'trajectory.npz');spec=json.loads(str(z['spec_json']))
            result=score(z['time'],z['actual_p'],z['actual_r'],z['installed_p'],z['installed_r'],spec,record['scorer']['safety_failure'])
            assert result==record['scorer'],path
            if spec['task']=='M3':
                broad,entry,arc=independent_entry(z['time'],z['actual_p'],z['actual_r'],spec)
                assert entry==result['entry_index'] and (None if broad is None else float(z['time'][broad]))==result['earliest_broad_readiness_s']
            error=np.linalg.norm(z['actual_p']-z['installed_p'],axis=1)
            rows.append(dict(point=point,family=record['family'],task=record['task'],command=record['command'],backend=record['backend'],
                success=record['success'],reason=record['stop_reason'] or result.get('reason'),queries=len(record['queries']),elapsed_s=record['actual_elapsed_s'],
                bins=len(result['ordered_bins']),reverse=result['reverse'],shortcut=result['gap_shortcut'],V2_success=result['legacy_v2']['success'],
                entry_time_s=result.get('entry_time_s'),final_xyz_m=result['final_xyz_m'],final_so3_rad=result['final_so3_rad'],
                same_tick_tracking_xyz_rms_m=float(np.sqrt(np.mean(error**2))),dangerous_collision=bool(record['collision_events'])))
            checks.append(dict(point=point,case=path.parent.name,shared_scorer_replay=True,independent_entry_replay=True,arrays_sha256=record['arrays_sha256']))
    prediction_checks=[]
    for path in sorted(RAW.rglob('prediction.npz')):
        if '/offline/' not in str(path) and '/virtual/' not in str(path):continue
        z=np.load(path);raw=z['raw'];finite=bool(np.isfinite(raw).all() and np.isfinite(z['decoded']).all())
        assert np.array_equal(z['actual_initial_noise'],torch.from_numpy(z['noise']).to(torch.bfloat16).float().numpy())
        tx,tv=z['trace_x'],z['trace_v'];assert tx.shape==tv.shape==(4,1,40,132)
        worst=0.
        for i in range(4):
            expected=(torch.from_numpy(tx[i]).to(torch.bfloat16)+(.25*torch.from_numpy(tv[i]).to(torch.bfloat16))*torch.ones_like(torch.from_numpy(tx[i]).to(torch.bfloat16))).float().numpy()
            actual=tx[i+1] if i<3 else raw
            assert np.array_equal(expected,actual,equal_nan=True),('ACTUAL_NATIVE_BF16_EULER_REPLAY',path,i)
        if finite:
            decoded=codec.decode(processor,raw[0],z['state']);difference=float(np.max(abs(decoded-z['decoded'][0])))
            assert difference<=1e-7,('ACTUAL_CODEC_CPU_REPLAY',path,difference)
        else:difference=None
        prediction_checks.append(dict(file=str(path),sha256=sha(path),native_Euler_four_steps_exact=True,codec_max_abs=difference))
    offline=[];metric_checks=[]
    for point in points:
        for domain,n in (('new',256),('legacy',440)):
            records=sorted((RAW/'offline'/point/domain).glob('*/record.json'));assert len(records)==n
            for path in records:
                rec=read(path);assert rec['status']=='VALID' and sha(rec['tensor'])==rec['tensor_sha256'];z=np.load(rec['tensor'])
                if domain=='new':expert=np.load(RAW/'offline/source'/rec['row']['uid']/'input.npz')['expert']
                else:expert=z['expert'] if 'expert' in z else None
                if expert is None:
                    sys.path.insert(0,str(BASE/'PI0264/code'));from pi0264_freeze_eval import source
                    _,expert=source(rec['row'])
                metrics=rec['metrics']
                if metrics is not None:
                    computed=independent_metrics(z['decoded'][0].astype(float),expert.astype(float))
                    worst=max(abs(computed[k]-metrics[k]) for k in computed);assert worst<=1e-6,(point,domain,rec['row']['uid'],worst)
                else:worst=None
                metric_checks.append(dict(point=point,domain=domain,uid=rec['row']['uid'],independent_metrics_max_abs=worst))
                offline.append(dict(point=point,domain=domain,uid=rec['row']['uid'],gate=rec['gate']['decision'],**(metrics or {})))
    groups=[]
    for point in points:
        own=[r for r in rows if r['point']==point]
        for task in ('M1','M3'):
            for backend in ('E0','E1'):
                g=[r for r in own if r['task']==task and r['backend']==backend]
                c=[sum(r['success'] for r in g if r['command']==x) for x in (0,1)];assert len(g)==16
                groups.append(dict(point=point,task=task,backend=backend,success=sum(r['success'] for r in g),total=16,command_success=c,
                    dangerous_collisions=sum(r['dangerous_collision'] for r in g),PASS=sum(r['success'] for r in g)>=13 and min(c)>=6 and not any(r['dangerous_collision'] for r in g)))
    candidates=[p for p in ('N4096','R4096') if all(g['PASS'] for g in groups if g['point']==p)]
    out=RAW/'delivery';out.mkdir(exist_ok=True);csv_write(out/'VIRTUAL192.csv',rows);csv_write(out/'OFFLINE2088.csv',offline)
    write(out/'CPU_REPLAY.json',dict(status='PASS_ACTUAL_NATIVE_EULER_CODEC_INDEPENDENT_METRICS_AND_SHARED_SCORER_REPLAY',episodes=checks,predictions=prediction_checks,
        offline_metrics=metric_checks,model_queries=0,gradient_queries=0,new_physics_steps=0,scorer_shared=True,entry_geometry_independent=True))
    write(out/'PRIMARY_VERDICT.json',dict(status='PRIMARY_COMPLETE',quality='VIRTUAL_RESEARCH_CANDIDATE_QUALIFIED' if candidates else 'VALID_QUALITY_FAIL',
        groups=groups,virtual_research_candidates=candidates,virtual_episodes=192,offline_rows=2088,legacy_donor_cache=440,
        old_V2_FAIL_preserved=True,V3_contract_change_not_learning_gain=True,scope='New geometries / procedural virtual tool / one fixed noise stream / not cross-robot',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    lines=['# PI0267-R1 fixed primary quality','','Actual N4096/R4096; common pinned donor; selected/deployment NONE.','',
        'Original PRIMARY_PARTIAL/V2/64 remain unchanged. V3 readmission is an entry-contract change; it is not a learned gain.','',
        '| Point | Task/backend | Success | Command0/1 | Dangerous collision | Original qualification |','|---|---|---|---|---|---|']
    for g in groups:lines.append(f"| {g['point']} | {g['task']}/{g['backend']} | {g['success']}/16 | {g['command_success'][0]}/8, {g['command_success'][1]}/8 | {g['dangerous_collisions']} | {'PASS' if g['PASS'] else 'FAIL'} |")
    lines+=['','Qualified virtual research candidates: '+(', '.join(candidates) or 'NONE')+'. No formal selection or deployment.',
        '','## Fixed paired scene outcomes','','| Family/task/command/backend | Donor | N | R |','|---|---|---|---|']
    for key in sorted({(r['family'],r['task'],r['command'],r['backend']) for r in rows}):
        own={r['point']:r for r in rows if (r['family'],r['task'],r['command'],r['backend'])==key}
        lines.append('| '+ '/'.join(map(str,key))+' | '+' | '.join(('PASS' if own[p]['success'] else 'FAIL: '+str(own[p]['reason'])) for p in points)+' |')
    lines+=['','## Offline diagnostics under separate reference contracts','',
        '| Point | Domain | Median XYZ RMSE m | Median SO3 RMSE rad | REJECT/rows |','|---|---|---|---|---|']
    for point in points:
        for domain in ('new','legacy'):
            own=[r for r in offline if (r['point'],r['domain'])==(point,domain)];finite=[r for r in own if 'xyz_rmse_m' in r]
            xyz=float(np.median([r['xyz_rmse_m'] for r in finite])) if finite else None;angle=float(np.median([r['so3_rmse_rad'] for r in finite])) if finite else None
            lines.append(f"| {point} | {domain} | {xyz} | {angle} | {sum(r['gate']=='REJECT' for r in own)}/{len(own)} |")
    lines+=['','New-domain arclength-teacher RMSE and OLD time-paced-reference RMSE answer different questions. Donor legacy440 is exact-cache reuse.',
        'VIRTUAL192.csv preserves the full fixed denominator and failures; OFFLINE2088.csv preserves all rows. Candidate qualification uses each task/backend and command subgroup, without replacing them by averages.',
        'Independent CPU path replays saved BF16 Euler operations and decoder outputs, separately computes row-6D/geodesic metrics and entry geometry; the task scorer remains shared and is disclosed.',
        'This is procedural virtual-tool/new-geometry transfer with one fixed sampler stream. It does not establish physical, hardware or cross-embodiment generalization. Bounded WBC results are reported separately.','']
    (out/'PRIMARY_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PRIMARY_COMPLETE',groups=groups,candidates=candidates)),flush=True)


def secondary():
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from pi0258_geometry import full_fk,homogeneous
    from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
    from pi0265_metrics import so3
    cases=read(R1/'design/TEACHER_CARD.json')['diagnostics'];assert len(cases)==32
    identity=read(R1/'evidence/SOURCE_IDENTITY.json');urdf_path=next(r['path'] for r in identity['files'] if r['name']=='native_urdf')
    urdf=ET.parse(urdf_path).getroot();kin=NativeArmKinematics.from_urdf(urdf_path);table=[];checks=[]
    for case in cases:
        paths=sorted((RAW/'diagnostics'/case['case_id']).glob('attempt-*/result.json'));valid=[p for p in paths if read(p)['validity']=='VALID']
        assert len(valid)<=1
        if not valid:
            attempted=sum(1 for p in (RAW/'diagnostics'/case['case_id']).glob('attempt-*/model/*prediction.receipt.json'))
            table.append(dict(case=case['case_id'],point=case['point'],family=case['family']['family_id'],task=case['task'],command=case['command'],
                validity='NOT_RUN_OR_ENGINEERING_INCOMPLETE',success=False,reason='SECONDARY_SYSTEM_DEFECT' if paths else 'NOT_RUN',new_model_queries=attempted));continue
        path=valid[0];result=read(path);states=read(path.parent/'rows.json');entrycheck=None
        if (path.parent/'world_tcp.npz').exists():
            z=np.load(path.parent/'world_tcp.npz');pp=[];rr=[]
            for s in states:
                root=np.array(s['root']);joints=dict(zip(result['dof_names'],np.asarray(s['DOF']).reshape(-1,2)[:,0]))
                transform=homogeneous(Rotation.from_quat(root[3:7]).as_matrix(),root[:3]);pose=full_fk(urdf,joints,transform)['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
                pp.append(pose[:3,3]);rr.append(pose[:3,:3])
            pe=float(np.linalg.norm(np.array(pp)-z['position'],axis=1).max());re=float(so3(rr,z['rotation']).max());assert pe<=1e-5 and re<=1e-4
            spec=read(path.parent/'task_spec.json');computed=score(z['elapsed'],z['position'],z['rotation'],z['installed_position'],z['installed_rotation'],spec,result['safety_failure'])
            if result.get('failure') in ('ORIGINAL_RAW_GATE_REJECT','CLOCK_GATED_REFERENCE_REJECT') or (case['point']!='expert' and result.get('failure') is not None):computed['success']=False
            assert computed==result['metrics']
            if case['task']=='M3':
                broad,entry,arc=independent_entry(z['elapsed'],z['position'],z['rotation'],spec);assert entry==computed['entry_index']
                entrycheck=dict(broad=broad,acquired=entry)
            checks.append(dict(case=case['case_id'],independent_full_URDF_FK_xyz_m=pe,independent_full_URDF_FK_SO3_rad=re,scorer_shared=True,entry=entrycheck))
        metrics=result.get('metrics',{})
        table.append(dict(case=case['case_id'],point=case['point'],family=case['family']['family_id'],task=case['task'],command=case['command'],validity='VALID',
            success=result['success'],reason=result.get('failure'),safety=result.get('safety_failure'),new_model_queries=result['GR00T_queries'],
            elapsed_s=result.get('actual_task_s'),final_xyz_m=metrics.get('final_xyz_m'),final_so3_rad=metrics.get('final_so3_rad'),
            bins=len(metrics.get('ordered_bins',[])),V2_success=metrics.get('legacy_v2',{}).get('success')))
    out=RAW/'delivery';out.mkdir(exist_ok=True);csv_write(out/'SECONDARY32.csv',table)
    budget=read(R1/'evidence/RAW_BUDGET.json');caps=read(R1/'design/BUDGET_CARD.json')
    query_ledger=R1/'evidence/model_attempts.jsonl'
    actual_queries=sum(json.loads(s)['request'].get('noise_registry')=='SECONDARY_CARD.json' for s in query_ledger.read_text().splitlines()) if query_ledger.exists() else 0
    assert budget['starts']<=36 and budget['simulated_seconds']<=1476+1e-6 and actual_queries<=2400
    accounting=[]
    for path in sorted((RAW/'diagnostics').glob('*/attempt-*/physics_started.json')):
        folder=path.parent;result=read(folder/'result.json');n=sum(1 for _ in (folder/'physics_steps.jsonl').open())
        actors=len(read(folder/'actor_calls.json'));assert actors==result.get('WBC_actor_calls',actors)
        accounting.append(dict(attempt=str(folder),ordinal=read(path)['ordinal'],ticks=n,actor_calls=actors,simulated_seconds=n*.02))
    assert sum(r['ticks'] for r in accounting)==budget['control_ticks'] and sum(r['actor_calls'] for r in accounting)==budget['actor_calls']
    assert abs(sum(r['simulated_seconds'] for r in accounting)-budget['simulated_seconds'])<=1e-5
    write(out/'SECONDARY_CPU_REPLAY.json',dict(status='PASS_INDEPENDENT_FULL_TREE_FK_AND_SHARED_SCORER',checks=checks,actual_physics_accounting=accounting,
        original_budget=budget,model_queries=0,new_physics=0,local_witness_not_global_reachability_proof=True))
    write(out/'SECONDARY_VERDICT.json',dict(status='SECONDARY_COMPLETE' if all(r['validity']=='VALID' for r in table) else 'SECONDARY_PARTIAL',
        outcomes=table,actual_model_attempts_including_invalid=actual_queries,scope='First-two-TRAIN WBC executor-transfer diagnostic / domain shift and following separate',
        does_not_change_primary_learning=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(rows=len(table),valid=sum(r['validity']=='VALID' for r in table),successes=sum(r['success'] for r in table),physics=budget)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('primary','secondary'));a=p.parse_args()
    primary() if a.mode=='primary' else secondary()
