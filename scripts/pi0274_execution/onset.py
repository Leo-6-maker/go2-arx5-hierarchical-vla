"""Read all128 saved trajectories and original visibility; no new query or rasterization."""
import csv,hashlib,json,os,resource,sys
from collections import Counter
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274');OLD=ROOT.parent/'PI0273'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')
sys.path[:0]=[str(OLD/'code'),str(ROOT.parent/'PI0267/code'),str(ROOT.parent/'PI0266/code'),str(ROOT.parent/'PI0250/runtime/reference-317a/src')]
from quality_gate import admitted,read,sha
from pi0259_observation import visibility
from pi0255_motion3 import write
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

def first(t,mask):
    indexes=np.flatnonzero(mask);return float(t[indexes[0]]) if len(indexes) else None

def onset(t,p,r,spec,success):
    xyz=np.linalg.norm(p-np.asarray(spec['goal']),axis=1);so3=Rotation.from_matrix(r@np.asarray(spec['goal_rotation']).T).magnitude()
    after=t>success if success is not None else np.zeros(len(t),dtype=bool)
    x=first(t,after&(xyz>.035));a=first(t,after&(so3>.1))
    order='NO_ORIGINAL_SUCCESS' if success is None else 'NEITHER' if x is None and a is None else 'SO3_ONLY' if x is None else 'XYZ_ONLY' if a is None else 'SO3_FIRST' if a<x else 'XYZ_FIRST' if x<a else 'SAME_TICK'
    return dict(first_XYZ_exit_s=x,first_SO3_exit_s=a,exit_order=order,XYZ_at_SO3_exit_m=float(xyz[np.argmin(abs(t-a))]) if a is not None else None,
        SO3_at_XYZ_exit_rad=float(so3[np.argmin(abs(t-x))]) if x is not None else None)

def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    admitted();old_cpu=read(RAW/'delivery/CPU_REPLAY.json');assert old_cpu['status']=='PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES'
    saved=RAW/'delivery/SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv'
    with saved.open() as f:old_transitions={(r['point'],r['case'],int(r['query_index'])):r for r in csv.DictReader(f)}
    scene=read(ROOT.parent/'PI0267/design/SCENE_CARD.json');anchor=np.asarray(scene['policy_anchor_world_m'])
    episodes=[];queries=[];sources=[]
    for point in ('BROAD5120','TARGET5120'):
        paths=sorted((RAW/'virtual'/point).glob('*/result.json'));assert len(paths)==64
        for path in paths:
            rec=read(path);traj=path.parent/'trajectory.npz';assert rec['arrays_sha256']==sha(traj)
            with np.load(traj) as z:t=z['time'].copy();p=z['actual_p'].copy();r=z['actual_r'].copy();ip=z['installed_p'].copy();ir=z['installed_r'].copy();spec=json.loads(str(z['spec_json']))
            hit=rec['scorer']['time_to_success_s'];actual=onset(t,p,r,spec,hit);installed=onset(t,ip,ir,spec,hit)
            row=dict(point=point,case=path.parent.name,task=rec['task'],command=rec['command'],backend=rec['backend'],first_original_success_s=hit,
                **actual,installed_first_XYZ_exit_s=installed['first_XYZ_exit_s'],installed_first_SO3_exit_s=installed['first_SO3_exit_s'],stop_reason=rec['stop_reason'],
                actual_elapsed_s=rec['actual_elapsed_s'],visibility_resolution_s=.4,visibility_order_not_causality=True,trajectory_sha256=sha(traj))
            vis=[];unobserved=0;last_accepted=None;gate={v['replan']:v['result']['decision'] for v in rec['gates']}
            for q in rec['queries']:
                request=q['query'];folder=Path(request['input']).parent;prediction=Path(request['output']);assert sha(prediction)==q['prediction_sha256']
                value=dict(old_transitions[point,path.parent.name,q['replan']]);now=float(value['timestamp_s'])
                with np.load(prediction) as z:d=z['decoded'][0].copy()
                if np.isfinite(d).all():
                    try:rr=rot6d_rows_to_matrix(d[:,3:9].astype(float))
                    except ValueError:value['reference0_status']='INVALID_ROTATION'
                    else:value.update(reference0_target_XYZ_m=float(np.linalg.norm(d[0,:3]+anchor-np.asarray(spec['goal']))),reference0_target_SO3_rad=float(Rotation.from_matrix(rr[0]@np.asarray(spec['goal_rotation']).T).magnitude()))
                else:value['reference0_status']='NONFINITE'
                if (folder/'debug.npz').exists() and (folder/'camera.json').exists():
                    with np.load(folder/'debug.npz') as z:frames={k:z[k].copy() for k in z.files}
                    with np.load(request['input']) as z:frames.update(body=z['body_rgb'].copy(),wrist=z['wrist_rgb'].copy())
                    camera=read(folder/'camera.json');v=visibility(frames,camera['scene_ids'],rec['task'],rec['command'],common_start=False)
                    value.update(visibility_observed=True,visibility_pass=v['pass_necessary'],visibility_failures=v['failures'],visibility_counts=v['counts']);vis.append((now,v['pass_necessary']))
                    sources.extend([dict(path=str(folder/'debug.npz'),sha256=sha(folder/'debug.npz')),dict(path=str(folder/'camera.json'),sha256=sha(folder/'camera.json'))])
                else:value.update(visibility_observed=False,visibility_status='UNOBSERVED');unobserved+=1
                if gate.get(q['replan']) in ('ACCEPT','WARN'):last_accepted=dict(query_index=q['replan'],timestamp_s=now,prediction=str(prediction),sha256=sha(prediction))
                queries.append(value)
            row.update(visibility_samples=len(vis),visibility_UNOBSERVED_queries=unobserved,first_visibility_failure_s=next((x for x,passed in vis if not passed),None),
                first_visibility_failure_after_success_s=next((x for x,passed in vis if hit is not None and x>hit and not passed),None),
                last_accepted_reference=last_accepted,collision_events=rec['collision_events'])
            vf=row['first_visibility_failure_after_success_s'];exits=[x for x in (actual['first_XYZ_exit_s'],actual['first_SO3_exit_s']) if x is not None]
            row['visibility_vs_first_exit']='UNOBSERVED' if not vis else 'NO_POST_SUCCESS_VISIBILITY_FAILURE' if vf is None else 'NO_POSE_EXIT' if not exits else 'VISIBILITY_OBSERVED_FIRST' if vf<min(exits) else 'POSE_EXIT_FIRST' if min(exits)<vf else 'SAME_OBSERVATION_TIME'
            for quantity in ('XYZ','SO3'):
                at=actual[f'first_{quantity}_exit_s']
                if at is not None:
                    k=int(np.argmin(abs(t-at)));row[f'{quantity}_exit_same_tick_tracking_XYZ_m']=float(np.linalg.norm(p[k]-ip[k]));row[f'{quantity}_exit_same_tick_tracking_SO3_rad']=float(Rotation.from_matrix(r[k]@ir[k].T).magnitude())
                    row[f'{quantity}_exit_installed_target_XYZ_m']=float(np.linalg.norm(ip[k]-np.asarray(spec['goal'])));row[f'{quantity}_exit_installed_target_SO3_rad']=float(Rotation.from_matrix(ir[k]@np.asarray(spec['goal_rotation']).T).magnitude())
            episodes.append(row);sources.append(dict(path=str(traj),sha256=sha(traj)))
        print('SAVED_ONSET_COMPLETE',point,len(episodes),flush=True)
    assert len(episodes)==128 and len(queries)==12368
    out=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274/delivery');out.mkdir(parents=True,exist_ok=True)
    def csv_write(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    csv_write('A0_ONSET128.csv',episodes);csv_write('A0_REFERENCE_AND_VISIBILITY12368.csv',queries)
    summaries=[dict(point=point,task=task,exit_orders=dict(Counter(r['exit_order'] for r in episodes if r['point']==point and r['task']==task)),
        visibility_orders=dict(Counter(r['visibility_vs_first_exit'] for r in episodes if r['point']==point and r['task']==task))) for point in ('BROAD5120','TARGET5120') for task in ('M1','M3')]
    write(ROOT/'evidence/A0_ONSET_ANALYSIS.json',dict(status='PASS_ALL128_SAVED_ONSETS_AND_OBSERVED_VISIBILITY',episodes=128,queries=len(queries),summaries=summaries,
        observed_source_hashes=sources,tables={n:sha(out/n) for n in ('A0_ONSET128.csv','A0_REFERENCE_AND_VISIBILITY12368.csv')},shared_visibility_source=str(ROOT.parent/'PI0266/code/pi0259_observation.py'),
        model_queries=0,teacher_calls=0,renders=0,physics=0,temporal_associations_not_causal_proof=True))
    print('PASS_ALL128_SAVED_ONSETS_AND_OBSERVED_VISIBILITY',json.dumps(summaries),flush=True)

if __name__=='__main__':
    t=np.arange(8)*.02;p=np.zeros((8,3));r=Rotation.from_euler('x',[0,0,.11,.12,.13,.14,.15,.16]).as_matrix();p[4:,0]=.04
    check=onset(t,p,r,dict(goal=[0,0,0],goal_rotation=np.eye(3)),0.)
    assert check['exit_order']=='SO3_FIRST' and check['first_SO3_exit_s']==.04 and check['first_XYZ_exit_s']==.08 and check['XYZ_at_SO3_exit_m']==0.
    assert onset(t,p,r,dict(goal=[0,0,0],goal_rotation=np.eye(3)),None)['exit_order']=='NO_ORIGINAL_SUCCESS'
    main()
