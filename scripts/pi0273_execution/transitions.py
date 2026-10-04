"""Saved same-tick transitions and post-completion failures; never changes V3."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from quality_worker import BASE,ROOT,read,sha,write,admitted
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')


def angle(a,b):return float(Rotation.from_matrix(a@b.T).magnitude())


def classify(rec,t,p,r,spec):
    scorer=rec['scorer'];goal=np.asarray(spec['goal']);target_r=np.asarray(spec['goal_rotation'])
    xyz=np.linalg.norm(p-goal,axis=1);so3=Rotation.from_matrix(r@target_r.T).magnitude()
    window=t>=35.-1e-9;complete=abs(t[-1]-40.)<=1e-6 and window.sum()==251
    hit=scorer['time_to_success_s'];after=t>hit if hit is not None else np.zeros(len(t),dtype=bool)
    exits=np.flatnonzero(after&((xyz>.035)|(so3>.10)))
    flags=[]
    if not scorer['success']:flags.append('ORIGINAL_TASK_NOT_COMPLETED')
    if rec['stop_reason']:flags.append('SAFETY_STOP')
    if rec['task']=='M3' and scorer['entry_index'] is None:flags.append('NO_OBSERVABLE_ENTRY')
    if rec['task']=='M3' and scorer['entry_index'] is not None and not scorer['success']:flags.append('POST_ENTRY_PATH_OR_HOLD_FAILURE')
    if hit is not None and hit>35.:flags.append('LATE_COMPLETION')
    if not complete:flags.append('INCOMPLETE_FINAL_WINDOW')
    if np.any(window&(xyz>.035)):flags.append('FINAL_WINDOW_XYZ_EXCEEDED')
    if np.any(window&(so3>.10)):flags.append('FINAL_WINDOW_SO3_EXCEEDED')
    if len(exits):flags.append('EXIT_AFTER_FIRST_SUCCESS')
    reverse=None;negative=None
    if rec['task']=='M3' and after.any():
        arc=np.unwrap(np.arctan2(p[:,1]-spec['center'][1],p[:,0]-spec['center'][0]))
        progress=(1 if rec['command']==0 else -1)*arc*spec['radius']
        post=progress[t>=hit]
        reverse=float(np.max(np.maximum.accumulate(post)-post));negative=float(-np.minimum(np.diff(post),0).sum())
    return dict(task=rec['task'],command=rec['command'],backend=rec['backend'],original_success=scorer['success'],
        original_completion_s=hit,flags=flags,final_window_complete=bool(complete),observed_final_window_ticks=int(window.sum()),
        final_window_xyz_outside_ticks=int(np.sum(window&(xyz>.035))),final_window_so3_outside_ticks=int(np.sum(window&(so3>.10))),
        first_exit_after_success_s=float(t[exits[0]]) if len(exits) else None,post_success_max_backward_excursion_m=reverse,
        post_success_total_negative_variation_m=negative,original_V3_unchanged=True,descriptive_not_new_qualification=True)


def main():
    admitted();assert read(RAW/'delivery/CPU_REPLAY.json')['status']=='PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES'
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    sys.path.insert(0,str(BASE/'PI0267/code'))
    from pi0267_eval_report import csv_write
    diagnoses=[]
    anchor=np.asarray(read(BASE/'PI0267/design/SCENE_CARD.json')['policy_anchor_world_m']);transitions=[];m3=[]
    for path in sorted(p for point in ('BROAD5120','TARGET5120') for p in (RAW/'virtual'/point).glob('*/result.json')):
        rec=read(path);assert rec['arrays_sha256']==sha(path.parent/'trajectory.npz')
        if rec['task']=='M3':
            s=rec['scorer'];m3.append(dict(point=rec['point'],case=path.parent.name,family=rec['family'],command=rec['command'],backend=rec['backend'],
                original_success=s['success'],entry_index=s['entry_index'],entry_time_s=s['entry_time_s'],ordered_bins=s['ordered_bins'],
                reverse_violation=s['reverse'],max_backward_excursion_m=s['max_backward_excursion_m'],
                total_negative_variation_m=s['total_negative_variation_m'],reason=s['reason'],
                gap_shortcut=s['gap_shortcut'],original_V3_and_V2_shared=True))
        with np.load(path.parent/'trajectory.npz',allow_pickle=False) as z:
            times=z['time'].copy();actual=z['actual_p'].copy();installed=z['installed_p'].copy();spec=json.loads(str(z['spec_json']))
            actual_r=z['actual_r'].copy();installed_r=z['installed_r'].copy()
        goal=np.asarray(spec['goal']);goal_r=np.asarray(spec['goal_rotation']);gates={g['replan']:g['result'] for g in rec['gates']}
        diagnoses.append(dict(point=rec['point'],case=path.parent.name,**classify(rec,times,actual,actual_r,spec)))
        for q in rec['queries']:
            request=q['query'];prediction=Path(request['output']);assert sha(prediction)==q['prediction_sha256']
            with np.load(request['input'],allow_pickle=False) as z:current=z['position_world'].copy();current_r=z['rotation_world'].copy();now=float(z['timestamp'])
            with np.load(prediction,allow_pickle=False) as z:decoded=z['decoded'][0].copy()
            row=dict(point=rec['point'],case=path.parent.name,family=rec['family'],task=rec['task'],command=rec['command'],backend=rec['backend'],
                     query_index=q['replan'],timestamp_s=now,prediction_sha256=sha(prediction),current_target_xyz_m=float(np.linalg.norm(current-goal)),
                     current_target_so3_rad=angle(current_r,goal_r),
                     gate_decision=gates.get(q['replan'],{}).get('decision','NOT_INSTALLED'),
                     actual_prefix_executed=False,predicted_reference_is_not_achieved=True)
            if np.isfinite(decoded).all():
                try:reference_r=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
                except ValueError:row['prediction_status']='INVALID_ROTATION'
                else:
                    endpoint=decoded[8,:3].astype(float)+anchor
                    row.update(prediction_status='FINITE_VALID_ROTATION',original_model_reference_0p4_target_xyz_m=float(np.linalg.norm(endpoint-goal)),
                               original_model_reference_first_point_bias_m=float(np.linalg.norm(decoded[0,:3]+anchor-current)),
                               original_model_reference_0p4_target_so3_rad=angle(reference_r[8],goal_r),
                               original_model_reference_first_point_bias_rad=angle(reference_r[0],current_r))
            else:row['prediction_status']='NONFINITE'
            endpoint_tick=np.flatnonzero(np.abs(times-(now+.4))<=1e-9)
            if row['gate_decision'] in ('ACCEPT','WARN') and len(endpoint_tick)==1:
                k=int(endpoint_tick[0]);row.update(actual_prefix_executed=True,
                    installed_reference_same_tick_target_xyz_m=float(np.linalg.norm(installed[k]-goal)),
                    achieved_same_tick_target_xyz_m=float(np.linalg.norm(actual[k]-goal)),
                    same_tick_tracking_xyz_m=float(np.linalg.norm(actual[k]-installed[k])),
                    achieved_target_error_change_m=float(np.linalg.norm(actual[k]-goal)-np.linalg.norm(current-goal)),
                    installed_reference_same_tick_target_so3_rad=angle(installed_r[k],goal_r),
                    achieved_same_tick_target_so3_rad=angle(actual_r[k],goal_r),
                    same_tick_tracking_so3_rad=angle(actual_r[k],installed_r[k]),
                    achieved_target_error_change_rad=angle(actual_r[k],goal_r)-angle(current_r,goal_r))
            else:row['execution_missingness']='GATE_NOT_INSTALLED' if row['gate_decision']=='REJECT' else 'NO_COMPLETE_0P4_INTERVAL'
            transitions.append(row)
    assert len(diagnoses)==128 and len(m3)==64
    out=RAW/'delivery'
    csv_write(out/'SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv',transitions)
    csv_write(out/'FAILURE_AND_POST_SUCCESS_DIAGNOSIS128.csv',diagnoses)
    write(out/'TRANSITION_ANALYSIS.json',dict(status='PASS_SAVED_TRANSITIONS_AND_POST_SUCCESS_DIAGNOSIS',
        transitions=len(transitions),M3_entry_and_coverage=m3,episodes=diagnoses,
        tables={name:sha(out/name) for name in ('SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv','FAILURE_AND_POST_SUCCESS_DIAGNOSIS128.csv')},
        source_sha256=sha(__file__),shared_original_V3=True,model_queries=0,teacher_calls=0,renders=0,physics=0))
    print('PASS_SAVED_TRANSITIONS_AND_POST_SUCCESS_DIAGNOSIS',len(transitions),flush=True)


def self_check():
    t=np.arange(2001)*.02;p=np.zeros((2001,3));r=np.repeat(np.eye(3)[None],2001,axis=0)
    rec=dict(task='M1',command=0,backend='E0',stop_reason=None,scorer=dict(success=True,time_to_success_s=34.))
    spec=dict(goal=[0,0,0],goal_rotation=np.eye(3))
    assert classify(rec,t,p,r,spec)['flags']==[]
    p[1750,0]=.03500001;result=classify(rec,t,p,r,spec)
    assert result['final_window_xyz_outside_ticks']==1 and result['first_exit_after_success_s']==35.
    assert 'FINAL_WINDOW_SO3_EXCEEDED' not in result['flags']
    p[1750,0]=.035;assert classify(rec,t,p,r,spec)['flags']==[]
    assert 'INCOMPLETE_FINAL_WINDOW' in classify(rec,t[:-1],p[:-1],r[:-1],spec)['flags']
    rx=Rotation.from_euler('x',90,degrees=True).as_matrix();rz=Rotation.from_euler('z',90,degrees=True).as_matrix()
    assert abs(angle(rx,rz)-2*np.pi/3)<1e-12
    print('PASS_CONSTRUCTED_SAME_TICK_AND_POST_SUCCESS_BOUNDARIES',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    self_check() if parser.parse_args().self_check else main()
