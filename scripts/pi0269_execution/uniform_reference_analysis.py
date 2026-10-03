"""Completed UNIFORM main64 only: source-bound reference, tracking and drift."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/liuyu-nfs/Projects/go2-arx5/PI0269/code')
from time_analysis import BASE, ROOT, RAW, sha, errors
sys.path.append(str(BASE/'PI0268/R1/code'))
from reliability import reliable

out = RAW/'delivery'
out.mkdir(exist_ok=True)
paths = sorted((RAW/'virtual/UNIFORM4096').glob('*/result.json'))
assert len(paths) == 64
rows = []
for path in paths:
    record = json.loads(path.read_text())
    assert record['point'] == 'UNIFORM4096' and record['validity'] == 'VALID'
    assert sha(path.parent/'trajectory.npz') == record['arrays_sha256']
    with np.load(path.parent/'trajectory.npz', allow_pickle=False) as z:
        t,p,r,ip,ir = (z[k] for k in ('time','actual_p','actual_r','installed_p','installed_r'))
        spec = json.loads(str(z['spec_json']))
        e = errors(t,p,r,ip,ir,np.asarray(spec['goal']),np.asarray(spec['goal_rotation']))
        assert len(t) == 2001 and abs(t[-1]-40) < 1e-6
        score = record['scorer']
        hit = score['time_to_success_s']
        retention = reliable(t,p,r,spec['goal'],spec['goal_rotation'],score,
            gate_refusals=sum(g['result']['decision']=='REJECT' for g in record['gates']),
            collisions=len(record['collision_events']),execution_failure=record['stop_reason'] is not None)
        exit_time = retention['first_pose_exit_after_success_s']
        rows.append(dict(point='UNIFORM4096',case=path.parent.name,task=record['task'],command=record['command'],backend=record['backend'],
            original_success=score['success'],reliable=retention['reliable'],first_success_s=hit,
            first_pose_exit_after_success_s=exit_time,
            dwell_until_first_exit_s=exit_time-hit if exit_time is not None else None,
            final_achieved_target_xyz_m=float(e['achieved_terminal_target_xyz_m'][-1]),
            final_reference_target_xyz_m=float(e['reference_terminal_target_xyz_m'][-1]),
            final_achieved_target_so3_rad=float(e['achieved_terminal_target_so3_rad'][-1]),
            max_same_tick_tracking_xyz_m=float(e['tracking_xyz_m'].max()),
            max_same_tick_tracking_so3_rad=float(e['tracking_so3_rad'].max()),
            final5s_outside_ticks=retention['final5s_outside_ticks'],
            broad_ready=score.get('earliest_broad_readiness_s') is not None,
            entered=score.get('entry_time_s') is not None,
            ordered_bins=len(score['ordered_bins']),reverse=score['reverse'],gap_shortcut=score['gap_shortcut'],
            source_result_sha256=sha(path),source_arrays_sha256=record['arrays_sha256']))
csv_path = out/'UNIFORM_MAIN64_REFERENCE_TRACKING_DRIFT.csv'
with csv_path.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

family = next(f['family_id'] for f in json.loads((BASE/'PI0267/design/SCENE_CARD.json').read_text())['families'] if f['split']=='NEW_SIM_EVAL')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
chapters=[]
for column,(command,backend) in enumerate((c,b) for c in (0,1) for b in ('E0','E1')):
    case=f'{family}-M1-c{command}-{backend}'
    folder=RAW/'virtual/UNIFORM4096'/case
    record=json.loads((folder/'result.json').read_text())
    with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
        spec=json.loads(str(z['spec_json']));t=z['time']
        e=errors(t,z['actual_p'],z['actual_r'],z['installed_p'],z['installed_r'],np.asarray(spec['goal']),np.asarray(spec['goal_rotation']))
        hit=record['scorer']['time_to_success_s'];row=next(r for r in rows if r['case']==case)
        for metric,scale,tolerance,ylabel,axis in [('xyz_m',1000,35,'XYZ distance (mm)',axes[0,column]),('so3_rad',1,.10,'SO3 distance (rad)',axes[1,column])]:
            axis.plot(t,e['reference_terminal_target_'+metric]*scale,color='#e68613',label='Installed reference to target')
            axis.plot(t,e['achieved_terminal_target_'+metric]*scale,color='#2874b2',linestyle='--',label='Achieved pose to target')
            axis.plot(t,e['tracking_'+metric]*scale,color='#31913c',label='Same-tick tracking error')
            axis.axhline(tolerance,color='gray',linestyle=':',label='Original pose bound')
            axis.axvspan(35,40,color='gray',alpha=.12)
            if hit is not None:axis.axvline(hit,color='#8d4ba7',linestyle=':',label='Original completion')
            if row['first_pose_exit_after_success_s'] is not None:axis.axvline(row['first_pose_exit_after_success_s'],color='#b83232',linestyle=':',label='First later pose exit')
            axis.set_xlim(0,40);axis.set_ylim(bottom=0);axis.set_ylabel(ylabel);axis.grid(alpha=.2)
        axes[0,column].set_title(f'M1 c{command}/{backend}: original={record["success"]}, reliable={row["reliable"]}')
        axes[1,column].set_xlabel('Saved virtual task time (s)')
        chapters.append(dict(case=case,arrays_sha256=record['arrays_sha256']))
fig.suptitle('PI0269 UNIFORM4096: fixed first EVAL family, all four M1 conditions\nVIRTUAL MATHEMATICAL EXECUTOR / NOT WBC / NOT PHYSICS / all late motion retained')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
fig.tight_layout(rect=(0,.10,1,.92))
figure=out/'UNIFORM_FIXED_FIRST_FAMILY_REFERENCE_ACHIEVED.png'
fig.savefig(figure,dpi=150);plt.close(fig)
summary={}
for task in ('M1','M3'):
    own=[r for r in rows if r['task']==task]
    success=[r for r in own if r['original_success']]
    durations=[r['dwell_until_first_exit_s'] for r in success if r['dwell_until_first_exit_s'] is not None]
    summary[task]=dict(total=len(own),original_success=len(success),successful_then_exited=sum(r['first_pose_exit_after_success_s'] is not None for r in success),
        median_first_exit_delay_s=float(np.median(durations)) if durations else None,
        median_final_xyz_m=float(np.median([r['final_achieved_target_xyz_m'] for r in own])),
        broad_ready=sum(r['broad_ready'] for r in own),entered=sum(r['entered'] for r in own),
        reversed=sum(r['reverse'] for r in own),shortcut=sum(r['gap_shortcut'] for r in own),
        max_ordered_bins=max(r['ordered_bins'] for r in own))
receipt=dict(status='PASS_COMPLETE_UNIFORM64_SAVED_REFERENCE_TRACKING_DRIFT_ANALYSIS_PREFIX_PENDING',
    rows=64,summary=summary,max_same_tick_tracking_xyz_m=max(r['max_same_tick_tracking_xyz_m'] for r in rows),
    max_same_tick_tracking_so3_rad=max(r['max_same_tick_tracking_so3_rad'] for r in rows),
    csv_sha256=sha(csv_path),figure_sha256=sha(figure),figure_selection='First frozen EVAL family, all four M1 commands/backends; not selected by result',
    figure_sources=chapters,source_sha256=sha(Path(__file__)),same_tick_helper_sha256=sha(ROOT/'code/time_analysis.py'),
    target_semantics='Original static final task pose, not a teacher time trajectory',
    model_queries=0,teacher_calls=0,render_calls=0,physics_steps=0,not_WBC=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
(out/'UNIFORM_REFERENCE_TRACKING_DRIFT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
