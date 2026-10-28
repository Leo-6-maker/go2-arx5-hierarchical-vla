"""Read complete fixed-point arrays; separate installed motion, tracking and task events."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(ROOT/'code'),str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import sha,write
from pi0265_metrics import so3
from pi0267_r1_cpu import independent_entry


def read(p):return json.loads(p.read_text())


def main(point):
    families=[f['family_id'] for f in read(ROOT/'design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
    expected={f'{f}-{t}-c{c}-{b}' for f in families for t in ('M1','M3') for c in (0,1) for b in ('E0','E1')}
    paths=sorted((RAW/'virtual'/point).glob('*/result.json'));assert len(expected)==64 and {p.parent.name for p in paths}==expected
    rows=[]
    for p in paths:
        r=read(p);assert r['validity']=='VALID' and r['point']==point
        assert r['evaluation_card_sha256']==sha(R1/'design/EVALUATION_CARD.json') and r['arrays_sha256']==sha(p.parent/'trajectory.npz')
        with np.load(p.parent/'trajectory.npz') as z:
            spec=json.loads(str(z['spec_json']));t=z['time'];actual=z['actual_p'];installed=z['installed_p'];s=r['scorer']
            xyz=np.linalg.norm(actual-spec['goal'],axis=1);angle=so3(z['actual_r'],np.asarray(spec['goal_rotation']))
            error=np.linalg.norm(actual-installed,axis=1)
            assert abs(float(xyz[-1])-s['final_xyz_m'])<1e-12 and abs(float(angle[-1])-s['final_so3_rad'])<1e-12
            acquired=installed_acquired=broad=None
            if r['task']=='M3':
                broad,acquired,_=independent_entry(t,actual,z['actual_r'],spec)
                _,installed_acquired,_=independent_entry(t,installed,z['installed_r'],spec)
                assert acquired==s['entry_index']
            hit=s['time_to_success_s'];after=np.flatnonzero(t>=hit-1e-9) if hit is not None else np.array([],dtype=int)
            good=(xyz<=.035)&(angle<=.10);left=after[~good[after]]
            rows.append(dict(case=p.parent.name,point=point,task=r['task'],command=r['command'],backend=r['backend'],original_success=r['success'],
                reason=r['stop_reason'] or s.get('reason') or ('ORIGINAL_TASK_HOLD_NOT_MET' if not r['success'] else None),
                queries=len(r['queries']),gate_counts=dict(Counter(g['result']['decision'] for g in r['gates'])),
                achieved_entry_s=None if acquired is None else float(t[acquired]),installed_entry_s=None if installed_acquired is None else float(t[installed_acquired]),
                broad_ready=broad is not None,ordered_bins=len(s['ordered_bins']),reverse=s['reverse'],shortcut=s['gap_shortcut'],
                original_first_hold_s=hit,actual_elapsed_s=float(t[-1]),achieved_final_xyz_m=float(xyz[-1]),achieved_final_so3_rad=float(angle[-1]),
                installed_final_xyz_m=float(np.linalg.norm(installed[-1]-spec['goal'])),same_tick_tracking_rms_m=float(np.sqrt(np.mean(error**2))),
                same_tick_tracking_max_m=float(error.max()),final_pose_inside_original_bounds=bool(good[-1]),
                first_pose_exit_after_original_hold_s=None if not len(left) else float(t[left[0]]),
                result_sha256=sha(p),trajectory_sha256=r['arrays_sha256']))
    groups=[]
    for task in ('M1','M3'):
        for backend in ('E0','E1'):
            own=[r for r in rows if (r['task'],r['backend'])==(task,backend)];assert len(own)==16
            gates=Counter()
            for r in own:gates.update(r['gate_counts'])
            groups.append(dict(task=task,backend=backend,episodes=16,original_success=sum(r['original_success'] for r in own),gate_counts=dict(gates),
                reasons=dict(Counter(str(r['reason']) for r in own)),installed_entry=sum(r['installed_entry_s'] is not None for r in own),
                achieved_entry=sum(r['achieved_entry_s'] is not None for r in own),broad_ready=sum(r['broad_ready'] for r in own),
                ordered_bins_histogram=dict(Counter(str(r['ordered_bins']) for r in own)),
                median_achieved_final_xyz_m=float(np.median([r['achieved_final_xyz_m'] for r in own])),
                median_installed_final_xyz_m=float(np.median([r['installed_final_xyz_m'] for r in own])),
                median_same_tick_tracking_rms_m=float(np.median([r['same_tick_tracking_rms_m'] for r in own])),
                max_same_tick_tracking_m=max(r['same_tick_tracking_max_m'] for r in own),
                successful_episodes_outside_pose_bounds_at_end=sum(r['original_success'] and not r['final_pose_inside_original_bounds'] for r in own)))
    receipt=dict(status='COMPLETE_FIXED_POINT_ARRAY_DECOMPOSITION',point=point,groups=groups,rows=rows,source_sha256=sha(Path(__file__)),
        qualification_unchanged=True,post_hold_pose_retention_is_diagnostic_only=True,installed_entry_is_array_diagnostic_not_an_extra_episode=True,
        model_queries=0,new_renders=0,new_physics_steps=0,scope='VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP')
    write(R1/'evidence'/('VIRTUAL_DECOMPOSITION_'+point+'.json'),receipt);print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point',choices=('donor','N4096','R4096'));main(p.parse_args().point)
