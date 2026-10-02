"""Finite PI0267 S0/S1 only; uses frozen sources, no model, renderer or physics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE/'PI0267'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267')
OLD_RAW = RAW.parent/'PI0266'
sys.path.insert(0, str(BASE/'PI0250/runtime/reference-317a/src'))
sys.path.append(str(BASE/'PI0266/code'))
from pi0266_expert import ObservableCorrection, quintic
from pi0266_metrics import score, self_check as scorer_check
from pi0265_metrics import so3
from pi0255_motion3 import scene, write, sha
from pi0267_expert import ArclengthCorrection, self_check as teacher_check


def read(path):
    return json.loads(Path(path).read_text())


def interpolate(p, r, grid, time):
    time = np.clip(np.atleast_1d(time), grid[0], grid[-1])
    return (np.column_stack([np.interp(time, grid, p[:, j]) for j in range(3)]),
            Slerp(grid, Rotation.from_matrix(r))(time).as_matrix())


def prefix_audit():
    out = RAW/'S0'
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    diagnostics = sorted((OLD_RAW/'diagnostics').iterdir())
    assert len(diagnostics) == 16
    for case in diagnostics:
        attempts = [d for d in sorted(case.glob('attempt-*')) if not (d/'INVALIDATION.json').exists()]
        assert len(attempts) == 1, (case, attempts)
        d = attempts[0]
        result, rows = read(d/'result.json'), read(d/'rows.json')
        assert result['validity'] == 'VALID'
        z = np.load(d/'world_tcp.npz')
        windows = read(d/'windows.json')
        frames = sorted((d/'frames').glob('*.npz'))
        indexes = [0, len(frames)//2, len(frames)-1]
        selected = []
        for rank in indexes:
            frame = frames[rank]
            tick = int(frame.stem)
            assert tick+20 < len(rows)
            t = z['elapsed'][tick:tick+21] - z['elapsed'][tick]
            assert np.max(abs(t-np.arange(21)*.02)) < 1e-6
            raw = np.load(frame)
            assert abs(float(raw['time'])-rows[tick]['time']) < 1e-6
            sources = {str(frame): sha(frame)}
            if windows:
                w = windows[rank]
                proposal = Path(w['proposal'])
                assert proposal.parent.name == frame.stem and sha(proposal) == w['proposal_sha256']
                c = np.load(proposal)
                p, r = c['position_world'], c['rotation_world']
                grid = np.arange(40)*.05
                metadata = w['metadata']
                event_path = proposal.parent/'reference_only_install.json'
                installed_path = proposal.parent/'installed_table.npz'
                table = np.load(installed_path)
                assert np.isfinite(table['gamma_p']).all() and np.isfinite(table['tl_t']).all()
                sources.update({str(proposal): sha(proposal), str(installed_path): sha(installed_path)})
                assert np.linalg.norm(p[0]-z['position'][tick]) < 1e-12
                assert so3(r[0], z['rotation'][tick]) < 1e-10
                query_t = t
            else:
                proposal = d/'references/clock_full20s.npz'
                c = np.load(proposal)
                p, r = c['position'], c['rotation']
                grid = c['point_timestamps']-c['point_timestamps'][0]
                metadata = dict(label_rule='CLOCK_DRIVEN', source_phase_s=float(z['elapsed'][tick]),
                                projection_used=False, reference_installed_once=True)
                event_path = d/'clock_reference_only_install.json'
                sources[str(proposal)] = sha(proposal)
                query_t = t+z['elapsed'][tick]
            event = read(event_path)
            sources[str(event_path)] = sha(event_path)
            assert event['physics_root_state_unchanged'] and event['physics_dof_state_unchanged']
            assert not event['physics_reset'] and not event['physics_teleport'] and not event['policy_history_reset']
            requested_p, requested_r = interpolate(p, r, grid, query_t)
            installed_p = z['installed_position'][tick:tick+21].copy()
            installed_r = z['installed_rotation'][tick:tick+21].copy()
            # Right boundary can contain the NEXT chunk's h0. Preserve that raw rotation separately.
            # The old snapshot retained the previous installed position explicitly, but not its rotation.
            boundary = rows[tick+20]
            boundary_has_next = 'installed_before_handoff' in boundary and bool(windows)
            if boundary_has_next:
                installed_p[-1] = boundary['installed_before_handoff']
            achieved_p, achieved_r = z['position'][tick:tick+21], z['rotation'][tick:tick+21]
            name = case.name+'-'+frame.stem+'.npz'
            np.savez_compressed(out/name, elapsed=t, current_p=achieved_p[0], current_r=achieved_r[0],
                requested_p=requested_p, requested_r=requested_r, installed_p=installed_p,
                raw_installed_r=installed_r, achieved_p=achieved_p, achieved_r=achieved_r,
                initial_state_v1=raw['state_v1'])
            rotation_end = 20 if boundary_has_next else 21
            selected.append(dict(selection=('first','middle','final')[len(selected)], tick=tick,
                source_elapsed_s=float(z['elapsed'][tick]), sources=sources, metadata=metadata,
                raw_npz=name, requested_displacement_m=float(np.linalg.norm(requested_p[-1]-requested_p[0])),
                installed_displacement_m=float(np.linalg.norm(installed_p[-1]-installed_p[0])),
                actual_displacement_m=float(np.linalg.norm(achieved_p[-1]-achieved_p[0])),
                actual_target_before_m=float(np.linalg.norm(achieved_p[0]-read(d/'task_spec.json')['goal'])),
                actual_target_after_m=float(np.linalg.norm(achieved_p[-1]-read(d/'task_spec.json')['goal'])),
                requested_installed_xyz_max_m=float(np.max(np.linalg.norm(requested_p-installed_p,axis=1))),
                requested_installed_so3_max_rad=float(np.max(so3(requested_r[:rotation_end],installed_r[:rotation_end]))),
                actual_installed_xyz_rms_m=float(np.sqrt(np.mean(np.sum((achieved_p-installed_p)**2,axis=1)))),
                actual_installed_so3_rms_rad=float(np.sqrt(np.mean(so3(achieved_r[:rotation_end],installed_r[:rotation_end])**2))),
                install_preserved_physics_history=True, right_boundary_rotation_excluded=boundary_has_next,
                next_pose=achieved_p[-1].tolist(), current_pose=achieved_p[0].tolist()))
        cases.append(dict(case=result['case'], result_source=str(d/'result.json'), result_sha256=sha(d/'result.json'),
            historical_success=result['success'], historical_failure=result['failure'], replans=selected))
    report = dict(schema='PI0267_FINITE_ACTUAL_PREFIX_AUDIT_V1', cases=cases, diagnostics=16,
        selected_replans=48, source_raw_modified=False, historical_verdict_changed=False,
        interpretation='Requested/installed differences measure binding/interpolation/timing; they do not alone prove a worker defect. '
                       'Actual tracking uses SAME tick/reference. Projection is label metadata, not physical traversal.',
        no_missing_array_reconstruction=True, new_model_queries=0, robot_physics_steps=0)
    write(ROOT/'evidence/S0_PREFIX_AUDIT.json',report)
    print(json.dumps(dict(stage='S0',diagnostics=16,selected_replans=48)),flush=True)


def constructed_probe():
    expected=read(ROOT/'design/RECEDING_PREFIX_PROBE.json')
    rows=[]
    for variant in ('PI0266_POSITION_SPECIALIZATION','CONSTANT_ARCLENGTH_EXPONENTIAL_CORRECTION'):
        for bias in (0.,.0001,.0005):
            p=0.;hit=None;first=None
            for k in range(100):
                if variant=='PI0266_POSITION_SPECIALIZATION':
                    s=0. if p<=0 else 6. if p>=.1 else __import__('scipy').optimize.brentq(lambda t:.1*quintic(t/6)-p,0.,6.)
                    bar0=.1*quintic(s/6)
                    target=.1*quintic((s+.4)/6)+(1-quintic(.4))*(p-bar0)
                else:
                    bar0=np.clip(p,0.,.1)
                    target=min(.1,bar0+.008)+np.exp(-.8)*(p-bar0)
                next_p=float(target-bias)
                if first is None:first=next_p-p
                p=next_p
                if hit is None and abs(.1-p)<=.005:hit=(k+1)*.4
            rows.append(dict(variant=variant,endpoint_tracking_error_m=bias,first_executed_progress_m=first,
                endpoint_m=p,target_error_m=abs(.1-p),time_within_5mm_s=hit,elapsed_s=40.))
    for actual,want in zip(rows,expected['rows']):
        for k,v in want.items():
            assert actual[k]==v if v is None or isinstance(v,str) else abs(actual[k]-v)<1e-10,(k,actual,want)
    write(ROOT/'evidence/S0_CONSTRUCTED_PROBE.json',dict(expected,rows=rows,
        status='PASS_REEXECUTED_SUPPLIED_NUMERICAL_PROBE',expected_sha256=sha(ROOT/'design/RECEDING_PREFIX_PROBE.json')))


def additional_scorer_fixture():
    t=np.arange(0,40.0001,.02);radius=.04;center=np.array([.3,0,.6]);r=np.repeat(np.eye(3)[None],len(t),axis=0)
    theta=np.linspace(.2,1.5*np.pi,len(t))
    p=center+np.column_stack([radius*np.cos(-3*np.pi/4+theta),radius*np.sin(-3*np.pi/4+theta),np.zeros(len(t))])
    spec=dict(task='M3',command=0,center=center,radius=radius,goal=p[-1],goal_rotation=np.eye(3))
    result=score(t,p,r,p,r,spec)
    assert not result['success'] and result['entry_time_s'] is None
    return dict(status='PASS_MISSED_ENTRY_NEGATIVE',scorer_sha256=sha(BASE/'PI0266/code/pi0266_metrics.py'))


def teacher_matrix():
    card=read(ROOT/'design/SCENE_CARD.json');reg=card['registry']
    families=[f for f in card['families'] if f['split']=='CONTROLLER_DEV'];assert len(families)==4
    write(ROOT/'evidence/S1_FIXTURES.json',dict(scorer=scorer_check(),missed_entry=additional_scorer_fixture(),
        teacher=teacher_check(reg,families)))
    out=RAW/'S1';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for family in families:
        for task in ('M1','M3'):
            for command in (0,1):
                for name,cls in [('OLD',ObservableCorrection),('NEW',ArclengthCorrection)]:
                    for backend in ('E0','E1'):
                        key=f"{family['family_id']}-{task}-c{command}-{name}-{backend}"
                        assert not (out/(key+'.json')).exists(),'first-valid episode already exists'
                        teacher=cls(reg,family,task,command)
                        initial=np.asarray(reg['initial_TCP_world']);p=initial[:3,3].copy();r=initial[:3,:3].copy()
                        positions=[p.copy()];rotations=[r.copy()];installed_p=[p.copy()];installed_r=[r.copy()]
                        replans=[];labels_p=[];labels_r=[];invalid=None
                        for replan in range(100):
                            lp,lr,meta=teacher.label(p,r)
                            assert np.linalg.norm(lp[0]-p)<=1e-12 and so3(lr[0],r)<=1e-10
                            if not meta['valid_label']:
                                invalid='UNOBSERVABLE_LABEL';replans.append(dict(replan=replan,**meta));break
                            desired_p,desired_r=interpolate(lp,lr,np.arange(40)*.05,np.arange(1,21)*.02)
                            prefix_metric=float(np.sqrt(np.sum((desired_p[-1]-p)**2)+(.05*so3(desired_r[-1],r))**2))
                            if name=='NEW' and replan==0:assert prefix_metric>1e-8
                            before=p.copy()
                            for dp,dr in zip(desired_p,desired_r):
                                if backend=='E0':p,r=dp.copy(),dr.copy()
                                else:
                                    alpha=1-np.exp(-.02/.15)
                                    p=p+alpha*(dp-p)
                                    r=Rotation.from_rotvec(alpha*Rotation.from_matrix(dr@r.T).as_rotvec()).as_matrix()@r
                                positions.append(p.copy());rotations.append(r.copy());installed_p.append(dp);installed_r.append(dr)
                            replans.append(dict(replan=replan,reference_prefix_metric_m=prefix_metric,
                                actual_prefix_displacement_m=float(np.linalg.norm(p-before)),current_p=before.tolist(),next_p=p.tolist(),**meta))
                            labels_p.append(lp);labels_r.append(lr)
                        t=np.arange(len(positions))*.02
                        gp,gr=teacher.curve([teacher.end]);data=scene(reg,family)
                        spec=dict(task=task,command=command,goal=gp[0],goal_rotation=gr[0],center=data['center'],radius=data['radius'])
                        result=score(t,np.array(positions),np.array(rotations),np.array(installed_p),np.array(installed_r),spec)
                        if invalid:result['success']=False
                        phases=[m.get('source_phase_s',m.get('phase_s')) for m in replans]
                        case=dict(key=key,family=family['family_id'],task=task,command=command,teacher=name,backend=backend,
                            actual_elapsed_s=float(t[-1]),invalid_label=invalid,success=result['success'],
                            maximum_projection_source_phase_jump_s=float(max(np.diff(phases),default=0.)),
                            result={k:v for k,v in result.items() if k!='progress_rad'},replans=replans)
                        np.savez_compressed(out/(key+'.npz'),time=t,actual_p=positions,actual_r=rotations,
                            installed_p=installed_p,installed_r=installed_r,label_p=labels_p,label_r=labels_r,
                            actual_directed_progress_rad=np.array([np.nan if x is None else x for x in result['progress_rad']]),
                            spec_json=np.array(json.dumps({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in spec.items()})))
                        write(out/(key+'.json'),case);records.append(case)
                        print(json.dumps({k:case[k] for k in ['key','success','actual_elapsed_s','maximum_projection_source_phase_jump_s']}),flush=True)
    groups=[]
    for name in ('OLD','NEW'):
        for backend in ('E0','E1'):
            for task in ('M1','M3'):
                subset=[x for x in records if (x['teacher'],x['backend'],x['task'])==(name,backend,task)]
                groups.append(dict(teacher=name,backend=backend,task=task,success=sum(x['success'] for x in subset),total=len(subset)))
    def wins(backend,task=None):return sum(x['success'] for x in records if x['teacher']=='NEW' and x['backend']==backend and (task is None or x['task']==task))
    passed=wins('E0')==16 and wins('E1')>=12 and all(wins('E1',task)>=6 for task in ('M1','M3'))
    write(ROOT/'evidence/S1_TEACHER_LIVENESS.json',dict(schema='PI0267_FIXED_64_CPU_EPISODES_V1',
        status='PASS_TEACHER_LIVENESS' if passed else 'PRIMARY_BLOCKED_TEACHER_LIVENESS',scientific_overall='CONTINUE_S2' if passed else 'PRIMARY_PARTIAL',
        groups=groups,episodes=[{k:v for k,v in x.items() if k!='replans'} for x in records],
        original_fixed_scorer=True,robot_physics_steps=0,new_GR00T_queries=0,training_updates=0,
        primary_data_rendering=0,teacher_parameter_search=False,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['S0','S1']);args=parser.parse_args()
    binding=read(ROOT/'design/SOURCE_BINDING.json')
    for p,digest in binding['files'].items():assert sha(p)==digest,(p,'source drift')
    if args.stage=='S0':prefix_audit();constructed_probe()
    else:teacher_matrix()
