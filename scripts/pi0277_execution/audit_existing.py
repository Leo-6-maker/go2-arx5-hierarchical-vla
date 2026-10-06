"""Read saved arrays only; never import a simulator, model or prior stage entrypoint."""
import argparse, ast, csv, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

RELEASE = '63151a4548243ab0fb4ba172eea6f7d92e02c576'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.partial')
    q.write_text(json.dumps(v, indent=2, sort_keys=True) + '\n')
    q.replace(p)

def table(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def intervals(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [dict(start_s=float(a*dt), last_sample_s=float((b-1)*dt), duration_s=float((b-a)*dt))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]

def first(mask, dt):
    indices = np.flatnonzero(mask)
    return float(indices[0]*dt) if len(indices) else None

def rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=-1)))) if len(x) else None

def pure_reconstruct(path):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'reconstruct')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['reconstruct']

def audit(old, raw, out):
    assert not any(n in __import__('sys').modules for n in ('torch', 'sapien', 'source_worker'))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'AUDIT_EXISTING.json').exists():
        receipt = read(out/'AUDIT_EXISTING.json')
        for name, digest in receipt['outputs'].items(): assert sha(out/name) == digest
        print('REUSED_CLOSED_A0', flush=True); return
    reconstruct = pure_reconstruct(old/'code/replay.py')
    with (raw/'delivery/PHYSICAL144.csv').open() as f:
        records = [r for r in csv.DictReader(f) if r['point'] == 'TASK8192']
    assert len(records) == 48 and Counter(r['lane'] for r in records) == {'TEST': 32, 'TRAIN_PHYSICS': 16}
    rows, events, bindings = [], {}, []
    for r in records:
        folder = Path(r['path']); rec = read(folder/'RESULT.json')
        assert sha(folder/'RESULT.json') == r['result_sha256']
        for name in ('physics.npz', 'commands.npz', 'installed.npz'):
            assert sha(folder/name) == rec['files'][name]
        with np.load(folder/'physics.npz', allow_pickle=False) as z:
            native = reconstruct(z)
            for key, values in native.items(): assert np.array_equal(values, z[key]), (folder, key)
            t = z['time_s'].copy(); red = z['cubeA_pose'][:,:3].copy()
            tcp = z['tcp'].copy(); q = z['qpos'].copy()
            forces = [np.linalg.norm(z[s+'_force'], axis=1) for s in ('left', 'right')]
        assert np.allclose(t, np.arange(len(t))*.01, rtol=0, atol=1e-9)
        with np.load(folder/'commands.npz', allow_pickle=False) as z:
            cmd = z['action'].copy(); holds = z['ik_hold'].copy()
        with np.load(folder/'installed.npz', allow_pickle=False) as z:
            installed = z['installed_target_fk'].copy(); native_action = z['action'].copy()
        assert len(cmd) == rec['control_calls'] == len(installed)
        assert int(holds.sum()) == rec['ik_holds']
        grasp, on, success = [native[k] for k in ('is_cubeA_grasped','is_cubeA_on_cubeB','success')]
        lifted = red[:,2] - red[0,2] >= .02
        grasp_runs, lift_runs = intervals(grasp, .01), intervals(lifted, .01)
        lost = np.flatnonzero(grasp[:-1] & ~grasp[1:]) + 1
        acquired = np.flatnonzero(~grasp[:-1] & grasp[1:]) + 1
        carried = grasp & lifted
        carrying_losses = [int(i) for i in lost if carried[:i].any() and not on[i]]
        success_first = first(success[::5][:601], .05)
        persistent = bool(len(success[::5]) >= 601 and success[::5][560:601].all())
        assert success_first == rec['first_success_s'] and persistent == rec['persistent_success']
        actual = tcp[5::5]; opening = (q[5::5,-2:].sum(1)-.00004)/.08
        assert len(actual) == len(cmd)
        if len(cmd):
            assert np.allclose(native_action[:,-1], 2*np.clip(cmd[:,9], 0, 1)-1, rtol=0, atol=1e-6)
        closed = cmd[:,9] <= .2; opened = cmd[:,9] >= .8
        close_events = (np.flatnonzero(~closed[:-1] & closed[1:])+1)*.05
        open_events = (np.flatnonzero(~opened[:-1] & opened[1:])+1)*.05
        sustained_grasp = any(x['duration_s'] >= .3-1e-9 for x in grasp_runs)
        sustained_lift = any(x['duration_s'] >= .3-1e-9 for x in lift_runs)
        classification = ('GATE_REJECT' if 'GATE' in rec['status'] else 'TASK_SUCCESS' if persistent else
                          'NO_SUSTAINED_GRASP' if not sustained_grasp else 'NO_SUSTAINED_LIFT' if not sustained_lift else
                          'NO_ON_TOP' if not on.any() else 'NO_RELEASE_ON_TOP' if not (on & ~grasp).any() else 'NO_STABLE_STACK')
        row = dict(seed=int(r['seed']), partition='REPAIR_DEV' if r['lane']=='TEST' else 'TRAIN', historical_lane=r['lane'],
                   status=rec['status'], raw_gate_outcome='REJECT' if 'GATE' in rec['status'] else 'ALL_QUERIES_ACCEPTED',
                   diagnostic_failure_stage=classification, first_grasp_s=first(grasp,.01),
                   longest_grasp_s=max((x['duration_s'] for x in grasp_runs),default=0.), sustained_grasp=sustained_grasp,
                   first_lift_2cm_s=first(lifted,.01), longest_lift_2cm_s=max((x['duration_s'] for x in lift_runs),default=0.),
                   sustained_lift_2cm=sustained_lift, max_lift_m=float((red[:,2]-red[0,2]).max()),
                   carrying_loss_count=len(carrying_losses), first_carrying_loss_s=carrying_losses[0]*.01 if carrying_losses else None,
                   first_on_top_s=first(on,.01), release_before_any_on_top=sum(not on[:i+1].any() for i in lost),
                   release_on_top=sum(bool(on[i]) for i in lost), release_after_first_on_top=sum(bool(on[:i+1].any()) for i in lost),
                   first_success_s=success_first, persistent_success=persistent, IK_holds=int(holds.sum()),
                   first_IK_hold_s=first(holds,.05), command_open_fraction_mean=float(cmd[:,9].mean()) if len(cmd) else None,
                   achieved_open_fraction_mean=float(opening.mean()) if len(opening) else None,
                   opening_vs_command_MAE=float(np.abs(opening-cmd[:,9]).mean()) if len(cmd) else None,
                   command_installed_XYZ_RMS_m=rms(cmd[:,:3]-installed[:,:3,3]),
                   installed_achieved_XYZ_RMS_m=rms(installed[:,:3,3]-actual[:,:3,3]),
                   command_achieved_XYZ_RMS_m=rms(cmd[:,:3]-actual[:,:3,3]),
                   command_achieved_SO3_RMS_rad=r['command_actual_SO3_RMS_rad'], contact_points=int(r['contact_points']),
                   max_contact_impulse_Ns=float(r['max_contact_impulse_Ns']), min_contact_separation_m=r['min_contact_separation_m'],
                   source_result_sha256=r['result_sha256'], source_physics_sha256=r['physics_sha256'])
        rows.append(row)
        events[f"{row['partition']}-{r['seed']}"] = dict(grasp_intervals=grasp_runs, lift_intervals=lift_runs,
            grasp_acquisition_s=(acquired*.01).tolist(), grasp_to_release_s=(lost*.01).tolist(),
            carrying_loss_s=(np.array(carrying_losses)*.01).tolist(), close_command_transitions_s=close_events.tolist(),
            open_command_transitions_s=open_events.tolist(), on_top_intervals=intervals(on,.01),
            IK_hold_intervals=intervals(holds,.05), left_contact_force_intervals_ge_0p5N=intervals(forces[0]>=.5,.01),
            right_contact_force_intervals_ge_0p5N=intervals(forces[1]>=.5,.01))
        timeline=[dict(time_s=float(i*.05),command_g=float(cmd[i,9]),installed_native_g=float(native_action[i,-1]),
                       achieved_opening=float(opening[i]),grasp=bool(grasp[5*(i+1)]),on_top=bool(on[5*(i+1)]),
                       lift_m=float(red[5*(i+1),2]-red[0,2]),IK_hold=bool(holds[i])) for i in range(len(cmd))]
        if timeline: table(out/'timelines'/f"{row['partition']}-{r['seed']}.csv",timeline)
        bindings.append(dict(path=str(folder),result_sha256=sha(folder/'RESULT.json'),physics_sha256=sha(folder/'physics.npz')))
        print('A0_PHYSICAL',len(rows),r['seed'],classification,flush=True)
    table(out/'FAILURE_FUNNEL.csv',rows)
    write(out/'FAILURE_EVENTS.json',dict(events=events,interval_duration='sample occupancy count * dt; endpoint timestamps retained',
        release_is_observed_transition_not_intention=True,phase_of_diverged_policy='UNOBSERVED',
        diagnostic_lift_definition='>=0.02m above own initial red z for >=0.3s; does not change native task success'))
    write(out/'FAILURE_FUNNEL_SUMMARY.json',dict(counts={lane:dict(Counter(r['diagnostic_failure_stage'] for r in rows if r['partition']==lane))
        for lane in ('TRAIN','REPAIR_DEV')},sustained_grasps={lane:sum(r['sustained_grasp'] for r in rows if r['partition']==lane)
        for lane in ('TRAIN','REPAIR_DEV')},shared_native_math_source=dict(path=str(old/'code/replay.py'),sha256=sha(old/'code/replay.py')),
        timestamps_from_saved_physics=True,diagnostics_not_unique_causal_proof=True))
    outputs={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='AUDIT_EXISTING.json'}
    write(out/'AUDIT_EXISTING.json',dict(status='PASS_SAVED48_EVENT_RECONSTRUCTION_AND_EXECUTED_GRIPPER_TRANSPORT',release=RELEASE,
        script_sha256=sha(__file__),sources=bindings,outputs=outputs,original_physical_table_sha256=sha(raw/'delivery/PHYSICAL144.csv'),
        new_model_queries=0,new_physics_steps=0,new_teacher_calls=0,full_A0='EXPOSURE_AND_OBJECTIVE_AUDITS_REQUIRED_SEPARATELY'))

def check():
    assert intervals(np.array([False,True,True,False]),.1)==[dict(start_s=.1,last_sample_s=.2,duration_s=.2)]
    assert intervals(np.zeros(3,bool),.01)==[]
    assert first(np.array([False,True]),.05)==.05

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path);p.add_argument('--raw',type=Path);p.add_argument('--out',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    check()
    if a.check: print('PASS_EVENT_BOUNDARIES_NO_MODEL_OR_PHYSICS')
    else: assert a.old and a.raw and a.out; audit(a.old,a.raw,a.out)
