"""Independent saved-array task/installation audit, zero model or physics calls."""
import argparse,hashlib,itertools,json,os
from pathlib import Path
import numpy as np

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0282')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def longest(flags):return max((sum(1 for _ in g) for b,g in itertools.groupby(flags) if b),default=0)


def predicates(p):
    delta=p['cubeA_pose'][:,:3]-p['cubeB_pose'][:,:3]
    top=(np.linalg.norm(delta[:,:2],axis=1)<=np.linalg.norm(np.array([.02,.02],dtype=np.float32))+.005)&(np.abs(delta[:,2]-.04)<=.005)
    static=(np.linalg.norm(p['cubeA_velocity'],axis=1)<=.01)&(np.linalg.norm(p['cubeA_angular'],axis=1)<=.5)
    flags=[]
    for side in ('left','right'):
        force=p[side+'_force'];direction=p[side+'_direction'];norm=np.linalg.norm(force,axis=1)
        cosine=(force*direction).sum(1)/(np.maximum(norm,1e-12)*np.linalg.norm(direction,axis=1))
        flags.append((norm>=.5)&(np.arccos(np.clip(cosine,-1,1))*180/np.pi<=85))
    grasp=flags[0]&flags[1]
    return dict(is_cubeA_on_cubeB=top,is_cubeA_static=static,is_cubeA_grasped=grasp,success=top&static&~grasp)


def audit(point,panel='TRAIN'):
    sources=[ROOT.parent/'PI0276/ManiSkill/mani_skill/envs/tasks/tabletop/stack_cube.py',ROOT.parent/'PI0276/ManiSkill/mani_skill/agents/robots/panda/panda.py']
    text=sources[0].read_text();assert 'ang_thresh=0.5' in text and 'lin_thresh=1e-2' in text
    assert 'cube_half_size' in text and '0.02' in text
    text=sources[1].read_text();assert 'min_force=0.5, max_angle=85' in text
    source=ROOT/'source/evidence'/f'RESULT_{point}_{panel}.json';v=json.loads(source.read_text());rows=[]
    for r in v['rows']:
        folder=Path(r['folder'])
        for n,h in r['files'].items():assert sha(folder/n)==h,(r['condition'],n)
        with np.load(folder/'physics.npz',allow_pickle=False) as f:p={n:f[n] for n in f.files}
        computed=predicates(p)
        for n,x in computed.items():assert np.array_equal(x,p[n]),(point,r['condition'],n,'NATIVE_PREDICATE_ARRAY_MISMATCH')
        control=computed['success'][::5][20:]
        persistent=bool(r['status']=='VALID_SOURCE_COMPLETE' and r['task_control_calls']==640 and len(control)==641 and control[560:601].all())
        first=float(np.flatnonzero(control)[0]*.05) if control.any() else None
        assert persistent==r['persistent_success'] and first==r['first_success_s']
        grasp=longest(computed['is_cubeA_grasped'])>=30;assert grasp==r['sustained_grasp_0p3']
        gates=[json.loads((q/'GATE.json').read_text()) for q in sorted(folder.glob('q*')) if (q/'GATE.json').exists()]
        with np.load(folder/'commands.npz',allow_pickle=False) as f:cmd=f['action']
        with np.load(folder/'installed.npz',allow_pickle=False) as f:installed=f['installed_target_fk']
        actual=p['tcp'][5::5];assert len(actual)==len(cmd)==len(installed)==r['control_calls']
        start=20
        def rms(a,b):return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))) if len(a) else None
        rows.append(dict(condition=r['condition'],status=r['status'],persistent=persistent,first_success_s=first,sustained_grasp_0p3=grasp,
            task_controls=r['task_control_calls'],observed_hold_samples=len(control[560:601]),hold_pass_samples=int(control[560:601].sum()),
            max_cube_lift_above_initial_m=float((p['cubeA_pose'][:,2]-p['cubeA_pose'][0,2]).max()),
            reference_to_installed_XYZ_RMS_m=rms(cmd[start:,:3],installed[start:,:3,3]),
            installed_to_achieved_after_control_XYZ_RMS_m=rms(installed[start:,:3,3],actual[start:,:3,3]),
            reference_to_achieved_after_control_XYZ_RMS_m=rms(cmd[start:,:3],actual[start:,:3,3]),
            tracking_boundary='Actual TCP after five physics steps of the same issued control; not a difference of minima',
            rejected_gate_codes=[i['code'] for g in gates for i in g['issues'] if i['severity']=='REJECT'],result_sha256=sha(folder/'RESULT.json')))
    assert sum(r['persistent'] for r in rows)==v['persistent_count']
    result=dict(status='PASS_INDEPENDENT_CPU_NATIVE_TASK_AND_ACTUAL_INSTALLATION_RECOMPUTE',point=point,panel=panel,rows=rows,
        source_panel_sha256=sha(source),native_source_sha256={str(p):sha(p) for p in sources},persistent_count=sum(r['persistent'] for r in rows),
        new_model_queries=0,new_physics_steps=0,selected='NONE',deployment='NONE')
    out=ROOT/'audit'/f'{point}_{panel}_CLOSED_LOOP_CPU.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result


if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    a=argparse.ArgumentParser();a.add_argument('point');a.add_argument('--panel',default='TRAIN');v=a.parse_args();audit(v.point,v.panel)
