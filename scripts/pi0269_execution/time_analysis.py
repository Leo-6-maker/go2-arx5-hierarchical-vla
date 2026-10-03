"""Full saved-tick tracking and terminal-target distances; no model or teacher call."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0269'; RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def errors(t,p,r,reference_p,reference_r,goal,goal_r):
    n=len(t)
    assert n and p.shape==reference_p.shape==(n,3) and r.shape==reference_r.shape==(n,3,3)
    assert all(np.isfinite(x).all() for x in (t,p,r,reference_p,reference_r,goal,goal_r))
    assert np.allclose(t,np.arange(n)*.02,rtol=0,atol=1e-10)
    def angle(a,b):
        return np.arccos(np.clip((np.sum(a*b,axis=(-2,-1))-1)/2,-1,1))
    return dict(tracking_xyz_m=np.linalg.norm(p-reference_p,axis=1),tracking_so3_rad=angle(r,reference_r),
        reference_terminal_target_xyz_m=np.linalg.norm(reference_p-goal,axis=1),
        reference_terminal_target_so3_rad=angle(reference_r,goal_r),
        achieved_terminal_target_xyz_m=np.linalg.norm(p-goal,axis=1),
        achieved_terminal_target_so3_rad=angle(r,goal_r))


def main():
    verdict=json.loads((RAW/'delivery/PRIMARY_VERDICT.json').read_text())
    assert verdict['status']=='PRIMARY_EVALUATION_COMPLETE'
    output=RAW/'delivery/SAME_TICK_REFERENCE_ACHIEVED_TARGET.csv'; receipts=[];count=0
    with output.open('w',newline='') as stream:
        writer=None
        for point in ('ROT2048','UNIFORM4096','PREFIX4096'):
            source=RAW.parent/'PI0268/R1' if point=='ROT2048' else RAW
            cases=sorted((source/'virtual'/point).glob('*/result.json'));assert len(cases)==64
            for path in cases:
                record=json.loads(path.read_text());arrays=path.parent/'trajectory.npz'
                assert record['validity']=='VALID' and record['point']==point and sha(arrays)==record['arrays_sha256']
                with np.load(arrays,allow_pickle=False) as z:
                    t,p,r,ip,ir=(z[k] for k in ('time','actual_p','actual_r','installed_p','installed_r'))
                    spec=json.loads(str(z['spec_json']));values=errors(t,p,r,ip,ir,np.asarray(spec['goal']),np.asarray(spec['goal_rotation']))
                    for i,time in enumerate(t):
                        row=dict(point=point,case=path.parent.name,task=record['task'],backend=record['backend'],tick=i,time_s=float(time),
                            **{f'{kind}_{axis}_m':float(v[j]) for kind,v in [('reference',ip[i]),('achieved',p[i]),('terminal_target',spec['goal'])] for j,axis in enumerate('xyz')},
                            **{k:float(v[i]) for k,v in values.items()})
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);count+=1
                    receipts.append(dict(point=point,case=path.parent.name,source_arrays_sha256=record['arrays_sha256'],ticks=len(t),
                        elapsed_s=float(t[-1]),tracking_xyz_rms_m=float(np.sqrt(np.mean(values['tracking_xyz_m']**2))),
                        tracking_so3_rms_rad=float(np.sqrt(np.mean(values['tracking_so3_rad']**2))),stop_reason=record['stop_reason']))
    result=dict(status='PASS_FULL_SAVED_TICK_REFERENCE_ACHIEVED_TERMINAL_TARGET_ANALYSIS',episodes=len(receipts),rows=count,csv_sha256=sha(output),sources=receipts,
        target_semantics='Static final task pose from original saved spec; not an expert time trajectory. Tracking compares installed reference and achieved pose at the same saved tick.',
        execution_domain='VIRTUAL_EEF_KINEMATIC_NOT_WBC',early_failures_retained=True,model_queries=0,teacher_calls=0,physics_steps=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    (RAW/'delivery/SAME_TICK_ANALYSIS_RECEIPT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'],len(receipts),count,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    if parser.parse_args().self_check:
        t=np.arange(3)*.02;p=np.zeros((3,3));ip=p.copy();ip[:,0]=.01
        r=np.repeat(np.eye(3)[None],3,axis=0);ir=r.copy();ir[:,0,0]=ir[:,1,1]=0;ir[:,0,1]=-1;ir[:,1,0]=1
        e=errors(t,p,r,ip,ir,np.array([.03,0,0]),np.eye(3))
        assert np.allclose(e['tracking_xyz_m'],.01) and np.allclose(e['tracking_so3_rad'],np.pi/2)
        assert np.allclose(e['reference_terminal_target_xyz_m'],.02) and np.allclose(e['achieved_terminal_target_xyz_m'],.03)
        assert np.allclose(errors(t,p,r,p,r,p[0],r[0])['tracking_so3_rad'],0)
        try:errors(t,p[:2],r,ip,ir,p[0],r[0])
        except AssertionError:pass
        else:raise AssertionError('MISMATCHED_TICK_ARRAY_ACCEPTED')
        print('PASS_SAME_TICK_TRANSLATION_ROTATION_AND_SHAPE_CPU_ZERO_QUERY')
    else:main()
