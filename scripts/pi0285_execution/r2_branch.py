"""One frozen T1 branch decision from existing fixed raw-truth CPU evidence."""
import argparse,json,os
from pathlib import Path
import numpy as np
from contract import ROOT,sha,write

KEYS=('XYZ_RMS_m','SO3_RMS_rad','gripper_MAE')
TOL=np.array([.01,.10,.10])


def dynamic(rows):
    rows=[r for r in rows if r['task']=='T1' and r['split']=='TRAIN' and r['phase'] in range(4)]
    assert len(rows)==32 and len({r['uid'] for r in rows})==32
    assert all(sum(r['phase']==p for r in rows)==8 for p in range(4))
    values=np.array([[r['prefix'][k] for k in KEYS] if r['prefix'] is not None else [np.inf]*3 for r in rows])
    passed=(values<=TOL).all(1)
    return rows,values,passed


def decide(previous,current,finish_failures):
    invalid=lambda rows:sum(r['prefix'] is None for r in rows)
    rejects=lambda rows:sum(r['gate']['decision']=='REJECT' for r in rows)
    invalid_counts=[invalid(previous),invalid(current)];reject_counts=[rejects(previous),rejects(current)]
    p,x,_=dynamic(previous);c,y,passed=dynamic(current)
    assert [r['uid'] for r in p]==[r['uid'] for r in c]
    counts=[int(sum(ok and r['phase']==phase for r,ok in zip(c,passed))) for phase in range(4)]
    recovery=bool(passed.sum()>=24 and min(counts)>=4 or finish_failures>=6)
    j0=float(np.median(np.mean((x/TOL)**2,axis=1)));j1=float(np.median(np.mean((y/TOL)**2,axis=1)))
    with np.errstate(invalid='ignore'):p0,p1=np.percentile(x,95,axis=0),np.percentile(y,95,axis=0)
    p95_ok=bool(np.isfinite(p0).all() and np.isfinite(p1).all() and (p1<=1.25*p0).all())
    dose=bool(not recovery and passed.sum()<24 and np.isfinite(j0) and np.isfinite(j1) and j1<=.85*j0 and p95_ok and invalid_counts[1]<=invalid_counts[0] and reject_counts[1]<=reject_counts[0])
    return dict(branch='RECOVERY2048' if recovery else ('DOSE2048' if dose else 'STOP_NO_ADMITTED_EXTRA_BRANCH'),dynamic_pass=int(passed.sum()),pass_per_nonterminal_group=counts,finish_failures=finish_failures,RECOVERY_priority=recovery,DOSE_allowed=dose,J_median_2048=j0 if np.isfinite(j0) else 'NONFINITE',J_median_4096=j1 if np.isfinite(j1) else 'NONFINITE',dynamic_p95_growth_gate=p95_ok,invalid_decode_counts=invalid_counts,DIAG_reject_counts=reject_counts)


def freeze():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    from r2 import admission
    admission();assert not (ROOT/'qualification/CONFIRM_A_CANDIDATE.json').exists(),'No extra learning after a confirmation candidate'
    route=ROOT/'qualification/PLACE4096-4096_ROUTE.json';assert json.loads(route.read_text())['next_action']=='SELECT_R2_SINGLE_BRANCH_FROM_FROZEN_EVIDENCE'
    paths=[ROOT/'audit'/f'PLACE4096-{step}_DIAG_CPU.json' for step in (2048,4096)]
    reports=[json.loads(p.read_text()) for p in paths]
    assert all(v['status']=='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE' for v in reports)
    for step,report in zip((2048,4096),reports):
        result=ROOT/'diagnostics'/f'PLACE4096-{step}'/'RESULT.json'
        assert report['point']==f'PLACE4096-{step}' and report['source_sha256']==sha(result)
        assert json.loads(result.read_text())['plan_sha256']==sha(ROOT/'design/DIAG96_PER_TASK.json')
    source=ROOT/'source/evidence/PLACE4096-4096_TRAIN.json';panel=json.loads(source.read_text());cpu=ROOT/'audit/PLACE4096-4096_TRAIN_CPU.json';check=json.loads(cpu.read_text())
    assert check['source_sha256']==sha(source) and check['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS'
    assert len(panel['rows'])==8
    finish=0
    for row in panel['rows']:
        score=row['task_score'];h=score['history'];zone=score['subpredicates']['zone']
        finish+=bool(h['grasp'] and h['lift'] and h['horizontal_transfer'] and zone['first_true_s'] is not None and score['hold_pass']<41 and not score['persistent_success'])
    value=dict(status='FROZEN_R2_SINGLE_BRANCH_DECISION_NOT_DATA_OR_TRAINING_ADMISSION',**decide(reports[0]['rows'],reports[1]['rows'],finish),proofs={str(p.relative_to(ROOT)):sha(p) for p in paths+[source,cpu,route]},launch_allowed=False,new_model_queries=0,new_physics=0,selected='NONE',deployment='NONE')
    target=ROOT/'qualification/R2_A_SINGLE_BRANCH.json'
    if target.exists():assert json.loads(target.read_text())==value
    else:write(target,value)
    return value


def self_check():
    def rows(scale):return [dict(uid=str(i),task='T1',split='TRAIN',phase=i//8,prefix={k:float(t*scale) for k,t in zip(KEYS,TOL)},gate=dict(decision='ACCEPT')) for i in range(32)]
    assert decide(rows(2),rows(.8),0)['branch']=='RECOVERY2048'
    assert decide(rows(2),rows(1.7),0)['branch']=='DOSE2048'
    assert decide(rows(2),rows(1.7),6)['branch']=='RECOVERY2048'
    a=rows(1.7);a[0]['gate']['decision']='REJECT';assert decide(rows(2),a,0)['branch']=='STOP_NO_ADMITTED_EXTRA_BRANCH'
    a,b=rows(2),rows(1.7);a[0]['prefix']=b[0]['prefix']=None
    assert decide(a,b,0)['branch']=='DOSE2048' # unchanged invalid count; finite median and p95 still prove the gate
    assert decide(rows(2),rows(2.1),0)['branch']=='STOP_NO_ADMITTED_EXTRA_BRANCH'
    late=dict(uid='DEV-hold',task='T1',split='DEV',phase=4,prefix={k:0. for k in KEYS},gate=dict(decision='ACCEPT'))
    assert decide(rows(2)+[late],rows(1.7)+[dict(late,gate=dict(decision='REJECT'))],0)['branch']=='STOP_NO_ADMITTED_EXTRA_BRANCH'
    assert decide(rows(2)+[late],rows(1.7)+[dict(late,prefix=None)],0)['branch']=='STOP_NO_ADMITTED_EXTRA_BRANCH'
    print('PASS_FIXED_DYNAMIC_GROUPS_RECOVERY_PRIORITY_DOSE_AND_NO_THIRD_BRANCH_CPU_ONLY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');a=p.parse_args()
    self_check() if a.self_check else print(json.dumps(freeze()))
