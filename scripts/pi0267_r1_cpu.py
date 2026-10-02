"""Bounded R1 production fixtures and old 64-array rescoring; no new episodes."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267')
sys.path.insert(0,str(BASE/'PI0250/runtime/reference-317a/src'))
sys.path.append(str(BASE/'PI0266/code'));sys.path.append(str(ROOT/'code'))
from pi0266_metrics import score as v2, self_check as old_fixtures
from pi0266_expert import reference
from pi0255_motion3 import scene, sha, write
from pi0267_metrics_v3 import score as v3
import pi0267_r1_entry_contract_probe as probe


def read(path):return json.loads(Path(path).read_text())


def independent_entry(t,p,r,spec):
    center=np.asarray(spec['center']);rad=spec['radius'];d=1 if spec['command']==0 else -1;a=-d*3*np.pi/4
    entry=center+[rad*np.cos(a),rad*np.sin(a),0]
    # Independent geometry spelling, trace orientation predicate and vectorized continuous runs.
    angle=d*(np.arctan2(p[:,1]-center[1],p[:,0]-center[0])-a)
    arc=rad*np.arctan2(np.sin(angle),np.cos(angle))
    trace=np.trace(r@np.asarray(spec['goal_rotation']).T,axis1=1,axis2=2)
    near=(np.sum((p-entry)**2,axis=1)<=.015**2)&(trace>=1+2*np.cos(.10))&(abs(arc)<=.005)
    run_starts=np.maximum.accumulate(np.where(~near,np.arange(len(t)),-1))+1
    elapsed=t-t[np.minimum(run_starts,len(t)-1)]
    broad=np.flatnonzero(near&(elapsed>=.10-1e-9))
    narrow=np.flatnonzero(near&(elapsed>=.10-1e-9)&(abs(arc)<=.001+1e-12))
    return (None if not len(broad) else int(broad[0]),None if not len(narrow) else int(narrow[0]),arc)


def check_entry(t,p,r,spec,result):
    if spec['task']=='M1':return None
    broad,acquired,arc=independent_entry(t,p,r,spec)
    assert acquired==result['entry_index']
    assert (None if broad is None else float(t[broad]))==result['earliest_broad_readiness_s']
    if acquired is not None:assert abs(arc[acquired]-result['entry_arc_m'])<1e-12
    return dict(broad_index=broad,acquired_index=acquired,entry_arc_m=None if acquired is None else float(arc[acquired]))


def fixtures():
    t=np.arange(0,20.0001,.02);r=np.repeat(np.eye(3)[None],len(t),axis=0);rows=[]
    def add(name,p,rad,cmd,want,safety=False,goal_shift=False,wrong_orientation=False):
        a=(-3 if cmd==0 else 3)*np.pi/4;d=1 if cmd==0 else -1;c=np.array([.30,0,.60])
        goal=c+[rad*np.cos(a+d*1.5*np.pi),rad*np.sin(a+d*1.5*np.pi),0]
        if goal_shift:goal=goal+[.1,.1,.1]
        actual_r=r if not wrong_orientation else Rotation.from_rotvec([.2,0,0]).as_matrix()@r
        spec=dict(task='M3',command=cmd,center=c,radius=rad,goal=goal,goal_rotation=np.eye(3))
        result=v3(t,p,actual_r,p,r,spec,safety);event=check_entry(t,p,actual_r,spec,result)
        assert result['success']==want,(name,result)
        rows.append(dict(name=name,expected=want,actual=result['success'],entry=event,
            reverse=result['reverse'],gap=result['gap_shortcut'],bins=result['ordered_bins']))
    supplied=probe.run();assert supplied['fixture_count']==28
    for cmd in (0,1):
        for rad in (.04,.05):
            for side in (-1,1):add(f'approach-c{cmd}-r{rad}-side{side}',probe.path(t,rad,cmd,side=side),rad,cmd,True)
        for kind in ('valid','jitter','reverse','wrong_direction','missing_bin','shortcut','missing_entry','late_reverse_recover','unsafe','wrong_goal'):
            p=probe.path(t,.05,cmd,kind='valid' if kind in ('unsafe','wrong_goal') else kind)
            add(f'{kind}-c{cmd}',p,.05,cmd,kind in ('valid','jitter'),kind=='unsafe',kind=='wrong_goal')
        add(f'wrong-orientation-c{cmd}',probe.path(t,.05,cmd),.05,cmd,False,wrong_orientation=True)
        # A second physically revisited entry cannot erase the first acquired negative excursion.
        p=probe.path(t,.05,cmd,kind='late_reverse_recover');add(f'reacquire-after-failure-c{cmd}',p,.05,cmd,False)
    card=read(ROOT/'design/SCENE_CARD.json');reg=card['registry'];full_t=np.arange(0,40.0001,.02)
    for family in (f for f in card['families'] if f['split']=='CONTROLLER_DEV'):
        for task in ('M1','M3'):
            for cmd in (0,1):
                p,rr=reference(reg,family,task,cmd,full_t);gp,gr=reference(reg,family,task,cmd,[40.]);sc=scene(reg,family)
                spec=dict(task=task,command=cmd,goal=gp[0],goal_rotation=gr[0],center=sc['center'],radius=sc['radius'])
                result=v3(full_t,p,rr,p,rr,spec);event=check_entry(full_t,p,rr,spec,result)
                assert result['success'],(family['family_id'],task,cmd,result)
                rows.append(dict(name=f"nominal-{family['family_id']}-{task}-c{cmd}",expected=True,actual=True,entry=event))
    for kind in ('gap','duplicate','nonfinite'):
        bad_t=t.copy()
        if kind=='gap':bad_t[100:]+=.02
        elif kind=='duplicate':bad_t[100]=bad_t[99]
        else:bad_t[100]=np.nan
        p=probe.path(t,.05,0);spec=dict(task='M3',command=0,center=np.array([.3,0,.6]),radius=.05,goal=p[-1],goal_rotation=np.eye(3))
        try:v3(bad_t,p,r,p,r,spec)
        except AssertionError:rows.append(dict(name='integrity-'+kind,expected='INPUT_INTEGRITY_REJECT',actual='INPUT_INTEGRITY_REJECT'))
        else:raise AssertionError(kind+' accepted')
    assert len(rows)==51 and len(rows)<=96
    write(R1/'evidence/FIXTURES.json',dict(status='PASS_PRODUCTION_V3_AND_INDEPENDENT_ENTRY_FIXTURES',fixture_cases=51,rows=rows,
        original_v2_fixtures=old_fixtures(),publication_constructed_probe=supplied,production_scorer_sha256=sha(ROOT/'code/pi0267_metrics_v3.py'),
        new_teacher_episodes=0,new_teacher_plans=0,model_queries=0))
    print('R1_PRODUCTION_FIXTURES_PASS_51',flush=True)


def replay():
    assert read(R1/'evidence/FIXTURES.json')['status'].startswith('PASS')
    historical=read(ROOT/'evidence/S1_TEACHER_LIVENESS.json');records=[];out=RAW/'R1/rescore';out.mkdir(parents=True,exist_ok=True)
    for rec in historical['episodes']:
        path=RAW/'S1'/(rec['key']+'.npz');z=np.load(path);spec=json.loads(str(z['spec_json']))
        args=(z['time'],z['actual_p'],z['actual_r'],z['installed_p'],z['installed_r'],spec)
        old=v2(*args)
        assert {k:v for k,v in old.items() if k!='progress_rad'}==rec['result'],rec['key']
        result=v3(*args);assert result['legacy_v2']==old
        event=check_entry(z['time'],z['actual_p'],z['actual_r'],spec,result)
        # Derive old first violation from saved actual progress; no phase/time-based entry rule.
        events={}
        if spec['task']=='M3':
            valid=np.flatnonzero([x is not None for x in old['progress_rad']]);th=np.array([old['progress_rad'][i] for i in valid])
            if len(valid):
                draw=(np.maximum.accumulate(np.maximum(th,0))-th)*spec['radius']
                bad=np.flatnonzero((draw>.001+1e-12)|(th*spec['radius']<-.001-1e-12)|((th-1.5*np.pi)*spec['radius']>.001+1e-12))
                events=dict(old_first_violation_index=None if not len(bad) else int(valid[bad[0]]),
                    old_first_violation_precedes_v3_acquisition=None if not len(bad) or result['entry_index'] is None else bool(valid[bad[0]]<result['entry_index']))
        row=dict(key=rec['key'],family=rec['family'],task=rec['task'],command=rec['command'],teacher=rec['teacher'],backend=rec['backend'],
            source=str(path),source_sha256=sha(path),v2_success=old['success'],v3_success=result['success'],independent_entry=event,**events,
            v3={k:v for k,v in result.items() if k not in ('progress_rad','legacy_v2')})
        write(out/(rec['key']+'.json'),dict(row,v3_full=result));records.append(row)
    groups=[]
    for teacher in ('OLD','NEW'):
        for backend in ('E0','E1'):
            for task in ('M1','M3'):
                rows=[x for x in records if (x['teacher'],x['backend'],x['task'])==(teacher,backend,task)]
                groups.append(dict(teacher=teacher,backend=backend,task=task,v2=sum(x['v2_success'] for x in rows),v3=sum(x['v3_success'] for x in rows),total=len(rows)))
    def wins(backend,task=None):return sum(x['v3_success'] for x in records if x['teacher']=='NEW' and x['backend']==backend and (task is None or x['task']==task))
    passed=wins('E0')==16 and wins('E1')>=12 and all(wins('E1',task)>=6 for task in ('M1','M3'))
    receipt=dict(status='PASS_V3_SAVED_ARRAY_READMISSION_CONTINUE_S2' if passed else 'R1_CONTRACT_OR_TEACHER_UNRESOLVED',
        groups=groups,rows=records,original_v2_rows_exactly_reproduced=64,original_arrays_unchanged=True,
        actual_new_teacher_episodes=0,actual_new_teacher_plans=0,GR00T_queries=0,cumulative_old_CPU_episodes=64,
        production_scorer_sha256=sha(ROOT/'code/pi0267_metrics_v3.py'),new_model_learning_claim=False,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(R1/'evidence/SAVED_ARRAY_READMISSION.json',receipt)
    with (R1/'evidence/V2_V3_64_ROWS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['key','teacher','backend','task','command','v2_success','v3_success']);writer.writeheader()
        writer.writerows({k:x[k] for k in writer.fieldnames} for x in records)
    print(json.dumps(dict(status=receipt['status'],groups=groups)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fixtures','replay']);args=parser.parse_args()
    fixtures() if args.stage=='fixtures' else replay()
