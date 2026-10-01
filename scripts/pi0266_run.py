"""Finite CPU/reset dispatch using the already accepted native environment."""
import json
import subprocess
import sys
import numpy as np
from pi0266_common import ROOT,BASE,RAW,read,write,sha
sys.path.insert(0,str(BASE/'PI0256/runtime'))
import pi0255_run as previous


def environment():
    previous.ROOT=ROOT
    py,env=previous.environment();env['PYTHONPATH']+=':'+str(ROOT/'code')
    return py,env


def seal():
    assert not (ROOT/'evidence/RAW_BUDGET.json').exists()
    for name in ('OBSERVATION_CODEC_AUDIT.json','EXPERT_CPU.json','PATH_V2_CPU.json','NEUTRAL_NATIVE_CPU.json'):
        assert read(ROOT/'evidence'/name)['status'].startswith('PASS')
    files={str(p.relative_to(ROOT)):sha(p) for folder in ('code','design','evidence') for p in (ROOT/folder).glob('*') if p.is_file() and p.name!='RESET_EXECUTION_SEAL.json'}
    write(ROOT/'design/RESET_EXECUTION_SEAL.json',dict(authority=read(ROOT/'code/PUBLICATION_RECEIPT.json'),files=files,physics_starts=0))
    print('PASS_RESET_SOURCE_AND_CPU_ADMISSION_SEALED',flush=True)


def resets():
    py,env=environment();rows=[]
    for case in read(ROOT/'design/SCENE_CARD.json')['reset_cases']:
        folder=RAW/'resets'/case['case_id'];prior=list(folder.glob('attempt-*/result.json'))
        valid=[p for p in prior if read(p)['validity']=='VALID']
        if not valid:
            log=ROOT/'logs'/(case['case_id']+'-%02d.log'%len(prior))
            with log.open('wb') as stream:
                done=subprocess.run([str(py),str(ROOT/'code/pi0266_reset.py'),'--case',case['case_id']],env=env,stdout=stream,stderr=subprocess.STDOUT)
            if done.returncode:raise SystemExit(done.returncode)
            valid=[p for p in folder.glob('attempt-*/result.json') if read(p)['validity']=='VALID']
        assert len(valid)==1
        row=read(valid[0]);rows.append(dict(result=row,attempt=str(valid[0].parent)))
        print(json.dumps(dict(case=case['case_id'],safe=row['safe'],failure=row.get('failure'))),flush=True)
    states=[dict(np.load(row['attempt']+'/reset_state.npz',allow_pickle=False)) for row in rows]
    differences=[]
    for a in states[1:]:
        first=states[0];physical_keys=['root','dof','TCP','rotation']
        for key in physical_keys+rows[0]['result']['history_fields']:
            error=float(np.max(abs(a[key]-first[key])));differences.append(dict(key=key,max_abs=error,bitwise_equal=np.array_equal(a[key],first[key])))
    reproducible=all(x['max_abs']<=1e-5 for x in differences)
    safe=all(r['result']['safe'] for r in rows)
    status='PASS_FIXED_RESET_SAFE_CONTINUE_TEACHER_DIAGNOSTICS' if safe and reproducible else 'FIXED_RESET_NOT_REPRODUCIBLY_SAFE'
    write(ROOT/'evidence/FIXED_RESET_VERDICT.json',dict(status=status,overall='IN_PROGRESS' if safe and reproducible else 'PARTIAL',cases=rows,
        reproducible=reproducible,all_safe=safe,differences=differences,numerical_tolerance=1e-5,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status=status,safe=safe,reproducible=reproducible)),flush=True)


def teacher_seal():
    assert read(ROOT/'evidence/FIXED_RESET_VERDICT.json')['status']=='PASS_FIXED_RESET_SAFE_CONTINUE_TEACHER_DIAGNOSTICS'
    assert not list((RAW/'diagnostics').glob('*/attempt-*')) and not list((RAW/'collection').glob('*/attempt-*'))
    card=read(ROOT/'design/SCENE_CARD.json');cases=[];collection=[]
    for family in card['families'][:2]:
        for task in ('M1','M3'):
            for command in (0,1):
                for expert in ('CLOCK_DRIVEN','OBSERVABLE_CORRECTION_V1'):
                    cases.append(dict(case_id='D-'+family['family_id']+'-'+task+'-c'+str(command)+'-'+expert,stage='DIAGNOSTIC',family=family,task=task,
                        command=command,expert=expert,reset_seed=card['reset_seed'],max_task_s=40.,version='ROW6D'))
    for family in [f for f in card['families'] if f['split']=='FEEDBACK_TRAIN']:
        for task in ('M1','M3'):
            for command in (0,1):collection.append(dict(case_id='T-'+family['family_id']+'-'+task+'-c'+str(command),stage='COLLECTION',family=family,task=task,
                command=command,expert='OBSERVABLE_CORRECTION_V1',reset_seed=card['reset_seed'],max_task_s=40.,version='ROW6D'))
    assert len(cases)==16 and len(collection)==96
    write(ROOT/'design/TEACHER_CARD.json',dict(diagnostics=cases,collection=collection,first_valid_retained=True,full_task_success_prerequisite=False,
        teacher=dict(grid_source_seconds=.01,refine_xatol=1e-10,tie='earliest objective within1e-15',residual_blend_seconds=1.,SO3_position_weight_m=.05,
            alias_rule='separate local minima >.4s apart, objective within1e-10m2 and future differing beyond codec1e-5m/1e-4rad',one_candidate=True),
        local_witness='original fixed DLS200iterations/2mm/.01rad, actual moving root held only for conservative local witness; actual other DOFs; all40points and original URDF velocity'))
    files={str(p.relative_to(ROOT)):sha(p) for folder in ('code','design','evidence') for p in (ROOT/folder).glob('*')
        if p.is_file() and p.name not in ('TEACHER_EXECUTION_SEAL.json','RAW_BUDGET.json') and not p.name.startswith('result-')}
    for p in (ROOT/'code').glob('pi0266_*.py'):compile(p.read_text(),str(p),'exec')
    write(ROOT/'design/TEACHER_EXECUTION_SEAL.json',dict(files=files,source_reset_seal=sha(ROOT/'design/RESET_EXECUTION_SEAL.json'),
        immutable_protocol=read(ROOT/'code/PUBLICATION_RECEIPT.json'),baseline_PI0266_counters=read(ROOT/'evidence/RAW_BUDGET.json')))
    print('PASS_FIXED_16_AND_96_SOURCE_TEACHER_SEAL',flush=True)


def teachers(stage):
    py,env=environment();card=read(ROOT/'design/TEACHER_CARD.json');completed=[]
    if stage=='collection':
        prior=read(ROOT/'evidence/DIAGNOSTIC_VERDICT.json');assert prior['all_registered_cases_valid'] and prior['runtime_integrity']
    for case in card['diagnostics' if stage=='diagnostics' else 'collection']:
        folder=RAW/stage/case['case_id'];valid=[p for p in folder.glob('attempt-*/result.json') if read(p)['validity']=='VALID']
        if not valid:
            prior=list(folder.glob('attempt-*'));log=ROOT/'logs'/(case['case_id']+'-%02d.log'%len(prior))
            with log.open('wb') as stream:
                done=subprocess.run([str(py),str(ROOT/'code/pi0266_control.py'),'--case',case['case_id']],env=env,stdout=stream,stderr=subprocess.STDOUT)
            if done.returncode:raise SystemExit(done.returncode)
            valid=[p for p in folder.glob('attempt-*/result.json') if read(p)['validity']=='VALID']
        assert len(valid)==1
        result=read(valid[0]);completed.append(dict(attempt=str(valid[0].parent),result=result))
        print(json.dumps(dict(case=case['case_id'],success=result['success'],failure=result.get('failure'),valid_windows=result.get('valid_windows'))),flush=True)
    write(ROOT/'evidence'/('DIAGNOSTIC_VERDICT.json' if stage=='diagnostics' else 'COLLECTION_VERDICT.json'),dict(
        status='COMPLETE_FIXED_TEACHER_DIAGNOSTICS_CONTINUE_COLLECTION' if stage=='diagnostics' else 'COMPLETE_96_COLLECTION_WINDOW_ADMISSION_PENDING',
        rows=completed,all_registered_cases_valid=True,runtime_integrity=True,quality_successes=sum(x['result']['success'] for x in completed),
        complete_task_quality_does_not_block_window_learning=True,GR00T_queries=0,optimizer_updates=0))


def teacher_repair_seal():
    import ast
    path=ROOT/'design/TEACHER_EXECUTION_SEAL.json';old=read(path);script=ROOT/'code/pi0266_control.py'
    node=next(n for n in ast.parse(script.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_cache')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'source_cache_guard','exec'),scope)
    check=scope['check_cache'];assert check(0.,.1424,.3)=='PRE_FIRST_POST_PHYSICS_OFFSET_UPDATE'
    assert check(.02,1e-7,1e-7)=='POST_INITIALIZATION_OFFSET_VERIFIED'
    try:check(.02,.1424,.3)
    except AssertionError:pass
    else:raise AssertionError('bad post-initialization cache accepted')
    changes=[]
    for rel,digest in old['files'].items():
        if sha(ROOT/rel)!=digest:
            assert rel in ('code/pi0266_control.py','code/pi0266_run.py'),rel
            changes.append(dict(path=rel,previous_sha256=digest,new_sha256=sha(ROOT/rel)))
    assert changes
    previous=ROOT/'evidence/TEACHER_EXECUTION_SEAL_V0.json';assert not previous.exists();write(previous,old)
    attempts=[]
    for folder in (RAW/'diagnostics').glob('*/attempt-*'):
        result=read(folder/'result.json')
        assert result['validity']=='ENGINEERING_INCOMPLETE' and not (folder/'physics_started.json').exists()
        item=dict(attempt=str(folder),classification='ENGINEERING_INVALID_PRE_PHYSICS_CACHE_AUDIT_LIFECYCLE',physics_starts=0,actor_calls=0,raw_preserved=True)
        write(folder/'INVALIDATION.json',item);attempts.append(item)
    deviation=dict(status='SAME_SEMANTICS_CACHE_AUDIT_GUARD_REPAIR',reason='native post_physics_step applies TCP offset after first implicit step; mechanical FK was recorded before it',
        changes=changes,attempts=attempts,native_source_or_control_changed=False,post_initialization_strict_checks_retained=True,narrow_negative_test=True)
    write(ROOT/'evidence/TEACHER_CACHE_AUDIT_DEVIATION.json',deviation)
    updated=dict(old,files={rel:sha(ROOT/rel) for rel in old['files']},previous_seal_sha256=sha(previous),deviation_sha256=sha(ROOT/'evidence/TEACHER_CACHE_AUDIT_DEVIATION.json'))
    write(path,updated);print(json.dumps(deviation),flush=True)


if __name__=='__main__':
    if '--seal' in sys.argv:seal()
    elif '--resets' in sys.argv:resets()
    elif '--teacher-seal' in sys.argv:teacher_seal()
    elif '--teacher-repair-seal' in sys.argv:teacher_repair_seal()
    elif '--teachers' in sys.argv:
        teachers('diagnostics');teachers('collection')
    elif '--diagnostics' in sys.argv:teachers('diagnostics')
    elif '--collection' in sys.argv:teachers('collection')
    else:
        py,env=environment();raise SystemExit(subprocess.run([str(py),str(ROOT/'code/pi0266_reset.py')]+sys.argv[1:],env=env).returncode)
