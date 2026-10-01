"""Finite native dispatcher using the accepted environment, isolated project caches."""
from pathlib import Path
import ast
import json
import subprocess
import sys
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0256/runtime')
import pi0255_run as original
from pi0265_source import ROOT,sha,write


def seal():
    from pi0258_kinematic import instructions
    card=json.loads((ROOT/'design/SCENE_CARD.json').read_text())
    assert not (ROOT/'evidence/RAW_BUDGET.json').exists()
    texts=[]
    for family in card['families']:
        for task in ('M1','M3'):
            for command in (0,1):texts.append(dict(family_id=family['family_id'],task=task,command=command,
                instruction=instructions(task,command,family['factors']['colors'])[0],supervision_templates='eight original M1 TRAIN forms only; M3 unchanged'))
    write(ROOT/'design/TEXT_CARD.json',dict(rows=texts,model_input_excludes_geometry_phase_source=True))
    confirm=[]
    for version in ('LEGACY','ROW6D'):
        for task in ('M1','M3'):
            for command in (0,1):confirm.append(dict(case_id='CONFIRM-'+task+'-c'+str(command)+'-'+version,
                stage='CONFIRM',version=version,route='REFERENCE_PROVIDER',family=card['families'][2],task=task,
                command=command,reset_seed=card['families'][2]['reset_seed'],max_task_s=20.,conditional=True))
    write(ROOT/'design/CONFIRM_CARD.json',dict(conditional_cases=confirm,execute_one_version_only=True,fourth_DEV_unexecuted=True))
    files={str(p.relative_to(ROOT)):sha(p) for folder in ('code','design','evidence') for p in (ROOT/folder).glob('*')
        if p.is_file() and p.name!='EXECUTION_SEAL.json'}
    write(ROOT/'design/EXECUTION_SEAL.json',dict(authority='edec521de2087d0ab96abfa8a13afc79d108912c',
        canonical_comment=5926263967,files=files,new_physics_starts=0))
    print('PASS_BEFORE_PHYSICS_EXECUTION_SEAL',flush=True)


def repair_seal():
    source=ROOT/'code/pi0265_control.py';tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forbidden_hits')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);forbidden_hits=scope['forbidden_hits']
    assert not forbidden_hits([dict(pair=['RL_foot','ground'])])
    assert forbidden_hits([dict(pair=['x5_link3','grey_obstacle'])])
    current=ROOT/'design/EXECUTION_SEAL.json';old=json.loads(current.read_text())
    repair=[];raw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0265')
    for path in sorted((raw/'controls').glob('*/attempt-*/result.json')):
        result=json.loads(path.read_text());rows=json.loads((path.parent/'rows.json').read_text())
        if result.get('failure')=='PRE_STEP_SAFETY' and result.get('WBC_actor_calls')==0 and rows and not rows[-1]['hard_unsafe'] and rows[-1]['collision']['hits'] and not forbidden_hits(rows[-1]['collision']['hits']):
            item=dict(original_result_sha256=sha(path),classification='ENGINEERING_INVALID_SUPPORT_CONTACT_CLASSIFIER',
                raw_preserved=True,raw_budget_retained=True,proof='only ordinary ground support was classified by added diagnostic; zero actor forwards',
                attempt=str(path.parent),original_ground_hits=rows[-1]['collision']['hits'])
            write(path.parent/'INVALIDATION.json',item);repair.append(item)
    assert len(repair)<=8,'technical physics reserve exhausted'
    write(ROOT/'evidence/GROUND_CLASSIFIER_DEVIATION.json',dict(status='MINIMAL_SAME_CONTRACT_REPAIR',attempts=repair))
    write(ROOT/'evidence/EXECUTION_SEAL_V0.json',old)
    files={path:sha(ROOT/path) for path in old['files']}
    for path,digest in old['files'].items():
        if path not in ('code/pi0265_control.py','code/pi0265_run.py'):assert files[path]==digest,path
    write(current,dict(old,files=files,previous_seal_sha256=sha(ROOT/'evidence/EXECUTION_SEAL_V0.json'),
        repair='support contacts excluded from added self/obstacle check; native ground termination unchanged'))
    print(json.dumps(dict(repaired_invalid_starts=len(repair),raw=json.loads((ROOT/'evidence/RAW_BUDGET.json').read_text()))),flush=True)


def g1():
    import numpy as np
    py,env=environment();card=json.loads((ROOT/'design/SCENE_CARD.json').read_text());raw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0265')
    def execute(case):
        existing=list((raw/'controls'/case['case_id']).glob('attempt-*/result.json'))
        valid=[json.loads(p.read_text()) for p in existing if json.loads(p.read_text())['validity']=='VALID' and not (p.parent/'INVALIDATION.json').exists()]
        if valid:return valid[0]
        log=ROOT/'logs'/(case['case_id']+'-%02d.log'%len(existing))
        with log.open('wb') as stream:
            status=subprocess.run([str(py),str(ROOT/'code/pi0265_control.py'),'--case',case['case_id']],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if status.returncode:raise SystemExit(status.returncode)
        result=json.loads(sorted((raw/'controls'/case['case_id']).glob('attempt-*/result.json'))[-1].read_text())
        assert result['validity']=='VALID'
        print(json.dumps(dict(case=case['case_id'],success=result.get('success'),failure=result.get('failure'))),flush=True);return result
    c0=[case for case in card['G1_initial_cases'] if case['stage']=='C0']
    results=[execute(case) for case in c0]
    parity=[]
    for version in ('LEGACY','ROW6D'):
        roots=[];tables=[]
        for route in ('DIRECT_NATIVE','REFERENCE_PROVIDER'):
            folder=raw/'controls'/('C0-'+version+'-'+route)
            valid=[p.parent for p in folder.glob('attempt-*/result.json') if json.loads(p.read_text())['validity']=='VALID' and not (p.parent/'INVALIDATION.json').exists()]
            assert len(valid)==1
            roots.append(dict(np.load(valid[0]/'pre_first_step.npz',allow_pickle=False)))
            tables.append(dict(np.load(valid[0]/'installed_table.npz',allow_pickle=False)))
        for key in ('root','dof','torch_rng','cuda_rng'):assert np.array_equal(roots[0][key],roots[1][key]),('C0_RESET_PARITY',version,key)
        assert tables[0].keys()==tables[1].keys()
        for key in tables[0]:assert np.array_equal(tables[0][key],tables[1][key]),('C0_TABLE_PARITY',version,key)
        parity.append(dict(version=version,reset_and_RNG_exact=True,all_installed_table_bytes_exact=True,table_fields=list(tables[0])))
    write(ROOT/'evidence/C0_ROUTE_PARITY.json',dict(status='PASS_ACTUAL_SOURCE_RESET_AND_INSTALLED_TABLE_PARITY',rows=parity))
    results += [execute(case) for case in card['G1_initial_cases'] if case['stage']=='DEV']
    qualification={}
    for version in ('LEGACY','ROW6D'):
        group=[r for r in results if r['case']['stage']=='DEV' and r['case']['version']==version]
        good=[r for r in group if r['success'] and r['tracking_xyz_rms_m']<=.030 and r['tracking_xyz_p95_m']<=.050 and r['tracking_so3_rms_rad']<=.10]
        qualification[version]=dict(successes=sum(r['success'] for r in group),all_successful_tracking_pass=len(good)==sum(r['success'] for r in group),
            M1=sum(r['case']['task']=='M1' for r in good),M3=sum(r['case']['task']=='M3' for r in good))
    eligible=[v for v,r in qualification.items() if r['successes']>=7 and r['all_successful_tracking_pass'] and r['M1']>=3 and r['M3']>=3]
    chosen=eligible[0] if eligible else None
    confirmation=[]
    if chosen:
        rows=json.loads((ROOT/'design/CONFIRM_CARD.json').read_text())['conditional_cases']
        confirmation=[execute(case) for case in rows if case['version']==chosen]
    passed=bool(chosen and all(r['success'] and r['tracking_xyz_rms_m']<=.030 and r['tracking_xyz_p95_m']<=.050 and r['tracking_so3_rms_rad']<=.10 for r in confirmation))
    write(ROOT/'evidence/G1_VERDICT.json',dict(status='PASS_CONTINUE_FEEDBACK' if passed else 'PI0265_CONTROLLER_ENVELOPE_BLOCKED',
        overall='IN_PROGRESS' if passed else 'PARTIAL',qualification=qualification,chosen_runtime=chosen,
        initial_results=results,confirmation=confirmation,selected_checkpoint='NONE',deployment='NONE'))
    print(json.dumps(dict(G1_pass=passed,qualification=qualification,chosen=chosen)),flush=True)


def environment():
    (ROOT/'runtime').mkdir(exist_ok=True)
    original.ROOT=ROOT
    py,env=original.environment()
    env['PYTHONPATH']+=':'+str(ROOT/'code')
    return py,env


if __name__=='__main__':
    if '--seal' in sys.argv:seal();raise SystemExit(0)
    if '--repair-seal' in sys.argv:repair_seal();raise SystemExit(0)
    if '--g1' in sys.argv:g1();raise SystemExit(0)
    py,env=environment()
    script=ROOT/'code'/('pi0265_g0.py' if '--cpu' in sys.argv else 'pi0265_control.py')
    done=subprocess.run([str(py),str(script)]+[s for s in sys.argv[1:] if s!='--cpu'],env=env)
    raise SystemExit(done.returncode)
