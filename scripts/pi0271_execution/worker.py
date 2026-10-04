"""Narrow reuse of PI0270 inference; registered inputs and separate PI0271 counters."""
import argparse
import ast
import os
import sys
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0271';OLD=BASE/'PI0270'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0271')
sys.path.insert(0,str(OLD/'code'))
import quality_worker as mature
read,sha=mature.read,mature.sha


def admitted():
    a0=read(ROOT/'evidence/A0_DECISION.json')
    assert a0['status']=='PROCEED_REGISTERED_DIAGNOSIS'
    for row in a0['source_files']:assert sha(row['path'])==row['sha256'],row['path']
    verdict=read(RAW.with_name('PI0270')/'delivery/PRIMARY_VERDICT.json')
    assert not (verdict['ORIGINAL_VIRTUAL_QUALIFICATION']['AGG4096']['PASS'] and
                verdict['RELIABLE_VIRTUAL_QUALIFICATION']['AGG4096']['PASS'])
    close=read(OLD/'evidence/final-publication/FINAL_CLOSEOUT_RECEIPT.json')
    assert close['execution']=='COMPLETE' and close['missing_required_work']==[]
    assert read(OLD/'evidence/final-publication/FINAL_CANONICAL_RECEIPT.json')['status']=='PASS_FINAL_CANONICAL_EXACT_BODY_READBACK'
    seal=read(ROOT/'design/EXECUTION_FILESET.json')
    assert seal['status']=='FROZEN_PI0271_AFTER_ACTUAL_A0'
    for row in seal['files']:assert sha(row['path'])==row['sha256'],row['path']
    assert read(ROOT/'design/QUERY_PLAN.json')['fixed_scientific_queries']<=2400


def registered(request,point):
    # ponytail: one finite worker per policy, with the predecessor's first-valid save path.
    plan=read(ROOT/'design/QUERY_PLAN.json')
    rows=[r for r in plan['rows'] if r['point']==point and r['request']['output']==request['output']]
    assert len(rows)==1 and rows[0]['request']==request,'ONLY_EXACT_FROZEN_QUERY_REQUESTS'
    assert Path(request['output']).resolve().is_relative_to(RAW.resolve())
    assert sha(request['input'])==rows[0]['input_file_sha256']


def prepared():
    source=OLD/'code/quality_worker.py'
    assert sha(source)=='7e70551b613b218986efae1dfe6f2fa9be2d347779352166274e8d31686ceeb4'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=node.body[-1];assert isinstance(loop,ast.For) and ast.unparse(loop.iter)=='sys.stdin'
    loop.body.insert(1,ast.parse('registered(request,point)').body[0])
    changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant):
            if n.value=='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270':n.value=str(RAW);changed.append('root')
            elif n.value==30720:n.value=2560;changed.append('attempt_cap')
            elif n.value==122880:n.value=10240;changed.append('head_cap')
    assert sorted(changed)==['attempt_cap','head_cap','root']
    routes={
        "input_path.resolve().is_relative_to(raw_root)": "assert input_path.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')) and output.resolve().is_relative_to(raw_root)",
        'noise_path.resolve().is_relative_to(p)': "assert any(noise_path.resolve().is_relative_to(p) for p in (Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/noise'), raw_root/'noise-support'))",
        'registry in': "assert registry in ('LATE_SUPPORT_CARD.json','VISITED_QUERY_CARD.json')",
        "point == 'AGG4096'": "assert point in ('PREFIX4096','AGG4096')",
    }
    for match,replacement in routes.items():
        indices=[i for i,n in enumerate(loop.body) if isinstance(n,ast.Assert) and match in ast.unparse(n)]
        assert len(indices)==1,(match,indices)
        loop.body[indices[0]]=ast.parse(replacement).body[0]
    original=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def tail(function):
        body=function.body[-1].body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='handle' for t in n.targets))
        return ast.dump(ast.Module(body=body[start:],type_ignores=[]),include_attributes=False)
    assert tail(node)==tail(original),'COMPLETE_NATIVE_S4_DECODE_NOISE_AND_FIRST_VALID_SAVE_MUST_BE_UNCHANGED'
    scope=vars(mature).copy();scope.update(ROOT=ROOT,R1=ROOT,admitted=admitted,registered=registered)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
    return scope['main']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('PREFIX4096','AGG4096'));p.add_argument('--self-check',action='store_true');a=p.parse_args()
    run=prepared()
    if a.self_check:
        try:admitted()
        except FileNotFoundError as error:
            assert Path(error.filename)==ROOT/'evidence/A0_DECISION.json'
            print('PASS_UNCHANGED_S4_DECODE_SAVE_AND_MISSING_A0_BLOCK_NO_MODEL_QUERY')
        else:print('PASS_UNCHANGED_S4_DECODE_SAVE_SOURCE_ACTUAL_ADMISSION_PRESENT')
    else:
        assert a.point;run(a.point)
