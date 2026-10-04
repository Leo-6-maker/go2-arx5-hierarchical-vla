"""Preserve only new diagnosis outputs with the qualified public full-readback path."""
import argparse
import ast
import importlib.util
import os
import shutil
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0271'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0271')
SOURCE_SHA='7bc5044261252c588394b3dfc6796ca92ef123efcd4500bd3a4e5f23d917dd26'


def roots(part):
    return {'support':[(RAW/'support','support'),(RAW/'noise-support','noise-support')],
        'visited':[(RAW/'visited','visited'),(RAW/'labels-visited','labels-visited')],
        'summary':[(RAW/'delivery','delivery'),(ROOT/'contracts','contracts'),(ROOT/'design','design'),
                   (ROOT/'evidence','evidence'),(ROOT/'model_bindings','model-bindings'),(ROOT/'code','code')]}[part]


def admitted(part):
    from worker import admitted as scientific_admitted,read,sha
    scientific_admitted()
    complete=read(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json')
    assert complete['status']=='ALL_FIXED_PI0271_OUTPUTS_COMPLETE'
    plan=read(ROOT/'design/QUERY_PLAN.json');assert complete['query_plan_sha256']==sha(ROOT/'design/QUERY_PLAN.json')
    assert len(complete['rows'])==len(plan['rows'])<=2400
    for row in complete['rows']:
        assert sha(row['path'])==row['sha256'] and sha(Path(row['path']).with_suffix('.receipt.json'))==row['receipt_sha256']
    if part in ('visited','summary'):
        labels=read(ROOT/'evidence/VISITED_LABELS_COMPLETE.json');assert len(labels['records'])==64
        for row in labels['records']:assert sha(row['path'])==row['sha256']
    if part=='summary':
        assert read(RAW/'delivery/CPU_REPLAY.json')['status']=='PASS_ALL_NEW_S4_DECODE_METRICS_AND_FIXED_K8_REPLAY'
        assert read(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json')['status']=='PASS_COMPLETE_FIXED_DIAGNOSIS_ARTIFACTS_AND_BUDGETS'
        assert (RAW/'delivery/FINAL_RESEARCH_RETURN.md').is_file()
        for previous in ('support','visited'):
            assert read(ROOT/'evidence'/('RESULT_ARCHIVE_'+previous+'.json'))['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
    sizes=[p.stat().st_size for folder,_ in roots(part) for p in folder.rglob('*') if p.is_file()]
    assert sizes
    # One uncompressed tar plus manifest overhead; original sources stay in place.
    extra=sum(((n+511)//512+1)*512 for n in sizes)+len(sizes)*1024+10240+10*2**30
    available=shutil.disk_usage(RAW).free;assert available>=extra,('ARCHIVE_SPACE',available,extra)
    print('ACTUAL_ARCHIVE_SPACE',part,'source_bytes',sum(sizes),'required_extra_with_10GiB',extra,'available',available,'quota_UNKNOWN',flush=True)


def prepared():
    path=BASE/'PI0269/code/archive_results.py';spec=importlib.util.spec_from_file_location('qualified_diagnosis_archive',path)
    mature=importlib.util.module_from_spec(spec);spec.loader.exec_module(mature);assert mature.sha(path)==SOURCE_SHA
    mature.ROOT=mature.R1=ROOT;mature.RAW=RAW
    os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'),HF_HUB_OFFLINE='0')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='roots' for t in n.targets))
    node.body[node.body.index(assignment)]=ast.parse('roots=registered_roots(part)').body[0]
    changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='evidence/PI0269/prefix-results-20261003/':
            n.value='evidence/PI0271/frozen-policy-diagnosis-20261004/';changed.append('prefix')
        elif isinstance(n,ast.Constant) and n.value=='PI0269 actual fixed prefix results ':
            n.value='PI0271 actual fixed diagnosis results ';changed.append('message')
    assert sorted(changed)==['message','prefix']
    scope=vars(mature).copy();scope.update(admitted=admitted,registered_roots=roots)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope['main']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=('support','visited','summary'));parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    run=prepared()
    if args.self_check:
        assert roots('support')[0][0]==RAW/'support' and roots('visited')[1][0]==RAW/'labels-visited'
        print('PASS_QUALIFIED_NEW_OUTPUT_PACK_AND_FULL_READBACK_SOURCE_NO_UPLOAD')
    else:run(args.part)
