"""New predictions and small evidence only, through the qualified HF readback helper."""
import argparse
import ast
import importlib.util
import os
import shutil
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0272'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0272')


def roots(part):
    assert part in ('predictions','summary')
    return [(RAW/'quality','quality')] if part=='predictions' else [(RAW/'delivery','delivery'),
        (ROOT/'contracts','contracts'),(ROOT/'design','design'),(ROOT/'evidence','evidence'),
        (ROOT/'model_bindings','model-bindings'),(ROOT/'code','code')]


def admitted(part):
    from quality_worker import admitted as actual_admission,read,sha
    actual_admission()
    result=read(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json');assert result['status']=='ALL_FIXED_PI0272_OUTPUTS_COMPLETE'
    assert result['evaluation_card_sha256']==sha(ROOT/'design/EVALUATION_CARD.json') and len(result['rows'])==2148
    for r in result['rows']:
        assert sha(r['path'])==r['sha256'] and sha(Path(r['path']).with_suffix('.receipt.json'))==r['receipt_sha256']
    if part=='summary':
        assert read(RAW/'delivery/CPU_REPLAY.json')['status']=='PASS_PI0272_SAVED_READOUT_DECODE_AND_PHYSICAL_METRICS'
        assert read(ROOT/'evidence/RESULT_ARCHIVE_predictions.json')['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
        assert read(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json')['status']=='PASS_PI0272_ALL_FIXED_RESULTS_TRAINERS_AND_BUDGETS'
        assert (RAW/'delivery/FINAL_RESEARCH_RETURN.md').is_file()
    sizes=[p.stat().st_size for folder,_ in roots(part) for p in folder.rglob('*') if p.is_file()]
    required=sum(((n+511)//512+1)*512 for n in sizes)+len(sizes)*1024+10240+10*2**30
    assert sizes and shutil.disk_usage(RAW).free>=required


def prepared():
    path=BASE/'PI0269/code/archive_results.py';spec=importlib.util.spec_from_file_location('pi0272_qualified_archive',path)
    mature=importlib.util.module_from_spec(spec);spec.loader.exec_module(mature)
    assert mature.sha(path)=='7bc5044261252c588394b3dfc6796ca92ef123efcd4500bd3a4e5f23d917dd26'
    mature.ROOT=mature.R1=ROOT;mature.RAW=RAW
    os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'),HF_HUB_OFFLINE='0')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='roots' for t in n.targets))
    node.body[node.body.index(assignment)]=ast.parse('roots=registered_roots(part)').body[0]
    changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='evidence/PI0269/prefix-results-20261003/':
            n.value='evidence/PI0272/matched-direct-fitting-20261004/';changed.append('prefix')
        elif isinstance(n,ast.Constant) and n.value=='PI0269 actual fixed prefix results ':
            n.value='PI0272 fixed readout fitting results ';changed.append('message')
    assert sorted(changed)==['message','prefix']
    scope=vars(mature).copy();scope.update(admitted=admitted,registered_roots=roots)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope['main']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('predictions','summary'));p.add_argument('--self-check',action='store_true');a=p.parse_args()
    run=prepared()
    if a.self_check:assert roots('predictions')==[(RAW/'quality','quality')];print('PASS_QUALIFIED_ARCHIVE_SOURCE_NO_UPLOAD')
    else:run(a.part)
