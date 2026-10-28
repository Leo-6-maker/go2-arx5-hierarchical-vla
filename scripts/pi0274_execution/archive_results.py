"""New predictions/field and small final evidence through qualified HF readback."""
import argparse,ast,importlib.util,os,shutil
from pathlib import Path
from quality_gate import BASE,ROOT,RAW,read,sha

def roots(part):
    assert part in ('predictions','summary')
    return [(RAW/lane,lane) for lane in ('field','offline','newfit','late','virtual','component')] if part=='predictions' else [
        (RAW/'delivery','delivery'),(ROOT/'contracts','contracts'),(ROOT/'design','design'),(ROOT/'evidence','evidence'),
        (ROOT/'model_bindings','model-bindings'),(ROOT/'authority','authority'),(ROOT/'code','code')]

def admitted(part):
    from quality_gate import admitted as actual_admission
    actual_admission()
    inventory=read(ROOT/'evidence/QUALITY_OUTPUT_INVENTORY.json');assert inventory['status']=='PASS_EXACT_ALL_REGISTERED_RESULTS_AND_LITERAL_BUDGETS'
    for r in inventory['outputs']:
        assert sha(r['path'])==r['sha256'] and sha(Path(r['path']).with_suffix('.receipt.json'))==r['receipt_sha256']
    if part=='summary':
        assert read(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json')['status']=='PASS_PI0274_ALL_FIXED_RESULTS_TRAINERS_DATA_AND_BUDGETS'
        assert read(ROOT/'evidence/RESULT_ARCHIVE_predictions.json')['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
        assert (RAW/'delivery/FINAL_RESEARCH_RETURN.md').is_file()
    sizes=[p.stat().st_size for folder,_ in roots(part) for p in folder.rglob('*') if p.is_file()]
    assert sizes and shutil.disk_usage(RAW).free>=sum(((n+511)//512+1)*512 for n in sizes)+len(sizes)*1024+10240+10*2**30

def prepared():
    # Reuse the exact registered old archive main; only its root list and prefix differ.
    path=BASE/'PI0269/code/archive_results.py';spec=importlib.util.spec_from_file_location('pi0274_result_archive_helper',path)
    mature=importlib.util.module_from_spec(spec);spec.loader.exec_module(mature);assert mature.sha(path)=='7bc5044261252c588394b3dfc6796ca92ef123efcd4500bd3a4e5f23d917dd26'
    mature.ROOT=mature.R1=ROOT;mature.RAW=RAW;os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'),HF_HUB_OFFLINE='0')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    allowed=next(n for n in ast.walk(node) if isinstance(n,ast.Tuple) and {'.json','.jsonl','.npz','.pbs'}<={e.value for e in n.elts if isinstance(e,ast.Constant)})
    allowed.elts.extend(ast.Constant(v) for v in ('.sh','.yaml','.yml'))
    assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='roots' for t in n.targets));node.body[node.body.index(assignment)]=ast.parse('roots=registered_roots(part)').body[0]
    changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='evidence/PI0269/prefix-results-20261003/':n.value='evidence/PI0274/pose-equilibrium-20261005/';changed.append('prefix')
        elif isinstance(n,ast.Constant) and n.value=='PI0269 actual fixed prefix results ':n.value='PI0274 actual fixed pose-equilibrium results ';changed.append('message')
    assert sorted(changed)==['message','prefix'];scope=vars(mature).copy();scope.update(admitted=admitted,registered_roots=roots)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope);return scope['main']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('predictions','summary'));p.add_argument('--self-check',action='store_true');a=p.parse_args();run=prepared()
    if a.self_check:assert len(roots('predictions'))==6;print('PASS_REGISTERED_NEW_FIELD_AND_ENDPOINT_ARCHIVE_SOURCE_NO_UPLOAD',flush=True)
    else:run(a.part)
