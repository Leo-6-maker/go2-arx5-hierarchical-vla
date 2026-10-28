"""Reuse the qualified public fixed-revision/file/member archive implementation."""
import argparse,ast,importlib.util,json,os,shutil
from pathlib import Path
from run import ROOT,sha
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')

def roots(part):
    if part=='training_data':
        return [(RAW/'working_data','source/data')]+[(ROOT/n,n) for n in ('design','contracts','evidence','authority','code')]+[(ROOT/'licenses','licenses')]
    if part=='source':return [(RAW/'episodes','episodes'),(RAW/'evidence','raw_evidence')]+[(ROOT/n,n) for n in ('design','evidence','authority','licenses')]
    if part=='predictions':return [(RAW/'quality','quality'),(RAW/'noise','noise'),(RAW/'episodes','episodes')]
    assert part=='summary'
    return [(RAW/'delivery','delivery'),(RAW/'evidence','raw_evidence')]+[(ROOT/n,n) for n in ('design','contracts','evidence','model_bindings','authority','code','licenses')]

def admitted(part):
    if part in ('training_data','source'):
        a=json.loads((ROOT/'evidence/DATA_ADMISSION.json').read_text());assert a['launch_allowed'] and a['status']=='PASS_PI0276_ACTUAL_REPLAY_INPUTS_CODEC_AND_TASK_STATISTICS'
    else:assert json.loads((ROOT/'evidence/PHYSICAL_EVALUATION_COMPLETE.json').read_text())['status']=='ALL_FIXED_PHYSICAL_LANES_EXECUTED'
    if part=='training_data':
        card=json.loads((ROOT/'design/DATASET_CARD.json').read_text())
        for e in card['episodes']:
            if e['split']!='TRAIN' or not e['admitted']:continue
            dst=RAW/'working_data'/f"TRAIN-{e['seed']}";dst.mkdir(parents=True,exist_ok=True)
            r=next(r for r in card['rows'] if r['seed']==e['seed'] and r['split']=='TRAIN')
            for name,key in [('commands.npz','command_sha256'),('observations.npz','observation_sha256')]:
                src=Path(e['path'])/name;target=dst/name;assert sha(src)==r[key]
                if not target.exists():os.link(src,target)
                assert os.path.samefile(src,target) and sha(target)==r[key]
    size=sum(p.stat().st_size for root,_ in roots(part) for p in root.rglob('*') if p.is_file())
    assert size>0 and shutil.disk_usage(RAW).free>=size+10*2**30

def prepared():
    src=ROOT.parent/'PI0269/code/archive_results.py'
    assert sha(src)=='7bc5044261252c588394b3dfc6796ca92ef123efcd4500bd3a4e5f23d917dd26'
    spec=importlib.util.spec_from_file_location('pi0276_archive_helper',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    node=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='roots' for t in n.targets))
    start=node.body.index(assignment);assert isinstance(node.body[start+1],ast.If)
    node.body[start:start+2]=ast.parse('roots=registered_roots(part)').body
    allowed=next(n for n in ast.walk(node) if isinstance(n,ast.Tuple) and {'.json','.jsonl','.npz','.pbs'}<={e.value for e in n.elts if isinstance(e,ast.Constant)})
    allowed.elts.extend(ast.Constant(v) for v in ('.sh','.yaml','.yml','.txt'))
    replacements={'evidence/PI0269/prefix-results-20261003/':'evidence/PI0276/maniskill-stackcube-20261006/',
        'PI0269 actual fixed prefix results ':'PI0276 actual StackCube evidence ',
        'preserves_original_V2_FAIL':'preserves_all_original_source_and_model_failures',
        'not_physical_generalization':'not_target_robot_transfer'}
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in replacements:n.value=replacements[n.value]
        elif isinstance(n,ast.keyword) and n.arg in replacements:n.arg=replacements[n.arg]
    scope=vars(m).copy();scope.update(ROOT=ROOT,R1=ROOT,RAW=RAW,admitted=admitted,registered_roots=roots)
    os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(RAW/'hf_online/xet'),HF_HUB_OFFLINE='0')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(src),'exec'),scope)
    return scope['main']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('training_data','source','predictions','summary'));p.add_argument('--check-only',action='store_true');a=p.parse_args()
    run=prepared()
    if a.check_only:admitted(a.part);print('ACTUAL_ARCHIVE_INPUTS_CHECKED_NO_UPLOAD',a.part)
    else:run(a.part)
