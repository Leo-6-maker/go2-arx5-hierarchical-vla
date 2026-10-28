"""Reuse full-file plus streamed all-member readback for new registered material."""
import argparse,ast,importlib.util,os,shutil
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0274';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274')
def roots(part):
    assert part=='data'
    return [(RAW/'data','data'),(RAW/'noise-field','noise-field'),(ROOT/'design','design'),(ROOT/'evidence','evidence'),(ROOT/'authority','authority'),(ROOT/'code','code')]
def admitted(part):
    from quality_gate import read,sha
    receipt=read(ROOT/'evidence/DATA_ADMISSION.json');assert receipt['status']=='PASS_ALL672_REAL_POSES_CODEC_INPUTS_AND_FIXED_SCHEDULE' and receipt['launch_allowed']
    pre=read(ROOT/'evidence/DATA_PRECODEC.json');assert pre['total_proposals']==1248 and not pre['training_invalid']
    for row in pre['samples']:assert sha(row['path'])==row['sha256'] and Path(row['path']).with_name('processed_inputs.json').is_file()
    sizes=[p.stat().st_size for folder,_ in roots(part) for p in folder.rglob('*') if p.is_file()]
    assert shutil.disk_usage(RAW).free>=sum(sizes)+len(sizes)*2048+10*2**30
def prepared():
    path=BASE/'PI0269/code/archive_results.py';spec=importlib.util.spec_from_file_location('pi0274_qualified_archive',path);mature=importlib.util.module_from_spec(spec);spec.loader.exec_module(mature)
    assert mature.sha(path)=='7bc5044261252c588394b3dfc6796ca92ef123efcd4500bd3a4e5f23d917dd26'
    mature.ROOT=mature.R1=ROOT;mature.RAW=RAW;os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'),HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'),HF_HUB_OFFLINE='0')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='roots' for t in n.targets));node.body[node.body.index(assignment)]=ast.parse('roots=registered_roots(part)').body[0]
    changed=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='evidence/PI0269/prefix-results-20261003/':n.value='evidence/PI0274/pose-equilibrium-20261005/';changed.append('prefix')
        elif isinstance(n,ast.Constant) and n.value=='PI0269 actual fixed prefix results ':n.value='PI0274 frozen actual pose dataset ';changed.append('message')
    assert sorted(changed)==['message','prefix'];scope=vars(mature).copy();scope.update(admitted=admitted,registered_roots=roots)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope);return scope['main']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=('data',));a=p.parse_args();prepared()(a.part)
