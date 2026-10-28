"""Only the three compared EQUIL canary payloads, through the qualified helper."""
import argparse,ast,importlib.util
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0274')
source=ROOT.with_name('PI0273')/'code/retire_discarded_canary.py'
spec=importlib.util.spec_from_file_location('pi0273_canary_retirement',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');changes=0
for n in ast.walk(node):
    if isinstance(n,ast.Constant) and n.value=='PASS_PI0273_REAL_EXACT_RESUME_CANARY':n.value='PASS_PI0274_REAL_EXACT_RESUME_CANARY';changes+=1
assert changes==1
scope=vars(old).copy();scope['ROOT']=ROOT
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('EQUIL',));scope['main'](p.parse_args().arm)
