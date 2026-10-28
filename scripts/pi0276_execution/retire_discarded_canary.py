"""Only this task's three compared and explicitly discarded canary payloads."""
import ast,importlib.util
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0276')
source=ROOT.with_name('PI0273')/'code/retire_discarded_canary.py'
spec=importlib.util.spec_from_file_location('pi0276_canary_retirement',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
for n in ast.walk(node):
    if isinstance(n,ast.Constant) and n.value=='PASS_PI0273_REAL_EXACT_RESUME_CANARY':n.value='PASS_PI0276_REAL_EXACT_RESUME_CANARY'
scope=vars(m).copy();scope['ROOT']=ROOT
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
if __name__=='__main__':scope['main']('TASK')
