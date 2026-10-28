"""Seal completed predecessor journals using the existing small-evidence packer."""
import ast
import importlib.util
import json
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0270')
SOURCE_SHA='661aa72054a6c119ebec5f8291165d9a27088afff5448a8060774b7a0d22b774'


def main():
    path=Path(__file__).with_name('pi0269_training_evidence.py')
    spec=importlib.util.spec_from_file_location('qualified_small_evidence',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.sha(path)==SOURCE_SHA
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    arms=[n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='arm']
    assert len(arms)==1 and ast.literal_eval(arms[0].iter)==('UNIFORM','PREFIX')
    arms[0].iter=ast.parse("('AGG',)",mode='eval').body
    statuses=[n for n in ast.walk(node) if isinstance(n,ast.Constant) and n.value=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY']
    assert len(statuses)==1;statuses[0].value='PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY'
    scope=vars(module).copy();scope['ROOT']=ROOT
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    scope['main']()
    print(json.dumps(dict(source_sha256=SOURCE_SHA,wrapper_sha256=module.sha(Path(__file__)),
        model_weights=0,optimizer_payloads=0,model_queries=0,teacher_calls=0)),flush=True)


if __name__=='__main__':main()
