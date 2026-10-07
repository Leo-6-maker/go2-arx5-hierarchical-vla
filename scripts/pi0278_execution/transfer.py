"""Reuse qualified byte transfer/readback; no local weight payload or new transport."""
import argparse,ast,importlib.util
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'pi0277_execution/r2/transfer.py'
spec=importlib.util.spec_from_file_location('qualified_transfer',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run';old.RAW='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2'
node=next(n for n in ast.parse(source.read_text()).body if getattr(n,'name',None)=='copy');assert isinstance(node.body[0],ast.Assert)
node.body[0]=ast.parse("assert point in [f'COVER93{n}' for n in (512,1024,2048,4096)]").body[0]
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),vars(old))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point',nargs='?');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:old.check()
    else:assert a.point;old.copy(a.point)
