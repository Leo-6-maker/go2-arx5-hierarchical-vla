"""Only new EQUIL saved trajectories; unchanged executor, gate and V3 arithmetic."""
import argparse,ast,importlib.util
from quality_gate import BASE,ROOT,RAW
source=BASE/'PI0273/code/trajectory_replay.py'
spec=importlib.util.spec_from_file_location('pi0273_closed_trajectory_replay',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
tuples=[n for n in ast.walk(node) if isinstance(n,ast.Tuple) and all(isinstance(e,ast.Constant) for e in n.elts) and ast.literal_eval(n)==('BROAD5120','TARGET5120')]
assert len(tuples)==1;tuples[0].elts=[ast.Constant('EQUIL5120')]
scope=vars(previous).copy();scope.update(ROOT=ROOT,RAW=RAW,__file__=__file__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope);main=scope['main']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=('virtual','component'),required=True);p.add_argument('--point',choices=('EQUIL5120',),required=True);a=p.parse_args();main(a.lane,a.point)
