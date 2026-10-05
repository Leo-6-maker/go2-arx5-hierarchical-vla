"""New EQUIL same-tick transitions; original failure/retention boundaries reused."""
import argparse,ast,importlib.util
from quality_gate import BASE,ROOT,RAW
source=BASE/'PI0273/code/transitions.py'
spec=importlib.util.spec_from_file_location('pi0273_closed_transitions',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.ROOT=ROOT;previous.RAW=RAW
class Registered(ast.NodeTransformer):
    def visit_Tuple(self,n):
        if all(isinstance(e,ast.Constant) for e in n.elts) and ast.literal_eval(n)==('BROAD5120','TARGET5120'):n.elts=[ast.Constant('EQUIL5120')]
        return self.generic_visit(n)
    def visit_Constant(self,n):
        mapping={128:64,64:32,'FAILURE_AND_POST_SUCCESS_DIAGNOSIS128.csv':'FAILURE_AND_POST_SUCCESS_DIAGNOSIS64.csv',
            'PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES':'PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES'}
        return ast.copy_location(ast.Constant(mapping[n.value]),n) if n.value in mapping else n
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
scope=vars(previous).copy();scope.update(ROOT=ROOT,RAW=RAW,__file__=__file__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[Registered().visit(node)],type_ignores=[])),str(source),'exec'),scope)
main=scope['main'];self_check=previous.self_check
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');self_check() if p.parse_args().self_check else main()
