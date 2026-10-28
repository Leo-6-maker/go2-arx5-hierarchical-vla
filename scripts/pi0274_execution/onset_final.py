"""EQUIL saved onset/visibility analysis using the already qualified A0 functions."""
import ast,importlib.util
from quality_gate import ROOT,RAW,read
source=ROOT/'code/onset.py'
spec=importlib.util.spec_from_file_location('pi0274_closed_A0_onset',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
class Registered(ast.NodeTransformer):
    def visit_Tuple(self,n):
        if all(isinstance(e,ast.Constant) for e in n.elts) and ast.literal_eval(n)==('BROAD5120','TARGET5120'):n.elts=[ast.Constant('EQUIL5120')]
        return self.generic_visit(n)
    def visit_Assert(self,n):
        if any(isinstance(e,ast.Name) and e.id=='queries' for e in ast.walk(n)):
            return ast.copy_location(ast.parse('assert len(episodes)==64 and len(queries)==len(old_transitions)').body[0],n)
        return self.generic_visit(n)
    def visit_Constant(self,n):
        mapping={128:64,'PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES':'PASS_PI0274_ALL_NEW_S4_DECODE_AND_TRAJECTORIES',
            'A0_ONSET128.csv':'EQUIL_ONSET64.csv','A0_REFERENCE_AND_VISIBILITY12368.csv':'EQUIL_REFERENCE_AND_VISIBILITY.csv',
            'evidence/A0_ONSET_ANALYSIS.json':'evidence/EQUIL_ONSET_ANALYSIS.json','PASS_ALL128_SAVED_ONSETS_AND_OBSERVED_VISIBILITY':'PASS_ALL64_EQUIL_SAVED_ONSETS_AND_OBSERVED_VISIBILITY'}
        return ast.copy_location(ast.Constant(mapping[n.value]),n) if n.value in mapping else n
    def visit_keyword(self,n):
        if n.arg=='shared_visibility_source':n.value=ast.parse("str(ROOT.parent/'PI0267/code/pi0259_observation.py')",mode='eval').body
        return self.generic_visit(n)
scope=vars(previous).copy();scope.update(ROOT=ROOT,RAW=RAW)
exec(compile(ast.fix_missing_locations(ast.Module(body=[Registered().visit(node)],type_ignores=[])),str(source),'exec'),scope)
if __name__=='__main__':scope['main']()
