"""Exercise the frozen source termination guard against false post-reset hits."""
import ast
import copy
import json
from types import SimpleNamespace
import numpy as np
from pi0266_common import ROOT,read,sha,write
from pi0266_metrics import score


def main():
    path=ROOT/'code/pi0266_control.py';tree=ast.parse(path.read_text())
    guards=[node for node in ast.walk(tree) if isinstance(node,ast.If) and any(
        isinstance(child,ast.Assign) and isinstance(child.value,ast.Constant) and child.value.value=='NATIVE_TERMINATION' for child in node.body)]
    assert len(guards)==1;guard=copy.deepcopy(guards[0])
    # Only remove the loop's break so the actual source flag assignments can be exercised on CPU.
    guard.body=[node for node in guard.body if not isinstance(node,ast.Break)]
    prefix=ast.parse('safety=False\nfailure=None\nresult={}').body
    suffix=ast.parse('return dict(safety=safety,failure=failure,result=result)').body
    fn=ast.FunctionDef(name='source_guard',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ('done','event','native')],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=prefix+[guard]+suffix,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={};exec(compile(module,str(path),'exec'),scope)
    t=np.arange(0,1.001,.02);p=np.tile([.3,0,.6],(len(t),1));r=np.tile(np.eye(3),(len(t),1,1))
    spec=dict(task='M1',command=0,goal=p[0],goal_rotation=np.eye(3));assert score(t,p,r,p,r,spec)['success']
    checks=[]
    for done,event,reset in ((True,None,False),(False,{'pre_autoreset':True},False),(False,None,True)):
        native=SimpleNamespace(reset_buf=[SimpleNamespace(item=lambda reset=reset:reset)])
        flags=scope['source_guard'](done,event,native)
        assert flags['safety'] and flags['failure']=='NATIVE_TERMINATION'
        assert not score(t,p,r,p,r,spec,flags['safety'])['success']
        checks.append(dict(done=done,event=event,reset_buf=reset,false_post_reset_goal_hit_rejected=True))
    flags=scope['source_guard'](False,None,SimpleNamespace(reset_buf=[SimpleNamespace(item=lambda:False)]))
    assert not flags['safety'] and flags['failure'] is None
    report=dict(status='PASS_FROZEN_TERMINATION_SOURCE_GUARD_FALSE_POST_RESET_HIT_NEGATIVES',source_control_sha256=sha(path),
        checked_source_AST=True,guard_routes=checks,unaltered_nonterminal_positive=True,simulation_steps=0,actor_forwards=0,GR00T_queries=0,optimizer_updates=0)
    write(ROOT/'evidence/LIFECYCLE_CPU_NEGATIVES.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
