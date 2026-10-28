"""Execute the actual suffix valid statements against sealed errors and boundaries."""
import ast,copy,json
from pathlib import Path

def checker(source):
    tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='suffix')
    nodes=sorted([n for n in ast.walk(fn) if isinstance(n,ast.Assign) and not isinstance(n.value,ast.Constant) and any(isinstance(t,ast.Name) and t.id=='valid' for t in n.targets)],key=lambda n:n.lineno)
    assert len(nodes)==2 and 'native_preclip_out_of_range_count' not in ast.unparse(ast.Module(body=nodes,type_ignores=[]))
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_suffix_valid_statements','exec')
    def valid(err):
        if 'error' in err:return False
        scope=dict(err=err,maxima=dict(XYZ_max_m=0.,SO3_max_rad=0.,gripper_max_fraction=0.))
        exec(code,scope);return bool(scope['valid'])
    return valid

def boundary(valid,real_windows):
    base=dict(XYZ_max_m=0.,SO3_max_rad=0.,gripper_max_fraction=0.,nonfinite=False,
              decoded_rot6d_validity=dict(degenerate_count=0),command_rot6d_validity=dict(degenerate_count=0),native_preclip_out_of_range_count=80)
    cases=[]
    def check(label,err,expected):
        actual=valid(err);assert actual==expected,(label,actual,expected)
        cases.append(dict(case=label,expected=expected,actual=actual))
    check('diagnostic_preclip80_with_valid_physical_error',base,True)
    for key,t in [('XYZ_max_m',1e-5),('SO3_max_rad',1e-4),('gripper_max_fraction',1e-5)]:
        x=copy.deepcopy(base);x[key]=t;check(key+'_at_boundary',x,True)
        x[key]=t*(1+1e-12);check(key+'_above_boundary',x,False)
    x=copy.deepcopy(base);x['nonfinite']=True;check('nonfinite',x,False)
    for key in ['decoded_rot6d_validity','command_rot6d_validity']:
        x=copy.deepcopy(base);x[key]['degenerate_count']=1;check(key+'_degenerate',x,False)
    real=[]
    for w in real_windows:
        expected=w['slot'].endswith('12s');actual=valid(w['sealed_roundtrip_errors']);assert actual==expected
        real.append(dict(slot=w['slot'],anchor=w['anchor'],expected=expected,actual=actual))
    assert len(real)==15 and sum(x['actual'] for x in real)==12
    return dict(status='PASS_ACTUAL_SOURCE_VALID_STATEMENTS_BOUNDARIES_AND_REAL_SO3_FAILURE',synthetic=cases,real_sealed_failed_windows=real,
                model_calls=0,physics_calls=0,codec_calls=0)

if __name__=='__main__':
    here=Path(__file__).resolve().parent;repo=here.parents[2]
    check=checker((repo/'scripts/pi0286_execution/recovery.py').read_text())
    windows=json.loads((here/'PRECLIP_FAILED15_DETAILS.json').read_text())['window_details']
    result=boundary(check,windows)
    (here/'NATIVE_CONTRACT_V2_LOCAL_BOUNDARY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
