"""Isolated convergence diagnostics from the original certifier's AST.

Only iteration/convergence constants and read-only trace collection change.
All path points, history, IK method, contact rules and .7/1.0 joint gates stay.
Alternate results never replace original certificates or authorize execution.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np

from scripts.pi0294 import task_path_model as original
from scripts.pi0294.task_state import ActualState,parse_state
from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.sla.audit_query_feasibility_v7 import joint_rates

ORIGINAL_SHA='edfb96620f91b6e2e583168bb1e157e8b8a9b5dab884917985af3d68d8a82e93'
CONFIGS=[(5,1e-5,1e-4),(20,1e-7,1e-6),(50,1e-9,1e-8)]


def diagnostic_function(settings,record):
    if tuple(settings) not in CONFIGS:raise ValueError('ONLY_PREDECLARED_CONVERGENCE_DIAGNOSTICS')
    count,position,rotation=settings;tree=ast.parse(inspect.getsource(original.certify_chunk));changes=dict(loop=0,ep=0,er=0,trace=0)
    class Transform(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node.name='certify_diagnostic';return self.generic_visit(node)
        def visit_For(self,node):
            if isinstance(node.target,ast.Name) and node.target.id=='iteration':
                assert ast.dump(node.iter)==ast.dump(ast.parse('range(5)',mode='eval').body)
                node.iter.args[0]=ast.Constant(count);changes['loop']+=1
            return self.generic_visit(node)
        def visit_Compare(self,node):
            for name,before,after in [('ep',1e-5,position),('er',1e-4,rotation)]:
                if ast.dump(node.left)==ast.dump(ast.parse('np.linalg.norm('+name+')',mode='eval').body):
                    assert isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==before
                    node.comparators[0]=ast.Constant(after);changes[name]+=1
            return self.generic_visit(node)
        def visit_Expr(self,node):
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and isinstance(value.func.value,ast.Name) and value.func.value.id=='qs' and value.func.attr=='append':
                changes['trace']+=1
                return [ast.Expr(ast.Call(ast.Name('_diagnostic_record',ast.Load()),[ast.Name(k,ast.Load()) for k in ('data','qa','target','tick','site')],[])),node]
            return self.generic_visit(node)
        def visit_Dict(self,node):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value in ('maximum_IK_updates_per_point','position_tolerance_m','rotation_tolerance_rad'):
                    node.values[i]=ast.Constant({'maximum_IK_updates_per_point':count,'position_tolerance_m':position,'rotation_tolerance_rad':rotation}[key.value])
            return self.generic_visit(node)
    tree=Transform().visit(tree);assert changes==dict(loop=1,ep=2,er=2,trace=1),changes
    namespace=dict(original.certify_chunk.__globals__,_diagnostic_record=record)
    exec(compile(ast.fix_missing_locations(tree),'<isolated_ik_convergence_diagnostic>','exec'),namespace)
    return namespace['certify_diagnostic'],changes


def evaluate(model,sample,installed,history,stage,settings,trace_path):
    import mujoco
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    q=[];residual=[]
    def record(data,qa,target,tick,site):
        q.append(data.qpos[qa].copy());residual.append([float(np.linalg.norm(target[:3,3]-data.site_xpos[site])),
            float(geodesic_angle(target[:3,:3],data.site_xmat[site].reshape(3,3)))])
    function,changes=diagnostic_function(settings,record)
    motion=(original.PlanningObjectMotion(sample,mode='ACTUAL_GRASP_ATTACHMENT')
        if stage in ('close','lift','hold','release') and sample['native_grasp'] else None)
    before=[sample[k].copy() for k in ('qpos','qvel','ctrl')];start=time.time();code='DIAGNOSTIC_PASS_ONLY';certificate=None
    try:
        with forbid_integration(mujoco) as calls:certificate=function(model,sample,matrix10(installed),stage=stage,history=history,planning_object_motion=motion)
        assert not calls
    except ValueError as error:code=str(error)
    assert all(np.array_equal(v,sample[k]) for v,k in zip(before,('qpos','qvel','ctrl')))
    q=np.asarray(q).reshape((-1,6));residual=np.asarray(residual).reshape((-1,2));prefix=len(history);full=prefix+40
    np.savez(trace_path,joint_q_2ms=q,position_residual_m=residual[:,0],rotation_residual_rad=residual[:,1])
    rates=None
    if len(q)>=full*25:
        command=q[np.arange(full)*25];rates=joint_rates(command,prefix)
    return dict(settings=dict(max_updates=settings[0],position_m=settings[1],rotation_rad=settings[2]),
        result=code,certificate=certificate,full_path_solved=len(q)==full*25+1,solved_points=len(q),
        maximum_position_residual_m=float(residual[:,0].max()) if len(residual) else None,
        maximum_rotation_residual_rad=float(residual[:,1].max()) if len(residual) else None,
        rates=rates,trace_file=trace_path.name,trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        AST_changes=changes,wall_s=time.time()-start,formal_native_admission=False)


def run_original16(payload,output):
    import mujoco
    assert mujoco.__version__=='3.2.7' and mujoco.get_mjcb_control() is None
    assert hashlib.sha256(Path(inspect.getsourcefile(original.certify_chunk)).read_bytes()).hexdigest()==ORIGINAL_SHA
    output=Path(output);output.mkdir(exist_ok=False);start=time.time();cpu=time.process_time();models={};rows=[]
    for i,r in enumerate(payload['rows']):
        root=Path(payload['roots'][r['native_run']]);native=root/'native-01';spec=json.loads((root/'EXECUTION_SPEC.json').read_bytes())
        result=json.loads((native/'RESULT.json').read_bytes());scene=Path(spec['model_path'])
        assert hashlib.sha256(scene.read_bytes()).hexdigest()==result['scene_sha256']
        if str(scene) not in models:models[str(scene)]=mujoco.MjModel.from_binary_path(str(scene))
        model=models[str(scene)];capture=json.loads((native/'TASK_LEARNING_CAPTURES.private.json').read_bytes())['records'][r['query']]
        blob=(Path(capture['path']).parent/'STATE.private.bin').read_bytes();assert hashlib.sha256(blob).hexdigest()==r['state_sha256']
        with forbid_integration(mujoco):sample=ActualState(model).update(parse_state(blob))
        past=list(json.loads((native/'STABLE_TASK_CYCLES/-1/RESULT.private.json').read_bytes())['windows'])
        for v in json.loads((native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json').read_bytes())['records']:
            past.append(dict(v,actual_installed_consumed_EEF10_BASE7=v['installed_consumed_EEF10']))
        past=[v for v in past if v['native_end_step']<=sample['native_step']][-2:]
        assert [v['native_end_step'] for v in past]==[sample['native_step']-25,sample['native_step']]
        history=[dict(end_native_step=v['native_end_step'],pose=matrix10(v['actual_installed_consumed_EEF10_BASE7'][:10])) for v in past]
        configs=[]
        for j,settings in enumerate(CONFIGS):
            value=evaluate(model,sample,np.asarray(r['installed']),history,r['stage'],settings,output/f'{i:02d}-precision{j}.npz')
            configs.append(value)
            if j==0:
                assert value['result']=='ORIGINAL_JOINT_ACCELERATION_REJECT'
                assert np.allclose(value['rates']['command_q'],r['command_q'],rtol=0,atol=1e-12),'BASELINE_CLONE_MUST_REPRODUCE_SAVED_PATH'
        row=dict(native_run=r['native_run'],query=r['query'],state_sha256=r['state_sha256'],settings=configs,
            original_result_unchanged=True,full_WBC_state_restored=False,same_state_historical_expert_plan_available=False)
        rows.append(row)
        with(output/'ROWS.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    out=dict(schema='SLA_ORIGINAL16_IK_CONVERGENCE_SENSITIVITY_V7_1',rows=rows,original_certifier_sha256=ORIGINAL_SHA,
        paths=16,clone_certifier_calls=48,new_model_generations=0,optimizer_updates=0,training_visits=0,
        physical_requests=0,physics_integrations=0,full_WBC_state_restored=False,
        formal_results_or_thresholds_changed=False,wall_s=time.time()-start,CPU_s=time.process_time()-cpu)
    (output/'RESULT.private.json').write_text(json.dumps(out,indent=2,allow_nan=False));return out
