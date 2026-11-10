import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_endpoint_native_v3 as native
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_endpoint_policy_v3 import compile_proposal,consumed_grip,METHODS
from go2arx5_vla.world_model.storage import digest


def example(tmp_path,goal=.01):
    state=np.array([0.,0.,0.,1.,0.,0.,0.,1.,0.,.954545454545])
    raw=np.repeat(state[None],40,axis=0);raw[:,9]=1.;raw[-1,0]=goal
    action=tmp_path/'ACTION.npz';np.savez(action,physical_H40=raw,normalized=np.zeros((1,40,132)))
    return state,raw,action


def test_native_basis_preserves_raw_goal_and_uses_nominal_consumed_grip(tmp_path):
    state,raw,action=example(tmp_path)
    program,proof=compile_proposal(action,digest(action),consumed_nominal_grip=1.,source_from_target=np.eye(4),
        query_state10=state,actual_TCP=np.eye(4),consumed_reference_TCP=np.eye(4),base_hold=[0,0,0,1,0,0,0])
    assert np.array_equal(program.decoded_H40[-1],raw[-1])
    assert program.decoded_H40[0,9]==1. and program.reference_path[0,9]==pytest.approx(42/44)
    assert proof['GR00T_raw_H40_consumed']is False and proof['explicit_decoder_applied']is True
    assert proof['command_clips']==0 and proof['teacher_fallback']is False
    with np.load(action)as packet:assert np.array_equal(packet['physical_H40'],raw)
    from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
    _,_,far=example(tmp_path,goal=1.)
    with pytest.raises(StudentProposalRejected,match='ORIGINAL_FULL_H40_GATE'):
        compile_proposal(far,digest(far),consumed_nominal_grip=1.,source_from_target=np.eye(4),query_state10=state,
            actual_TCP=np.eye(4),consumed_reference_TCP=np.eye(4),base_hold=[0,0,0,1,0,0,0])


def test_actual_consumer_receipt_required_not_measured_finger_fraction(tmp_path):
    state,raw,action=example(tmp_path)
    program,_=compile_proposal(action,digest(action),consumed_nominal_grip=1.,source_from_target=np.eye(4),
        query_state10=state,actual_TCP=np.eye(4),consumed_reference_TCP=np.eye(4),base_hold=[0,0,0,1,0,0,0])
    from scripts.pi0292.eval_contract import matrix10
    row=dict(native_end_step=6703,actual_installed_consumed_EEF10_BASE7=np.r_[program.reference_path[-1],[0,0,0,1,0,0,0]],
        solver_policies_with_exact_installed_target=[1],gripper={'command_consumed':True})
    cycle=dict(windows=[row]*40,all_40_knots_consumed=True)
    value,proof=consumed_grip(cycle,program,{'native_step':6703},matrix10(program.reference_path[-1]))
    assert value==1. and proof['measured_state_used_as_nominal']is False
    for change in ['stale','unconsumed','different_reference']:
        bad=copy.deepcopy(cycle);reference=matrix10(program.reference_path[-1])
        if change=='stale':bad['windows'][-1]['native_end_step']-=25
        if change=='unconsumed':bad['windows'][-1]['gripper']['command_consumed']=False
        if change=='different_reference':reference[0,3]+=.001
        with pytest.raises(ValueError):consumed_grip(bad,program,{'native_step':6703},reference)


def panel():
    from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id
    rows=[dict(native_run=f'endpoint-{i}',layout=l,method=m,parent=scene_id(l),split='TRAIN',purpose=native.PURPOSE,
        restart=1,command=task.COMMAND,query_seeds=list(range(s,s+15)),model_sha256='a'*64)for i,(l,m,s)in enumerate(native.ORDER)]
    return dict(schema=native.SCHEMA,task_learning_role=native.PROTOCOL,protocol=task.protocol(native.PROTOCOL),
        B1_global_step=2000,preregistration={'sha256':native.CONTRACT_SHA},source_sha='b'*40,
        policies={m:{'arm':m}for m in METHODS},assignments=rows)


def test_two_condition_identity_never_weakens_original_conic_or_dev_population():
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import selection
    from go2arx5_vla.world_model.task_learning_conic_native_v3 import validate_panel as conic
    from go2arx5_vla.world_model.task_learning_student_v1 import validate as dev
    p=panel();inp=dict(schema=native.INPUT_SCHEMA,assignment_slice=[0,2],population_predecessor=None,source_sha=p['source_sha'])
    assert selection(inp,p)==p['assignments']
    assert native.summarize(p,[])['requested']==2
    for validate in(conic,dev):
        with pytest.raises(ValueError):validate(p)
    bad=copy.deepcopy(p);bad['assignments'][0]['query_seeds'][0]+=1
    with pytest.raises(ValueError):native.validate_panel(bad)
    from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    assert teacher_fingerprint()==FROZEN_TEACHER


def test_native_representation_imports_and_compiles_without_neural_or_solver_dependencies(tmp_path):
    import os,subprocess,sys
    _,_,action=example(tmp_path)
    code="""
import sys,builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0]in('torch','gr00t','scipy','clarabel'):raise ImportError('FORBIDDEN_NATIVE_DEPENDENCY:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
import numpy as np
from go2arx5_vla.world_model.task_learning_endpoint_policy_v3 import compile_proposal
from go2arx5_vla.world_model.storage import digest
program,proof=compile_proposal(sys.argv[1],digest(sys.argv[1]),consumed_nominal_grip=1.,source_from_target=np.eye(4),
 query_state10=np.array([0.,0.,0.,1.,0.,0.,0.,1.,0.,42/44]),actual_TCP=np.eye(4),consumed_reference_TCP=np.eye(4),base_hold=[0,0,0,1,0,0,0])
assert proof['GR00T_raw_H40_consumed']is False and program.decoded_H40.shape==(40,10)
assert not({'torch','gr00t','scipy','clarabel'}&set(sys.modules))
"""
    q=subprocess.run([sys.executable,'-B','-c',code,str(action)],capture_output=True,text=True,
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path)),timeout=30)
    assert q.returncode==0,q.stderr
