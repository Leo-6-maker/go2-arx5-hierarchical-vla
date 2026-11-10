import json
import os,subprocess,sys
import numpy as np
import pytest
from go2arx5_vla.world_model.task_learning_conic_policy_v3 import action_packet,compile_proposal
from go2arx5_vla.world_model.task_learning_conic_qualification_v3 import identity
from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
from go2arx5_vla.world_model.storage import digest


def value():
    raw=np.zeros((40,10));raw[:,3]=raw[:,7]=raw[:,9]=1.
    raw[1:-1,0]=.0008*(-1.)**np.arange(38)
    return dict(physical_H40=raw,normalized=np.zeros((1,40,132)))


def compile(path,method='C500_PLUS_CONIC'):
    return compile_proposal(path,digest(path),method=method,source_from_target=np.eye(4),
        query_state10=np.r_[np.zeros(3),[1,0,0,0,1,0],1.],actual_TCP=np.eye(4),
        consumed_reference_TCP=np.eye(4),base_hold=np.array([0.,0.,0.,1.,0.,0.,0.]))


def test_raw_rejection_remains_and_composed_output_is_separately_labelled(tmp_path):
    v=value();raw=v['physical_H40'].copy();packet=action_packet(v,'C500_PLUS_CONIC')
    path=tmp_path/'candidate.npz';np.savez(path,**packet)
    with pytest.raises(StudentProposalRejected):compile(path,'C500_RAW')
    program,proof=compile(path)
    assert proof['GR00T_raw_H40_consumed']is False and proof['explicit_decoder_applied']is True
    assert proof['command_clips']==0 and proof['teacher_fallback']is False
    assert np.array_equal(v['physical_H40'],raw)and np.array_equal(packet['physical_H40'],raw)
    assert np.array_equal(program.decoded_H40,packet['explicit_conic_H40'])


def test_solver_failure_is_valid_rejection_with_no_raw_fallback(tmp_path):
    v=value();v['physical_H40'][-1,0]=1.
    packet=action_packet(v,'C500_PLUS_CONIC');assert 'explicit_conic_H40'not in packet
    path=tmp_path/'infeasible.npz';np.savez(path,**packet)
    with pytest.raises(StudentProposalRejected,match='SOLVER_REJECTED_NO_FALLBACK'):compile(path)


def test_independent_native_checks_catch_modified_packet_even_with_new_file_hash(tmp_path):
    packet=action_packet(value(),'C500_PLUS_CONIC');packet['explicit_conic_H40'][10,0]+=.1
    path=tmp_path/'changed.npz';np.savez(path,**packet)
    with pytest.raises(ValueError,match='OUTPUT_IDENTITY')as error:compile(path)
    assert not isinstance(error.value,StudentProposalRejected)


def test_endpoint_and_original_rate_gates_do_not_trust_solver_status(tmp_path):
    packet=action_packet(value(),'C500_PLUS_CONIC');packet['explicit_conic_H40'][10,0]+=.1
    audit=json.loads(str(packet['conic_fit_audit_json']));audit['decoded_identity']=identity(packet['explicit_conic_H40'])
    packet['conic_fit_audit_json']=np.asarray(json.dumps(audit));path=tmp_path/'false-proof.npz';np.savez(path,**packet)
    with pytest.raises(StudentProposalRejected,match='ORIGINAL_FULL_H40_GATE'):compile(path)


def test_native_compilation_has_no_solver_torch_or_scipy_dependency(tmp_path):
    path=tmp_path/'packet.npz';np.savez(path,**action_packet(value(),'C500_PLUS_CONIC'))
    code="""
import sys,builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0]in('torch','scipy','clarabel'):raise ImportError('FORBIDDEN_NATIVE_DEPENDENCY:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
import numpy as np
from go2arx5_vla.world_model.task_learning_conic_policy_v3 import compile_proposal
from go2arx5_vla.world_model.storage import digest
program,proof=compile_proposal(sys.argv[1],digest(sys.argv[1]),method='C500_PLUS_CONIC',source_from_target=np.eye(4),
 query_state10=np.r_[np.zeros(3),[1,0,0,0,1,0],1.],actual_TCP=np.eye(4),consumed_reference_TCP=np.eye(4),base_hold=np.array([0.,0.,0.,1.,0.,0.,0.]))
assert proof['GR00T_raw_H40_consumed']is False
assert not({'torch','scipy','clarabel'}&set(sys.modules))
"""
    q=subprocess.run([sys.executable,'-B','-c',code,str(path)],capture_output=True,text=True,
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path)),timeout=30)
    assert q.returncode==0,q.stderr
