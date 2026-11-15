import copy
import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import pd4_student_v71 as raw_profile, pd4_join_v72 as join_profile
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal as raw_compile
from go2arx5_vla.world_model.sla_reference_join_v72 import (
    compile_proposal, configuration, ReferenceJoinProgram)
from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
from go2arx5_vla.world_model.storage import digest
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.sla.reference_join10_v71 import connect


def inputs(tmp_path):
    actual=np.eye(4);reference=np.eye(4);reference[0,3]=.025
    raw=np.repeat(pose10(actual,.6)[None],40,axis=0)
    raw[:,0]=.0005*np.sin(np.linspace(0,np.pi,40))**2
    action=tmp_path/'ACTION.npz'
    np.savez(action,physical_H40=raw,normalized=np.zeros((1,40,32),np.float32))
    kwargs=dict(source_from_target=np.eye(4),query_state10=pose10(actual,.6),
        actual_TCP=actual,consumed_reference_TCP=reference,base_hold=np.array([0,0,.3,1,0,0,0]))
    past=reference.copy();past[0,3]-=.00002
    history=[dict(end_native_step=5678,pose=past),dict(end_native_step=5703,pose=reference)]
    return action,kwargs,history


def test_native_candidate_equals_fixed_offline_method_and_keeps_raw_grip_suffix(tmp_path):
    action,kwargs,history=inputs(tmp_path);pin=digest(action)
    baseline,_=raw_compile(action,pin,**kwargs)
    program,proposal=compile_proposal(action,pin,reference_history=history,query_native_step=5703,**kwargs)
    assert type(program)is ReferenceJoinProgram
    assert np.array_equal(program.reference_path,connect(baseline.reference_path,[r['pose']for r in history]))
    assert np.array_equal(program.decoded_H40,baseline.decoded_H40)
    assert np.array_equal(program.reference_path[9:],baseline.reference_path[9:])
    assert np.array_equal(program.reference_path[:,9],baseline.reference_path[:,9])
    assert proposal['installed_reference_changed'] and not proposal['original_H40_interior_preserved']
    assert proposal['SLA_codec_decode_calls']==1 and proposal['additional_gripper_decode_calls']==0
    assert not proposal['teacher_fallback'] and not proposal['native_execution_authorized']
    evidence=program.evidence()
    assert evidence['history_native_steps']==[5678,5703]
    assert evidence['original_raw_H40_unchanged'] and evidence['suffix_k9_through39_byte_exact']
    assert not program.reference_path.flags.writeable
    # Renewals shift the same candidate. They never recompute a correction.
    assert np.array_equal(program.preview(7)[:33],program.reference_path[7:])
    assert np.array_equal(program.preview(7)[33:],np.repeat(program.reference_path[-1:],7,axis=0))
    assert digest(action)==pin


def test_raw_invalid_suffix_remains_rejected_before_join(tmp_path):
    action,kwargs,history=inputs(tmp_path)
    with np.load(action)as z:raw=z['physical_H40'].copy()
    raw[30,0]=.1
    np.savez(action,physical_H40=raw,normalized=np.zeros((1,40,32),np.float32))
    with pytest.raises(StudentProposalRejected,match='SLA_DENSE_ORIGINAL_FULL_H40_GATE'):
        compile_proposal(action,digest(action),reference_history=history,query_native_step=5703,**kwargs)


def test_candidate_original_full40_gate_rejects_without_raw_fallback(tmp_path):
    action,kwargs,history=inputs(tmp_path)
    raw_compile(action,digest(action),**kwargs)
    history[0]['pose'][0,3]-=.02
    with pytest.raises(StudentProposalRejected,match='JOIN10_FULL_H40_GATE')as error:
        compile_proposal(action,digest(action),reference_history=history,query_native_step=5703,**kwargs)
    evidence=error.value.reference_connection_evidence
    assert evidence['installed_reference_changed'] and len(evidence['candidate_reference_H40'])==40
    assert not evidence['native_execution_authorized']


def test_stale_history_is_engineering_failure_not_policy_rejection(tmp_path):
    action,kwargs,history=inputs(tmp_path);history[0]['end_native_step']-=25
    with pytest.raises(ValueError,match='EXACT_COMPLETED_REFERENCE_HISTORY')as error:
        compile_proposal(action,digest(action),reference_history=history,query_native_step=5703,**kwargs)
    assert not isinstance(error.value,StudentProposalRejected)


def test_join_and_raw_have_distinct_admission_and_program_types():
    spec=dict(task_learning_role=join_profile.PROTOCOL,task_learning_grasp=join_profile.protocol())
    assert task.native_body_profile(task.SLA_JOIN_ROLE)is join_profile
    assert raw_profile.profile(spec)is join_profile
    assert task.student_role(task.SLA_JOIN_ROLE) and task.dense_sla_role(task.SLA_JOIN_ROLE)
    assert task.program_type(spec,0)is ReferenceJoinProgram
    assert task.program_type(spec,-1)is task.FiftyMsProgram
    assert task.program_type(spec,60)is task.FiftyMsProgram
    spec['task_learning_role']=raw_profile.PROTOCOL
    assert task.program_type(spec,0)is task.FiftyMsProgram
    spec['reference_join10_v72']=configuration()
    with pytest.raises(PermissionError,match='CANNOT_MODIFY_RAW'):raw_profile.profile(spec)


def test_new_panel_cannot_be_scored_as_old_raw():
    from scripts.wm02.sla_pd4_student_queue_v71 import summarize
    rows=[dict(native_run=f'{layout}-{arm}',method=arm,layout=layout,restart=17,
               query_seeds=list(range(60)))
          for layout in join_profile.LAYOUTS for arm in('PD4_EXECUTED_REF','PD4_FINAL_PLAN')]
    panel=dict(schema=join_profile.VERSION+'_PANEL',B1_global_step=500,protocol=join_profile.protocol(),
               assignments=rows,old_RP8_passed=False,automatic_retry=False,
               policies={'PD4_EXECUTED_REF':{},'PD4_FINAL_PLAN':{}})
    result=summarize(panel,[])
    assert result['schema']==join_profile.VERSION+'_SUMMARY' and result['terminal']==0
    stale=copy.deepcopy(panel);stale['schema']=raw_profile.VERSION+'_PANEL'
    with pytest.raises(PermissionError):raw_profile.panel_contract(stale)
