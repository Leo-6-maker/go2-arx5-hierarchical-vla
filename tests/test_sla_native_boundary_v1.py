import json
import numpy as np
import pytest
from scripts.pi0291_arx5_data.native_dataset import pose10
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.sla_native_policy_v1 import compile_proposal
from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
from go2arx5_vla.world_model.task_learning_endpoint_basis_v3 import decode
from go2arx5_vla.world_model.sla_population_v1 import declaration, DEV_OFFSETS
from go2arx5_vla.world_model import task_learning_grasp_v1 as task


def inputs(tmp_path, width=32, grip=0.2):
    actual=np.eye(4);actual[:3,3]=[.2,.01,.7]
    reference=actual.copy();reference[:3,3]+=[.003,-.002,.001]
    frame=np.eye(4);frame[:3,3]=[.2,-.3,.1]
    state=pose10(frame@actual, .7)
    raw=np.tile(state,(40,1));raw[-1,:3]+=[.003,.002,-.001];raw[:,9]=grip
    # Raw middle knots are retained even when they differ from the declared basis.
    raw[15:20,0]+=.1
    path=tmp_path/'action.npz';np.savez(path,physical_H40=raw,normalized=np.zeros((1,40,width)))
    args=dict(source_from_target=frame,query_state10=state,actual_TCP=actual,
              consumed_reference_TCP=reference,base_hold=np.r_[0.,0.,.3,1.,0.,0.,0.],consumed_nominal_grip=.8)
    return path,args,raw


def test_sla32_retains_raw_and_uses_original_basis_transport(tmp_path):
    path,args,raw=inputs(tmp_path)
    before=path.read_bytes();program,audit=compile_proposal(path,digest(path),**args)
    expected,_=decode(raw,args['query_state10'],consumed_nominal_grip=.8)
    assert type(program) is task.FiftyMsProgram
    np.testing.assert_allclose(program.decoded_H40[:,:3],expected[:,:3]-args['source_from_target'][:3,3])
    np.testing.assert_array_equal(np.asarray(audit['raw_policy_H40']),raw)
    assert path.read_bytes()==before and not audit['teacher_fallback'] and audit['command_clips']==0
    assert len(program.evidence()['clock']['knot_native_tick_offsets'])==40


def test_legacy132_does_not_pass_sla_boundary(tmp_path):
    path,args,_=inputs(tmp_path,width=132)
    with pytest.raises(ValueError,match='NATIVE32_CONTAINER'):
        compile_proposal(path,digest(path),**args)


def test_sla_out_of_range_grip_is_policy_rejection_without_clip(tmp_path):
    path,args,raw=inputs(tmp_path,grip=1.1)
    before=path.read_bytes()
    with pytest.raises(StudentProposalRejected,match='no clipping'):
        compile_proposal(path,digest(path),**args)
    assert path.read_bytes()==before


def test_fixed_dev_population_is_separate_and_not_an_old_role():
    from go2arx5_vla.world_model.task_learning_scene_v1 import geometry,TRAIN_OFFSETS,QUALIFICATION_OFFSETS,RECOVERY_OFFSETS
    from go2arx5_vla.world_model.task_learning_dev_v1 import DEV_OFFSETS as old_dev
    assert not set(DEV_OFFSETS.values()) & set((TRAIN_OFFSETS|QUALIFICATION_OFFSETS|RECOVERY_OFFSETS|old_dev).values())
    declared=declaration()
    assert len(declared['population'])==16 and len({r['initial_geometry_sha256']for r in declared['population']})==8
    assert all(geometry(k)['role']=='SLA_INDEPENDENT_DEV_LAYOUT_NOT_TRAIN'for k in DEV_OFFSETS)
    assert task.student_role(task.SLA_ROLE) and task.native_body_profile(task.SLA_ROLE).PROTOCOL==task.SLA_ROLE
    assert task.protocol(task.SLA_ROLE)['normalized_container_dimensions']==32
