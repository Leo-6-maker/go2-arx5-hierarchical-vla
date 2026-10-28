import copy

import numpy as np
import pytest

from scripts.pi0291_arx5_data.teacher import Teacher
from go2arx5_vla.flowdagger.abi import SOURCE_ABI
from go2arx5_vla.flowdagger.oracle import OracleRequest,SourceOracle,legacy_saved_teacher_provider
from go2arx5_vla.task_data.codec import pose10
from .test_codec import action
from .test_contracts import proof


def request():
    return OracleRequest('obs-1',0,'T1',action()[0],'SYNTHETIC_CPU_FIXTURE',SOURCE_ABI)


def provider(req,truth,history):
    assert 'object_pose' in truth
    return dict(valid=True,expert_H40=action(),phase='approach',reason='SYNTHETIC_MOCK',observation_id=req.observation_id,tick=req.tick)


def test_mock_full_plan_same_state_and_explicit_nonphysical_identity():
    oracle=SourceOracle(provider,version='SYNTHETIC_ONLY',safety_check=lambda c,r:proof(c))
    reply=oracle.query(request(),dict(object_pose=[0,0,0]),[])
    assert reply['valid'] and len(reply['expert_H40'])==40 and reply['validity_ticks']==[0,975]
    assert not reply['scientific_recovery_qualified'] and not reply['physically_executed']
    assert reply['evidence_kind']=='SYNTHETIC_CPU_FIXTURE'


def test_no_certificate_no_valid_training_label():
    reply=SourceOracle(provider,version='MOCK_NO_S1').query(request(),dict(object_pose=[]),[])
    assert not reply['valid'] and reply['expert_H40'] is None and reply['proposal_H40_audit_only'] is not None


@pytest.mark.parametrize('reason',['UNRECOVERABLE','JOINT_OR_STATE_RESTRICTED','NO_COMPLETE_H40'])
def test_invalid_oracle_is_first_class_not_fake_label(reason):
    oracle=SourceOracle(lambda *args:dict(valid=False,expert_H40=None,phase=None,reason=reason),version='MOCK')
    reply=oracle.query(request(),{},[])
    assert not reply['valid'] and reply['reason']==reason and reply['active_dim_mask'] is None


def test_partial_horizon_and_stale_proposal_rejected():
    def short(req,truth,history):
        value=provider(req,truth,history);value['expert_H40']=action()[:8];return value
    assert not SourceOracle(short,version='MOCK').query(request(),dict(object_pose=[]),[])['valid']
    def stale(req,truth,history):
        value=provider(req,truth,history);value['tick']=200;return value
    with pytest.raises(ValueError):SourceOracle(stale,version='MOCK').query(request(),dict(object_pose=[]),[])


def test_original_t1_teacher_plan_reused_on_synthetic_saved_state_without_engine():
    initial=np.eye(4);initial[:3,3]=[.4,0,.42]
    teacher=Teacher(dict(cube_yaw=0,source_xy=[.4,0],goal_xy=[.6,0]),initial)
    tcp=teacher.target({})[0]
    state=dict(tcp=tcp.copy(),gripper_fraction=1.,time_s=0.)
    before=copy.deepcopy(teacher.__dict__)
    req=OracleRequest('saved-synthetic',0,'T1',pose10(tcp,1.),'SYNTHETIC_CPU_FIXTURE',SOURCE_ABI)
    oracle=SourceOracle(legacy_saved_teacher_provider(teacher,state),version='ORIGINAL_T1_PLAN_NO_RECOVERY_CERTIFICATE')
    reply=oracle.query(req,dict(object_pose=initial.tolist()),[])
    assert len(reply['proposal_H40_audit_only'])==40 and not reply['valid']
    assert teacher.stage==before['stage'] and teacher.events==before['events']
    assert not reply['scientific_recovery_qualified']


def test_synthetic_perturbation_does_not_become_live_student_state():
    r=request()
    with pytest.raises(ValueError):OracleRequest(r.observation_id,r.tick,r.task,r.measured_state10,'LIVE_STUDENT',SOURCE_ABI).validate()
