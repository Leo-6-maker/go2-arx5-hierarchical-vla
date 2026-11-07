"""C01 course uses real poses; old domain and scores stay distinct."""
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_source_phase_c01_capability_v1 as c
from go2arx5_vla.world_model.fusion_graded_short_skill_v1 import screen_proposal, score_contract as old_score
from go2arx5_vla.world_model.fusion_task_phase_live_skill_contract_v9 import terminal_step, score_contract, task_document


def geometry(distance=.080):
    tcp=np.eye(4);tcp[:3,3]=(.3,0.,.6)
    cube=tcp.copy();cube[0,3]+=distance
    return tcp,cube,[0.,0.,.4,1.,0.,0.,0.]


@pytest.mark.parametrize('method,delta',[('HOLD',0.),('GEOMETRIC_TEACHER_5MM',.005)])
def test_full40_fixed_reference_and_old_domain_rejection_remain(method,delta):
    tcp,cube,base=geometry()
    program,receipt=c.build(tcp,tcp,base,cube,method)
    assert program.decoded_H40.shape==(40,10)
    np.testing.assert_allclose(program.decoded_H40[-1,:3]-tcp[:3,3],[delta,0,0],atol=1e-15)
    assert receipt['minimum_full_path_clearance_m']>=.07
    assert program.end_atom+8<=176
    assert receipt['native_execution_authorized'] is False
    with pytest.raises(ValueError,match='START_OUTSIDE_PRECONTACT'):
        screen_proposal(program,cube)
    assert old_score()['precontact_offset_from_cube_m']==.12
    assert terminal_step(c.CONDITION)==38703
    assert score_contract(task_condition=c.CONDITION)==c.score_contract()


@pytest.mark.parametrize('distance',[.069,.074,.091])
def test_course_or_whole_step_outside_declared_domain_rejected(distance):
    tcp,cube,base=geometry(distance)
    with pytest.raises(ValueError,match='COURSE_START'):
        c.build(tcp,tcp,base,cube,'GEOMETRIC_TEACHER_5MM')


def test_cpu_course_task_is_explicit_and_cannot_relabel_full_source_language():
    from go2arx5_vla.world_model.fusion_short_skill import LANGUAGE
    value=task_document('a'*64,LANGUAGE,task_condition=c.CONDITION)
    assert value['language_provenance']=='EXPLICIT_PRIVILEGED_C01_CAPABILITY_DIAGNOSTIC'
    with pytest.raises(ValueError,match='CAPABILITY_LANGUAGE'):
        task_document('a'*64,'Pick and place the cube',task_condition=c.CONDITION)
