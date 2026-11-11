import copy
import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_sequence_v1 import sequence,map_policy_frame
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,matrix_to_quat,quat_slerp,quat_to_matrix


def records(rotation=None):
    decisions=[];captures=[];rows=[]
    for index in range(15):
        actual=np.eye(4);actual[:3,3]=[.2,.01,.7+index*.001]
        if rotation is not None:actual[:3,:3]=rotation
        reference=actual.copy();reference[:3,3]+=[.003,-.002,.001]
        raw=np.tile(np.r_[actual[:3,3],rotation_matrix_to_rot6d_rows(actual[:3,:3]),1.],(40,1))
        u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u))
        raw[:,3:9]=rotation_matrix_to_rot6d_rows(quat_to_matrix(quat_slerp(
            matrix_to_quat(actual[:3,:3]),matrix_to_quat(actual[:3,:3]),w[:,None])))
        raw[:,2]+=w*.001;raw[:,9]=1.-w if index==8 else 1.
        p=task.FiftyMsProgram(raw,actual,reference,np.r_[0.,0.,.3,1.,0.,0.,0.])
        step=task.B+1000*index;pin=f'{index:064x}'
        capture=dict(native_step=step,actual_cube_truth_is_student_input=False,path=f'/record/{index}',sha256=pin)
        captures.append(capture)
        decisions.append(dict(submitted_policy_H40=raw.tolist(),original_actual_TCP=actual.tolist(),
            native_step=step,observation=copy.deepcopy(capture),no_future_achieved_state_labels=True,
            diagnostic_control=None,_sha256=pin,stage='CLOSE'if index==8 else'DESCEND'))
        for j in range(40):
            rows.append(dict(native_start_step=step+25*j,native_end_step=step+25*(j+1),
                solver_gripper_confirmed=True,Teacher_decision_sha256=pin,
                policy_EEF10=raw[j].tolist(),installed_consumed_EEF10=p.reference_path[j].tolist()))
    return decisions,dict(records=rows,complete=True,period_s=.05,unexecuted_suffix_included=False,
        future_achieved_labels=False),captures


def test_fifteen_current_observations_retain_nominal_policy_not_reference():
    d,c,obs=records();v=sequence(d,c,obs)
    assert v['policy_absolute10'].shape==(600,10)
    assert v['native_command_anchor'].tolist()==list(range(0,600,40))
    assert np.array_equal(v['policy_absolute10'][320:360],d[8]['submitted_policy_H40'])
    assert not np.array_equal(v['policy_absolute10'],v['installed_audit_only'])
    assert v['policy_absolute10'][359,9]==0
    assert v['installed_audit_only'][359,9]>0
    assert v['expert_training_eligible']is False
    assert v['termination_hold_used_as_new_Teacher_decision']is False


@pytest.mark.parametrize('change', ['missing_receipt','extra_hold','unexecuted','wrong_clock',
    'wrong_decision','future_capture','future_pose','unconfirmed','privileged','double_grip_affine','reference_as_policy'])
def test_corrupt_or_unexecuted_commands_cannot_be_labels(change):
    d,c,obs=records()
    if change=='missing_receipt':c['records'].pop()
    elif change=='extra_hold':c['records']+=copy.deepcopy(c['records'][-20:])
    elif change=='unexecuted':c['unexecuted_suffix_included']=True
    elif change=='wrong_clock':c['records'][41]['native_end_step']+=1
    elif change=='wrong_decision':c['records'][41]['Teacher_decision_sha256']='f'*64
    elif change=='future_capture':obs[1]['native_step']+=25
    elif change=='future_pose':d[1]['original_actual_TCP'][2][3]+=.001
    elif change=='unconfirmed':c['records'][41]['solver_gripper_confirmed']=False
    elif change=='privileged':obs[1]['actual_cube_truth_is_student_input']=True
    elif change=='double_grip_affine':c['records'][359]['installed_consumed_EEF10'][9]=0.
    elif change=='reference_as_policy':d[1]['submitted_policy_H40'][5]=c['records'][45]['installed_consumed_EEF10']
    with pytest.raises(ValueError):sequence(d,c,obs)


def test_rigid_policy_frame_conversion_does_not_reapply_gripper_mapping():
    checked=sequence(*records());frame=np.eye(4);frame[:3,3]=[.1,-.2,.3]
    mapped=map_policy_frame(checked,frame)
    assert np.allclose(mapped['policy_absolute10'][:,:3],checked['policy_absolute10'][:,:3]+frame[:3,3])
    assert np.array_equal(mapped['policy_absolute10'][:,9],checked['policy_absolute10'][:,9])
    assert np.array_equal(mapped['installed_audit_only'][:,9],checked['installed_audit_only'][:,9])


def test_recorded_intervention_is_audited_but_never_expert_supervision():
    d,c,obs=records();d[8]['label_role']='INTERVENTION_NOT_EXPERT'
    v=sequence(d,c,obs)
    assert v['confirmed_intervals']==600
    assert v['actual_high_level_decisions']==15
    assert v['supervised_policy_decisions']==14 and v['excluded_intervention_decisions']==1
    assert not v['valid_window'][8] and v['valid_window'].sum()==14
    assert v['expert_training_eligible']is False


def test_unknown_label_producer_is_rejected():
    d,c,obs=records();d[8]['label_role']='UNEXECUTED_MODEL_PREDICTION'
    with pytest.raises(ValueError):sequence(d,c,obs)


def test_all_rotation_knots_must_match_frozen_teacher_conversion_exactly():
    rotation=quat_to_matrix(np.array([.1,.9,.3,.01]))
    d,c,obs=records(rotation)
    assert sequence(d,c,obs)['rotation_audit']=='EXACT_FROZEN_TEACHER_QUATERNION_MATH_ALL40_V2'
    # A later knot still has to be the actual Teacher output, even when its
    # first pose and its consumption receipt would otherwise be accepted.
    d[4]['submitted_policy_H40'][21][3]+=1e-14
    c['records'][4*40+21]['policy_EEF10']=d[4]['submitted_policy_H40'][21]
    with pytest.raises(ValueError,match='NOMINAL_POLICY_BOUNDARY'):
        sequence(d,c,obs)
