"""Corrupted receipt checks only; fixtures are not native certification."""
import copy
import numpy as np
import pytest

from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.sla.audit_pd4_join_v72 import (
    paths,preview,verify_adaptation,verify_install,verify_consumption)
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.sla_reference_join_v72 import ReferenceJoinProgram


def fixture():
    actual=np.eye(4);reference=np.eye(4);reference[0,3]=.025
    raw=np.repeat(pose10(actual,.6)[None],40,axis=0)
    raw[:,0]=.0005*np.sin(np.linspace(0,np.pi,40))**2
    past=reference.copy();past[0,3]-=.00002
    history=np.stack([past,reference]);target,_,baseline,candidate,_=paths(raw,np.eye(4),actual,history)
    base=np.array([0,0,.3,1,0,0,0])
    program=ReferenceJoinProgram(task.FiftyMsProgram(target,actual,reference,base),
        [dict(end_native_step=5678,pose=past),dict(end_native_step=5703,pose=reference)],5703)
    return baseline,candidate,history,program


def certificate():
    return dict(status='PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF',
        certifier_revision='FIVE_UPDATE_FINAL_FK_CHECK_V1',maximum_IK_updates_per_point=5,
        executed_reference_history_points=2,points=1051,position_tolerance_m=1e-5,
        rotation_tolerance_rad=1e-4,physics_integrations=0,actual_state_modified=False,
        Cartesian_reference_values_changed=False,max_joint_speed_rad_s=.01,
        max_joint_acceleration_rad_s2=.1)


def install(candidate,elapsed=0,rejected=False):
    return dict(native_step=5703+25*elapsed,action_EEF10=preview(candidate,elapsed).tolist(),
        phase='reference-certificate-rejected'if rejected else'proposed',
        native_reference_certificate=certificate(),wire_sha256=f'wire-{elapsed}',
        submitted_or_executed=False,reason='ORIGINAL_JOINT_ACCELERATION_REJECT')


def windows(candidate,count):
    return [dict(native_start_step=5703+25*k,native_end_step=5703+25*(k+1),actual_ticks=25,
        actual_installed_consumed_EEF10_BASE7=np.r_[candidate[k],[0,0,.3,1,0,0,0]].tolist(),
        reference_sha256=f'wire-{0 if k<7 else 7}',
        full_final_installed_H40=preview(candidate,0 if k<7 else 7).tolist(),
        solver_policies_with_exact_installed_target=['actual-policy'],gripper=dict(command_consumed=True))
        for k in range(count)]


def test_saved_adaptation_is_reproduced_and_changed_suffix_is_detected():
    baseline,candidate,history,program=fixture();value=program.evidence()
    verify_adaptation(value,baseline,candidate,history,5703)
    # A suffix edit must fail even though none of that suffix was consumed.
    value['candidate_reference_H40'][35][0]+=.001
    with pytest.raises(ValueError,match='FULL40'):verify_adaptation(value,baseline,candidate,history,5703)


def test_later_renewal_is_full40_not_a_new_connection_or_truncated_certificate():
    _,candidate,_,_=fixture();event=install(candidate,7)
    assert verify_install(event,candidate,5703)['elapsed']==7
    changed=copy.deepcopy(event);changed['action_EEF10'][25][1]+=.001
    with pytest.raises(ValueError,match='RENEWAL'):verify_install(changed,candidate,5703)
    event['native_reference_certificate']['points']=251
    with pytest.raises(ValueError,match='COMPLETE_H40'):verify_install(event,candidate,5703)


def test_seven_confirmed_points_remain_real_when_renewal_is_rejected():
    _,candidate,_,_=fixture();first=install(candidate);failed=install(candidate,7,True)
    assert verify_install(failed,candidate,5703)['status']=='REJECTED'
    assert verify_consumption(windows(candidate,7),candidate,[first,failed],5703)==7
    # The eighth point cannot be counted against a rejected second install.
    with pytest.raises(ValueError,match='ACCEPTED_CERTIFIED'):
        verify_consumption(windows(candidate,8),candidate,[first,failed],5703)


@pytest.mark.parametrize('fault',['clock','gripper','wire','suffix'])
def test_consumption_requires_real_time_gripper_reference_and_entire_preview(fault):
    _,candidate,_,_=fixture();events=[install(candidate),install(candidate,7)]
    rows=windows(candidate,10)
    assert verify_consumption(rows,candidate,events,5703)==10
    if fault=='clock':rows[3]['native_end_step']+=1
    elif fault=='gripper':rows[3]['gripper']['command_consumed']=False
    elif fault=='wire':rows[3]['reference_sha256']='unrelated-policy'
    else:rows[3]['full_final_installed_H40'][39][2]+=.001
    with pytest.raises(ValueError):verify_consumption(rows,candidate,events,5703)
