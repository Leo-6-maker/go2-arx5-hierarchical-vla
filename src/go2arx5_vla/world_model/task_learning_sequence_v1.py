"""Audit continuous nominal Teacher labels against consumed 50 ms commands.

This reader creates no BC admission. Construction and qualification episodes
retain their roles even when their command correspondence is complete.
"""
import numpy as np

from . import task_learning_grasp_v1 as task
from .fusion_trajectory_adapter import WORKING_CLOSED_M,WORKING_OPEN_M,MECHANICAL_TRAVEL_M
from ..trajectory.so3 import rot6d_rows_to_matrix,rotation_matrix_to_rot6d_rows


def sequence(decisions,confirmed,captures):
    """Return the policy intent, not the transported controller reference."""
    task.need(len(decisions)==len(captures)==task.QUERIES,'FIFTEEN_REAL_DECISIONS_AND_CAPTURES')
    task.need(confirmed.get('complete')is True and confirmed.get('period_s')==.05
        and confirmed.get('unexecuted_suffix_included')is False
        and confirmed.get('future_achieved_labels')is False,
        'COMPLETE_CONFIRMED_COMMAND_SEQUENCE_NOT_PREDICTED_SUFFIX')
    rows=confirmed.get('records')or[]
    task.need(len(rows)==task.QUERIES*40,'EXACTLY_SIX_HUNDRED_TEACHER_COMMANDS_EXCLUDES_TERMINATION_HOLD')
    policy=[];installed=[];anchors=[];stages=[]
    for index,(decision,capture)in enumerate(zip(decisions,captures)):
        step=task.B+1000*index
        raw=np.asarray(decision['submitted_policy_H40'],float)
        actual=np.asarray(decision['original_actual_TCP'],float)
        task.need(raw.shape==(40,10)and actual.shape==(4,4)
            and np.isfinite(raw).all()and np.isfinite(actual).all()
            and np.all((raw[:,9]>=0)&(raw[:,9]<=1)),'FINITE_ACTUAL_NOMINAL_H40')
        task.need(decision['native_step']==capture['native_step']==step
            and decision['observation']==capture and capture.get('actual_cube_truth_is_student_input')is False
            and decision.get('no_future_achieved_state_labels')is True
            and decision.get('diagnostic_control')is None,
            'REAL_CURRENT_OBSERVATION_DECISION_CLOCK_NO_PRIVILEGED_STUDENT_DATA')
        task.need(np.array_equal(raw[0,:3],actual[:3,3])
            and np.allclose(rot6d_rows_to_matrix(raw[0,3:9]),actual[:3,:3],rtol=0,atol=2e-15),
            'NOMINAL_POLICY_BOUNDARY_ANCHORED_TO_CURRENT_ACTUAL_TCP')
        chunk=rows[40*index:40*(index+1)]
        for offset,row in enumerate(chunk):
            task.need(row['native_start_step']==step+25*offset
                and row['native_end_step']==step+25*(offset+1)
                and row.get('solver_gripper_confirmed')is True
                and row['Teacher_decision_sha256']==decision['_sha256']
                and np.array_equal(np.asarray(row['policy_EEF10']),raw[offset]),
                'EVERY_POLICY_KNOT_HAS_ITS_OWN_REAL_CONSUMPTION_RECEIPT')
        consumed=np.asarray([row['installed_consumed_EEF10']for row in chunk],float)
        task.need(consumed.shape==(40,10)and np.isfinite(consumed).all(),'FINITE_INSTALLED_COMMANDS')
        expected=raw.copy()
        expected[:,:3]=consumed[0,:3]+raw[:,:3]-actual[:3,3]
        initial_reference=rot6d_rows_to_matrix(consumed[0,3:9])
        expected[:,3:9]=rotation_matrix_to_rot6d_rows(initial_reference@actual[:3,:3].T@rot6d_rows_to_matrix(raw[:,3:9]))
        expected[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*raw[:,9])/MECHANICAL_TRAVEL_M
        # Rotation round-trips and the recovered first translation anchor may
        # differ at FP64 reduction roundoff. Raw nominal commands above are exact.
        task.need(np.allclose(consumed,expected,rtol=0,atol=2e-15),
            'ORIGINAL_REFERENCE_TRANSPORT_AND_ONE_WORKING_GRIPPER_AFFINE')
        policy.extend(raw);installed.extend(consumed);anchors.append(index*40);stages.append(decision['stage'])
    return dict(policy_absolute10=np.asarray(policy),installed_audit_only=np.asarray(installed),
        native_command_anchor=np.asarray(anchors,dtype=np.int64),stages=np.asarray(stages),
        valid_window=np.ones(task.QUERIES,dtype=bool),policy_period_s=.05,
        supervised_policy_decisions=task.QUERIES,confirmed_intervals=len(rows),
        termination_hold_used_as_new_Teacher_decision=False,expert_training_eligible=False)


def map_policy_frame(checked,frame):
    """Apply the existing rigid target-to-policy map once to both audit views."""
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
    from .fusion_observation import rigid
    frame=rigid(frame)
    return dict(checked,policy_absolute10=pose10(frame@matrix_from10(checked['policy_absolute10']),
            checked['policy_absolute10'][:,9]),
        installed_audit_only=pose10(frame@matrix_from10(checked['installed_audit_only']),
            checked['installed_audit_only'][:,9]))
