"""Separate smooth predicted-goal policy; raw predictions remain immutable."""
import numpy as np
from . import task_learning_grasp_v1 as task
from .storage import digest
from .task_learning_endpoint_basis_v3 import decode,VERSION

PROTOCOL='CORE_LOCAL_C500_ENDPOINT_BASIS_DIAGNOSTIC_NOT_DEV_TEST_V3'
METHODS=('C500_PLUS_ENDPOINT_BASIS',)


def consumed_grip(cycle,program,sample,reference):
    """Read a completed original consumer receipt, never measured finger state."""
    from scripts.pi0292.eval_contract import matrix10
    rows=cycle.get('windows')or[]
    task.need(len(rows)==40 and cycle.get('all_40_knots_consumed')is True,
        'ENDPOINT_COMPLETE_ACTUALLY_CONSUMED_PREVIOUS_H40')
    last=rows[-1];actual=np.asarray(last['actual_installed_consumed_EEF10_BASE7'])
    task.need(last['native_end_step']==sample['native_step']
        and last['solver_policies_with_exact_installed_target']and last['gripper']['command_consumed']
        and actual.shape==(17,)and np.array_equal(actual[:10],program.reference_path[-1])
        and np.array_equal(matrix10(actual[:10]),reference),
        'ENDPOINT_CURRENT_SOLVER_AND_GRIP_CONSUMPTION_WITNESS')
    nominal=float(program.decoded_H40[-1,9])
    task.need(0<=nominal<=1,'ENDPOINT_CONSUMED_NOMINAL_GRIP_RANGE')
    return nominal,dict(native_end_step=last['native_end_step'],nominal_grip=nominal,
        mechanical_command=float(actual[9]),source='ORIGINAL_FULL_H40_SOLVER_AND_GRIPPER_CONSUMPTION',
        measured_state_used_as_nominal=False)


def compile_proposal(action,pin,*,consumed_nominal_grip,source_from_target,query_state10,
                     actual_TCP,consumed_reference_TCP,base_hold):
    from .task_learning_student_v1 import StudentProposalRejected
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_observation import rigid
    task.need(digest(action)==pin,'ENDPOINT_ORIGINAL_ACTION_BYTES')
    with np.load(action,allow_pickle=False)as packet:
        raw=packet['physical_H40'].copy();normalized=packet['normalized']
        task.need(raw.shape==(40,10)and normalized.shape==(1,40,132)
            and np.isfinite(raw).all()and np.isfinite(normalized).all(),'ENDPOINT_ORIGINAL_COMPLETE_PROCESSOR_DECODE')
    frame=np.asarray(source_from_target,float);rigid(frame)
    task.need(np.allclose(matrix10(query_state10),frame@np.asarray(actual_TCP),rtol=0,atol=1e-9),
        'ENDPOINT_SAME_CURRENT_POLICY_OBSERVATION_AND_NATIVE_STATE')
    try:
        decoded,audit=decode(raw,np.asarray(query_state10),consumed_nominal_grip=consumed_nominal_grip)
        target=pose10(np.linalg.inv(frame)@matrix10(decoded),decoded[:,9])
        program=task.FiftyMsProgram(target,actual_TCP,consumed_reference_TCP,base_hold)
    except ValueError as exc:
        raise StudentProposalRejected('TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: ENDPOINT_ORIGINAL_FULL_H40_GATE:'+str(exc))from exc
    return program,dict(raw_policy_H40=raw.tolist(),explicit_endpoint_policy_H40=decoded.tolist(),
        mapped_target_H40=target.tolist(),original_Processor_decode_calls=1,action_sha256=pin,
        policy_frame_mapping_count=1,command_clips=0,teacher_fallback=False,expert_training_eligible=False,
        external_inference_adapter=False,composed_policy_method=METHODS[0],explicit_decoder_applied=True,
        explicit_decoder_execution_version=VERSION,representation_audit=audit,GR00T_raw_H40_consumed=False,
        composed_H40_requires_original_native_reference_and_consumption_audits=True)
