"""Current-query equivalent of actually consumed EEF reference commands.

Issuing-query nominal commands use changing measured TCP anchors. Concatenating
them is not a single-query trajectory. Invert the unchanged reference transport
at one current anchor instead;all40 installed commands and their clock remain
identical. No achieved future state,unexecuted suffix,clipping or retiming.
"""
from types import SimpleNamespace
import numpy as np

from scripts.pi0292.eval_contract import matrix10
from ..trajectory.so3 import rot6d_rows_to_matrix,rotation_matrix_to_rot6d_rows
from ..world_model.task_learning_grasp_v1 import (
    FiftyMsProgram,WORKING_CLOSED_M,WORKING_OPEN_M,MECHANICAL_TRAVEL_M,
)

VERSION='SLA_CURRENT_ANCHOR_EQUIVALENT_CONSUMED_REFERENCE_V6'


def derive(state10,issuing_nominal,installed_reference,*,anchor_native_step,command_native_steps):
    state,old,installed=(np.asarray(v,dtype=np.float64)for v in(state10,issuing_nominal,installed_reference))
    steps=np.asarray(command_native_steps)
    if (state.shape!=(10,)or old.shape!=(40,10)or installed.shape!=(40,10)
            or type(anchor_native_step)is not int or anchor_native_step<5703
            or steps.shape!=(40,)or not np.issubdtype(steps.dtype,np.integer)
            or not np.array_equal(steps,anchor_native_step+np.arange(40)*25)
            or not all(np.isfinite(v).all()for v in(state,old,installed))
            or np.any((old[:,9]<0)|(old[:,9]>1))):
        raise ValueError('FULL40_REAL_CONSUMED_COMMANDS_AT_ORIGINAL_20HZ_REQUIRED')
    actual=matrix10(state);reference=matrix10(installed[0])
    first_error=float(np.max(np.abs(matrix10(old[0])-actual)))
    if first_error>1e-12:
        raise ValueError('ISSUING_FIRST_NOMINAL_MUST_EQUAL_SAME_CURRENT_MEASURED_TCP')
    result=old.copy()
    result[:,:3]=actual[:3,3]+installed[:,:3]-reference[:3,3]
    result[:,3:9]=rotation_matrix_to_rot6d_rows(actual[:3,:3]@reference[:3,:3].T@
        rot6d_rows_to_matrix(installed[:,3:9]))
    desired=result.copy()
    desired[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*result[:,9])/MECHANICAL_TRAVEL_M
    reconstructed=desired.copy()
    reconstructed[:,:3]=reference[:3,3]+result[:,:3]-actual[:3,3]
    reconstructed[:,3:9]=rotation_matrix_to_rot6d_rows(reference[:3,:3]@actual[:3,:3].T@
        rot6d_rows_to_matrix(result[:,3:9]))
    error=float(np.max(np.abs(reconstructed-installed)))
    if error>1e-12 or not np.array_equal(result[:,9],old[:,9]):
        raise ValueError('ALL40_ACTUAL_INSTALLED_REFERENCES_AND_ORIGINAL_NOMINAL_GRIP_REQUIRED')
    # Invoke the original envelope validator without inventing a base target.
    # It uses only these two EEF paths/anchors;native execution still uses its
    # complete FiftyMsProgram constructor,actual base and original safety chain.
    FiftyMsProgram.validate_envelope(SimpleNamespace(desired_path=desired,
        decision_TCP_target=actual,reference_path=reconstructed,initial_reference_TCP_target=reference))
    result=np.frombuffer(result.tobytes(),dtype=np.float64).reshape(40,10)
    return result,dict(version=VERSION,first_nominal_vs_actual_error=first_error,
        reconstructed_reference_max_error=error,full40_original_envelope_passes=True,
        current_measured_TCP_is_only_state_anchor=True,future_achieved_state_labels=False,
        actual_consumed_command_knots=40,unexecuted_suffix_added=False,original_nominal_grip_unchanged=True,
        native_transport_unchanged=True,geometry_clips=0,time_retiming=0,endpoint_basis=False,
        native_execution_authorized=False)
