"""First actual roundtrip violation, separate from relative-coordinate clipping."""
import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from scripts.pi0294.pv1.export import GATES,codec_packet


class Trace:
    def __init__(self):
        self.windows=0
        self.first_error=None
        self.first_inverse_clip=None

    def __call__(self,*,anchor,state,target,expected_relative,native_relative,normalized,packed,actual):
        if anchor!=self.windows:
            raise ValueError('DATA_COV1_TRACE_EXACT_ASCENDING_601_ANCHORS')
        self.windows+=1
        left,right=matrix_from10(target),matrix_from10(actual)
        cosine=np.clip((np.trace(left[:,:3,:3]@np.swapaxes(right[:,:3,:3],-1,-2),axis1=-2,axis2=-1)-1)/2,-1,1)
        errors=dict(XYZ_m=np.max(np.abs(target[:,:3]-actual[:,:3]),axis=-1),
            SO3_rad=np.arccos(cosine),gripper=np.abs(target[:,9]-actual[:,9]),
            relative_SE3_max_abs=np.max(np.abs(native_relative-expected_relative[:,:9]),axis=-1))
        clipped=np.abs(packed)>1
        def evidence(index):
            return dict(window_anchor=anchor,anchor_time_s=anchor/20,action_index=int(index),
                reference_time_s=(anchor+int(index))/20,
                normalized_channel_indices_with_inverse_clip=np.flatnonzero(clipped[index]).tolist(),
                normalized_inverse_clip_mask=clipped[index].tolist(),
                normalized_float64_values=normalized[index].tolist(),
                normalized_float32_values=packed[index].tolist(),
                target_absolute_EEF10=target[index].tolist(),decoded_absolute_EEF10=actual[index].tolist(),
                actual_anchor_state_EEF10=state.tolist(),
                error_at_action={k:float(v[index]) for k,v in errors.items()},
                world_XYZ_absolute_error=np.abs(target[index,:3]-actual[index,:3]).tolist(),
                relative_coordinate_2_is_world_Z=False,
                actual_event_join='UNKNOWN_UNTIL_JOINED_TO_SAVED_NATIVE_PHYSICS_NOT_TEACHER_PHASE')
        if self.first_inverse_clip is None and clipped.any():
            self.first_inverse_clip=evidence(np.flatnonzero(np.any(clipped,axis=1))[0])
        failed=np.column_stack([errors[k]>threshold for k,threshold in GATES.items()])
        if self.first_error is None and failed.any():
            index=int(np.flatnonzero(np.any(failed,axis=1))[0])
            self.first_error=dict(evidence(index),
                failed_physical_gates=[k for k,v in errors.items() if v[index]>GATES[k]],
                ordering='ASCENDING_WINDOW_ANCHOR_THEN_ACTION_INDEX_NOT_TRAJECTORY_MAXIMUM',
                normalized_offending_channel_unknown_if_clip_mask_empty=True)


def audit(arrays,processor,guard):
    trace=Trace()
    result=codec_packet(arrays,processor,guard,diagnostic_sink=trace)
    if trace.windows!=601:
        raise ValueError('DATA_COV1_COMPLETE_CODEC_TRACE_REQUIRED')
    if (trace.first_error is None)!=(result['status']=='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'):
        raise ValueError('DATA_COV1_TRACE_MUST_AGREE_WITH_ORIGINAL_NUMERIC_GATES')
    return dict(result,diagnostics=dict(first_error=trace.first_error,
        first_native_inverse_clip=trace.first_inverse_clip,windows_actually_observed=trace.windows,
        clipping_is_not_automatically_physical_roundtrip_failure=True,
        diagnostic_only_original_values_and_admission_unchanged=True))
