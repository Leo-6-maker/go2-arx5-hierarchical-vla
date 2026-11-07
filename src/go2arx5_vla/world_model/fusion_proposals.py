"""Audit frozen source proposals without changing their values or authorizing motion."""
import numpy as np


def source_execution_schedule(observation_s,*,points=40,committed_points=8,point_dt_s=.05,native_dt_s=.002):
    """Original source issue0 at observation; state0 is AFTER its 25 integrations.

    Returned future times are a command/sampling schedule, not observed outcomes.
    Uncommitted preview points are never labeled as measured source responses.
    """
    if (not np.isfinite(observation_s)or type(points)is not int or points!=40
            or type(committed_points)is not int or committed_points!=8
            or point_dt_s!=.05 or native_dt_s!=.002):
        raise ValueError('FROZEN_SOURCE_H40_COMMIT8_25_NATIVE_TICKS_REQUIRED')
    issue=float(observation_s)+np.arange(points)*point_dt_s
    sampled=float(observation_s)+np.arange(1,committed_points+1)*point_dt_s
    return dict(command_issue_timestamps_s=issue,committed_command_issue_s=issue[:committed_points].copy(),
        expected_post_command_sample_s=sampled,native_integrations_per_command=25,
        committed_native_integrations=200,uncommitted_preview_points=32,
        measured_source_outcomes=False,physical_execution_authorized=False,
        forecast_or_command_times_are_not_measured_response_labels=True)


def frozen_inverse_normalization_audit(normalized,physical_h40):
    """Expose the original saved processor's inverse clipping; do not redo it.

    Actual source uses the first10 physical channels from its padded132 vector.
    This records the raw values outside[-1,1] and leaves saved model/processor
    outputs unchanged. Fusion must not add projection or gripper rescaling.
    """
    raw=np.asarray(normalized,float);physical=np.asarray(physical_h40,float)
    if (raw.shape!=(1,40,132)or physical.shape!=(40,10)
            or not np.isfinite(raw).all()or not np.isfinite(physical).all()):
        raise ValueError('ACTUAL_UNMODIFIED_FROZEN_SOURCE_ACTION_ARRAYS_REQUIRED')
    used=raw[0,:,:10];outside=(used<-1)|(used>1)
    return dict(schema='WM02_FUSION_FROZEN_PROCESSOR_CLIP_AUDIT_V1',
        physical_channels=['x','y','z','r00','r01','r02','r10','r11','r12','mechanical_open_fraction'],
        outside_unit_components=int(outside.sum()),outside_unit_components_by_channel=outside.sum(0).tolist(),
        affected_points=int(outside.any(-1).sum()),raw_used_min_by_channel=used.min(0).tolist(),
        raw_used_max_by_channel=used.max(0).tolist(),original_processor_inverse_clip=True,
        inverse_clip_source='FROZEN_UNNORMALIZE_VALUES_MINMAX_IMPLEMENTATION',
        original_processor_or_weights_changed=False,fusion_clipping_applied=False,
        source_command_is_measured_outcome=False,physical_execution_authorized=False)


def target_gripper_stroke_audit(mechanical_open_fraction):
    """A raw mechanical fraction is not the target Teacher's logical fraction."""
    values=np.asarray(mechanical_open_fraction,float)
    if values.ndim!=1 or not len(values)or not np.isfinite(values).all()or np.any(values<0)or np.any(values>1):
        raise ValueError('FINITE_SOURCE_MECHANICAL_OPEN_FRACTIONS_REQUIRED')
    low,high=.002/.044,.042/.044
    rejected=(values<low)|(values>high)
    return dict(source_semantics='MECHANICAL_OPEN_FRACTION_ABSOLUTE',
        source_min=float(values.min()),source_max=float(values.max()),
        target_working_mechanical_fraction=[low,high],outside_working_stroke_points=int(rejected.sum()),
        outside_working_stroke_indices=np.flatnonzero(rejected).tolist(),
        target_stroke_compatible=not bool(rejected.any()),rescaled_or_clamped=False,
        teacher_logical_fraction_conversion_applied=False,
        native_admission='NOT_GRANTED_BY_STROKE_AUDIT')
