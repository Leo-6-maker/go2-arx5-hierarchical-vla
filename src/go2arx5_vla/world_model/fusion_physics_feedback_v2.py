"""Offline feedback with explicit freejoint-root versus MPC BASE semantics.

The V1 producer and its historical records stay unchanged. Their numeric BASE
comparison established record consistency, not physical-point equivalence.
This wrapper retains that comparison only as an unqualified diagnostic. It
implements no positive point binding, guessed extrinsic, control or grant.
"""
import copy

from . import fusion_physics_feedback as original

SCHEMA = 'WM02_FUSION_PHYSICS_FEEDBACK_PACKET_V2'
MEASURED_POINT = 'NATIVE_QPOS0_FREEJOINT_ROOT'
REFERENCE_POINT = 'MPC_BASE_REFERENCE_POINT_FROZEN_SEMANTICS_UNQUALIFIED'
BASE_REASON = 'BASE_PHYSICAL_POINT_FRAME_AND_MODEL_OUTPUT_SEMANTICS_NOT_QUALIFIED'


def make_packet(*, identity, state_before, state_after, consumption=None, saved_prediction=None,
                consumed_since_prediction=(), lifecycle_evidence=None, causal_timing=None,
                forecast_candidates=()):
    """Keep the V1 checks and demote BASE fields without changing input bytes.

    The measured freejoint root remains an available privileged measurement.
    An unqualified MPC-point comparison cannot become BASE tracking or a model
    innovation. No boolean or claimed receipt can enable positive applicability
    in this version. A future qualified point/model binding needs its own code.
    """
    packet = original.make_packet(identity=identity, state_before=state_before, state_after=state_after,
        consumption=consumption, saved_prediction=saved_prediction,
        consumed_since_prediction=consumed_since_prediction, lifecycle_evidence=lifecycle_evidence,
        causal_timing=causal_timing, forecast_candidates=forecast_candidates)
    packet['schema'] = SCHEMA
    measured = packet['measured']; values = measured['values']; masks = measured['validity_mask']
    values['freejoint_root_pose'] = values.pop('base_pose')
    masks['freejoint_root_pose'] = masks.pop('base_pose')
    measured['deployable_vs_privileged']['freejoint_root_pose'] = measured['deployable_vs_privileged'].pop('base_pose')
    raw_comparison = values['tracking_error']['base_pose']
    values['tracking_error']['base_pose'] = None
    masks['tracking_base_pose'] = False
    measured['freejoint_root_point_semantic_id'] = MEASURED_POINT
    measured['BASE_tracking_semantically_qualified'] = False
    measured['unqualified_BASE_point_diagnostic'] = dict(
        status='UNQUALIFIED_ROOT_VS_MPC_POINT', actual_point=MEASURED_POINT,
        reference_point=REFERENCE_POINT, numeric_SE3_comparison=raw_comparison,
        is_tracking_error=False, is_innovation=False, model_or_control_label=False,
        external_point_frame_evidence_required=True)

    innovation = packet['innovation']
    innovation['values']['base_pose'] = None
    innovation['validity_mask']['base_pose'] = False
    innovation['field_invalid_reasons'] = dict(base_pose=BASE_REASON)
    if innovation['valid'] and not any(innovation['validity_mask'].values()):
        innovation['valid'] = False
        innovation['invalid_reasons'].append('NO_SEMANTICALLY_APPLICABLE_PREDICTION_FIELDS')
    for forecast in packet['forecast']:
        estimate = forecast['values'].get('base_pose')
        forecast['values']['base_pose'] = None
        forecast['validity_mask']['base_pose'] = False
        forecast['field_invalid_reasons'] = dict(base_pose=BASE_REASON)
        forecast['unqualified_BASE_model_estimate'] = dict(value=copy.deepcopy(estimate),
            model_output_point='UNQUALIFIED', semantic_applicability=False,
            is_measurement=False, is_command=False, training_authorized=False)

    packet['BASE_point_semantics'] = dict(measured_point=MEASURED_POINT,
        installed_reference_point=REFERENCE_POINT, prediction_output_point='UNQUALIFIED',
        forecast_output_point='UNQUALIFIED', physical_point_equivalence_qualified=False,
        frame_equivalence_qualified=False, positive_point_binding_implemented=False,
        extrinsic_transform_applied=False, BASE_reference_modified=False,
        measured_root_relabelled_as_controller_BASE=False,
        original_V1_record_unchanged=True, guessed_offset_or_boolean_grant_allowed=False)
    return packet
