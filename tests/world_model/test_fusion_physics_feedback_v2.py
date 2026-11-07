"""Point semantics are not inferred from coincident numbers or pass flags."""
import copy
import json

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_physics_feedback as original
from go2arx5_vla.world_model import fusion_physics_feedback_v2 as feedback
from go2arx5_vla.world_model.target_reference import trajectory_identity
from test_fusion_physics_feedback import identity, state, consumption, saved_prediction, forecast, timing, lifecycle, rewire, seal_prediction


def inputs():
    before, after = state(3503), state(3528, .302)
    consumed = consumption(before, after)
    return dict(identity=identity(), state_before=before, state_after=after, consumption=consumed,
        saved_prediction=None, consumed_since_prediction=[consumed],
        lifecycle_evidence=lifecycle(before, after), causal_timing=timing(before, after), forecast_candidates=[])


def test_valid_privileged_freejoint_measurement_is_named_and_never_qualified_BASE_tracking():
    args = inputs(); old = original.make_packet(**args); new = feedback.make_packet(**args)
    assert old['measured']['validity_mask']['tracking_base_pose'] is True
    assert new['schema'] == feedback.SCHEMA
    assert 'base_pose' not in new['measured']['values']
    assert 'base_pose' not in new['measured']['validity_mask']
    assert new['measured']['values']['freejoint_root_pose'] == old['measured']['values']['base_pose']
    assert new['measured']['validity_mask']['freejoint_root_pose'] is True
    assert new['measured']['deployable_vs_privileged']['freejoint_root_pose'] == 'SIMULATOR_PRIVILEGED'
    assert new['measured']['freejoint_root_point_semantic_id'] == 'NATIVE_QPOS0_FREEJOINT_ROOT'
    assert new['measured']['validity_mask']['tracking_base_pose'] is False
    assert new['measured']['values']['tracking_error']['base_pose'] is None
    diagnostic = new['measured']['unqualified_BASE_point_diagnostic']
    assert diagnostic['status'] == 'UNQUALIFIED_ROOT_VS_MPC_POINT'
    assert diagnostic['numeric_SE3_comparison'] == old['measured']['values']['tracking_error']['base_pose']
    assert diagnostic['numeric_SE3_comparison']['translation_world_m'] == [0., 0., 0.]
    assert not diagnostic['is_tracking_error'] and not diagnostic['model_or_control_label']


def test_different_root_and_MPC_points_are_retained_only_as_unqualified_numeric_diagnostic():
    args = inputs(); c = args['consumption']; base = [.05, 0., .5, 0., 0., 0., 1.]
    def change(doc):
        for point in doc['base']['waypoints']: point['position'] = base[:3]
    rewire(c, change)
    c['window']['actual_installed_consumed_EEF10_BASE7'][10:] = base
    for event in c['proposed']['receiver_events']:
        if event['kind'] == 'installed':
            for pose in event['poses']: pose[:7] = base
            pin = trajectory_identity(event['times'], event['poses'], event['inputs'])
            c['window']['target_trajectory_sha256'] = pin
            c['window']['solver_policies_with_exact_installed_target'][0]['trajectory_sha256'] = pin
    new = feedback.make_packet(**args)
    assert new['consumption_invalid_reason'] is None
    assert new['measured']['command_domain_evidence']['SOLVER_TARGET_CONFIRMED']
    raw = new['measured']['unqualified_BASE_point_diagnostic']['numeric_SE3_comparison']
    assert raw['translation_world_m'] == pytest.approx([-.05, 0., -.1])
    assert new['measured']['values']['tracking_error']['base_pose'] is None
    assert not new['BASE_point_semantics']['extrinsic_transform_applied']
    assert not new['BASE_point_semantics']['BASE_reference_modified']


def test_TCP_gripper_commands_and_all_causal_identities_are_unchanged_without_mutating_V1_or_inputs():
    args = inputs(); args['saved_prediction'] = saved_prediction(args['state_before'], args['state_after'], [args['consumption']])
    untouched = copy.deepcopy(args); old = original.make_packet(**args); old_copy = copy.deepcopy(old)
    new = feedback.make_packet(**args)
    assert args == untouched and old == old_copy
    for key in ('identity', 'observation_timestamp', 'native_step', 'history_span', 'consumed_reference',
                'actual_state_document_sha256', 'WM_checkpoint_sha256', 'units'):
        assert new[key] == old[key]
    for key in ('tcp', 'finger_q', 'object_pose', 'velocity', 'events', 'reference_age_s'):
        assert new['measured']['values'][key] == old['measured']['values'][key]
    for key in ('tcp', 'finger_error_m'):
        assert new['measured']['values']['tracking_error'][key] == old['measured']['values']['tracking_error'][key]
    assert old['innovation']['validity_mask']['base_pose'] is True
    assert new['innovation']['validity_mask']['base_pose'] is False
    for key in ('tcp', 'finger_q', 'object_pose'):
        assert new['innovation']['values'][key] == old['innovation']['values'][key]
        assert new['innovation']['validity_mask'][key] == old['innovation']['validity_mask'][key]
    assert new['innovation']['valid'] and new['innovation']['hidden_controller_selection_observed'] is False
    assert not new['native_execution_authorized'] and new['new_integrations'] == new['new_queries'] == 0


def test_saved_BASE_only_prediction_cannot_claim_any_valid_innovation_without_model_point_semantics():
    args = inputs(); pred = saved_prediction(args['state_before'], args['state_after'], [args['consumption']])
    record = json.loads(pred['bytes']); record['predicted'] = {'base_pose': record['predicted']['base_pose']}
    args['saved_prediction'] = seal_prediction(record)
    assert original.make_packet(**args)['innovation']['valid']
    new = feedback.make_packet(**args)
    assert not new['innovation']['valid'] and new['innovation']['values']['base_pose'] is None
    assert not any(new['innovation']['validity_mask'].values())
    assert new['innovation']['invalid_reasons'] == ['NO_SEMANTICALLY_APPLICABLE_PREDICTION_FIELDS']
    assert new['innovation']['field_invalid_reasons']['base_pose'] == feedback.BASE_REASON


def test_forecast_available_BASE_output_is_not_semantically_qualified_by_candidate_or_model_domain_hash():
    args = inputs(); args['forecast_candidates'] = [forecast(args['state_after'])]
    old = original.make_packet(**args); new = feedback.make_packet(**args)
    assert old['forecast'][0]['record_consistent'] and old['forecast'][0]['validity_mask']['base_pose']
    assert new['forecast'][0]['record_consistent'] and new['forecast'][0]['field_availability_mask']['base_pose']
    assert new['forecast'][0]['values']['base_pose'] is None
    assert new['forecast'][0]['validity_mask']['base_pose'] is False
    diagnostic = new['forecast'][0]['unqualified_BASE_model_estimate']
    assert diagnostic['value'] == old['forecast'][0]['values']['base_pose']
    assert diagnostic['model_output_point'] == 'UNQUALIFIED'
    assert not diagnostic['semantic_applicability'] and not diagnostic['training_authorized']
    for key in ('tcp', 'finger_q', 'object_pose', 'event_probabilities'):
        assert new['forecast'][0]['values'][key] == old['forecast'][0]['values'][key]
        assert new['forecast'][0]['validity_mask'][key] == old['forecast'][0]['validity_mask'][key]


@pytest.mark.parametrize('field,value', [('base_semantics_qualified', True), ('base_point_binding', True),
    ('base_point_binding', {'same_point': True, 'evidence_sha256': 'a'*64}),
    ('base_point_binding', {'transform': np.eye(4).tolist(), 'qualified': True})])
def test_no_boolean_receipt_or_guessed_transform_can_unlock_positive_binding(field, value):
    args = inputs(); args[field] = value
    with pytest.raises(TypeError): feedback.make_packet(**args)


def test_missing_root_state_stays_missing_and_forecast_never_fills_it():
    args = inputs(); args['forecast_candidates'] = [forecast(args['state_after'])]
    args['state_after'].pop('qpos')
    args['consumption'] = None
    new = feedback.make_packet(**args)
    assert new['measured']['values']['freejoint_root_pose'] is None
    assert new['measured']['validity_mask']['freejoint_root_pose'] is False
    assert new['measured']['unqualified_BASE_point_diagnostic']['numeric_SE3_comparison'] is None
    assert new['forecast'][0]['values']['base_pose'] is None
    assert json.loads(json.dumps(new, allow_nan=False)) == new


def test_original_clock_rejection_and_no_prediction_masks_are_preserved():
    args = inputs(); new = feedback.make_packet(**args)
    assert new['innovation']['valid'] is False and new['innovation']['invalid_reasons'] == ['NO_SAVED_PREDICTION']
    args['state_after']['time_s'] += .002
    with pytest.raises(ValueError, match='ACTUAL_ABSOLUTE_NATIVE_CLOCK'): feedback.make_packet(**args)
