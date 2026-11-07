"""Pure declared-reference tests, without a native admission or model fit."""
from dataclasses import replace,FrozenInstanceError
import json
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_reference_intervention_v1 as intervention
from go2arx5_vla.world_model.fusion_live_program_replay_v1 import PHASE
from test_fusion_trajectory_adapter import fixture


def document(candidate):
    return dict(schema=intervention.SCHEMA,candidate=candidate,
        offset_target_xy_m=list(intervention.OFFSETS[candidate]),ramp_atoms=16)


@pytest.mark.parametrize('candidate',['NOMINAL','X_PLUS_050','X_MINUS_050','Y_PLUS_050','Y_MINUS_050'])
@pytest.mark.parametrize('time_factor',[1,2])
def test_whole_intent_and_clock_survive_explicit_xy_reference_changes(candidate,time_factor):
    original=replace(fixture(),time_stretch_factor=time_factor)
    changed=replace(original,reference_intervention=document(candidate))
    for name in ('decoded_H40','mapped_H40','desired_knots','desired_path','base_pose_hold'):
        np.testing.assert_array_equal(getattr(changed,name),getattr(original,name))
    assert changed.knot_indices==original.knot_indices and changed.end_atom==original.end_atom
    np.testing.assert_array_equal(changed.reference_path[:,2:],original.reference_path[:,2:])
    np.testing.assert_array_equal(changed.reference_path[0],original.reference_path[0])
    delta=changed.reference_path[:,:2]-original.reference_path[:,:2]
    np.testing.assert_allclose(delta[16:],np.tile(intervention.OFFSETS[candidate],(len(delta)-16,1)),atol=3e-17,rtol=0)
    assert np.linalg.norm(delta,axis=1).max()<=.0005+1e-16
    assert changed.evidence()['raw_desired_path_unchanged'] is True
    assert changed.evidence()['native_execution_authorized'] is False
    with pytest.raises(FrozenInstanceError):changed.reference_intervention.candidate='HOLD'


def test_hold_is_a_control_with_unexecuted_raw_intent_never_a_clipped_skill():
    original=fixture();hold=replace(original,reference_intervention=document('HOLD'))
    np.testing.assert_array_equal(hold.decoded_H40,original.decoded_H40)
    np.testing.assert_array_equal(hold.desired_path,original.desired_path)
    np.testing.assert_array_equal(hold.reference_path,np.repeat(original.reference_path[:1],len(original.reference_path),axis=0))
    evidence=hold.evidence()
    assert evidence['post_query_HOLD_control'] is True
    assert evidence['raw_H40_installed_as_skill'] is False
    assert evidence['reference_translation']=='FIXED_INITIAL_REFERENCE_HOLD_RAW_INTENT_UNEXECUTED'
    assert evidence['gripper']['execution_rule'].endswith('RAW_GRIPPER_INTENT_UNEXECUTED')
    assert evidence['command_clips']==0


def test_outside_declared_reference_set_is_rejected_instead_of_clipped():
    value=document('X_PLUS_050');value['offset_target_xy_m']=[.001,0.]
    with pytest.raises(ValueError):replace(fixture(),reference_intervention=value)


def test_scope_requires_actual_pinned_panel_assignment(tmp_path):
    from go2arx5_vla.world_model.storage import digest
    config=document('HOLD');panel=tmp_path/'PANEL.json';root=tmp_path/'new-native'
    panel.write_text(json.dumps(dict(domain=dict(time_stretch_factor=1),assignments=[dict(native_run=root.name,parent='HOLD0',split='TRAIN',reference_intervention=config)])))
    pin=digest(panel)
    spec=dict(phase=PHASE,task_condition='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1',
        live_skill_geometry_scale=1.,live_skill_time_stretch_factor=1,registered_root=str(root),
        program_reference_intervention=config,response_learning_panel=str(panel),response_learning_panel_sha256=pin,
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',parent='HOLD0',split='TRAIN',panel_sha256=pin))
    assert intervention.validate_scope(spec).candidate=='HOLD'
    spec['program_reference_intervention']=document('X_PLUS_050')
    with pytest.raises(ValueError,match='PANEL_ROW'):intervention.validate_scope(spec)
