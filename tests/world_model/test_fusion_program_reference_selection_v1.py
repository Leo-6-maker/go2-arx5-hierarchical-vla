"""Bounded selection and scope semantics; these are not native qualification."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_reference_selection_v1 as selection
from go2arx5_vla.world_model.fusion_program_reference_intervention_v1 import SCHEMA
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.fusion_program_forecast_v1 import COUNTERFACTUAL_CONFIG
from test_fusion_trajectory_adapter import fixture
from test_fusion_live_program_replay_v1 import artifacts


def nominal():
    return replace(fixture(), time_stretch_factor=2, reference_intervention=dict(
        schema=SCHEMA, candidate='NOMINAL', offset_target_xy_m=[0., 0.], ramp_atoms=16))


def policy(method='WM_PRIMARY_SCALED'):
    return dict(schema=selection.POLICY_SCHEMA, method=method, candidates=selection.CANDIDATES.copy(),
                objective=selection.OBJECTIVE, cadence='FIRST_INSTALL_ONLY',
                excluded_minus_X='TRAIN_QX06_ORIGINAL_LATE_JOINT_LIMIT_FAILURE_RETAINED')


def test_one_shared_intent_and_clock_only_bounded_reference_changes(monkeypatch):
    program = nominal()
    _, prefix, decision, _ = artifacts()
    model_names = ['366104-SCALED', 'TRAIN_RIDGE_FULL_PROGRAM']
    costs = dict(NOMINAL=4., X_PLUS_050=3., X_MINUS_050=0., Y_PLUS_050=1., Y_MINUS_050=2.)
    seen = []
    def estimate(history, past, actual, models):
        assert history is decision['actual_history'] and len(past) == 8
        np.testing.assert_array_equal(actual.decoded_H40, program.decoded_H40)
        seen.append(actual)
        return dict(candidates={c: dict(predicted_H16_TCP_error_m={m: v for m in model_names})
                                for c, v in costs.items()}, future_observations_read=False)
    monkeypatch.setattr(selection, 'counterfactual_program_choices', estimate)
    for method, candidate in [('DIRECT', 'NOMINAL'), ('TRAIN_LINEAR', 'Y_PLUS_050'),
                              ('WM_PRIMARY_SCALED', 'Y_PLUS_050')]:
        chosen, report = selection.choose(program, decision['actual_history'], prefix['windows'][-8:],
                                          dict.fromkeys(model_names), method=method, primary_seed=366104)
        assert report['selected_candidate'] == candidate
        assert report['selected_candidate'] != 'X_MINUS_050'
        np.testing.assert_array_equal(chosen.decoded_H40, program.decoded_H40)
        np.testing.assert_array_equal(chosen.desired_path, program.desired_path)
        np.testing.assert_array_equal(chosen.base_pose_hold, program.base_pose_hold)
        assert chosen.end_atom == program.end_atom
        assert np.max(np.abs(chosen.reference_path[:, :2]-program.reference_path[:, :2])) <= .0005+1e-15
        assert report['actual_reference_installations'] == 0 and report['native_qualification'] is False
    assert len(seen) == 3


def test_real_counterfactual_with_action_blind_models_keeps_direct_tie_nominal():
    from go2arx5_vla.world_model.fusion_program_response_data_v1 import FEATURE_DIM
    program = nominal()
    _, prefix, decision, _ = artifacts()
    arrays = dict(x_mean=np.zeros(FEATURE_DIM), x_scale=np.ones(FEATURE_DIM),
        y_mean=np.zeros((16, 8)), y_scale=np.ones((16, 8)), w0=np.zeros((32, FEATURE_DIM)),
        b0=np.zeros(32), w1=np.zeros((128, 32)), b1=np.zeros(128), mask_future=np.array(True))
    chosen, report = selection.choose(program, decision['actual_history'], prefix['windows'][-8:],
        {'366104-SCALED': arrays}, method='WM_PRIMARY_SCALED', primary_seed=366104)
    assert report['selected_candidate'] == 'NOMINAL'
    np.testing.assert_array_equal(chosen.reference_path, program.reference_path)


def registered(tmp_path):
    run = tmp_path/'native'
    panel = tmp_path/'panel.json'
    cfg = policy()
    intervention = nominal().reference_intervention.document()
    panel.write_text(json.dumps(dict(domain=dict(time_stretch_factor=2), assignments=[dict(
        native_run=run.name, parent='NEW_DEV', split='DEV', reference_intervention=intervention,
        reference_selection=cfg)])))
    pin = digest(panel)
    return dict(phase='collect-target-fusion-task-phase-live-skill-v9', registered_root=str(run),
        task_condition='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1',
        live_skill_geometry_scale=1., live_skill_time_stretch_factor=2,
        program_reference_intervention=intervention, program_reference_selection=cfg,
        program_response_counterfactual=COUNTERFACTUAL_CONFIG,
        response_learning_panel=str(panel), response_learning_panel_sha256=pin,
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',
            split='DEV', parent='NEW_DEV', panel_sha256=pin),
        program_response_forecast=dict(schema='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2',
            seeds=[366104, 366105, 366106], primary_seed=366104))


@pytest.mark.parametrize('fault', ['method', 'candidate', 'cadence', 'seed', 'TRAIN', 'changed_panel', 'row'])
def test_no_implicit_policy_or_larger_action_domain(tmp_path, fault):
    spec = registered(tmp_path)
    assert selection.validate_registration(spec) == policy()
    if fault == 'method': spec['program_reference_selection']['method'] = 'DEV_BEST_SEED'
    elif fault == 'candidate': spec['program_reference_selection']['candidates'].append('X_MINUS_050')
    elif fault == 'cadence': spec['program_reference_selection']['cadence'] = 'EVERY_ATOM'
    elif fault == 'seed': spec['program_response_forecast']['primary_seed'] = 366106
    elif fault == 'TRAIN': spec['response_learning_assignment']['split'] = 'TRAIN'
    elif fault == 'changed_panel': Path(spec['response_learning_panel']).write_text('{}')
    elif fault == 'row': spec['program_reference_selection']['method'] = 'DIRECT'
    with pytest.raises(ValueError):
        selection.validate_registration(spec)


def test_chosen_program_outside_original_domain_is_recorded_before_any_install(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from go2arx5_vla.world_model import fusion_bounded_reference_tracking_v1 as bounds
    spec = registered(tmp_path)
    output = Path(spec['registered_root']); output.mkdir()
    adapter = SimpleNamespace(spec=spec, output=output,
        measured_task_state=lambda: dict(object_pose=np.eye(4)))
    class Recorder:
        def __init__(self,*args,**kwargs): self.models = {}
    monkeypatch.setattr(selection,'ShadowForecaster',Recorder)
    monkeypatch.setattr(selection,'choose',lambda program,*args,**kwargs:(program,dict(native_qualification=False)))
    monkeypatch.setattr(bounds,'screen',lambda *args:dict(eligible_for_original_native_certificate=False))
    with pytest.raises(ValueError,match='OUTSIDE_ORIGINAL_MECHANICAL_DOMAIN_NO_INSTALL'):
        selection.prepare(adapter,nominal(),{},[])
    saved=json.loads((output/'PROGRAM_REFERENCE_SELECTION.private.json').read_text())
    assert saved['selected_full_program_domain_screen']['eligible_for_original_native_certificate']is False

