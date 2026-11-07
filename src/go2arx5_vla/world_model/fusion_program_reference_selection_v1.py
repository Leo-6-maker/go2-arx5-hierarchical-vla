"""One bounded pre-effect reference choice, separate from frozen WBC and safety.

This is a finite diagnostic selector, not an Actor. Every selected whole
program still needs each original native certificate, ACK and consumption.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .fusion_program_forecast_v1 import counterfactual_program_choices, ShadowForecaster, COUNTERFACTUAL_CONFIG
from .fusion_program_reference_intervention_v1 import SCHEMA, OFFSETS, validate_scope

POLICY_SCHEMA = 'WM02_FIRST_PROGRAM_REFERENCE_SELECTION_V1'
METHODS = ('DIRECT', 'TRAIN_LINEAR', 'WM_PRIMARY_SCALED')
CANDIDATES = ['NOMINAL', 'X_PLUS_050', 'Y_PLUS_050', 'Y_MINUS_050']
OBJECTIVE = 'H16_GENERATED_DESIRED_TCP_TRANSLATION'


def validate_registration(spec):
    config = spec.get('program_reference_selection')
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError('TYPED_REFERENCE_SELECTION_REQUIRED')
    expected = dict(schema=POLICY_SCHEMA, method=config.get('method'), candidates=CANDIDATES,
                    objective=OBJECTIVE, cadence='FIRST_INSTALL_ONLY',
                    excluded_minus_X='TRAIN_QX06_ORIGINAL_LATE_JOINT_LIMIT_FAILURE_RETAINED')
    initial = validate_scope(spec)
    forecast = spec.get('program_response_forecast', {})
    seeds = forecast.get('seeds', [])
    if (config != expected or config['method'] not in METHODS
            or initial is None or initial.candidate != 'NOMINAL'
            or spec['response_learning_assignment']['split'] != 'DEV'
            or spec.get('live_skill_time_stretch_factor') != 2
            or forecast.get('schema') != 'WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2'
            or spec.get('program_response_counterfactual') != COUNTERFACTUAL_CONFIG
            or not seeds or forecast.get('primary_seed') != seeds[0]):
        raise ValueError('EXPLICIT_FROZEN_SINGLE_CHOICE_DEV_POLICY_REQUIRED')
    panel = json.loads(Path(spec['response_learning_panel']).read_text())
    rows = [r for r in panel['assignments'] if r.get('native_run') == Path(spec['registered_root']).name]
    if len(rows) != 1 or rows[0].get('reference_selection') != config:
        raise ValueError('EXACT_PRE_EFFECT_REGISTERED_REFERENCE_SELECTION_ROW')
    return config


def choose(program, history, past_windows, models, *, method, primary_seed):
    """Use only the current observed past and the same complete frozen intent."""
    if (method not in METHODS or type(primary_seed) is not int
            or program.geometry_scale != 1. or program.time_stretch_factor != 2
            or program.reference_intervention is None
            or program.reference_intervention.candidate != 'NOMINAL'):
        raise ValueError('SAME_FULL_TIME2_NOMINAL_PROGRAM_BEFORE_SELECTION_REQUIRED')
    estimates = counterfactual_program_choices(history, past_windows, program, models)
    model = (None if method == 'DIRECT' else 'TRAIN_RIDGE_FULL_PROGRAM'
             if method == 'TRAIN_LINEAR' else str(primary_seed)+'-SCALED')
    if model is not None and model not in models:
        raise ValueError('EXACT_PREDECLARED_MODEL_NOT_DEV_WINNER')
    scores = (None if model is None else
              {c: estimates['candidates'][c]['predicted_H16_TCP_error_m'][model] for c in CANDIDATES})
    candidate = 'NOMINAL' if scores is None else min(CANDIDATES, key=lambda c: scores[c])
    intervention = dict(schema=SCHEMA, candidate=candidate,
                        offset_target_xy_m=list(OFFSETS[candidate]), ramp_atoms=16)
    selected = replace(program, reference_intervention=intervention)
    if (not np.array_equal(program.decoded_H40, selected.decoded_H40)
            or not np.array_equal(program.desired_path, selected.desired_path)
            or selected.end_atom != program.end_atom
            or not np.array_equal(program.base_pose_hold, selected.base_pose_hold)):
        raise ValueError('ONLY_DECLARED_REFERENCE_MAY_CHANGE_NOT_INTENT_CLOCK_OR_BASE')
    report = dict(schema=POLICY_SCHEMA, native_step=34303, method=method,
                  selected_model=model, selected_candidate=candidate,
                  allowed_candidates=CANDIDATES, all_candidate_estimates=estimates,
                  original_nominal_reference_path=program.reference_path,
                  selected_reference_path_sha256=selected.evidence()['reference_path']['sha256'],
                  future_observations_read=False, proposed_reference_changes=int(candidate != 'NOMINAL'),
                  actual_reference_installations=0, native_qualification=False,
                  original_native_certificates_and_ACKs_still_required=True,
                  full_MPC_hidden_state_restored=False)
    return selected, report


def prepare(adapter, program, history, past_windows):
    """Called only inside a new admitted native worker, before any suffix effect."""
    from .storage import write_json
    config = validate_registration(adapter.spec)
    if config is None:
        raise ValueError('REFERENCE_SELECTION_WAS_NOT_REGISTERED')
    recorder = ShadowForecaster(adapter.spec, adapter.output, program=program)
    selected, report = choose(program, history, past_windows, recorder.models,
                              method=config['method'],
                              primary_seed=adapter.spec['program_response_forecast']['primary_seed'])
    from .fusion_bounded_reference_tracking_v1 import screen
    cube = np.asarray(adapter.measured_task_state()['object_pose'], float)
    report['selected_full_program_domain_screen'] = screen(selected, cube)
    path = adapter.output/'PROGRAM_REFERENCE_SELECTION.private.json'
    if path.exists():
        raise ValueError('NO_REPLAY_OF_REFERENCE_SELECTION')
    write_json(path, report)
    if not report['selected_full_program_domain_screen']['eligible_for_original_native_certificate']:
        raise ValueError('SELECTED_WHOLE_PROGRAM_OUTSIDE_ORIGINAL_MECHANICAL_DOMAIN_NO_INSTALL')
    recorder.counterfactual_program = selected
    return selected, recorder, report
