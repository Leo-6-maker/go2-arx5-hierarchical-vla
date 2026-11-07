"""Small NumPy response study for a NEW moving-trajectory identification panel.

All four regressors use fixed ridge=1. History interaction is compared with
the same-history additive lag/velocity model, not only with a blind mean.
No old XY/ARX parameters, model, normalization, threshold or dataset is reused.
"""
import json
import os
import hashlib
from pathlib import Path

import numpy as np

from ..trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
from .fusion_trajectory_response_contract import (
    ATOM_DT_S, DIRECTION, EPSILON_M, THETA_LEVELS, canonical_sha256, need, validate_panel,
)
from .fusion_trajectory_response_replay import validate_cohort, whole_parent_folds

SCHEMA = 'WM02_PAIRED_TRAJECTORY_RESPONSE_NUMPY_V1'
MODELS = ('NOMINAL', 'ADDITIVE_LAG_VELOCITY', 'CONDITIONAL', 'CORRECTION_BLIND')
DIMENSIONS = dict(NOMINAL=31, ADDITIVE_LAG_VELOCITY=40, CONDITIONAL=49, CORRECTION_BLIND=40)
RIDGE = 1.


def _ids(data, ids):
    result = np.arange(len(data['history_state'])) if ids is None else np.asarray(ids)
    need(result.ndim == 1 and result.dtype.kind in 'iu'
         and np.all((result >= 0) & (result < len(data['history_state']))), 'VALID_RESPONSE_ROW_IDS')
    return result


def _theta(data, ids, candidate_theta):
    value = np.asarray(data['theta_level'])[ids] if candidate_theta is None else np.asarray(candidate_theta)
    if value.ndim == 0:
        value = np.full(len(ids), value.item(), dtype=value.dtype)
    need(value.shape == (len(ids),) and value.dtype.kind in 'iu'
         and np.isin(value, THETA_LEVELS).all(), 'DECLARED_CORRECTION_LEVELS')
    return value


def mask_future_correction(data):
    """Unknown future correction retains nominal, history, and TRAIN prior."""
    result = dict(data)
    result['theta_level'] = np.zeros_like(data['theta_level'])
    result['corrected_H40'] = np.asarray(data['nominal_H40']).copy()
    if 'nominal_path' in data:
        result['corrected_path'] = np.asarray(data['nominal_path']).copy()
    return result


def _rotation_log(rotation):
    q = matrix_to_quat(rotation)
    q = np.where(q[..., 3:] < 0, -q, q)
    sine = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    return q[..., :3] * (2 * np.arctan2(sine, q[..., 3:])) / np.where(sine > 0, sine, 1.)


def causal_features(data, ids=None, *, model='CONDITIONAL', candidate_theta=None):
    """31 fixed nominal/clock summaries; nine observed lag/velocity scalars.

    All forty nominal points enter fixed bins. Corrected-path bytes, desired
    future outcomes, future states, parent/context/restart IDs and absolute
    world position/native time are never regression features.
    """
    need(model in MODELS, 'DECLARED_REGRESSOR')
    ids = _ids(data, ids)
    state = np.asarray(data['history_state'], float)[ids]
    past = np.asarray(data['past_reference'], float)[ids]
    nominal = np.asarray(data['nominal_H40'], float)[ids]
    need(state.shape == (len(ids), 9, 106) and past.shape == (len(ids), 8, 17)
         and nominal.shape == (len(ids), 40, 10)
         and np.isfinite(state).all() and np.isfinite(past).all() and np.isfinite(nominal).all(),
         'FINITE_CAUSAL_HISTORY_PAST_REFERENCES_AND_FULL_NOMINAL_H40')
    xyz = (nominal[:, :, :3] - nominal[:, :1, :3]) * 1000.
    rotation = rot6d_rows_to_matrix(nominal[:, :, 3:9])
    relative = np.swapaxes(rotation[:, :1], -1, -2) @ rotation
    rotation_log = _rotation_log(relative)
    grip = nominal[:, :, 9:10] - nominal[:, :1, 9:10]
    program = np.concatenate((xyz, rotation_log, grip), axis=-1)
    bins = ((1, 5), (5, 9), (9, 17), (17, 40))
    source_age = np.asarray(data['source_age_s'], float)[ids, None]
    remaining = np.asarray(data['ttl_remaining_s'], float)[ids, None]
    live = np.asarray(data['complete_live_atoms'], float)[ids, None]
    nominal_x = np.concatenate([program[:, start:end].mean(1) for start, end in bins]
                               + [source_age / .4, remaining / .4, live / 8.], axis=1)
    tcp = state[:, :, 86:89]
    history_x = np.concatenate(((tcp[:, -1] - nominal[:, 0, :3]) * 1000.,
                                (tcp[:, -1] - tcp[:, -2]) * 1000.,
                                (tcp[:, :8] - past[:, :, :3]).mean(1) * 1000.), axis=1)
    if model == 'NOMINAL':
        result = nominal_x
    elif model in ('ADDITIVE_LAG_VELOCITY', 'CORRECTION_BLIND'):
        result = np.concatenate((nominal_x, history_x), axis=1)
    else:
        theta = _theta(data, ids, candidate_theta)
        result = np.concatenate((nominal_x, history_x, history_x * theta[:, None]), axis=1)
    need(result.shape == (len(ids), DIMENSIONS[model]) and np.isfinite(result).all(),
         'EXACT_FINITE_CAUSAL_FEATURE_LAYOUT')
    return result


def _target(data, ids):
    current = np.asarray(data['history_state'], float)[ids, -1, 86:89]
    value = np.asarray(data['future_TCP'], float)[ids] - current[:, None]
    need(value.shape == (len(ids), 16, 3) and np.isfinite(value).all(), 'ACTUAL_OBSERVED_H16_XYZ_LABELS')
    return value


def _static(data, ids):
    need(np.all(np.asarray(data['split'])[ids] == 'TRAIN'), 'TRAIN_ONLY_STATIC_FIT')
    theta = np.asarray(data['theta_level'])[ids]
    target = _target(data, ids)
    # validate_cohort/folds ensure equal parent/context/restart counts per level.
    curves = np.stack([target[theta == level].mean(0) for level in THETA_LEVELS])
    need(np.isfinite(curves).all(), 'ALL_THREE_LEVELS_IN_TRAIN_FOLD')
    return curves


def _base(curves, levels, model):
    if model == 'CORRECTION_BLIND':
        return np.broadcast_to(curves.mean(0), (len(levels), 16, 3)).copy()
    return curves[levels + 1]


def _fit(data, ids, model, curves):
    x = causal_features(data, ids, model=model)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = (x - mean) / scale
    levels = _theta(data, ids, None)
    residual_mm = ((_target(data, ids) - _base(curves, levels, model)) * 1000.).reshape(len(ids), 48)
    weight = z.T @ np.linalg.solve(z @ z.T + RIDGE * np.eye(len(ids)), residual_mm)
    return dict(mean=mean, scale=scale, weight=weight)


def _displacement(data, ids, model, parameters, curves, candidate_theta=None):
    # The blind arm ignores candidate levels in BOTH features and its base.
    levels = (np.zeros(len(ids), dtype=int) if model == 'CORRECTION_BLIND'
              else _theta(data, ids, candidate_theta))
    x = causal_features(data, ids, model=model, candidate_theta=candidate_theta)
    z = (x - parameters['mean']) / parameters['scale']
    return _base(curves, levels, model) + .001 * (z @ parameters['weight']).reshape(-1, 16, 3)


def training_manifest_sha256(data, panel):
    train = validate_cohort(data, panel)['TRAIN']
    return canonical_sha256(dict(panel_sha256=canonical_sha256(panel), runs=[dict(
        run=str(data['run_id'][i]), parent=str(data['parent'][i]), context=str(data['context_id'][i]),
        theta=int(data['theta_level'][i]), restart=int(data['restart_index'][i]),
        source_record_sha256=str(data['source_record_sha256'][i]))
        for i in sorted(train, key=lambda j: str(data['run_id'][j]))]))


def train_noise_threshold(data, panel, *, fraction, floor_mm):
    """Predeclare fraction/floor BEFORE DEV; derive noise from NEW TRAIN only.

    Same-cell restart differences divided by sqrt(2) estimate single-run
    coordinate noise. This is descriptive scatter, not an independence proof.
    """
    need(np.isfinite(fraction) and fraction > 0 and np.isfinite(floor_mm) and floor_mm > 0,
         'POSITIVE_PREDECLARED_NOISE_FRACTION_AND_NEW_DOMAIN_FLOOR')
    train = validate_cohort(data, panel)['TRAIN']
    gaps = []
    for parent in sorted(set(np.asarray(data['parent'])[train])):
        for context in ('C0', 'C1'):
            for theta in THETA_LEVELS:
                ids = train[(np.asarray(data['parent'])[train] == parent)
                            & (np.asarray(data['context_id'])[train] == context)
                            & (np.asarray(data['theta_level'])[train] == theta)]
                need(len(ids) == 2, 'TWO_NEW_TRAIN_SAME_CELL_RESTARTS')
                target = _target(data, ids)
                gaps.append((target[1] - target[0]) * 1000.)
    noise = float(np.sqrt(np.mean(np.asarray(gaps) ** 2) / 2.))
    return dict(method='NEW_TRAIN_SAME_CELL_RESTART_COORDINATE_RMSE_OVER_SQRT2_V1',
                training_manifest_sha256=training_manifest_sha256(data, panel),
                independent_restart_pairs=len(gaps), noise_RMSE_mm=noise,
                predeclared_fraction=float(fraction), predeclared_floor_mm=float(floor_mm),
                min_lopo_gain_mm=max(float(floor_mm), float(fraction) * noise),
                TEST_read=False, DEV_labels_read=False, old_XY_threshold_reused=False)


class TrajectoryResponseStudy:
    def __init__(self, panel):
        validate_panel(panel)
        self.panel = json.loads(json.dumps(panel))
        self._reload_verified = False
        self._loaded_artifact_sha256 = None

    def fit(self, data, *, min_lopo_gain_mm, threshold_source):
        self._reload_verified = False
        self._loaded_artifact_sha256 = None
        train = validate_cohort(data, self.panel)['TRAIN']
        expected = train_noise_threshold(data, self.panel,
            fraction=threshold_source['predeclared_fraction'], floor_mm=threshold_source['predeclared_floor_mm'])
        need(expected == threshold_source and min_lopo_gain_mm == expected['min_lopo_gain_mm'],
             'EXACT_NEW_TRAIN_THRESHOLD_SOURCE_NO_DEV_OR_OLD_XY_THRESHOLD')
        errors = {model: [] for model in ('STATIC',) + MODELS}
        audit = []
        for parent, fit_ids, held in whole_parent_folds(data, self.panel):
            curves = _static(data, fit_ids)
            truth = _target(data, held)
            levels = _theta(data, held, None)
            errors['STATIC'].append(float(np.mean((1000. * (_base(curves, levels, 'STATIC') - truth)) ** 2)))
            normalizers = {}
            for model in MODELS:
                parameters = _fit(data, fit_ids, model, curves)
                prediction = _displacement(data, held, model, parameters, curves)
                errors[model].append(float(np.mean((1000. * (prediction - truth)) ** 2)))
                normalizers[model] = {key: parameters[key].tolist() for key in ('mean', 'scale')}
            audit.append(dict(held_parent=parent, train_parents=sorted(set(map(str, data['parent'][fit_ids]))),
                              train_runs=list(map(str, data['run_id'][fit_ids])),
                              static_curves_m=curves.tolist(), normalizers=normalizers))
        rmse = {model: float(np.sqrt(np.mean(value))) for model, value in errors.items()}
        if (rmse['STATIC'] - rmse['CONDITIONAL'] >= min_lopo_gain_mm
                and rmse['ADDITIVE_LAG_VELOCITY'] - rmse['CONDITIONAL'] >= min_lopo_gain_mm):
            selected = 'CONDITIONAL'
        elif rmse['STATIC'] - rmse['ADDITIVE_LAG_VELOCITY'] >= min_lopo_gain_mm:
            selected = 'ADDITIVE_LAG_VELOCITY'
        else:
            selected = 'STATIC'
        self.curves = _static(data, train)
        self.parameters = {model: _fit(data, train, model, self.curves) for model in MODELS}
        self.report = dict(schema=SCHEMA, ridge=RIDGE, selected_model=selected,
            threshold_source=threshold_source, training_manifest_sha256=training_manifest_sha256(data, self.panel),
            TRAIN_runs=48, DEV_runs=24, physical_TRAIN_parents=4, physical_DEV_parents=2,
            one_predeclared_probe_per_run=True, optimizer_updates=0, neural_model_loads=0,
            closed_form_ridge_solves=4 * (4 + 1), static_curve_fits=5, normalizer_fits=4 * (4 + 1),
            TRAIN_LOPO_RMSE_mm=rmse, TRAIN_LOPO_parent_MSE_mm2=errors, folds=audit,
            same_history_additive_comparator='ADDITIVE_LAG_VELOCITY',
            DEV_labels_read=False, TEST_read=False, native_gain_established=False,
            all_models_retained_including_negative=True, forecast_conditioning='PREDECLARED_PROGRAM_CONDITIONAL')
        return self

    def _ready(self):
        need(hasattr(self, 'parameters'), 'FIT_OR_RELOAD_FIRST')

    def predict(self, data, ids=None, *, model='SELECTED', candidate_theta=None):
        self._ready()
        ids = _ids(data, ids)
        need('split' in data and np.asarray(data['split']).shape == (len(data['history_state']),)
             and np.isin(np.asarray(data['split']), ('TRAIN', 'DEV')).all(),
             'EXPLICIT_OFFLINE_TRAIN_DEV_ROLES_REQUIRED')
        if np.any(np.asarray(data['split'])[ids] == 'DEV'):
            need(self._reload_verified, 'EXACT_TRAIN_SAVE_RELOAD_VERIFICATION_BEFORE_DEV')
        model = self.report['selected_model'] if model == 'SELECTED' else model
        if model == 'STATIC':
            levels = _theta(data, ids, candidate_theta)
            displacement = _base(self.curves, levels, model)
        else:
            need(model in MODELS, 'DECLARED_PREDICTOR')
            displacement = _displacement(data, ids, model, self.parameters[model], self.curves, candidate_theta)
        current = np.asarray(data['history_state'])[ids, -1, 86:89]
        prediction = current[:, None] + displacement
        need(np.isfinite(prediction).all(), 'FINITE_RESPONSE_FORECAST')
        return prediction

    def candidate_forecasts(self, data, ids=None, *, model='SELECTED'):
        return np.stack([self.predict(data, ids, model=model, candidate_theta=theta)
                         for theta in THETA_LEVELS], axis=1)

    def arrays(self):
        self._ready()
        result = dict(schema=np.asarray(SCHEMA), panel_json=np.asarray(json.dumps(self.panel, sort_keys=True)),
                      report_json=np.asarray(json.dumps(self.report, sort_keys=True, allow_nan=False)),
                      static_curves_m=self.curves.copy())
        for model, parameters in self.parameters.items():
            result.update({model + '__' + key: value.copy() for key, value in parameters.items()})
        return result

    def save(self, path):
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **self.arrays())
            stream.flush(); os.fsync(stream.fileno())

    @classmethod
    def from_arrays(cls, arrays):
        need(str(np.asarray(arrays['schema']).item()) == SCHEMA, 'EXACT_PORTABLE_MODEL_SCHEMA')
        result = cls(json.loads(str(np.asarray(arrays['panel_json']).item())))
        result.report = json.loads(str(np.asarray(arrays['report_json']).item()))
        need(result.report.get('ridge') == RIDGE and result.report.get('DEV_labels_read') is False
             and result.report.get('TEST_read') is False
             and result.report.get('selected_model') in ('STATIC', 'ADDITIVE_LAG_VELOCITY', 'CONDITIONAL'),
             'EXACT_FIXED_RIDGE_TRAIN_ONLY_SELECTION_METADATA')
        result.curves = np.asarray(arrays['static_curves_m'], float).copy()
        need(result.curves.shape == (3, 16, 3) and np.isfinite(result.curves).all(), 'FINITE_STATIC_CURVES')
        result.parameters = {}
        for model, dimension in DIMENSIONS.items():
            parameters = {}
            for key, shape in dict(mean=(dimension,), scale=(dimension,), weight=(dimension, 48)).items():
                value = np.asarray(arrays[model + '__' + key], float)
                need(value.shape == shape and np.isfinite(value).all(), 'EXACT_FINITE_MODEL_ARRAYS')
                parameters[key] = value.copy()
            need((parameters['scale'] > 0).all(), 'POSITIVE_OWN_TRAIN_NORMALIZER')
            result.parameters[model] = parameters
        return result

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            result = cls.from_arrays(archive)
        result._loaded_artifact_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return result

    def verify_reload(self, restored, data):
        self._ready(); restored._ready()
        need(restored is not self and restored._loaded_artifact_sha256 is not None,
             'DISTINCT_ACTUALLY_RELOADED_FILE_ARTIFACT_REQUIRED')
        before, after = self.arrays(), restored.arrays()
        need(before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before),
             'EXACT_SERIALIZED_ARRAYS_REQUIRED')
        train = validate_cohort(data, self.panel)['TRAIN']
        for model in ('STATIC',) + MODELS:
            need(np.array_equal(self.candidate_forecasts(data, train, model=model),
                                restored.candidate_forecasts(data, train, model=model)),
                 'EXACT_TRAIN_CANDIDATE_RELOAD_PREDICTIONS_REQUIRED')
        restored._reload_verified = True
        return dict(exact_arrays=True, TRAIN_candidate_max_error_m=0., DEV_evaluated=False,
                    optimizer_restore='NOT_APPLICABLE_NO_OPTIMIZER')


def ordinary_baselines(data, ids=None, *, candidate_theta=None):
    """Same-information deterministic forecasts, not executed feedback gains."""
    ids = _ids(data, ids)
    theta = _theta(data, ids, candidate_theta)
    nominal = np.asarray(data['nominal_H40'], float)[ids]
    basis = np.asarray(data['correction_basis_H40'], float)[ids]
    corrected = nominal[:, :, :3] + theta[:, None, None] * EPSILON_M * basis[:, :, None] * DIRECTION
    history = np.asarray(data['history_state'], float)[ids]
    current, previous = history[:, -1, 86:89], history[:, -2, 86:89]
    return dict(CURRENT_LAG_TRANSPORT=current[:, None] + corrected[:, 1:17] - corrected[:, :1],
                CONSTANT_VELOCITY=current[:, None] + np.arange(1, 17)[None, :, None] * (current - previous)[:, None],
                HOLD_CURRENT=np.repeat(current[:, None], 16, axis=1))


def forecast_metrics(prediction, truth):
    error = (np.asarray(prediction) - np.asarray(truth)) * 1000.
    need(error.ndim == 3 and error.shape[1:] == (16, 3) and np.isfinite(error).all(), 'FINITE_MATCHED_METRIC_ARRAYS')
    return dict(pooled_XYZ_coordinate_RMSE_mm=float(np.sqrt(np.mean(error ** 2))),
                pooled_XY_coordinate_RMSE_mm=float(np.sqrt(np.mean(error[..., :2] ** 2))),
                horizons={str(h): dict(XYZ_coordinate_RMSE_mm=float(np.sqrt(np.mean(error[:, h - 1] ** 2))),
                                      XY_coordinate_RMSE_mm=float(np.sqrt(np.mean(error[:, h - 1, :2] ** 2))))
                          for h in (1, 4, 8, 16)})
