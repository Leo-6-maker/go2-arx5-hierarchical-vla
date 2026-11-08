"""TRAIN-only structured H16 TCP displacement regression, using NumPy only.

This is a new offline hypothesis, not the frozen ARX/MLP experiment or native
admission. Absolute TCP predictions are anchored at the observed decision TCP.
The action-conditioned static base is TRAIN HOLD drift plus an explicit
action-minus-HOLD response. The blind base is the equal-action TRAIN marginal
response, rather than an assertion that an unknown future action is HOLD.

Only robot prefix summaries, actual past-reference errors and observed decision
clocks enter regression. Future states are targets; future goals, future clocks,
object state, parent identity and absolute world translation are never features.
Ridge is selected on whole-parent TRAIN LOPO predictions. Every fold refits its
static curves and normalizer without its held parent. A positive meaningful-gain
threshold must be fixed without DEV before fit; otherwise the model falls back
to its static base. Prediction accuracy is not native control benefit.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np


ARMS = ('ACTION_CONDITIONED', 'NO_FUTURE_ACTION')
RIDGE_GRID = (.1, 1., 10., 100., 1000., np.inf)
SCHEMA = 'WM02_FUSION_STRUCTURED_NUMPY_V1'
# The existing qualified discrete domain; this module never compiles references.
OFFSETS_M = np.array([[0., 0., 0.], [.001, 0., 0.],
                      [-.001, 0., 0.], [0., .001, 0.]])
_FIELDS = (('qpos', 34), ('qvel', 32), ('ctrl', 20),
           ('tcp_position_m', 3), ('tcp_rotation6d', 6),
           ('object_position_m', 3), ('object_rotation6d', 6),
           ('finger_position_m', 2))


def _layout(meta):
    fields = tuple((field['name'], field['size']) for field in meta['state_fields'])
    cubes = [joint for joint in meta['joints'] if joint['name'] == 'task_cube_free']
    if (fields != _FIELDS or len(cubes) != 1
            or (cubes[0]['qpos_address'], cubes[0]['qvel_address']) != (27, 26)):
        raise ValueError('ACTUAL_106_STATE_ROBOT_OBJECT_BOUNDARY_REQUIRED')
    result, offset = {}, 0
    for name, size in fields:
        result[name] = slice(offset, offset + size)
        offset += size
    return result


def validate_cohort(data, meta):
    """Check the original balanced cohort without reading DEV future labels.

    This does not authenticate physical evidence, hashes, leases or admission.
    Those remain the caller's responsibility under the original replay contract.
    """
    _layout(meta)
    shapes = dict(state=(48, 25, 106), goal=(48, 17),
                  history_goal=(48, 8, 17), source_age=(48, 25),
                  source_age_valid=(48, 25), future_prefix_remaining=(48,),
                  window_start_s=(48,), parent=(48,), split=(48,),
                  observed_action=(48,), repeat_index=(48,), run_id=(48,))
    if any(np.asarray(data[name]).shape != shape for name, shape in shapes.items()):
        raise ValueError('COMPLETE_BALANCED_48_WINDOW_COHORT_REQUIRED')
    parents = {}
    for parent, split in zip(data['parent'], data['split']):
        if parent in parents and parents[parent] != split:
            raise ValueError('WHOLE_PARENT_TRAIN_DEV_SPLIT_REQUIRED')
        parents[parent] = split
    if Counter(parents.values()) != dict(TRAIN=4, DEV=2):
        raise ValueError('FOUR_TRAIN_TWO_DEV_PARENTS_NO_TEST_REQUIRED')
    action, repeat = np.asarray(data['observed_action']), np.asarray(data['repeat_index'])
    if (action.dtype.kind not in 'iu' or repeat.dtype.kind not in 'iu'
            or np.any(action < 0) or np.any(action > 3)
            or np.any(repeat < 0) or np.any(repeat > 1)
            or len(set(data['run_id'])) != 48):
        raise ValueError('UNIQUE_RUNS_AND_QUALIFIED_ACTION_RESTART_IDS_REQUIRED')
    expected = Counter((a, r) for a in range(4) for r in range(2))
    for parent in parents:
        ids = np.flatnonzero(np.asarray(data['parent']) == parent)
        if Counter(zip(action[ids], repeat[ids])) != expected:
            raise ValueError('ALL_ACTIONS_TWO_RESTARTS_PER_WHOLE_PARENT_REQUIRED')
    past, goal = np.asarray(data['history_goal']), np.asarray(data['goal'])
    age, valid = np.asarray(data['source_age']), np.asarray(data['source_age_valid'])
    if (not np.isfinite(past).all() or not np.isfinite(goal).all()
            or not np.array_equal(past, np.repeat(past[:, :1], 8, axis=1))
            or not np.array_equal(goal[:, 3:], past[:, 0, 3:])
            or not np.allclose(goal[:, :3] - past[:, 0, :3], OFFSETS_M[action],
                               atol=1e-10, rtol=0)
            or valid.dtype.kind != 'b' or not valid[:, 8].all()
            or valid[:, :8].any() or valid[:, 9:].any()
            or not np.isfinite(age[:, 8]).all()
            or np.any(age[:, 8] < -1e-8) or np.any(age[:, 8] >= .4)
            or not np.all(np.asarray(data['future_prefix_remaining']) == 7.)
            or not np.all(np.asarray(data['window_start_s']) == 0.)):
        raise ValueError('ACTUAL_PAST_REFERENCE_EXECUTED_ACTION_AND_CLOCK_REQUIRED')
    return {split: np.flatnonzero(np.asarray(data['split']) == split)
            for split in ('TRAIN', 'DEV')}


def whole_parent_folds(data, meta):
    """Return four deterministic TRAIN-only (held parent, fit rows, held rows)."""
    train = validate_cohort(data, meta)['TRAIN']
    parents = np.asarray(data['parent'])
    return [(str(parent), train[parents[train] != parent], train[parents[train] == parent])
            for parent in sorted(set(parents[train]))]


def _ids(data, ids):
    n = len(data['state'])
    value = np.arange(n) if ids is None else np.asarray(ids)
    if value.ndim != 1 or value.dtype.kind not in 'iu' or np.any(value < 0) or np.any(value >= n):
        raise ValueError('VALID_WINDOW_ROW_IDS_REQUIRED')
    return value


def _actions(data, ids, candidate_actions):
    value = np.asarray(data['observed_action'])[ids] if candidate_actions is None else np.asarray(candidate_actions)
    if value.shape == ():
        value = np.full(len(ids), value.item(), dtype=value.dtype)
    if value.shape != (len(ids),) or value.dtype.kind not in 'iu' or np.any(value < 0) or np.any(value > 3):
        raise ValueError('QUALIFIED_FOUR_ACTION_CANDIDATES_REQUIRED')
    return value


def causal_features(data, meta, ids=None, *, candidate_actions=None,
                    arm='ACTION_CONDITIONED'):
    """177 causal scalars: robot summaries, relative motion/reference and clocks.

    Robot summaries retain current and mean past robot state, removing root XYZ
    and object qpos/qvel. TCP velocities use only observed prefix differences.
    Past references enter as TCP errors, never as absolute world positions.
    Candidate indicators are zero in the blind arm; its static base is also blind.
    """
    if arm not in ARMS:
        raise ValueError('STRUCTURED_RESPONSE_ARM_REQUIRED')
    layout = _layout(meta)
    ids = _ids(data, ids)
    prefix = np.asarray(np.asarray(data['state'])[ids, :9], float)
    past = np.asarray(np.asarray(data['history_goal'])[ids], float)
    if prefix.shape != (len(ids), 9, 106) or past.shape != (len(ids), 8, 17):
        raise ValueError('NINE_OBSERVED_STATES_EIGHT_PAST_REFERENCES_REQUIRED')
    columns = list(range(3, 27)) + list(range(34, 60))
    for name in ('ctrl', 'tcp_rotation6d', 'finger_position_m'):
        columns += list(range(layout[name].start, layout[name].stop))
    robot = prefix[:, :, columns]
    tcp = prefix[:, :, layout['tcp_position_m']]
    errors = (past[:, :, :3] - tcp[:, :8]) / .001
    age = np.asarray(data['source_age'], float)[ids, 8:9]
    valid = np.asarray(data['source_age_valid'])[ids, 8]
    remaining = np.asarray(data['future_prefix_remaining'], float)[ids, None]
    origin = np.asarray(data['window_start_s'], float)[ids, None]
    if not valid.all():
        raise ValueError('OBSERVED_DECISION_SOURCE_AGE_REQUIRED')
    actions = _actions(data, ids, candidate_actions)
    indicators = np.zeros((len(ids), 3))
    if arm == 'ACTION_CONDITIONED':
        indicators = (actions[:, None] == np.arange(1, 4)).astype(float)
    result = np.concatenate((robot[:, 8], robot[:, :8].mean(1),
                             (tcp[:, 8] - tcp[:, 7]) / .001,
                             (tcp[:, 8] - tcp[:, 0]) / (.001 * 8),
                             errors[:, -1], errors.mean(1),
                             (tcp[:, 8] - prefix[:, 8, :3]) / .001,
                             age, remaining / 7., origin, indicators), axis=1)
    if not np.isfinite(result).all():
        raise ValueError('FINITE_CAUSAL_ROBOT_PREFIX_REQUIRED')
    return result


def _targets(data, meta, ids):
    layout = _layout(meta)
    state = np.asarray(data['state'])
    value = (np.asarray(state[ids, 9:, layout['tcp_position_m']], float)
             - np.asarray(state[ids, 8:9, layout['tcp_position_m']], float))
    if value.shape != (len(ids), 16, 3) or not np.isfinite(value).all():
        raise ValueError('FINITE_ACTUALLY_OBSERVED_TRAIN_H16_TCP_TARGETS_REQUIRED')
    return value


def _static(data, meta, ids):
    if not len(ids) or not np.all(np.asarray(data['split'])[ids] == 'TRAIN'):
        raise ValueError('STATIC_CURVES_REQUIRE_TRAIN_ONLY')
    target = _targets(data, meta, ids)
    actions = np.asarray(data['observed_action'])[ids]
    curves = np.stack([target[actions == action].mean(0) for action in range(4)])
    if not np.isfinite(curves).all():
        raise ValueError('EVERY_QUALIFIED_ACTION_REQUIRED_IN_TRAIN_FOLD')
    return dict(hold_curve_m=curves[0], action_response_m=curves - curves[0],
                marginal_curve_m=curves.mean(0))


def _base(parameters, actions, arm):
    if arm == 'NO_FUTURE_ACTION':
        return np.broadcast_to(parameters['marginal_curve_m'], (len(actions), 16, 3)).copy()
    return parameters['hold_curve_m'][None] + parameters['action_response_m'][actions]


def _fit_fold(data, meta, ids, arm):
    x = causal_features(data, meta, ids, arm=arm)
    parameters = _static(data, meta, ids)
    parameters['mean'] = x.mean(0)
    parameters['scale'] = np.maximum(x.std(0), 1e-6)
    z = (x - parameters['mean']) / parameters['scale']
    residual_mm = (_targets(data, meta, ids) - _base(parameters, _actions(data, ids, None), arm)).reshape(len(ids), 48) * 1000.
    return parameters, z, residual_mm


def _ridge(z, residual_mm, ridge):
    if np.isinf(ridge):
        return np.zeros((z.shape[1], 48))
    # Sample-space solve keeps every LOPO solve at only 24 x 24.
    return z.T @ np.linalg.solve(z @ z.T + ridge * np.eye(len(z)), residual_mm)


class StructuredResponse:
    """Direct H16 TCP XYZ displacement model; no complete-state prediction."""

    def __init__(self, meta, arm='ACTION_CONDITIONED'):
        _layout(meta)
        if arm not in ARMS:
            raise ValueError('STRUCTURED_RESPONSE_ARM_REQUIRED')
        self.meta = json.loads(json.dumps(meta))
        self.arm = arm

    def fit(self, data, *, min_lopo_gain_mm):
        """Fit once from the original TRAIN cohort, never evaluating DEV targets.

        ``min_lopo_gain_mm`` is the predeclared minimum improvement in pooled
        held-TRAIN H16 XYZ RMSE. It must be positive and fixed without DEV. This
        fit never derives a threshold from DEV or silently substitutes a zero.
        Finite ties prefer the larger ridge; gain below threshold selects INF.
        """
        if not np.isfinite(min_lopo_gain_mm) or min_lopo_gain_mm <= 0:
            raise ValueError('POSITIVE_PREDECLARED_TRAIN_ONLY_GAIN_THRESHOLD_REQUIRED')
        split = validate_cohort(data, self.meta)
        squared_errors = np.zeros((len(RIDGE_GRID), 4))
        audit = []
        for fold, (held_parent, train, held) in enumerate(whole_parent_folds(data, self.meta)):
            parameters, z, residual = _fit_fold(data, self.meta, train, self.arm)
            held_x = causal_features(data, self.meta, held, arm=self.arm)
            held_z = (held_x - parameters['mean']) / parameters['scale']
            base = _base(parameters, _actions(data, held, None), self.arm)
            target = _targets(data, self.meta, held)
            for index, ridge in enumerate(RIDGE_GRID):
                prediction = base + (held_z @ _ridge(z, residual, ridge)).reshape(-1, 16, 3) * .001
                squared_errors[index, fold] = np.mean(((prediction - target) * 1000.) ** 2)
            audit.append(dict(held_parent=held_parent,
                              train_parents=sorted(set(map(str, np.asarray(data['parent'])[train]))),
                              train_run_ids=list(map(str, np.asarray(data['run_id'])[train])),
                              fit_rows=int(len(train)), held_rows=int(len(held)),
                              **{name: value.tolist() for name, value in parameters.items()}))
        errors = np.sqrt(squared_errors.mean(1))
        # Reversing before argmin resolves an exact tie toward stronger ridge.
        best = len(RIDGE_GRID) - 2 - int(np.argmin(errors[-2::-1]))
        gain = float(errors[-1] - errors[best])
        selected = best if gain >= min_lopo_gain_mm else len(RIDGE_GRID) - 1
        self.ridge = RIDGE_GRID[selected]
        self.parameters, z, residual = _fit_fold(data, self.meta, split['TRAIN'], self.arm)
        self.parameters['weight'] = _ridge(z, residual, self.ridge)
        self.selection_report = dict(
            method='WHOLE_PARENT_TRAIN_LOPO_POOLED_H16_XYZ_RMSE',
            min_lopo_gain_mm=float(min_lopo_gain_mm),
            ridge_grid=[str(ridge) for ridge in RIDGE_GRID],
            lopo_rmse_mm=errors.tolist(), parent_mse_mm2=squared_errors.tolist(),
            best_finite_gain_mm=gain, selected_ridge=str(self.ridge),
            fallback_to_static=bool(np.isinf(self.ridge)), DEV_targets_read=False,
            TEST_read=False, folds=audit)
        return self

    def _ready(self):
        if not hasattr(self, 'parameters'):
            raise ValueError('FIT_OR_RELOAD_STRUCTURED_RESPONSE_FIRST')

    def predict_displacement(self, data, ids=None, *, candidate_actions=None):
        self._ready()
        ids = _ids(data, ids)
        actions = _actions(data, ids, candidate_actions)
        x = causal_features(data, self.meta, ids, candidate_actions=actions, arm=self.arm)
        z = (x - self.parameters['mean']) / self.parameters['scale']
        result = _base(self.parameters, actions, self.arm) + (z @ self.parameters['weight']).reshape(-1, 16, 3) * .001
        if not np.isfinite(result).all():
            raise FloatingPointError('NONFINITE_RESPONSE_NO_CLIPPING')
        return result

    def predict(self, data, ids=None, *, candidate_actions=None):
        ids = _ids(data, ids)
        anchor = np.asarray(np.asarray(data['state'])[ids, 8:9,
                            _layout(self.meta)['tcp_position_m']], float)
        return anchor + self.predict_displacement(data, ids, candidate_actions=candidate_actions)

    def candidate_forecasts(self, data, ids=None):
        """Return [window, four candidates, H16, XYZ], preserving the same past."""
        ids = _ids(data, ids)
        return np.stack([self.predict(data, ids, candidate_actions=action)
                         for action in range(4)], axis=1)

    def arrays(self):
        self._ready()
        return dict(schema=np.asarray(SCHEMA), arm=np.asarray(self.arm),
                    meta_json=np.asarray(json.dumps(self.meta, sort_keys=True)),
                    ridge=np.asarray(self.ridge),
                    selection_json=np.asarray(json.dumps(self.selection_report, sort_keys=True, allow_nan=False)),
                    **{name: value.copy() for name, value in self.parameters.items()})

    def save(self, path):
        """Write a new portable artifact without pickle or overwriting evidence."""
        arrays = self.arrays()
        with Path(path).open('xb') as stream:
            np.savez(stream, **arrays)

    @classmethod
    def from_arrays(cls, arrays):
        if str(np.asarray(arrays['schema']).item()) != SCHEMA:
            raise ValueError('STRUCTURED_RESPONSE_SCHEMA_DIFFERS')
        model = cls(json.loads(str(np.asarray(arrays['meta_json']).item())),
                    str(np.asarray(arrays['arm']).item()))
        # Two 78-value robot summaries + 12 motion/reference + 3 relative TCP
        # + three clocks + three action indicators = 177 input features.
        shapes = dict(mean=(177,), scale=(177,), weight=(177, 48),
                      hold_curve_m=(16, 3), action_response_m=(4, 16, 3),
                      marginal_curve_m=(16, 3))
        model.parameters = {}
        for name, shape in shapes.items():
            value = np.asarray(arrays[name], float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('STRUCTURED_RESPONSE_FINITE_ARRAY_LAYOUT_REQUIRED')
            model.parameters[name] = value.copy()
        p = model.parameters
        if (not (p['scale'] > 0).all() or np.any(p['action_response_m'][0] != 0)
                or not np.allclose(p['marginal_curve_m'], p['hold_curve_m'] + p['action_response_m'].mean(0), atol=1e-14, rtol=0)):
            raise ValueError('STRUCTURED_RESPONSE_STATIC_OR_SCALE_INVALID')
        model.ridge = float(np.asarray(arrays['ridge']).item())
        if model.ridge not in RIDGE_GRID or (np.isinf(model.ridge) and np.any(p['weight'] != 0)):
            raise ValueError('STRUCTURED_RESPONSE_RIDGE_OR_FALLBACK_INVALID')
        model.selection_report = json.loads(str(np.asarray(arrays['selection_json']).item()))
        if (model.selection_report['selected_ridge'] != str(model.ridge)
                or model.selection_report['DEV_targets_read'] is not False
                or model.selection_report['TEST_read'] is not False):
            raise ValueError('STRUCTURED_RESPONSE_SELECTION_METADATA_DIFFERS')
        return model

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls.from_arrays(arrays)
