"""A small causal, command-relative contact-event baseline; NumPy only.

This is a clipped linear probability/FIR forecaster, not a simulator or a new
controller. Each horizon has one ridge head. Its inputs are a strict subset of
the existing contact WM's nine actual observations and the selected program's
opening commands up to that horizon. There is no episode clock, phase, parent,
goal, close-start field, future observation or later command suffix in a head.

The future commands are PREDECLARED_PROGRAM_CONDITIONAL. They are not claimed
to be already installed, consumed, or an adaptive policy's unmade decisions.
"""

from pathlib import Path
import os

import numpy as np

from .pilot2_contact_model import HORIZONS, replay


SCHEMA = 'WM02_CONTACT_TIMING_FIR_NUMPY_V1'
INFORMATION = 'PREDECLARED_PROGRAM_CONDITIONAL'
ARMS = ('ACTION_CONDITIONED', 'NO_FUTURE_PROGRAM')
RIDGE = 1.0  # Fixed before the new diagnostic; no DEV hyperparameter search.
HISTORY_ROWS = 9
ROW_DIMENSION = 103
HISTORY_DIMENSION = HISTORY_ROWS * ROW_DIMENSION
PAST_FEATURES = HISTORY_ROWS * 10
FEATURE_RECIPE = 'NINE_OBSERVED_GRIPPER_ROWS_PLUS_WITHIN_H_COMMAND_INCREMENTS_V1'


def validate_knowledge(episodes):
    """Require the actual builder's static/selected-policy knowledge boundary.

    The SHA-bound builder has already checked the policy decision against its
    actual installed effective program. This check prevents a caller from
    moving the knowledge boundary earlier while reusing its episode contents.
    """
    for ep in episodes:
        schema = ep.get('declared_episode_schema')
        if schema == 'WM02_PILOT2_SUPPORTED_EPISODE_V1':
            if (ep.get('program_known_from_anchor') != 8
                    or ep.get('policy_decision_sha256') is not None
                    or ep['episode'].get('purpose') != 'SUPPORTED_CONTACT_COLLECTION'):
                raise ValueError('ACTUAL_STATIC_PROGRAM_KNOWLEDGE_BOUNDARY_REQUIRED')
        elif schema == 'WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1':
            pin = ep.get('policy_decision_sha256')
            if (ep.get('program_known_from_anchor') != 111
                    or not isinstance(pin, str) or len(pin) != 64
                    or any(c not in '0123456789abcdef' for c in pin)
                    or ep['episode'].get('purpose') != 'SUPPORTED_POLICY_EVALUATION'):
                raise ValueError('ACTUAL_PAUSED_POLICY_PROGRAM_KNOWLEDGE_BOUNDARY_REQUIRED')
        else:
            raise ValueError('BUILDER_RECORDED_STATIC_OR_SELECTED_PROGRAM_PROVENANCE_REQUIRED')
        for row in ep['samples']:
            fields = [('qpos', 34), ('qvel', 32), ('ctrl', 20), ('TCP_m', 3), ('finger_q', 2)]
            if (any(np.asarray(row[name]).shape != (count,) for name, count in fields)
                    or np.asarray(row['object_pose']).shape != (4, 4)
                    or np.asarray(row['actual_gripper_internal_memory']['values']['target']).shape != (2,)):
                raise ValueError('ORIGINAL_CONTACT_HISTORY_FIELD_OFFSETS_REQUIRED')


def contact_replay(episodes):
    validate_knowledge(episodes)
    data = replay(episodes)
    history = np.asarray(data['history'])
    if history.shape != (len(history), HISTORY_DIMENSION):
        raise ValueError('ACTUAL_NINE_BY103_CONTACT_HISTORY_LAYOUT_REQUIRED')
    if not np.array_equal(history.reshape(-1, HISTORY_ROWS, ROW_DIMENSION)[:, -1, 99:],
                          data['current_events']):
        raise ValueError('SAME_OBSERVED_CURRENT_EVENTS_REQUIRED')
    return data


def causal_features(history, program, horizon, arm):
    """Use only nine past rows and within-h selected future commands.

    Original state_feature layout: qpos34/qvel32/ctrl20/TCP3/objectXYZ3/
    fingers2/force1/internalTarget2/internalOpening1/relativeTTL1/events4.
    This baseline deliberately uses fingers/force/target/opening/events only.
    The event values are privileged native measurements, as in the old WM;
    this is not a camera-only deployment claim.
    """
    past, future = np.asarray(history, np.float64), np.asarray(program, np.float64)
    if (arm not in ARMS or type(horizon) is not int or horizon not in HORIZONS
            or past.ndim != 2 or past.shape[1] != HISTORY_DIMENSION
            or future.shape != (len(past), max(HORIZONS))
            or not np.isfinite(past).all() or not np.isfinite(future).all()):
        raise ValueError('FINITE_NINE_HISTORY_AND_DECLARED64_PROGRAM_REQUIRED')
    rows = past.reshape(-1, HISTORY_ROWS, ROW_DIMENSION)
    if (np.any(rows[:, :, 94] < 0)
            or np.any((rows[:, :, 99:] != 0) & (rows[:, :, 99:] != 1))):
        raise ValueError('OBSERVED_NONNEGATIVE_FORCE_AND_BINARY_EVENTS_REQUIRED')
    selected = np.concatenate((rows[:, :, 92:94] / .044,
                               np.log1p(rows[:, :, 94:95]),
                               rows[:, :, 95:97] / .044,
                               rows[:, :, 97:98], rows[:, :, 99:103]), axis=-1)
    # A blind head receives the original blind_plan meaning: repeat the actual
    # current internal opening. Its future increments are therefore all zero.
    opening = rows[:, -1, 97:98]
    within = future[:, :horizon] if arm == ARMS[0] else np.repeat(opening, horizon, 1)
    increments = np.diff(np.concatenate((opening, within), axis=1), axis=1)
    return np.concatenate((selected.reshape(len(past), PAST_FEATURES), increments), axis=1)


class ContactTimingFIR:
    """Fixed-ridge event forecaster, with TRAIN-only normalization per head."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('DECLARED_CONTACT_TIMING_ARM_REQUIRED')
        self.arm, self.heads = arm, {}

    def fit(self, data):
        if (not len(data['history']) or not np.all(np.asarray(data['split']) == 'TRAIN')
                or tuple(data['horizons']) != HORIZONS):
            raise ValueError('NONEMPTY_TRAIN_ONLY_REPLAY_REQUIRED_BEFORE_CONTACT_TIMING_FIT')
        for j, h in enumerate(HORIZONS):
            ids = np.flatnonzero(data['valid'][:, j])
            if not len(ids):
                raise ValueError('ACTUALLY_EXECUTED_TRAIN_LABELS_AT_EVERY_HORIZON_REQUIRED')
            x = causal_features(data['history'][ids], data['plan'][ids], h, self.arm)
            y = np.asarray(data['events'][ids, j], np.float64)
            if y.shape != (len(ids), 4) or not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
                raise ValueError('FOUR_ACTUAL_BINARY_TRAIN_EVENTS_REQUIRED')
            mean, scale = x.mean(0), np.maximum(x.std(0), 1e-4)
            z, base = (x - mean) / scale, y.mean(0)
            coef = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ (y - base))
            if not np.isfinite(coef).all():
                raise ValueError('FINITE_CLOSED_FORM_CONTACT_TIMING_FIT_REQUIRED')
            self.heads[h] = dict(mean=mean, scale=scale, coef=coef, base=base,
                                 train_labels=np.asarray(len(ids), dtype=np.int64))
        return self

    def predict(self, history, program, *, return_unclipped=False):
        if set(self.heads) != set(HORIZONS):
            raise ValueError('ALL_SIX_FROZEN_CONTACT_TIMING_HEADS_REQUIRED')
        values = []
        for h in HORIZONS:
            head = self.heads[h]
            x = causal_features(history, program, h, self.arm)
            values.append((x - head['mean']) / head['scale'] @ head['coef'] + head['base'])
        raw = np.stack(values, axis=1)
        if not np.isfinite(raw).all():
            raise ValueError('FINITE_CONTACT_EVENT_PROBABILITY_FORECAST_REQUIRED')
        # This is an explicitly declared probability link. No action/reference
        # or observed physical state is clipped or changed by this module.
        return raw if return_unclipped else np.clip(raw, 0., 1.)

    def arrays(self):
        if set(self.heads) != set(HORIZONS):
            raise ValueError('ALL_SIX_FROZEN_CONTACT_TIMING_HEADS_REQUIRED')
        result = dict(schema=np.asarray(SCHEMA), arm=np.asarray(self.arm), ridge=np.asarray(RIDGE),
                      horizons=np.asarray(HORIZONS, np.int64), information=np.asarray(INFORMATION),
                      feature_recipe=np.asarray(FEATURE_RECIPE))
        result.update({name + '_' + str(h): np.asarray(value).copy()
                       for h, head in self.heads.items() for name, value in head.items()})
        return result

    def save(self, path):
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **self.arrays())
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in archive.files}
        expected = {'schema', 'arm', 'ridge', 'horizons', 'information', 'feature_recipe'}
        expected.update(name + '_' + str(h) for h in HORIZONS
                        for name in ('mean', 'scale', 'coef', 'base', 'train_labels'))
        if (set(values) != expected or values['schema'].item() != SCHEMA
                or values['information'].item() != INFORMATION
                or values['feature_recipe'].item() != FEATURE_RECIPE
                or values['ridge'].shape != () or values['ridge'].item() != RIDGE
                or not np.array_equal(values['horizons'], HORIZONS)):
            raise ValueError('EXACT_CONTACT_TIMING_NUMPY_SCHEMA_REQUIRED')
        model = cls(values['arm'].item())
        for h in HORIZONS:
            size = PAST_FEATURES + h
            shapes = dict(mean=(size,), scale=(size,), coef=(size, 4), base=(4,), train_labels=())
            head = {name: values[name + '_' + str(h)] for name in shapes}
            if (any(head[name].shape != shape or not np.isfinite(head[name]).all()
                    for name, shape in shapes.items())
                    or np.any(head['scale'] < 1e-4) or np.any(head['base'] < 0)
                    or np.any(head['base'] > 1) or head['train_labels'].dtype.kind not in 'iu'
                    or head['train_labels'].item() < 1):
                raise ValueError('FINITE_SIX_TRAIN_FITTED_CONTACT_TIMING_HEADS_REQUIRED')
            model.heads[h] = head
        return model
