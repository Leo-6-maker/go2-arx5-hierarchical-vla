"""Full original 927-value contact history with horizon-local program inputs.

This independent recipe leaves the original 90-feature FIR unchanged. It shares
the old neural replay's observed history and blind current executed-reference
meaning, not its architecture, objective or fitted normalizer. Future commands
are predeclared program information, not installed/consumed future references.
"""

from pathlib import Path
import os

import numpy as np

from .contact_timing_baseline import (
    ARMS, HISTORY_DIMENSION, HISTORY_ROWS, HORIZONS, INFORMATION, RIDGE,
    ROW_DIMENSION, contact_replay as _original_contact_replay,
)


SCHEMA = 'WM02_CONTACT_TIMING_FULL927_NUMPY_V1'
FEATURE_RECIPE = 'FULL_927_HISTORY_WITH_WITHIN_H_PROGRAM_PREFIX_V1'
BLIND_SOURCE = 'ANCHOR_ACTUAL_EXECUTED_REFERENCE_EEF10_LAST_GRIPPER_COMPONENT'


def contact_replay(episodes):
    """Preserve the original replay and bind blind inputs to raw anchor rows."""
    data = _original_contact_replay(episodes)
    rows = {(ep['run_id'], row['native_end_step']): row
            for ep in episodes for row in ep['samples']}
    current = np.asarray([rows[(v['run_id'], v['anchor_native_step'])]
                          ['executed_reference_EEF10'][-1]
                          for v in data['identities']], dtype=data['blind_plan'].dtype)
    if not np.array_equal(data['blind_plan'], np.broadcast_to(current[:, None], data['plan'].shape)):
        raise ValueError('BLIND_PLAN_MUST_REPEAT_RAW_ANCHOR_EXECUTED_REFERENCE')
    data['anchor_executed_gripper'] = current
    return data


def full_history_features(history, program, horizon, arm, *, blind_plan, anchor_executed_gripper):
    """Return the unchanged 927 history values and only this horizon's prefix.

    The caller supplies the raw-anchor binding separately from candidate future
    programs. Even a nonconstant blind plan is rejected rather than silently
    treating its first value as a current observation. Internal memory.opening
    is retained as an observed history feature, not used to redefine blind_plan.
    """
    past, future = np.asarray(history, np.float64), np.asarray(program, np.float64)
    blind = np.asarray(blind_plan, np.float64)
    current = np.asarray(anchor_executed_gripper, np.float64)
    if (arm not in ARMS or type(horizon) is not int or horizon not in HORIZONS
            or past.ndim != 2 or past.shape[1] != HISTORY_DIMENSION
            or future.shape != (len(past), max(HORIZONS))
            or blind.shape != future.shape or current.shape != (len(past),)
            or not np.isfinite(past).all() or not np.isfinite(current).all()
            or not np.array_equal(blind, np.broadcast_to(current[:, None], blind.shape))):
        raise ValueError('ORIGINAL927_HISTORY_AND_CONSTANT_RAW_ANCHOR_BOUND_BLIND_PLAN_REQUIRED')
    rows = past.reshape(-1, HISTORY_ROWS, ROW_DIMENSION)
    if (np.any(rows[:, :, 94] < 0)
            or np.any((rows[:, :, 99:] != 0) & (rows[:, :, 99:] != 1))):
        raise ValueError('ORIGINAL_OBSERVED_NONNEGATIVE_FORCE_AND_BINARY_EVENTS_REQUIRED')
    within = (future if arm == ARMS[0] else blind)[:, :horizon]
    if not np.isfinite(within).all():
        raise ValueError('FINITE_WITHIN_HORIZON_PROGRAM_PREFIX_REQUIRED')
    return np.concatenate((past, within), axis=1)


def features(data, horizon, arm, ids=None):
    selected = slice(None) if ids is None else ids
    return full_history_features(data['history'][selected], data['plan'][selected], horizon, arm,
        blind_plan=data['blind_plan'][selected],
        anchor_executed_gripper=data['anchor_executed_gripper'][selected])


class ContactTimingFullHistory:
    """Fixed ridge=1, four-event heads with own valid-TRAIN normalization."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('DECLARED_FULL_HISTORY_CONTACT_ARM_REQUIRED')
        self.arm, self.heads = arm, {}

    def fit(self, data):
        if (not len(data['history']) or not np.all(np.asarray(data['split']) == 'TRAIN')
                or tuple(data['horizons']) != HORIZONS):
            raise ValueError('NONEMPTY_TRAIN_ONLY_FULL_HISTORY_REPLAY_REQUIRED')
        for j, h in enumerate(HORIZONS):
            ids = np.flatnonzero(data['valid'][:, j])
            if not len(ids):
                raise ValueError('ACTUALLY_EXECUTED_TRAIN_LABELS_AT_EVERY_HORIZON_REQUIRED')
            x = features(data, h, self.arm, ids)
            y = np.asarray(data['events'][ids, j], np.float64)
            if y.shape != (len(ids), 4) or not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
                raise ValueError('FOUR_ACTUAL_BINARY_TRAIN_EVENTS_REQUIRED')
            mean, scale = x.mean(0), np.maximum(x.std(0), 1e-4)
            z, base = (x - mean) / scale, y.mean(0)
            # Equivalent primal/dual ridge; one solve per head. Tiny fixtures
            # need not factor a 991-square matrix. Real TRAIN uses the primal.
            if z.shape[1] <= len(z):
                coef = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ (y - base))
            else:
                coef = z.T @ np.linalg.solve(z @ z.T + RIDGE * np.eye(len(z)), y - base)
            if not np.isfinite(coef).all():
                raise ValueError('FINITE_FULL_HISTORY_CLOSED_FORM_FIT_REQUIRED')
            self.heads[h] = dict(mean=mean, scale=scale, coef=coef, base=base,
                                 train_labels=np.asarray(len(ids), dtype=np.int64))
        return self

    def predict(self, data, *, return_unclipped=False):
        if set(self.heads) != set(HORIZONS):
            raise ValueError('ALL_SIX_FULL_HISTORY_FROZEN_HEADS_REQUIRED')
        values = []
        for h in HORIZONS:
            head = self.heads[h]
            x = features(data, h, self.arm)
            values.append((x - head['mean']) / head['scale'] @ head['coef'] + head['base'])
        raw = np.stack(values, axis=1)
        if not np.isfinite(raw).all():
            raise ValueError('FINITE_FULL_HISTORY_EVENT_FORECAST_REQUIRED')
        return raw if return_unclipped else np.clip(raw, 0., 1.)

    def arrays(self):
        if set(self.heads) != set(HORIZONS):
            raise ValueError('ALL_SIX_FULL_HISTORY_FROZEN_HEADS_REQUIRED')
        result = dict(schema=np.asarray(SCHEMA), arm=np.asarray(self.arm), ridge=np.asarray(RIDGE),
            horizons=np.asarray(HORIZONS, np.int64), information=np.asarray(INFORMATION),
            feature_recipe=np.asarray(FEATURE_RECIPE), blind_source=np.asarray(BLIND_SOURCE))
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
        expected = {'schema', 'arm', 'ridge', 'horizons', 'information', 'feature_recipe', 'blind_source'}
        expected.update(name + '_' + str(h) for h in HORIZONS
                        for name in ('mean', 'scale', 'coef', 'base', 'train_labels'))
        if (set(values) != expected or values['schema'].item() != SCHEMA
                or values['information'].item() != INFORMATION
                or values['feature_recipe'].item() != FEATURE_RECIPE
                or values['blind_source'].item() != BLIND_SOURCE
                or values['ridge'].shape != () or values['ridge'].item() != RIDGE
                or not np.array_equal(values['horizons'], HORIZONS)):
            raise ValueError('EXACT_FULL927_CONTACT_NUMPY_SCHEMA_REQUIRED')
        model = cls(values['arm'].item())
        for h in HORIZONS:
            size = HISTORY_DIMENSION + h
            shapes = dict(mean=(size,), scale=(size,), coef=(size, 4), base=(4,), train_labels=())
            head = {name: values[name + '_' + str(h)] for name in shapes}
            if (any(head[name].shape != shape or not np.isfinite(head[name]).all()
                    for name, shape in shapes.items())
                    or np.any(head['scale'] < 1e-4) or np.any(head['base'] < 0)
                    or np.any(head['base'] > 1) or head['train_labels'].dtype.kind not in 'iu'
                    or head['train_labels'].item() < 1):
                raise ValueError('FINITE_SIX_VALID_TRAIN_FULL_HISTORY_HEADS_REQUIRED')
            model.heads[h] = head
        return model
