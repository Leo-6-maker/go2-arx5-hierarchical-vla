"""Synthetic schema fixtures only; no physical/native qualification is claimed."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from go2arx5_vla.world_model.fusion_structured import (
    OFFSETS_M, RIDGE_GRID, StructuredResponse, causal_features,
    validate_cohort, whole_parent_folds,
)


def synthetic_cohort(*, structured=True):
    """Original 48-window schema, deliberately synthetic causal response curves."""
    fields = [('qpos', 34), ('qvel', 32), ('ctrl', 20),
              ('tcp_position_m', 3), ('tcp_rotation6d', 6),
              ('object_position_m', 3), ('object_rotation6d', 6),
              ('finger_position_m', 2)]
    meta = dict(state_fields=[dict(name=name, size=size) for name, size in fields],
                joints=[dict(name='task_cube_free', qpos_address=27, qvel_address=26)])
    actions = np.tile(np.repeat(np.arange(4), 2), 6)
    repeats = np.tile(np.arange(2), 24)
    parent = np.repeat(['T0', 'T1', 'T2', 'T3', 'D0', 'D1'], 8)
    split = np.repeat(['TRAIN'] * 4 + ['DEV'] * 2, 8)
    state = np.zeros((48, 25, 106))
    history_goal = np.zeros((48, 8, 17))
    h = np.arange(1, 17)[:, None] / 16.
    for row in range(48):
        p = row // 8
        factor = (p - 1.5) if structured else 0.
        root = np.array([p * .001, -p * .002, .3])
        anchor = root + [.45, repeats[row] * .00001, .2]
        velocity = np.array([factor * .000001, -.0000005, 0.])
        state[row, :, :3] = root
        state[row, :, 3] = 1.
        state[row, :, 7] = p - 1.5
        state[row, :, 66] = (p - 1.5) * .3
        state[row, :, 89:95] = [1., 0., 0., 0., 1., 0.]
        state[row, :9, 86:89] = anchor - (8 - np.arange(9)[:, None]) * velocity
        hold = h * np.array([.0004, -.0002, .00003])
        response = h * OFFSETS_M[actions[row]] * .65
        residual = h * np.array([factor * .00018, factor * .00007, 0.])
        state[row, 9:, 86:89] = anchor + hold + response + residual
        history_goal[row, :, :3] = anchor + [.00002, -.00003, 0.]
        history_goal[row, :, 3:9] = [1., 0., 0., 0., 1., 0.]
        history_goal[row, :, 10:13] = root
        # Excluded object channels deliberately vary independently of responses.
        state[row, :, 27:34] = row * .01
        state[row, :, 60:66] = row * .02
        state[row, :, 95:104] = row * .03
    goal = history_goal[:, 0].copy()
    goal[:, :3] += OFFSETS_M[actions]
    valid = np.zeros((48, 25), bool)
    valid[:, 8] = True
    data = dict(state=state, goal=goal, history_goal=history_goal,
                source_age=np.zeros((48, 25)), source_age_valid=valid,
                future_prefix_remaining=np.full(48, 7.), window_start_s=np.zeros(48),
                observed_action=actions, repeat_index=repeats, parent=parent,
                split=split, run_id=np.array([f'SYNTHETIC_ONLY_{i}' for i in range(48)]))
    return meta, data


def translated(data, shift):
    result = copy.deepcopy(data)
    # World root/TCP/object positions and their past/installed references move
    # together. Body-relative velocities, rotations and joints stay unchanged.
    for sl in (slice(0, 3), slice(27, 30), slice(86, 89), slice(95, 98)):
        result['state'][:, :, sl] += shift
    for name in ('goal', 'history_goal'):
        result[name][..., :3] += shift
        result[name][..., 10:13] += shift
    return result


class StructuredLearning(unittest.TestCase):
    def test_whole_parent_train_folds_and_complete_original_schema(self):
        meta, data = synthetic_cohort()
        self.assertEqual(len(validate_cohort(data, meta)['TRAIN']), 32)
        folds = whole_parent_folds(data, meta)
        self.assertEqual([held for held, _, _ in folds], ['T0', 'T1', 'T2', 'T3'])
        for held, train, test in folds:
            self.assertEqual((len(train), len(test)), (24, 8))
            self.assertTrue(np.all(data['split'][train] == 'TRAIN'))
            self.assertNotIn(held, data['parent'][train])
            self.assertEqual(set(data['parent'][test]), {held})
            self.assertFalse(set(train) & set(test))
        for field, row, value in [('split', 0, 'DEV'), ('split', 0, 'TEST'),
                                  ('repeat_index', 1, 0), ('observed_action', 0, 4)]:
            bad = copy.deepcopy(data)
            bad[field][row] = value
            with self.assertRaises(ValueError):
                validate_cohort(bad, meta)
        bad_meta = copy.deepcopy(meta)
        bad_meta['joints'][0]['qvel_address'] = 25
        with self.assertRaises(ValueError):
            StructuredResponse(bad_meta)

    def test_translation_equivariant_features_fit_and_anchored_predictions(self):
        meta, data = synthetic_cohort()
        shift = np.array([.27, -.31, .19])
        moved = translated(data, shift)
        np.testing.assert_allclose(causal_features(data, meta), causal_features(moved, meta),
                                   atol=2e-10, rtol=0)
        original = StructuredResponse(meta).fit(data, min_lopo_gain_mm=.00001)
        clone = StructuredResponse(meta).fit(moved, min_lopo_gain_mm=.00001)
        self.assertEqual(original.ridge, clone.ridge)
        np.testing.assert_allclose(original.predict(moved), original.predict(data) + shift,
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(clone.predict(moved), original.predict(data) + shift,
                                   atol=1e-12, rtol=0)

    def test_future_labels_clocks_candidate_goal_and_privileged_objects_are_not_inputs(self):
        meta, data = synthetic_cohort()
        model = StructuredResponse(meta).fit(data, min_lopo_gain_mm=.00001)
        changed = copy.deepcopy(data)
        changed['state'][:, 9:] = np.nan
        changed['source_age'][:, :8] = np.nan
        changed['source_age'][:, 9:] = np.nan
        changed['source_age_valid'][:, 9:] = True
        changed['goal'][:] = np.nan
        for sl in (slice(27, 34), slice(60, 66), slice(95, 104)):
            changed['state'][:, :9, sl] = 1e20
        changed['parent'][:] = 'ignored'
        changed['repeat_index'][:] = 999
        changed['run_id'][:] = 'ignored'
        np.testing.assert_array_equal(causal_features(data, meta), causal_features(changed, meta))
        np.testing.assert_array_equal(model.candidate_forecasts(data), model.candidate_forecasts(changed))
        changed = copy.deepcopy(data)
        changed['history_goal'][:, :, 0] += .0001
        self.assertFalse(np.array_equal(causal_features(data, meta), causal_features(changed, meta)))
        changed = copy.deepcopy(data)
        changed['state'][:, 0, 66] += .01
        self.assertFalse(np.array_equal(causal_features(data, meta), causal_features(changed, meta)))

    def test_blind_removes_candidate_from_features_and_static_base_with_fair_marginal(self):
        meta, data = synthetic_cohort()
        model = StructuredResponse(meta, 'NO_FUTURE_ACTION').fit(data, min_lopo_gain_mm=100.)
        all_predictions = model.candidate_forecasts(data)
        for action in range(1, 4):
            np.testing.assert_array_equal(all_predictions[:, 0], all_predictions[:, action])
        changed = copy.deepcopy(data)
        changed['observed_action'] = (changed['observed_action'] + 1) % 4
        changed['goal'][:] = 1e20
        np.testing.assert_array_equal(model.predict(data), model.predict(changed))
        for action in range(4):
            np.testing.assert_array_equal(causal_features(data, meta, candidate_actions=action,
                                                         arm='NO_FUTURE_ACTION'),
                                          causal_features(data, meta, candidate_actions=0,
                                                          arm='NO_FUTURE_ACTION'))
        p = model.parameters
        expected = p['hold_curve_m'] + p['action_response_m'].mean(0)
        np.testing.assert_allclose(p['marginal_curve_m'], expected, atol=1e-18, rtol=0)
        np.testing.assert_allclose(model.predict_displacement(data),
                                   np.broadcast_to(expected, (48, 16, 3)), atol=1e-18, rtol=0)
        self.assertGreater(np.max(np.abs(expected - p['hold_curve_m'])), 1e-5)
        ac = StructuredResponse(meta).fit(data, min_lopo_gain_mm=100.)
        self.assertGreater(np.max(np.abs(ac.candidate_forecasts(data)[:, 1]
                                        - ac.candidate_forecasts(data)[:, 0])), .0001)

    def test_train_only_selection_and_every_fold_refits_base_and_normalizer(self):
        meta, data = synthetic_cohort()
        model = StructuredResponse(meta).fit(data, min_lopo_gain_mm=.00001)
        changed_dev = copy.deepcopy(data)
        changed_dev['state'][32:, 9:] = np.nan
        changed_dev['state'][32:, :9, 7:27] += 1e10
        changed_dev['state'][32:, :9, 66:86] -= 1e10
        other = StructuredResponse(meta).fit(changed_dev, min_lopo_gain_mm=.00001)
        for name, value in model.parameters.items():
            np.testing.assert_array_equal(value, other.parameters[name])
        self.assertEqual(model.selection_report, other.selection_report)
        folds = whole_parent_folds(data, meta)
        for audit, (parent, train, held) in zip(model.selection_report['folds'], folds):
            self.assertEqual(audit['held_parent'], parent)
            self.assertNotIn(parent, audit['train_parents'])
            self.assertEqual(set(audit['train_run_ids']), set(data['run_id'][train]))
            x = causal_features(data, meta, train)
            np.testing.assert_array_equal(audit['mean'], x.mean(0))
            np.testing.assert_array_equal(audit['scale'], np.maximum(x.std(0), 1e-6))
            holds = train[data['observed_action'][train] == 0]
            displacement = data['state'][holds, 9:, 86:89] - data['state'][holds, 8:9, 86:89]
            np.testing.assert_array_equal(audit['hold_curve_m'], displacement.mean(0))
            # A held-parent perturbation changes that fold's evaluation but
            # cannot change either its feature normalizer or fitted static base.
            varied = copy.deepcopy(data)
            varied['state'][held, :, 66] += 100.
            varied['state'][held, 9:, 86] += .01
            probe = StructuredResponse(meta).fit(varied, min_lopo_gain_mm=.00001)
            actual_audit = next(row for row in probe.selection_report['folds']
                                if row['held_parent'] == parent)
            for name in ('mean', 'scale', 'hold_curve_m', 'action_response_m', 'marginal_curve_m'):
                self.assertEqual(audit[name], actual_audit[name])

    def test_meaningful_held_train_gain_or_exact_static_fallback(self):
        meta, data = synthetic_cohort()
        model = StructuredResponse(meta).fit(data, min_lopo_gain_mm=.00001)
        self.assertIn(model.ridge, RIDGE_GRID[:-1])
        report = model.selection_report
        self.assertGreater(report['best_finite_gain_mm'], .01)
        self.assertFalse(report['fallback_to_static'])
        fallback = StructuredResponse(meta).fit(data, min_lopo_gain_mm=100.)
        self.assertTrue(np.isinf(fallback.ridge))
        np.testing.assert_array_equal(fallback.parameters['weight'], 0.)
        expected = (fallback.parameters['hold_curve_m'][None]
                    + fallback.parameters['action_response_m'][data['observed_action']])
        np.testing.assert_array_equal(fallback.predict_displacement(data), expected)
        flat_meta, flat = synthetic_cohort(structured=False)
        static = StructuredResponse(flat_meta).fit(flat, min_lopo_gain_mm=.00001)
        self.assertTrue(np.isinf(static.ridge))
        for threshold in (0., -.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                StructuredResponse(meta).fit(data, min_lopo_gain_mm=threshold)

    def test_npz_serialization_parity_without_pickle_and_rejects_invalid_arrays(self):
        meta, data = synthetic_cohort()
        for arm, threshold in [('ACTION_CONDITIONED', .00001), ('NO_FUTURE_ACTION', 100.)]:
            model = StructuredResponse(meta, arm).fit(data, min_lopo_gain_mm=threshold)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'SYNTHETIC_ONLY_RESPONSE.npz'
                model.save(path)
                reloaded = StructuredResponse.load(path)
                np.testing.assert_array_equal(model.candidate_forecasts(data), reloaded.candidate_forecasts(data))
                self.assertEqual(model.selection_report, reloaded.selection_report)
                with self.assertRaises(FileExistsError):
                    model.save(path)
            broken = model.arrays()
            broken['scale'][0] = 0.
            with self.assertRaises(ValueError):
                StructuredResponse.from_arrays(broken)
            broken = model.arrays()
            broken['hold_curve_m'][0, 0] = np.nan
            with self.assertRaises(ValueError):
                StructuredResponse.from_arrays(broken)


if __name__ == '__main__':
    unittest.main()
