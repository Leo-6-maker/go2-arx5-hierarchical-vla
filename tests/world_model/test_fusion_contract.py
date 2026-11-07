import dataclasses
import unittest
import numpy as np
from go2arx5_vla.world_model.fusion_contract import (
    ActionSemantics, decode, validate_local_preview, OFFSETS)
from go2arx5_vla.world_model.fusion_corrector import ARMS, choose
from go2arx5_vla.trajectory.so3 import (
    quat_to_matrix, rotation_matrix_to_rot6d, rotation_matrix_to_rot6d_rows)


class FusionActionContract(unittest.TestCase):
    def setUp(self):
        self.s = ActionSemantics('WORLD', 'ABSOLUTE', 'QUAT_XYZW', 'OPEN_FRACTION',
                                 'm', .05, 1, 'actual-task-TCP', '1'*64)
        self.args = dict(positions=[[0, 0, 0], [.001, 0, 0]],
            orientations=[[0, 0, 0, 1]]*2, gripper=[.2, .8], timestamps=[3.05, 3.1],
            observation_timestamp=3., world_from_source=np.eye(4),
            observed_tcp_world=np.eye(4), semantics=self.s,
            provenance='CALIBRATION_INTENT', proposal_id='fixture-only')

    def test_identity_preserves_actual_source_times_pose_and_grip(self):
        v = decode(**self.args)
        np.testing.assert_array_equal(v.eef_position, self.args['positions'])
        np.testing.assert_array_equal(v.point_timestamps, self.args['timestamps'])
        np.testing.assert_array_equal(v.gripper, [.2, .8])
        self.assertEqual(v.validity_end, 3.4)
        self.assertEqual(v.provenance['execution_points'], 1)

    def test_nonunit_base_transform_and_nonzero_rotation(self):
        a = dict(self.args); base = np.eye(4)
        base[:3, :3] = quat_to_matrix([0, 0, np.sqrt(.5), np.sqrt(.5)])
        base[:3, 3] = [1, 2, .3]
        a.update(world_from_source=base, semantics=dataclasses.replace(self.s, frame='BASE'))
        v = decode(**a)
        np.testing.assert_allclose(v.eef_position, [[1, 2, .3], [1, 2.001, .3]], atol=1e-12)
        np.testing.assert_allclose(quat_to_matrix(v.eef_orientation),
                                   np.repeat(base[None, :3, :3], 2, 0), atol=1e-12)

    def test_observation_left_and_right_are_not_silently_interchanged(self):
        obs = np.eye(4); obs[:3, :3] = quat_to_matrix([0, 0, np.sqrt(.5), np.sqrt(.5)])
        a = dict(self.args, positions=[[.001, 0, 0]]*2, observed_tcp_world=obs)
        for mode, expected in [('OBSERVATION_LEFT', [.001, 0, 0]),
                               ('OBSERVATION_RIGHT', [0, .001, 0])]:
            with self.subTest(mode=mode):
                a['semantics'] = dataclasses.replace(self.s, pose_mode=mode)
                np.testing.assert_allclose(decode(**a).eef_position, [expected]*2, atol=1e-12)

    def test_incremental_actions_accumulate_but_observation_relative_does_not(self):
        for mode, final in [('INCREMENTAL_LEFT', .002), ('INCREMENTAL_RIGHT', .002),
                            ('OBSERVATION_LEFT', .001), ('OBSERVATION_RIGHT', .001)]:
            with self.subTest(mode=mode):
                a = dict(self.args, positions=[[.001, 0, 0]]*2,
                         semantics=dataclasses.replace(self.s, pose_mode=mode))
                self.assertAlmostEqual(decode(**a).eef_position[-1, 0], final)

    def test_row_and_column_rot6d_decode_same_nontrivial_rotation(self):
        matrix = quat_to_matrix([.2, -.3, .1, .9])
        for name, encoder in [('ROT6D_COLUMNS', rotation_matrix_to_rot6d),
                              ('ROT6D_ROWS', rotation_matrix_to_rot6d_rows)]:
            with self.subTest(name=name):
                a = dict(self.args, orientations=[encoder(matrix)]*2,
                         semantics=dataclasses.replace(self.s, rotation=name))
                np.testing.assert_allclose(quat_to_matrix(decode(**a).eef_orientation),
                                           [matrix]*2, atol=1e-12)

    def test_explicit_gripper_directions_and_execution_horizons(self):
        for grip, expected in [('OPEN_FRACTION', [.2, .8]), ('CLOSED_FRACTION', [.8, .2])]:
            with self.subTest(gripper=grip):
                a = dict(self.args, semantics=dataclasses.replace(self.s, gripper=grip, execution_points=2))
                np.testing.assert_allclose(decode(**a).gripper, expected)
        with self.assertRaisesRegex(ValueError, 'SEMANTICS'):
            decode(**dict(self.args, semantics=dataclasses.replace(self.s, execution_points=3)))

    def test_unknown_semantics_and_discontinuous_or_fabricated_time_rejected(self):
        for change in [dict(position_unit='mm'), dict(frame='UNKNOWN'), dict(rotation='EULER'),
                       dict(gripper='UNKNOWN'), dict(evidence_sha256='unknown')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                decode(**dict(self.args, semantics=dataclasses.replace(self.s, **change)))
        for ts in [[3.05, 3.2], [3.1, 3.05], [30.05, 30.1]]:
            with self.subTest(ts=ts), self.assertRaises(ValueError):
                decode(**dict(self.args, timestamps=ts))
        with self.assertRaisesRegex(ValueError, 'WORLD_FRAME'):
            bad = np.eye(4); bad[0, 3] = 1
            decode(**dict(self.args, world_from_source=bad))

    def test_malformed_frame_and_nan_action_rejected(self):
        bad = np.eye(4); bad[0, 0] = 2
        for change in [dict(world_from_source=bad), dict(positions=[[np.nan, 0, 0]]*2),
                       dict(gripper=[-.1, 1.1]), dict(orientations=[[0, 0, 0, 0]]*2)]:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                decode(**dict(self.args, **change))


class FusionLocalEnvelope(unittest.TestCase):
    def fixture(self, selected=1, renewed=False):
        from go2arx5_vla.world_model.target_reference import fixed_goal_poses
        from go2arx5_vla.task_data.codec import pose10
        from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
        initial = np.eye(4); initial[:3, 3] = [.5, .1, .3]
        goal = initial.copy(); goal[:3, 3] += OFFSETS[selected]
        action = np.stack([pose10(p, float(teacher_physical_fraction(1.))) for p in
                          fixed_goal_poses(initial, goal, renewed=renewed)])
        kw = dict(initial_tcp=initial, chosen_action=selected, renewed=renewed,
                  base_pose=[0, 0, .4, 0, 0, 0, 1], frozen_base_pose=[0, 0, .4, 0, 0, 0, 1])
        return action, kw

    def test_all_four_actual_programs_and_renewals_preserved(self):
        for a in range(4):
            for renewed in [False, True]:
                with self.subTest(action=a, renewed=renewed):
                    value, kw = self.fixture(a, renewed)
                    self.assertEqual(validate_local_preview(value, **kw)['full_preview_points_checked'], 40)

    def test_even_unexecuted_preview_suffix_or_tiny_delta_on_large_path_is_rejected(self):
        action, kw = self.fixture()
        for index, component, change in [(39, 0, .1), (39, 2, .00001),
                                          (39, 9, -.01), (0, 3, .00001)]:
            with self.subTest(index=index, component=component), self.assertRaises(ValueError):
                bad = action.copy(); bad[index, component] += change
                validate_local_preview(bad, **kw)
        with self.assertRaises(ValueError):
            validate_local_preview(action, **dict(kw, base_pose=[.001, 0, .4, 0, 0, 0, 1]))


class MatchedCorrection(unittest.TestCase):
    def fixture(self):
        return dict(arm=ARMS[0], nominal_action=1, desired_xy_m=[.0008, 0],
            provenance='CALIBRATION_INTENT', actual_tcp_xy_m=[0, 0],
            static_response_xy_m=[[.0008, 0], [.0018, 0], [-.0002, 0], [.0008, .001]],
            WM_endpoint_xy_m=[[0, .001], [.002, 0], [.0008, 0], [0, .002]],
            nonhold_cost_m=.00002, observation_time_s=7.406, source_time_s=7.4)

    def test_identity_static_feedback_and_history_model_have_declared_different_choices(self):
        args = self.fixture()
        for arm, expected in zip(ARMS, [1, 0, 1, 2]):
            with self.subTest(arm=arm):
                v = choose(**dict(args, arm=arm))
                self.assertEqual(v['selected_action'], expected)
                self.assertEqual(v['decision_opportunities'], 1)
                self.assertFalse(v['future_response_observed'])
                self.assertFalse(v['prediction_is_physical_evidence'])

    def test_stale_or_future_observation_holds_without_timestamp_rewrite_or_install(self):
        for source in [7., 8.]:
            with self.subTest(source=source):
                v = choose(**dict(self.fixture(), source_time_s=source))
                self.assertEqual(v['status'], 'EXPIRED_HOLD_NO_INSTALL')
                self.assertEqual(v['original_source_time_s'], source)
                self.assertFalse(v['timestamp_refreshed'])

    def test_unresolved_semantics_unsupported_nominal_and_nonfinite_models_rejected(self):
        for change in [dict(provenance='SEMANTICS_UNRESOLVED'), dict(nominal_action=4),
                       dict(WM_endpoint_xy_m=[[np.nan, 0]]*4)]:
            with self.subTest(change=list(change)), self.assertRaises((PermissionError, ValueError)):
                choose(**dict(self.fixture(), **change))


if __name__ == '__main__': unittest.main()
