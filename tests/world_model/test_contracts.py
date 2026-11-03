import copy
import unittest
import numpy as np

from go2arx5_vla.world_model.contracts import (
    compile_action, command_labels, macro_transition, actor_state, FRAME, TCP)
from go2arx5_vla.world_model.population import definitions, validate_parent_splits


def atom(i=0):
    before = dict(time_s=i*.05, qpos=np.zeros(15), qvel=np.zeros(14), tcp=np.eye(4),
                  object_pose=np.eye(4), reset=False, takeover=False)
    after = copy.deepcopy(before)
    after['time_s'] = (i+1)*.05
    action = np.r_[np.zeros(3), np.eye(3)[:2].ravel(), 1.]
    return dict(episode_id='e', scene_id='s', parent_id='p', split='TRAIN', executor_id='x', segment_id=0,
        frame=FRAME, tcp=TCP, t0=i*.05, t1=(i+1)*.05, physics_ticks=25,
        state_before=before, state_after=after, action_raw=np.zeros(7), reference_installed=action,
        reference_consumed=[dict(time_s=i*.05+j*.002, eef10=action.copy()) for j in range(25)],
        controller_memory={}, context={}, terminal=False, truncated=False, safety_reason=None,
        planned_dt=.05, actual_dt=.05)


class Contracts(unittest.TestCase):
    def test_first_point_is_at_observation_time_and_uses_actual_anchor(self):
        anchor = np.eye(4)
        anchor[:3, 3] = [.4, -.1, .6]
        raw = np.zeros((8, 7)); raw[:, 0] = .003; raw[:, 6] = 1.
        result = compile_action(raw, anchor, 1.2)
        self.assertEqual(result['timestamps_s'][0], 1.2)
        np.testing.assert_allclose(result['poses'][0, :3, 3], [.403, -.1, .6])
        np.testing.assert_array_equal(result['poses'][8:], np.repeat(result['poses'][7:8], 32, axis=0))

    def test_world_rotation_and_consumed_label_round_trip(self):
        from go2arx5_vla.world_model.contracts import exp_rotation
        anchor = np.eye(4); anchor[:3, :3] = exp_rotation([.2, 0, 0])
        raw = np.random.default_rng(7).normal(0, .001, (8, 7)); raw[:, 6] = .7
        result = compile_action(raw, anchor, 0)
        labels, represented = command_labels(result['poses'][:8], result['opening'][:8], anchor)
        self.assertTrue(represented)
        np.testing.assert_allclose(raw, labels, atol=1e-12)
        np.testing.assert_allclose(result['poses'][0, :3, :3], exp_rotation(raw[0, 3:6])@anchor[:3, :3])

    def test_bounds_are_norms_and_reject_without_clipping(self):
        for index, value in ((0, .011), (3, .051), (6, -1e-8), (6, 1.00000001)):
            raw = np.zeros((8, 7)); raw[0, index] = value
            with self.assertRaises(ValueError): compile_action(raw, np.eye(4), 0)
        raw = np.zeros((8, 7)); raw[0, :3] = .009
        with self.assertRaises(ValueError): compile_action(raw, np.eye(4), 0)

    def test_bad_frame_clock_and_rotation_reject(self):
        with self.assertRaises(ValueError): compile_action(np.zeros((8, 7)), np.eye(4), 0, frame='BODY')
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError): compile_action(np.zeros((8, 7)), bad, 0)
        rows = [atom(i) for i in range(8)]; rows[1]['t0'] += .002
        with self.assertRaises(ValueError): macro_transition(rows)

    def test_no_cross_episode_reset_executor_or_terminal(self):
        for key, value in (('episode_id', 'other'), ('segment_id', 1), ('executor_id', 'other'),
                           ('parent_id', 'other'), ('split', 'TEST'), ('terminal', True)):
            rows = [atom(i) for i in range(8)]; rows[3][key] = value
            with self.assertRaises(ValueError): macro_transition(rows)
        rows = [atom(i) for i in range(8)]; rows[2]['state_after']['reset'] = True
        with self.assertRaises(ValueError): macro_transition(rows)

    def test_labels_cannot_substitute_achieved_or_preview(self):
        rows = [atom(i) for i in range(8)]
        macro_transition(rows)
        rows[0]['reference_consumed'][0]['eef10'][0] = .2
        with self.assertRaises(ValueError): macro_transition(rows)
        with self.assertRaises(ValueError): command_labels(np.repeat(np.eye(4)[None], 40, axis=0), np.zeros(40), np.eye(4))

    def test_partial_macro_retains_event_without_state_regression_target(self):
        rows = [atom(i) for i in range(2)]; rows[-1]['terminal'] = True
        self.assertFalse(macro_transition(rows)['complete_state_target'])
        rows[-1]['terminal'] = False
        with self.assertRaises(ValueError): macro_transition(rows)

    def test_split_and_actor_whitelists(self):
        pop = definitions()
        self.assertEqual({s:len(c) for s,c in pop.items()}, dict(TRAIN=64, CALIB=16, DEV=20, TEST=50))
        validate_parent_splits([dict(parent_id='p', scene_id='s', split='TRAIN')])
        with self.assertRaises(ValueError): validate_parent_splits([
            dict(parent_id='p', scene_id='s', split='TRAIN'), dict(parent_id='p', scene_id='x', split='CALIB')])
        state = {k:0. for k in ('tcp', 'gripper_fraction', 'gripper_speed', 'tool_linear_velocity',
            'tool_angular_velocity', 'object_pose', 'object_linear_velocity', 'object_angular_velocity')}
        state.update(qpos=np.ones(15), teacher_phase='hold', native_grasp=True, robot_id='secret')
        actor = actor_state(state, [0, 0, 0], 32.)
        self.assertFalse(set(actor) & {'qpos', 'teacher_phase', 'native_grasp', 'robot_id'})


if __name__ == '__main__': unittest.main()
