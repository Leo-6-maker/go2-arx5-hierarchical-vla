import unittest
from types import SimpleNamespace
import numpy as np
import torch
from go2arx5_vla.world_model.features import STATE_DIM, ROTATIONS
from go2arx5_vla.world_model.evaluation import evaluate_predictions


class Movement:
    def __call__(self, state, action, goal, mean, scale):
        after = state.clone()
        after[:, :3] += action[:, :3]
        return after, state.new_zeros((len(state), 4))


class EvaluationTests(unittest.TestCase):
    def replay(self):
        actions = np.zeros((8, 4, 56), dtype=np.float32)
        actions[:, :, 0] = np.arange(1, 9)[:, None]*.001
        states = np.zeros((8, 4, STATE_DIM), dtype=np.float32)
        for sl in ROTATIONS:
            states[..., sl] = np.eye(3).ravel()
        targets = states.copy()
        for k in range(4):
            states[:, k, :3] = actions[:, 0, :3]*k
            targets[:, k, :3] = actions[:, 0, :3]*(k+1)
        events = np.zeros((8, 4, 4), dtype=np.float32)
        events[:3, :, :2] = 1
        return SimpleNamespace(split='CALIB', keys=list(range(8)), episodes=2,
            states=states, actions=actions, targets=targets, events=events,
            goals=np.zeros((8, 3), dtype=np.float32),
            stages=['carry']*3+['hold']*5, event_counts=[3,3,0,0])

    def test_physical_errors_causal_shuffle_and_zero_strata(self):
        model = SimpleNamespace(members=[Movement(), Movement(), Movement()], mean=None, scale=None)
        result = evaluate_predictions(model, self.replay())
        self.assertEqual(result['stratum_counts']['first_target_contact'], 3)
        self.assertEqual(result['stratum_counts']['teacher_stage_carry'], 3)
        self.assertEqual(result['stratum_counts']['teacher_stage_recovery'], 0)
        errors = result['prediction_errors']
        self.assertAlmostEqual(errors['model']['4']['all']['tcp_position_m']['mean'], 0, places=8)
        self.assertGreater(errors['action_shuffle']['4']['all']['tcp_position_m']['mean'], 0)
        self.assertAlmostEqual(errors['persistence']['4']['all']['tcp_position_m']['mean'], .018, places=7)
        self.assertIsNone(errors['model']['4']['teacher_stage_recovery']['tcp_position_m']['mean'])
        terminal = result['event_reliability']['model']['1']['all']['terminal']
        self.assertEqual(terminal['positives'], 0)
        self.assertEqual(terminal['calibration_status'], 'NO_POSITIVES_NOT_FAILURE_CALIBRATED')
        self.assertAlmostEqual(terminal['brier'], .25)
        self.assertEqual(terminal['bins'][0]['count'], 8)

    def test_training_and_test_views_are_not_held_out_evaluation(self):
        replay = self.replay()
        for split in ('TRAIN', 'TEST'):
            replay.split = split
            with self.assertRaises(ValueError):
                evaluate_predictions(None, replay)


if __name__ == '__main__': unittest.main()
