import unittest
import numpy as np
from go2arx5_vla.target_sim.limits import hard_limit_report


class HardLimitTests(unittest.TestCase):
    def test_soft_violation_is_not_hard_violation(self):
        # Delivered native reward reduces [0,3.24] to [.162,3.078].
        props = {'lower': np.array([0., 0.]), 'upper': np.array([3.24, .044])}
        result = hard_limit_report(['x5_joint3', 'x5_gripper_joint'], [0., .044], props)
        self.assertFalse(result['hard_limit'])
        result = hard_limit_report(['x5_joint3', 'x5_gripper_joint'], [-.001, .045], props)
        self.assertEqual(result['offending'], ['x5_joint3', 'x5_gripper_joint'])
        self.assertAlmostEqual(result['penetration'][0], .001)


if __name__ == '__main__': unittest.main()
