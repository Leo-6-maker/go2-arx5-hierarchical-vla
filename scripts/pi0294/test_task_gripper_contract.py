import unittest
import numpy as np
from .task_gripper_contract import (
    MECHANICAL_TRAVEL,SPEED,COMMAND_DT,MARGIN,teacher_physical_fraction,require_working_target,
)


class WorkingStrokeTests(unittest.TestCase):
    def test_one_declared_command_interval_margin_not_fitted_to_score(self):
        self.assertEqual(MARGIN,SPEED*COMMAND_DT)
        np.testing.assert_allclose(teacher_physical_fraction([0,.5,1])*MECHANICAL_TRAVEL,[.002,.022,.042],rtol=0,atol=1e-17)
        require_working_target(teacher_physical_fraction([0,1]))

    def test_no_clipping_or_measured_state_rewrite(self):
        for v in (0.,1.,float('nan')):
            with self.assertRaises(ValueError):require_working_target(v)
        original=np.array([-.00001,.0440263204416577])
        saved=original.copy()
        with self.assertRaises(ValueError):require_working_target(original/MECHANICAL_TRAVEL)
        np.testing.assert_array_equal(original,saved)


if __name__=='__main__':unittest.main()
