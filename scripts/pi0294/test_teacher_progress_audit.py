import unittest
import numpy as np
from .teacher_progress_audit import interpolate,residual


class ProgressAuditTests(unittest.TestCase):
    def test_same_time_interpolation_never_uses_future_endpoint(self):
        points=[dict(time=k*.05,position=[k*.003,0,0],quaternion_xyzw=[0,0,0,1]) for k in range(41)]
        actual=np.eye(4)
        self.assertAlmostEqual(residual(interpolate(points,.05),actual)[0],.003)
        self.assertAlmostEqual(residual(interpolate(points,1.95),actual)[0],.117)
        with self.assertRaises(ValueError):interpolate(points,2.1)

    def test_task_goal_is_distinct_from_local_preview(self):
        actual=np.eye(4);actual[:3,3]=[.2,0,.5]
        target=actual.copy();target[:3,3]=[.6,-.08,.54]
        distance,angle=residual(target,actual)
        self.assertGreater(distance,.4);self.assertEqual(angle,0.)
