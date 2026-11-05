import unittest
from go2arx5_vla.world_model.next2_contact import Continuity
class PrecontactContinuity(unittest.TestCase):
    def test_twenty_points_do_not_make_one_second(self):
        c=Continuity()
        for i in range(20):elapsed=c.observe(i*.05,True,.01)
        self.assertLess(elapsed,1.)
        self.assertAlmostEqual(c.observe(1.,True,.01),1.)
        self.assertAlmostEqual(c.observe(1.5,True,.01),1.5)
    def test_geometry_or_speed_loss_resets_without_teleport(self):
        c=Continuity();c.observe(0,True,.001);self.assertEqual(c.observe(.5,False,.001),0);self.assertIsNone(c.since)
        c.observe(.55,True,.001);self.assertEqual(c.observe(.6,True,.016),0);self.assertEqual(len(c.losses),2);self.assertEqual(c.observe(.65,True,.001),0)
if __name__=='__main__':unittest.main()

class FiniteStageBudget(unittest.TestCase):
    def test_descent_and_exit_need_reserved_time_and_smooth_reference(self):
        import numpy as np
        from go2arx5_vla.world_model.next2_contact import enough_time,stage_path
        start=np.eye(4);end=start.copy();end[2,3]=-.06;p=stage_path(start,end)
        self.assertEqual(p.shape,(51,4,4));np.testing.assert_array_equal(p[0],start);np.testing.assert_allclose(p[-1],end,atol=1e-15);self.assertLess(np.linalg.norm(np.diff(p[:,:3,3],axis=0)/.05,axis=1).max(),.06)
        self.assertFalse(enough_time(.8,'DESCEND'));self.assertFalse(enough_time(9.9,'DESCEND'));self.assertTrue(enough_time(10.,'DESCEND'));self.assertFalse(enough_time(6.,'CLOSE'));self.assertFalse(enough_time(2.9,'WITHDRAW'))
