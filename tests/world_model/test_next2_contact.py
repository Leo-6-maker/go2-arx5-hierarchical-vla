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
