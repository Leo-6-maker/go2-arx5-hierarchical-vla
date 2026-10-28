import unittest
import numpy as np

from .mp1.t2_saved import progress,ATTEMPTS
from scripts.pi0291_arx5_data.config import PARAMETERS as P


class SavedT2Tests(unittest.TestCase):
    def test_saved_goal_entry_is_not_complete_success(self):
        obj=np.repeat(np.eye(4)[None],3,axis=0);obj[:,2,3]=P['table_top']+.02;obj[:,0,3]=[0.,.1,.2]
        tcp=obj.copy();tcp[:,2,3]+=.03
        arrays=dict(time_s=np.array([0.,.05,.1]),object_pose=obj,tcp_world=tcp,
            robot_object_contact=np.array([False,True,True]),object_linear_velocity=np.zeros((3,3)),object_angular_velocity=np.zeros((3,3)))
        result=progress(arrays,dict(goal_xy=[.2,0.]))
        self.assertTrue(result['goal_zone_all_corners_ever']);self.assertEqual(result['first_goal_zone_all_corners_s'],.1)
        self.assertTrue(result['last_saved_robot_contact']);self.assertAlmostEqual(result['last_saved_TCP_object_distance_m'],.03)
        self.assertTrue(result['these_are_saved_frame_diagnostics_not_complete_task_success'])
        self.assertNotIn('persistent_success',result);self.assertEqual(len(ATTEMPTS),2)
