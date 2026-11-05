import unittest
import numpy as np
from scripts.wm02.evaluate_matched_target_models import delta_metrics

class MatchedActionDelta(unittest.TestCase):
    def test_action_absence_fails_even_when_absolute_common_motion_is_right(self):
        state=np.zeros((4,25,3));state[:,:,0]=np.arange(25)*.001
        goal=np.zeros((4,17));goal[1,0]=.001;goal[2,0]=-.001;goal[3,1]=.001
        for i in range(1,4):state[i,9:]+=goal[i,:3]
        data=dict(state=state,goal=goal,history_goal=np.zeros((4,8,17)))
        meta=dict(state_dimension=3,state_fields=[dict(name='tcp_position_m',size=3)],episodes=[dict(parent='P01',split='DEV',episode=str(i))for i in range(4)])
        truth=state[:,9:].copy();perfect=delta_metrics(truth,data,meta,'DEV')
        self.assertEqual(perfect['summary']['16']['coordinate_RMSE_m'],0.)
        self.assertEqual(perfect['summary']['16']['observed_direction_fraction'],1.)
        no_action=np.repeat(truth[0:1],4,axis=0);bad=delta_metrics(no_action,data,meta,'DEV')
        self.assertGreater(bad['summary']['16']['coordinate_RMSE_m'],.0005)
        self.assertEqual(bad['summary']['16']['observed_direction_fraction'],0.)
