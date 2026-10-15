import copy
import unittest
from .task_clock import require_pause,FROZEN_FIELDS,actual_source_age


class ClockTest(unittest.TestCase):
    def test_actual_pause_and_all_history_fields(self):
        a=dict(step=2500,time_s=5.,motiontime=3900,leg_history_steps=120,gait_indices=.3,
               running=False,clock_paused=True,ack_sequence=1)
        b=dict(a,ack_sequence=2)
        require_pause(a,b,2500)
        for key in FROZEN_FIELDS:
            changed=copy.deepcopy(b);changed[key]+=.001
            with self.assertRaises(ValueError):require_pause(a,changed,2500)

    def test_stale_ack_and_live_clock_rejected(self):
        a=dict(step=2500,time_s=5.,motiontime=1,leg_history_steps=1,gait_indices=0.,running=False,clock_paused=True,ack_sequence=1)
        for b in (dict(a),dict(a,running=True,ack_sequence=2),dict(a,clock_paused=False,ack_sequence=2)):
            with self.assertRaises(ValueError):require_pause(a,b,2500)

    def test_same_actual_integer_nanosecond_ros_clock_not_reanchored(self):
        self.assertEqual(actual_source_age(4.999999999999671,5.),0.)
        self.assertGreater(actual_source_age(5.,4.974),.025)
        self.assertLess(actual_source_age(5.,5.001),0.)
