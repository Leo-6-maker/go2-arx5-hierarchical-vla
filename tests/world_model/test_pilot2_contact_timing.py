import unittest
import numpy as np
from go2arx5_vla.world_model.pilot2_contact_timing import transition_timing


class EventTiming(unittest.TestCase):
    def data(self):
        y=np.zeros((3,3,4));y[0,1:,0]=1;y[1,:,0]=1;y[1,2,0]=0
        return dict(events=y,valid=np.ones((3,3),bool),current_events=np.array([[0]*4,[1,0,0,0],[0]*4]),
            horizons=(1,4,8),identities=[dict(parent=p)for p in ['T1','T2','T1']])

    def test_entries_exits_and_detected_only_error_are_in_seconds(self):
        d=self.data();p=d['events'].copy();p[0,0,0]=1
        r=transition_timing(p,d,[0,1,2])['events']['PAD_CONTACT_ANY']
        self.assertEqual(r['actual_grid_transitions'],2);self.assertEqual(r['detected_actual_transitions'],2)
        self.assertAlmostEqual(r['detected_only_timing_MAE_s'],.075)
        self.assertAlmostEqual(r['detected_only_mean_signed_error_s'],-.075)
        self.assertEqual(r['by_initial_state']['INITIALLY_TRUE_EXIT']['actual_grid_transitions'],1)

    def test_misses_and_false_alarms_do_not_become_zero_timing_error(self):
        d=self.data();p=np.zeros_like(d['events']);p[1,:,0]=1;p[2,2,0]=1
        r=transition_timing(p,d,[0,1,2])['events']['PAD_CONTACT_ANY']
        self.assertEqual(r['missed_actual_transitions'],2);self.assertEqual(r['false_transition_predictions'],1)
        self.assertIsNone(r['detected_only_timing_MAE_s']);self.assertEqual(r['detection_recall'],0)

    def test_masked_suffix_and_empty_anchor_are_not_future_evidence(self):
        d=self.data();d['valid'][0,1:]=False;d['valid'][1]=False
        r=transition_timing(d['events'],d,[0,1,2])['events']['PAD_CONTACT_ANY']
        self.assertEqual(r['valid_anchors'],2);self.assertEqual(r['actual_grid_transitions'],0)
        self.assertIsNone(r['detection_recall']);self.assertIsNone(r['parent_equal_detection_recall'])

    def test_probability_and_grid_contract(self):
        d=self.data();p=d['events'].copy();p[0,0,0]=np.nan
        with self.assertRaises(ValueError):transition_timing(p,d,[0])
        d['horizons']=(1,8,4)
        with self.assertRaises(ValueError):transition_timing(d['events'],d,[0])


if __name__=='__main__':unittest.main()
