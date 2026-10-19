"""Saved-array synthetic fixtures, never simulation or Teacher collection."""
from copy import deepcopy
import unittest
import numpy as np
from scripts.pi0294.pv1.event_exposure import analyze,summarize,_coverage


def fixture():
    obj=np.tile(np.eye(4),(641,1,1));obj[:,:3,3]=[.4,0,.46]
    tcp=obj.copy();tcp[:,2,3]+=.1
    micro=np.zeros((16000,5));micro[:,3]=.46
    micro[499:999,0]=1;micro[599:899,1]=1;micro[799:,2]=.12
    return dict(time_s=np.arange(641)/20,contact_time_s=np.arange(1,16001)*.002,
        contact_audit=micro,object_pose=obj,tcp_world=tcp),dict(id='fixture',group_id='TRAIN_FIXTURE',split='TRAIN',task='T1',goal_xy=[.4,0])


class EventExposure(unittest.TestCase):
    def test_real_flags_not_teacher_names_and_events_not_independent_windows(self):
        data,card=fixture();before=deepcopy(data);row=analyze(data,card)
        self.assertEqual(row['events']['contact_start']['first_s'],1.)
        self.assertEqual(row['events']['contact_end']['first_s'],2.)
        self.assertEqual(row['events']['native_grasp_end_after_10cm_transport']['count'],1)
        self.assertFalse(row['Teacher_phase_labels_used']);self.assertTrue(row['event_categories_overlap_and_must_not_be_summed'])
        for key in data:self.assertTrue(np.array_equal(data[key],before[key]))
        summary=summarize([row]);self.assertEqual(summary['tasks']['T1']['independent_trajectories'],1)

    def test_horizon_endpoint_and_prefix_tail_exposures_are_explicit(self):
        self.assertEqual(_coverage([1.95],1.95)['windows'],40)
        self.assertEqual(_coverage([1.95],.35)['windows'],8)
        self.assertEqual(_coverage([32.],1.95)['windows'],0)
        self.assertEqual(_coverage([],1.95)['fraction'],0.)

    def test_partial_nonfinite_safety_clock_and_dev_denied(self):
        data,card=fixture()
        for key in ('time_s','contact_time_s','contact_audit'):
            bad=deepcopy(data);bad[key]=bad[key][:-1]
            with self.assertRaises(PermissionError):analyze(bad,card)
        bad=deepcopy(data);bad['contact_audit'][1,4]=1
        with self.assertRaises(PermissionError):analyze(bad,card)
        with self.assertRaises(PermissionError):analyze(data,dict(card,split='DEV'))
        row=analyze(data,card)
        with self.assertRaises(PermissionError):summarize([row,row])
