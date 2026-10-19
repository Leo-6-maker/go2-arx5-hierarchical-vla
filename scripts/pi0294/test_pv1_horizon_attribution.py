"""Synthetic CPU attribution only; no policy inference or physics."""
import unittest

import numpy as np

from scripts.pi0294.pv1.horizon_attribution import analyze


def action():
    values=np.zeros((40,10));values[:,:3]=[.4,0,.6]
    values[:,3:9]=[1,0,0,0,1,0];values[:,9]=1
    return values


class HorizonAttribution(unittest.TestCase):
    def test_prefix_boundary_and_tail_are_not_conflated(self):
        for edge,region in [(6,'PREFIX8_INTERNAL'),(7,'PREFIX_TAIL_BOUNDARY'),(8,'TAIL_INTERNAL'),(38,'TAIL_INTERNAL')]:
            with self.subTest(edge=edge):
                a=action();a[edge+1:,0]+=.11;before=a.copy()
                v=analyze(a,a[0],28.8,'fixture')
                s=v['speed_attribution']['linear_speed_mps']
                self.assertEqual(s['max_edge_points_1based'],[edge+1,edge+2])
                self.assertEqual(s['max_region'],region)
                self.assertEqual(s['regions'][region]['reject_edges_points_1based'],[[edge+1,edge+2]])
                self.assertEqual(v['original_full_H40_gate']['decision'],'REJECT')
                self.assertTrue(np.array_equal(a,before));self.assertTrue(v['prefix_is_not_an_alternative_admission_gate'])

    def test_zero_expert_error_requires_explicit_matched_array(self):
        a=action();a[:,0]+=np.arange(40)*.001
        v=analyze(a,a[0],0,'fixture')
        self.assertIsNone(v['matched_expert_errors']);self.assertFalse(v['matched_expert_available'])
        v=analyze(a,a[0],0,'fixture',expert=a.copy())
        self.assertEqual(v['matched_expert_errors']['tail32'],dict(XYZ_mean_m=0.,SO3_mean_rad=0.,gripper_MAE=0.))

    def test_initial_jump_is_not_an_invented_velocity_edge(self):
        a=action();state=a[0].copy();state[0]-=.8
        v=analyze(a,state,0,'fixture')
        self.assertAlmostEqual(v['first_point_position_jump_m'],.8)
        self.assertEqual(v['speed_attribution']['linear_speed_mps']['max'],0.)
        self.assertEqual(v['original_full_H40_gate']['decision'],'REJECT')

    def test_original_thresholds_and_gripper_range_preserved(self):
        a=action();a[9,9]=1.01
        v=analyze(a,a[0],0,'fixture')
        self.assertEqual(v['gripper_outside_original_range_points_1based'],[10])
        self.assertEqual(v['original_full_H40_gate']['decision'],'REJECT')
        self.assertEqual(v['speed_attribution']['linear_speed_mps']['reject_threshold'],2.)
        self.assertEqual(v['speed_attribution']['angular_speed_radps']['reject_threshold'],4.)

    def test_malformed_partial_and_nonfinite_inputs_are_not_shortened_or_repaired(self):
        a=action()
        for bad in (a[:8],a.copy()):
            if bad.shape==(40,10):bad[1,0]=np.nan
            with self.assertRaises(ValueError):analyze(bad,a[0],0,'fixture')
        with self.assertRaises(ValueError):analyze(a,a[0],0,'fixture',expert=a[:8])
