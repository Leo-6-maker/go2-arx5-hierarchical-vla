import unittest
from go2arx5_vla.world_model.target_reference import deadline,require_live_window,require_install_clock,dynamic_limit,trajectory_identity,installed_target_identities
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
from scripts.pi0294.native_receiver import reference_expired


class TargetReferenceContract(unittest.TestCase):
    def test_actual_phase_and_source_relative_expiry(self):
        self.assertAlmostEqual(deadline(5.004),5.404)
        row=require_live_window(5.004,5.006,25)
        self.assertAlmostEqual(row['native_end_s'],5.056)
        self.assertAlmostEqual(row['remaining_at_install_s'],.398)
        require_live_window(5.004,5.006,175)
        with self.assertRaisesRegex(ValueError,'DEADLINE'):require_live_window(5.004,5.006,200)
        # At the receiver's expiry threshold and immediately afterwards.
        for native in (5.354,5.354000001):
            with self.assertRaises(ValueError):require_live_window(5.004,native,25)
        # Renew from a fresh *measured* source; old source cannot be retimed.
        require_live_window(5.354,5.356,25)
        # An installed prefix ages normally; 25ms is an *install* gate.
        require_live_window(5.004,5.306,25)
        with self.assertRaises(ValueError):require_install_clock(5.004,5.306)
        require_install_clock(5.004,5.006)
        self.assertFalse(reference_expired(5.404,5.402))
        self.assertTrue(reference_expired(5.404,5.404))
        self.assertTrue(reference_expired(5.404,5.406))
        self.assertFalse(reference_expired(None,5.406))
    def test_invalid_source_tick_count_or_partial_rejected(self):
        for args in ((5.007,5.006,25),(4.0,5.006,25),(5.004,5.006,24),(5.004,5.006,True)):
            with self.assertRaises(ValueError):require_live_window(*args)
    def test_stage_guards_no_arbitrary_limit(self):
        good=dict(phase='g2-reference-qualification',status='NATIVE_STAGE_PASSED',shutdown_clean=True,owned_process_exits=[dict(exit_code=0)])
        self.assertEqual(dynamic_limit('g2-hold-short',good),2528)
        for change in (dict(status='FAIL'),dict(shutdown_clean=False),dict(owned_process_exits=[]),dict(owned_process_exits=[dict(exit_code=139)])):
            with self.assertRaises(PermissionError):dynamic_limit('g2-hold-short',dict(good,**change))
        with self.assertRaises(PermissionError):dynamic_limit('anything',good)
    def test_only_exact_target_content_matches(self):
        a=trajectory_identity([5.004,5.054],[[1.,2.],[3.,4.]],[[0.],[0.]])
        self.assertEqual(a,trajectory_identity([5.004,5.054],[[1.,2.],[3.,4.]],[[0.],[0.]]))
        self.assertNotEqual(a,trajectory_identity([5.006,5.056],[[1.,2.],[3.,4.]],[[0.],[0.]]))
    def test_trace_cannot_resume_after_seal(self):
        adapter=TargetSequentialAdapter(dict(phase='g2-hold-short',max_ticks=2528),'.');adapter.trace_sealed=True
        with self.assertRaisesRegex(PermissionError,'SEALED'):adapter.advance_to(2528)
    def test_real_native_accumulation_and_integer_ROS_clock_same_tick(self):
        native=5.3559999999996375;source=5.356
        require_install_clock(source,native);require_live_window(source,native,25)
        with self.assertRaises(ValueError):require_install_clock(source+.001,native)
    def test_original_receiver_rolling_republication_retains_reference_identity(self):
        events=[dict(kind='installed',ref_id='a',times=[t,t+.05],poses=[[1.],[2.]],inputs=[[0.],[0.]],monotonic_s=t)for t in (5.004,5.104)]
        events.append(dict(events[-1],ref_id='b',times=[7.,7.05]))
        matched=installed_target_identities(events,'a')
        self.assertEqual(len(matched),2)
        self.assertIn(trajectory_identity([5.104,5.154],[[1.],[2.]],[[0.],[0.]]),matched)


if __name__=='__main__':unittest.main()
