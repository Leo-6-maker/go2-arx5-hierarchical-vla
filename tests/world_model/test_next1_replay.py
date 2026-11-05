import unittest,copy
from go2arx5_vla.world_model.next1_replay import sample_view


class Next1Replay(unittest.TestCase):
    def test_fixed_prefix_is_excluded_and_unstable_or_partial_is_rejected(self):
        windows=[dict(native_start_step=2503+25*i,native_end_step=2503+25*(i+1),actual_ticks=25,mode='WBC',solver_policies_with_exact_installed_target=[dict(actual=True)])for i in range(64)]
        d=dict(reference_rule='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1',complete_task_atoms=64,task_native_ticks=1600,decision_atom=48,late_decision_applied=True,steady_prefix=dict(qualified=True),actual_collection_start_step=2503,windows=windows)
        commands=[dict(phase='proposed',native_step=2503+25*i)for i in [0,7,14,21,28,35,40,47,48,55,62]]
        view,c=sample_view(d,commands);self.assertEqual(view['actual_collection_start_step'],3503);self.assertEqual(len(view['windows']),24);self.assertEqual(c[0]['native_step'],3503)
        self.assertEqual(d['complete_task_atoms'],64)
        for change in [dict(steady_prefix=dict(qualified=False)),dict(task_native_ticks=1599),dict(decision_atom=8)]:
            with self.assertRaises(ValueError):sample_view(dict(d,**change),commands)
        bad=copy.deepcopy(d);bad['windows'][1]['mode']='HOLD'
        with self.assertRaises(ValueError):sample_view(bad,commands)
