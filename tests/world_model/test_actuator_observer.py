import unittest
import numpy as np
from go2arx5_vla.world_model.actuator_observer import SourceFingerTarget

class ActuatorMemory(unittest.TestCase):
    def test_known_reset_matches_source_limiter_through_changes_and_saturation(self):
        observer=SourceFingerTarget(known_reset=True);expected=np.full(2,.044)
        for command in [0.]*600+[1.]*500+[.4]*300+[.045454545454545456]*400:
            expected+=np.clip(command*.044-expected,-.04*.002,.04*.002)
            low,high=observer.consume(command);self.assertTrue(np.array_equal(low,expected));self.assertTrue(np.array_equal(high,expected))
    def test_burn8_commands_do_not_invent_current_measured_gap_as_hidden_target(self):
        observer=SourceFingerTarget()
        for _ in range(8*25):observer.consume(0.)
        low,high=observer.interval();self.assertTrue(np.array_equal(low,[0.,0.]));self.assertAlmostEqual(float(high.mean()/.044),1-8*.05*.04/.044)
        # Two actual controller initial states remain compatible with commands.
        self.assertTrue(low[0]<=0.<.433*.044<high[0])
    def test_nonphysical_or_nonfinite_commands_are_rejected(self):
        for value in(-.01,1.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):SourceFingerTarget().consume(value)

if __name__=='__main__':unittest.main()
