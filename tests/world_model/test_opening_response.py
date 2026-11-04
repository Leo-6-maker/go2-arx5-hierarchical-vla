import unittest
import numpy as np
from go2arx5_vla.world_model.opening_response import reference_curve,predict_opening,fit_lag

class OpeningResponse(unittest.TestCase):
    def test_causal_slew_catchup_separates_release_timing_without_future_state(self):
        immediate=np.zeros((8,7));immediate[:,6]=.9545454545454545;delay=immediate.copy();delay[:2,6]=.045454545454545456
        a=reference_curve(np.zeros(2),immediate);b=reference_curve(np.zeros(2),delay)
        pa=predict_opening(.433,a,.005);pb=predict_opening(.433,b,.005)
        self.assertTrue(np.all(pa[:8]==.433));self.assertGreater(pa[-1],pb[-1]);self.assertAlmostEqual(a[-1],16*.05*.04/.044)
        self.assertAlmostEqual(pa[-1]-pb[-1],.05*.04/.044)
    def test_fit_only_moving_points_and_exact_saved_parameter_reuse(self):
        z=np.linspace(0,.8,16);truth=np.maximum(.433,z-.005)
        lag,n=fit_lag([(.433,z,truth)]);self.assertAlmostEqual(lag,.005);self.assertGreater(n,0);self.assertTrue(np.allclose(predict_opening(.433,z,lag),truth))
    def test_invalid_reference_is_not_silently_a_physical_command(self):
        with self.assertRaises(ValueError):reference_curve(np.zeros(2),np.zeros((16,7)))
        with self.assertRaises(ValueError):predict_opening(.433,[float('nan')],.005)
if __name__=='__main__':unittest.main()
