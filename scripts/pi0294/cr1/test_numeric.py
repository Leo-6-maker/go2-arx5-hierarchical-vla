import unittest
import numpy as np
from .numeric import diagnostic_inverse,reproduction_matches


class NumericTests(unittest.TestCase):
    def test_no_clamp_is_explicit_and_does_not_mutate(self):
        x=np.asarray([[2.,-2.]],dtype=np.float32);before=x.copy()
        p=dict(min=np.array([0.,0.]),max=np.array([1.,1.]))
        np.testing.assert_array_equal(diagnostic_inverse(x,p,True),[[1.,0.]])
        np.testing.assert_array_equal(diagnostic_inverse(x,p,False),[[1.5,-.5]])
        np.testing.assert_array_equal(x,before)

    def test_reproduction_not_rounded_or_wrong_window(self):
        from scripts.pi0294.pv1.export import GATES
        r=dict(H40_windows_actually_checked=601,frozen_numeric_gates=GATES,
            observation=dict(first_error=dict(window_anchor=209,action_index=31,reference_time_s=12.,
              error_at_action=dict(SO3_rad=.00013817330386000284),failed_physical_gates=['SO3_rad'])))
        self.assertTrue(reproduction_matches(r))
        r['observation']['first_error']['error_at_action']['SO3_rad']=.00013817
        self.assertFalse(reproduction_matches(r))
