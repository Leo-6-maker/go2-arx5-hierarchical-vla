import unittest
import numpy as np
from scripts.wm02.diagnose_saved_release import components

class ReleaseRanking(unittest.TestCase):
    def test_privileged_substitutions_separate_opening_bias_from_grasp_uncertainty(self):
        # Both candidates actually release; uncertain event prediction alone flips choice.
        c=components([.7,.65],[.2,.01],[.8,.6],[0,0])
        self.assertEqual(int(np.argmax(c['saved_WM_proxy'])),1)
        self.assertEqual(int(np.argmax(c['PRIVILEGED_true_grasp'])),0)
        self.assertEqual(int(np.argmax(c['PRIVILEGED_true_opening'])),0)
        self.assertTrue(np.array_equal(c['PRIVILEGED_true_both'],[.8,.6]))
    def test_nonfinite_misaligned_and_nonphysical_truth_are_rejected(self):
        for arrays in(([.5],[float('nan')],[.5],[0]),([.5],[.1,.2],[.5],[0]),([.5],[.1],[1.2],[0])):
            with self.assertRaises(ValueError):components(*arrays)

if __name__=='__main__':unittest.main()
