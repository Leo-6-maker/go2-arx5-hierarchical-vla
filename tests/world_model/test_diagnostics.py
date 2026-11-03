import unittest
import numpy as np
from go2arx5_vla.world_model.diagnostics import proposals, ACTION_SCALE, cluster_interval
from go2arx5_vla.world_model.contracts import compile_action


class DiagnosticsTests(unittest.TestCase):
    def test_actual_feasible_deltas_have_equal_norm_and_compile(self):
        base=np.zeros((8,7));base[:,0]=.009;base[:4,6]=1
        gradient=np.random.default_rng(4).normal(size=(8,7))
        result,details=proposals(base,gradient,np.zeros(56),seed=1103)
        norms=[]
        for arm in ('gradient','negative_gradient','random_matched_norm'):
            compile_action(result[arm],np.eye(4),0)
            norms.append(np.linalg.norm((result[arm]-base)/ACTION_SCALE))
        np.testing.assert_allclose(norms,norms[0],rtol=0,atol=1e-12)
        self.assertEqual(details['redraws'],0)

    def test_zero_gradient_retains_ties_and_all_denominators(self):
        base=np.zeros((8,7))
        result,details=proposals(base,np.zeros((8,7)),np.zeros(56),seed=5)
        for key in ('gradient','negative_gradient','random_matched_norm'):
            np.testing.assert_array_equal(result[key],base)
        self.assertTrue(details['zero_feasible_direction'])
        report=cluster_interval([1,0,0,0],['a','a','b','b'])
        self.assertEqual(report['states'],4);self.assertEqual(report['scene_clusters'],2)
        self.assertEqual(report['mean'],.25)


if __name__=='__main__':unittest.main()
