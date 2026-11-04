import unittest
from go2arx5_vla.world_model.temporal_assessment import assess
class TemporalAssessment(unittest.TestCase):
    def test_one_bad_parent_blocks_new_sampling_even_if_mean_effect_is_good(self):
        rows=[];ref=[]
        for h in(1,8,16):
            for scene in range(24,32):
                r=dict(partition='DIAGNOSTIC_DEV',anchor='transport',mode='FREE_CAUSAL',horizon=h,scene=scene,object_position=dict(actual_toward_component_m=.004,predicted_toward_component_m=.004,endpoint_RMSE_m=.0001));rows.append(r);ref.append(dict(r,object_position=dict(r['object_position'])))
        b={'models':{'M1':{'rows':ref}}};v={'models':{'M3':{'rows':rows}},'invariance':[dict(model='M3',canonical_candidate_permutation_max_abs=0)for _ in range(32)]};self.assertTrue(assess(v,b)['all_development_gates_pass'])
        rows[0]['object_position']['predicted_toward_component_m']+=.0006;self.assertFalse(assess(v,b)['all_development_gates_pass']);self.assertEqual(assess(v,b)['new_holdout_sampling'],'NOT_RUN_DEVELOPMENT_GATE_FAILED')
if __name__=='__main__':unittest.main()
