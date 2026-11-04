import unittest
import numpy as np
from go2arx5_vla.world_model.contact_analysis import position_pairs, progress_ranking, oracle_opportunities


class ContactAnalysis(unittest.TestCase):
    def test_zero_prediction_is_a_tie_and_flat_real_pairs_stay_in_denominator(self):
        a=np.array([[0.,0,0],[.003,0,0],[.003,0,0]])
        result=position_pairs(a,np.zeros_like(a))
        self.assertEqual((result['pairs'],result['informative'],result['flat']),(3,2,1))
        self.assertEqual(result['predicted_ties_informative'],2)
        self.assertEqual(result['positive_direction'],0)
        exact=position_pairs(a,a)
        self.assertEqual(exact['positive_direction'],2)
        self.assertEqual(exact['vector_difference_RMSE_m'],0.)

    def test_ranking_excludes_real_ties_and_reports_predicted_ties(self):
        r=progress_ranking([dict(real_progress_difference=a,predicted_progress_difference=b)
                            for a,b in ((1,2),(1,0),(-1,1),(0,2))])
        self.assertEqual(r,dict(declared_pairs=4,real_progress_informative=3,real_progress_flat=1,correct_sign=1,predicted_ties=1))

    def test_oracle_gate_cannot_be_met_when_nominal_is_already_optimal(self):
        pairs=[dict(scene=20,anchor='stable_grasp',first='hold',second=k,
                    real_progress_difference=-v) for k,v in [('nominal',1),('left',.5),('right',.5),('grip_timing',1)]]
        self.assertEqual(oracle_opportunities(pairs)['improvable_vs_nominal'],0)
        pairs[1]['real_progress_difference']=-2
        self.assertEqual(oracle_opportunities(pairs)['improvable_vs_nominal'],1)


if __name__=='__main__':unittest.main()
