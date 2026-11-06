import unittest
import numpy as np
from go2arx5_vla.world_model.actor_diagnostics import distributions,grouped_diagnostics,normalizer_diagnostics


class ActorDiagnostics(unittest.TestCase):
    def test_entropy_not_zero_ce_and_equal_teacher_has_zero_kl(self):
        q=np.array([[.7,.1,.1,.1],[.25,.25,.25,.25]])
        r=distributions(np.log(q),q,np.log(q)*.0001,high_margin_m=.00005)
        self.assertAlmostEqual(r['teacher_student_KL'],0.,places=14)
        self.assertGreater(r['soft_cross_entropy'],0.)
        self.assertAlmostEqual(r['soft_cross_entropy'],r['teacher_entropy'])

    def test_parent_and_goal_groups_keep_denominators_and_margins(self):
        q=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1]])
        logits=np.log(q[[1,1,2]]);scores=np.log(q)*.0001
        r=grouped_diagnostics(logits,q,scores,['A','A','B'],[0,1,0],high_margin_m=.00005)
        self.assertEqual(r['by_parent']['A']['examples'],2);self.assertEqual(r['by_parent']['B']['examples'],1)
        self.assertAlmostEqual(r['high_margin_error_rate'],1/3)
        self.assertEqual(r['confusion_teacher_rows_student_columns'][0][1],1)

    def test_normalizer_uses_given_train_statistics_and_reports_outliers(self):
        r=normalizer_diagnostics([[0.,11.]],np.zeros(2),np.ones(2))
        self.assertEqual(r['fraction_absolute_z_above_10'],.5)
        with self.assertRaises(ValueError):normalizer_diagnostics([[0.,1.]],np.zeros(2),np.zeros(2))


if __name__=='__main__':unittest.main()
