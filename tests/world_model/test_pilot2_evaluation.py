import copy,unittest
from collections import Counter
from go2arx5_vla.world_model.pilot2_evaluation import panel,analyze,minimum_practical_gain,METHODS


class RepeatedEvaluation(unittest.TestCase):
    def plan(self,split='DEV'):
        task=dict(goal_seed=36001,jitter_m=.0002,goals_per_center=64,
            mean_response_xy_m=[[0,0],[.001,0],[-.001,0],[0,.001]])
        return panel([dict(parent=f'P{i}',scene_id=f'S{i}',split=split)for i in range(6)],task,seed=363401,split=split)

    def rows(self,p):
        return [dict(r,attempted=True,native_passed=True,evidence_valid=True,safety_stop=False,task_success=True,
            XY_error_m=.0003 if r['method']==METHODS[0]else.0001,
            objective_m=.0003 if r['method']==METHODS[0]else.0001)for r in p['rows']]

    def test_full_parent_goal_method_repeat_distribution_balances_slots_and_is_frozen(self):
        p=self.plan();self.assertEqual(p,self.plan());self.assertEqual(len(p['rows']),144)
        c=Counter((r['parent'],r['goal_index'],r['method'],r['slot'])for r in p['rows'])
        self.assertTrue(all(v==1 for v in c.values()));self.assertEqual(len(c),144)
        self.assertFalse(p['exact_hidden_state_fork'])

    def test_dev_effect_is_not_test_control_gain_and_primary_test_only(self):
        for split in ['DEV','TEST']:
            p=self.plan(split);r=analyze(p,self.rows(p),minimum_gain_m=.00005,bootstrap=150)
            actor=r['comparisons']['TRAINED_ACTOR'];arx=r['comparisons']['FROZEN_ARX_SELECTOR']
            self.assertAlmostEqual(actor['parent_equal_mean_cost_improvement_m'],.0002)
            self.assertEqual(actor['LOCAL_CONTROL_GAIN'],split=='TEST');self.assertFalse(arx['LOCAL_CONTROL_GAIN'])
            self.assertEqual(actor['DEV_GAIN_CANDIDATE'],split=='DEV')

    def test_invalid_missing_and_failed_native_stay_in_denominators(self):
        p=self.plan('TEST');rows=self.rows(p);rows[0]['evidence_valid']=False;rows[1]['native_passed']=False
        r=analyze(p,rows[:-1],minimum_gain_m=.00005,bootstrap=150)
        self.assertEqual(r['attempted'],143);self.assertEqual(r['not_run'],1)
        self.assertEqual(r['invalid_evidence'],1);self.assertEqual(r['native_failed'],1)
        self.assertFalse(any(v['LOCAL_CONTROL_GAIN']for v in r['comparisons'].values()))
        with self.assertRaises(ValueError):analyze(p,rows+[rows[0]],minimum_gain_m=.00005,bootstrap=150)

    def test_practical_gain_uses_train_noise_and_original_task_utility(self):
        r=minimum_practical_gain(dict(fit_split='TRAIN',same_action_restart_gap_m=[[.0002]*4]*12),dict(success_tolerance_m=.00018))
        self.assertEqual(r['minimum_practical_improvement_m'],.0001);self.assertFalse(r['TEST_used'])


if __name__=='__main__':unittest.main()
