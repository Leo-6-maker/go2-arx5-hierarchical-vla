import copy,unittest
from go2arx5_vla.world_model.fusion_evaluation import panel,validate
from go2arx5_vla.world_model.fusion_corrector import ARMS


class FrozenFusionPanel(unittest.TestCase):
    def setUp(self):
        self.scenes=[dict(parent='newdev'+str(i),scene_id='D'+str(i),split='DEV')for i in range(6)]
        self.plan=panel(self.scenes)

    def test_all_four_known_intents_methods_restarts_and_cpu_slots_are_balanced(self):
        self.assertEqual(len(validate(self.plan)),192)
        self.assertEqual(panel(list(reversed(self.scenes))),self.plan)
        self.assertEqual(self.plan['baseline'],ARMS[1])

    def test_changed_intent_partial_panel_test_or_selected_best_method_rejected(self):
        for key,value in [('split','TEST'),('method','BEST_OF_DEV'),('nominal_action',True),('desired_delta_xy_m',[.1,0]),('slot',1)]:
            with self.subTest(key=key):
                p=copy.deepcopy(self.plan);p['rows'][0][key]=value
                with self.assertRaises(ValueError):validate(p)
        p=copy.deepcopy(self.plan);p['rows'].pop()
        with self.assertRaises(ValueError):validate(p)

    def test_original_parent_bootstrap_supports_explicit_static_baseline_without_test_gain_claim(self):
        from go2arx5_vla.world_model.pilot2_evaluation import analyze
        rows=[dict(row,attempted=True,native_passed=True,evidence_valid=True,safety_stop=False,
            task_success=row['method']==ARMS[3],XY_error_m=.0001 if row['method']==ARMS[3]else.0004,
            objective_m=.0001 if row['method']==ARMS[3]else.0004)for row in self.plan['rows']]
        value=analyze(self.plan,rows,minimum_gain_m=.00005,seed=366502,bootstrap=100,baseline=ARMS[1])
        self.assertEqual(value['planned'],192);self.assertEqual(value['not_run'],0)
        self.assertTrue(value['comparisons'][ARMS[3]]['DEV_GAIN_CANDIDATE'])
        self.assertFalse(value['comparisons'][ARMS[3]]['LOCAL_CONTROL_GAIN'])
        value=analyze(self.plan,rows[:-1],minimum_gain_m=.00005,seed=366502,bootstrap=100,baseline=ARMS[1])
        self.assertFalse(value['comparisons'][ARMS[3]]['DEV_GAIN_CANDIDATE'])


if __name__=='__main__':unittest.main()
