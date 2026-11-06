import copy
import unittest
from collections import Counter
from go2arx5_vla.world_model.fusion_identification import panel, validate


class BalancedResponse(unittest.TestCase):
    def setUp(self):
        self.scenes = [dict(parent='new-'+str(i), scene_id='F'+str(i), split='TRAIN' if i<4 else 'DEV') for i in range(6)]
        self.value = panel(self.scenes)

    def test_whole_parent_split_and_independent_restart_slot_balance(self):
        rows=validate(self.value)
        self.assertEqual(len(rows), 48)
        self.assertEqual(Counter(r['split'] for r in rows), dict(TRAIN=32, DEV=16))
        self.assertEqual(panel(list(reversed(self.scenes))), self.value)
        self.assertFalse(self.value['actual_counterfactual_forks'])

    def test_no_future_expert_test_or_score_selection(self):
        mutations=[('split','TEST'),('purpose','EVALUATION'),('method','B3_HISTORY_CONDITIONAL_WM_CORRECTION'),
            ('sampling_source','SELECTED_SUCCESSES'),('desired_delta_xy_m',[.1,0]),('nominal_action',True),
            ('inference_mode','REAL_VLA'),('repeat_index',2),('slot',1)]
        for key,value in mutations:
            with self.subTest(key=key):
                plan=copy.deepcopy(self.value);plan['rows'][0][key]=value
                with self.assertRaises(ValueError):validate(plan)

    def test_duplicate_response_and_parent_alias_rejected(self):
        v=copy.deepcopy(self.value);v['rows'][2]=copy.deepcopy(v['rows'][0])
        with self.assertRaises(ValueError):validate(v)
        v=copy.deepcopy(self.value)
        for row in v['rows']:
            if row['parent']=='new-5':row['scene_id']='F4'
        with self.assertRaises(ValueError):validate(v)


if __name__=='__main__':unittest.main()
