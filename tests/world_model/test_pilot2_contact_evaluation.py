import unittest
from collections import Counter
from go2arx5_vla.world_model.pilot2_contact_evaluation import panel,summarize


class ContactComparison(unittest.TestCase):
    def plan(self):
        variants=[dict(name=x,method=x)for x in ['GOAL_FSM','CONSTANT_KEEP','CONSTANT_RELEASE']]
        variants += [dict(name=x,method='TRAINED_GOAL_POLICY')for x in ['FSM_STUDENT','WM_STUDENT']]
        return panel([dict(parent='V'+str(i),scene_id='S'+str(i),split='DEV')for i in range(4)],variants,seed=363701)

    def test_single_then_repeat_balances_each_condition_across_slots(self):
        p=self.plan();self.assertEqual(p,self.plan());self.assertEqual(p['planned_evaluations'],80)
        self.assertEqual(len(p['single_rows']),40);self.assertEqual(len(p['paired_repeat_rows']),40)
        rows=p['single_rows']+p['paired_repeat_rows']
        c=Counter((r['parent'],r['task_goal'],r['variant'],r['slot'])for r in rows)
        self.assertTrue(all(v==1 for v in c.values()));self.assertEqual(len(c),80)
        for i in range(0,40,2):self.assertEqual([r['slot']for r in p['paired_repeat_rows'][i:i+2]],[0,1])
        self.assertTrue(p['qualification_requires_native_safe_eligible_prefix_not_task_success'])

    def test_invalid_prefix_failure_and_missing_results_keep_separate_denominators(self):
        p=self.plan();obs=[dict(r,attempted=True,native_passed=True,evidence_valid=True,
            decision_prefix_eligible=True,safety_stop=False,task_success=True)
            for r in p['single_rows']+p['paired_repeat_rows']]
        obs[0]['decision_prefix_eligible']=False;obs[1]['evidence_valid']=False;obs[2]['native_passed']=False
        r=summarize(p,obs[:-1]);self.assertEqual(r['attempted'],79);self.assertEqual(r['not_run'],1)
        self.assertEqual(r['ineligible_prefix'],1);self.assertEqual(r['invalid_evidence'],1);self.assertEqual(r['native_failed'],1)
        self.assertEqual(sum(v['strict_successes']for v in r['variants'].values()),76)
        self.assertFalse(r['CONTACT_CONTROL_GAIN'])
        with self.assertRaises(ValueError):summarize(p,obs+[obs[0]])


if __name__=='__main__':unittest.main()
