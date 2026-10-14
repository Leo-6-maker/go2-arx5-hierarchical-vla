"""CPU plans only; no model, real data or physics claim."""
import copy
import unittest
from .continuation import PARENT,STATS
from .policy_diagnostic import diagnostic_plan


class DiagnosticPlanTests(unittest.TestCase):
    def fixture(self):
        cards=[];episodes=[]
        for group in ('F0/000','F1/000'):
            for task in ('T1','T2'):
                identifier=group+'/'+task
                cards.append(dict(original_attempt_id=identifier,scene_group=group,task=task))
                episodes.append(dict(id=identifier,task=task,split='TRAIN',windows=601,
                    stage_anchors={p:[n] for n,p in enumerate(('approach','contact','transport','withdraw','hold'))}))
        return dict(run_uuid=PARENT,panels={'M512_TRAIN':cards}),dict(statistics_sha256=STATS,episodes=episodes)

    def test_48_real_stage_draws_and_same_noise_across_language_pair(self):
        panel,manifest=self.fixture();plan=diagnostic_plan(panel,manifest)
        self.assertEqual(len(plan['draws']),48)
        left=[r['seed'] for r in plan['draws'] if r['task']=='T1']
        right=[r['seed'] for r in plan['draws'] if r['task']=='T2']
        self.assertEqual(left,right)
        self.assertNotEqual(left[0],left[1])
        self.assertTrue(plan['not_closed_loop'])

    def test_reject_dev_stats_or_replaced_episode(self):
        panel,manifest=self.fixture()
        for key,value in [('split','DEV'),('task','other')]:
            bad=copy.deepcopy(manifest);bad['episodes'][0][key]=value
            with self.assertRaises(PermissionError):diagnostic_plan(panel,bad)
        manifest['statistics_sha256']='0'*64
        with self.assertRaises(PermissionError):diagnostic_plan(panel,manifest)


if __name__=='__main__':unittest.main()
