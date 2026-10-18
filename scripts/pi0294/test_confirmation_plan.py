from copy import deepcopy
import unittest

from scripts.pi0294.mp1 import confirmation_plan as p
from scripts.pi0294.scene_cards import digest,scene_group,task_card


class ConfirmationPlanTests(unittest.TestCase):
    def matrix(self):
        cells=[]
        for model,successes in [('S-BC-2048',(2,1)),('S-CORR-2048',(4,1))]:
            for task,total in zip(['T1','T2'],successes):
                for n in range(4):cells.append(dict(model=model+'-DEV8',task=task,valid_policy_outcome=True,
                    status='SUCCESS' if n<total else 'VALID_TASK_FAIL'))
        return dict(actual_safety_stop_present=False,cells=cells,checkpoint_manifest_identities=p.CHECKPOINTS)

    def fixture(self):
        candidate=p.rank_candidate(self.matrix());groups=p.candidate_groups([])['groups'];receipts=[]
        for g in groups:
            for task in ['T1','T2']:
                receipts.append(dict(scene_group=g['group_id'],task=task,status='STATIC_PASS_PENDING_LIVE_CONTACT',
                    geometry_passed=True,initial_safety_valid=True,field_of_view=dict(passed=True),integration_steps=0,
                    task_success_claim=False,object_pose_modified=False,card_sha256=digest(task_card(g,task))))
        return candidate,groups,receipts

    def test_original_ranking_does_not_invent_a_rate_threshold(self):
        value=p.rank_candidate(self.matrix());self.assertEqual(value['candidate'],'S-CORR-2048')
        self.assertEqual(value['ranking'][0]['worst_task'],.25);self.assertEqual(value['ranking'][0]['macro'],.625)
        self.assertIsNone(value['new_DEV_success_threshold']);self.assertFalse(value['selected_for_deployment'])
        for key,value in [('actual_safety_stop_present',True)]:
            m=self.matrix();m[key]=value
            with self.assertRaises(PermissionError):p.rank_candidate(m)
        m=self.matrix();m['cells'][0]['valid_policy_outcome']=False
        with self.assertRaises(PermissionError):p.rank_candidate(m)

    def test_ten_groups_two_tasks_two_banks_one_checkpoint(self):
        c,g,r=self.fixture();value=p.evaluation_plan(g,c,r,[])
        cards=[c for panel in value['panels'].values() for c in panel['cards']]
        self.assertEqual(len(cards),40);self.assertEqual(len({c['id'] for c in cards}),40)
        self.assertEqual(len({c['scene_group'] for c in cards}),10)
        for panel in value['panels'].values():self.assertEqual(panel['checkpoint_manifest_sha256'],p.CHECKPOINTS[p.CANDIDATE])
        self.assertEqual({c['confirmation_noise_bank'] for c in cards},{29201,29202})
        self.assertFalse(value['Teacher_success_screening']);self.assertFalse(value['confirmation_used_for_training'])

    def test_geometric_exclusion_ignores_appearance_family(self):
        first=p.candidate_groups([])['groups'];old=deepcopy(first[0]);old.update(group_id='OLD/TRAIN/F2/000',split='TRAIN',family='F2')
        revised=p.candidate_groups([old]);self.assertNotEqual(digest(first[0]),digest(revised['groups'][0]))
        self.assertTrue(revised['prefreeze_geometric_rejections']);self.assertEqual(revised['new_queries'],0)
        c,g,r=self.fixture()
        with self.assertRaises(PermissionError):p.evaluation_plan(g,c,r,[old])

    def test_static_identity_safety_fov_and_zero_step_rules(self):
        for key,value in [('status','STATIC_FAIL'),('geometry_passed',False),('initial_safety_valid',False),
                ('integration_steps',1),('task_success_claim',True),('object_pose_modified',True),('card_sha256','wrong')]:
            with self.subTest(key=key):
                c,g,r=self.fixture();r[0][key]=value
                with self.assertRaises(PermissionError):p.evaluation_plan(g,c,r,[])
        c,g,r=self.fixture();r[0]['field_of_view']['passed']=False
        with self.assertRaises(PermissionError):p.evaluation_plan(g,c,r,[])
        c,g,r=self.fixture()
        with self.assertRaises(PermissionError):p.evaluation_plan(g,c,r[:-1],[])
        c,g,r=self.fixture();c['confirmation_opened']=True
        with self.assertRaises(PermissionError):p.evaluation_plan(g,c,r,[])

    def test_original_scene_generation_still_deterministic(self):
        self.assertEqual(scene_group('TRAIN','F0',0),scene_group('TRAIN','F0',0))
        self.assertNotEqual(scene_group(p.SPLIT,'F0',0)['seed'],scene_group('TRAIN','F0',0)['seed'])


if __name__=='__main__':unittest.main()
