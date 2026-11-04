import unittest
from go2arx5_vla.world_model.decision_collection import cards
from go2arx5_vla.world_model.contact_identification import cards as old_cards

class NewDecisionData(unittest.TestCase):
    def test_frozen_new_physical_parents_whole_scene_split_and_workspace(self):
        c=cards();self.assertEqual(c,cards());self.assertEqual(len(c),40)
        self.assertEqual([sum(x['ident_partition']==p for x in c)for p in ('FIT','DIAGNOSTIC_DEV','ONCE_HOLDOUT')],[24,8,8])
        key=lambda x:tuple([*x['source_xy'],*x['goal_xy'],x['cube_yaw']])
        self.assertFalse({key(x)for x in c}&{key(x)for x in old_cards()})
        self.assertEqual(len({x['parent_id']for x in c}),40)
        self.assertTrue(all(x['split']=='TRAIN'and .442<=x['source_xy'][0]<=.458 for x in c))

if __name__=='__main__':unittest.main()
