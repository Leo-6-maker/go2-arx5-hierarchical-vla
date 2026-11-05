import unittest
from scripts.wm02.train_pilot2_matrix import validate_design,PLAN_SHA
from go2arx5_vla.world_model.actor_sampling import MODES


class Pilot2Matrix(unittest.TestCase):
    def spec(self):
        return dict(pilot2_plan_sha=PLAN_SHA,study_arms=[dict(dataset=d,sampler=m,seed=s)
            for d in ('OLD_24','EXPANDED_53')for m in MODES for s in (36001,36002,36003)],
            endpoint=96000,arm_endpoint=8000,batch_size=128,learning_rate=.001,physics=False,kind='ACTOR_STUDY',
            study_datasets=dict(OLD_24=dict(TRAIN_windows=24),EXPANDED_53=dict(TRAIN_windows=53)))

    def test_full_factorial_and_actual_total_updates(self):
        s=self.spec();self.assertEqual(len(validate_design(s)),12)
        self.assertEqual(len(s['study_arms'])*s['arm_endpoint'],s['endpoint'])
        for epochs in (20,40):
            self.assertEqual(epochs*24*256//128*128/(24*256),epochs)
            self.assertEqual(epochs*53*256//128*128/(53*256),epochs)

    def test_missing_duplicate_arm_or_wrong_recipe_is_rejected(self):
        for change in ('duplicate','endpoint','data'):
            s=self.spec()
            if change=='duplicate':s['study_arms'][-1]=s['study_arms'][0]
            elif change=='endpoint':s['endpoint']=8000
            else:s['study_datasets']['OLD_24']['TRAIN_windows']=25
            with self.assertRaises(ValueError):validate_design(s)


if __name__=='__main__':unittest.main()
