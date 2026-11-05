import json,unittest
from pathlib import Path
from go2arx5_vla.world_model.pilot2_supported import contract,stage_for,score,GOALS


class SupportedGoals(unittest.TestCase):
    def episode(self,mode='CLOSE_RELEASE',goal=GOALS[1],close=64):
        e=json.loads((Path(__file__).parents[2]/'experiments/wm02/pilot2_supported_collection_v1.json').read_text())
        e.update(command_program=mode,task_goal=goal,close_start_atom=close);return e

    def events(self,e):
        rows=[]
        for step in range(2504,6704):
            atom=(step-2504)//25;stage=stage_for(atom,e)
            grasp=stage=='SUPPORTED_HOLD'or stage=='CLOSE'and atom>=e['close_start_atom']+24
            rows.append(dict(native_step=step,stage=stage,native_grasp=grasp,table_support=True,
                contact_positive=grasp,finger_q=[.018,.018]if grasp else[.042,.042]))
        return rows

    def test_keep_and_release_have_different_predeclared_goal_outcomes(self):
        for close in (56,64,72):
            e=self.episode('CLOSE_KEEP',GOALS[0],close);rows=self.events(e)
            self.assertTrue(score(rows,e)['goal_success']);self.assertFalse(score(rows,dict(e,task_goal=GOALS[1]))['goal_success'])
            e=self.episode('CLOSE_RELEASE',GOALS[1],close);rows=self.events(e)
            self.assertTrue(score(rows,e)['goal_success']);self.assertFalse(score(rows,dict(e,task_goal=GOALS[0]))['goal_success'])

    def test_open_only_is_not_successful_release_from_grasp(self):
        e=self.episode('OPEN_ONLY');r=score(self.events(e),e)
        self.assertFalse(r['qualified_release']);self.assertFalse(r['goal_success'])

    def test_incomplete_native_tick_labels_and_new_action_ranges_rejected(self):
        e=self.episode()
        with self.assertRaises(ValueError):score(self.events(e)[:-1],e)
        for change in [dict(close_start_atom=48),dict(close_logical_opening=.2),dict(task_atoms=200),dict(task_goal='LIFT')]:
            with self.assertRaises(PermissionError):contract(dict(e,**change))


if __name__=='__main__':unittest.main()
