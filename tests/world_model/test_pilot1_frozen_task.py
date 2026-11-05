import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.wm02.train_pilot1_actor import task_definition
from go2arx5_vla.world_model.storage import digest


class FrozenTaskCoverage(unittest.TestCase):
    def test_coverage_changes_cannot_recompute_goals_or_success_threshold(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'task';task=dict(schema='LOCAL_EEF_REACH_PILOT_V1',success_tolerance_m=.00018584456642062617,goal_seed=36001);p.write_text(json.dumps(task));s=dict(frozen_task=str(p),frozen_task_sha256=digest(p))
            with patch('go2arx5_vla.world_model.pilot1_actor.task_from_train',side_effect=RuntimeError('must-not-refit-task')):
                self.assertEqual(task_definition(s,None,'new-train-data','new-metadata',[]),task)
                self.assertEqual(task_definition(s,dict(task=task),None,None,[]),task)
                with self.assertRaises(ValueError):task_definition(s,dict(task=dict(task,success_tolerance_m=.1)),None,None,[])
            p.write_text(json.dumps(dict(task,goal_seed=999)))
            with self.assertRaisesRegex(ValueError,'FROZEN_ACTOR_TASK_CHANGED'):task_definition(s,None,None,None,[])

    def test_legacy_resume_keeps_its_original_task(self):
        task=dict(schema='legacy',goal='unchanged');self.assertIs(task_definition({},dict(task=task),None,None,[]),task)


if __name__=='__main__':unittest.main()
