"""Small stdlib tests; no project runtime, network or heavy dependencies."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('wm01_check', ROOT/'check_plan.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BASE = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))


class PlanTests(unittest.TestCase):
    def reject(self, path, value):
        plan = copy.deepcopy(BASE)
        obj = plan
        for key in path[:-1]:
            obj = obj[key]
        obj[path[-1]] = value
        self.assertTrue(MODULE.validate(plan), path)

    def test_valid(self):
        self.assertEqual(MODULE.validate(BASE), [])

    def test_short_sha(self):
        self.reject(['baseline_commit'], '0a27753b')

    def test_runtime_claim(self):
        self.reject(['runtime_status'], 'NATIVE_READY')

    def test_wbc_change(self):
        self.reject(['scope','target_wbc_frozen'], False)

    def test_real_robot(self):
        self.reject(['scope','real_robot_execution'], True)

    def test_old_jobs(self):
        self.reject(['scope','touch_existing_runs'], True)

    def test_time(self):
        self.reject(['interface','macro_dt_s'], 0.2)

    def test_suffix(self):
        self.reject(['interface','future_suffix_is_executed_label'], True)

    def test_actor_shape(self):
        self.reject(['interface','actor_parameter_dim'], 400)

    def test_split(self):
        self.reject(['data','normalization_split'], 'TEST')

    def test_test_count(self):
        self.reject(['evaluation','final_test_episodes'], 50)

    def test_gradient_detach(self):
        self.reject(['imagination','retain_action_gradients'], False)

    def test_model_training_during_actor_update(self):
        self.reject(['imagination','freeze_wm_parameters'], False)

    def test_duplicate_seed(self):
        self.reject(['policy','seeds'], [1103,1103,1105])

    def test_stage_cycle(self):
        self.reject(['stages',0,'depends_on'], ['W4'])

    def test_invented_results(self):
        self.reject(['actual_results_this_publication','wm_updates'], 200)

    def test_malformed(self):
        self.assertTrue(MODULE.validate({}))


if __name__ == '__main__':
    unittest.main()
