import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tests.world_model.test_pilot2_supported_decision import supported
from go2arx5_vla.world_model.pilot2_supported_policy_runtime import choose,contract,SCHEMA


class PausedSupportedPolicy(unittest.TestCase):
    def episode(self,goal='SUPPORTED_GRASP_HOLD',method='GOAL_FSM'):
        e=supported('DEV','Q','CLOSE_KEEP')['episode']
        return dict(e,schema=SCHEMA,purpose='SUPPORTED_POLICY_EVALUATION',
            command_program='SELECT_KEEP_RELEASE_AFTER_SUPPORTED_HOLD',policy_decision_atom=112,
            method=method,task_goal=goal)

    def test_goal_rule_chooses_before_integration_using_only_actual_prefix(self):
        for goal,program in [('SUPPORTED_GRASP_HOLD','CLOSE_KEEP'),('RELEASE_FROM_SUPPORTED_GRASP','CLOSE_RELEASE')]:
            with tempfile.TemporaryDirectory() as tmp:
                rows=supported('DEV','Q','CLOSE_KEEP')['samples'][:112];events=[]
                a=SimpleNamespace(output=Path(tmp),spec=dict(source_sha='SYNTHETIC_FIXTURE'),
                    observe=lambda:dict(step=5303),command_event=events.append)
                e=choose(a,self.episode(goal),rows);self.assertEqual(e['command_program'],program)
                d=json.loads((Path(tmp)/'SUPPORTED_POLICY_DECISION.private.json').read_text())
                self.assertTrue(d['decision_prefix_eligible']);self.assertFalse(d['future_response_observed'])
                self.assertEqual(events[0]['phase'],'supported-policy-choice-before-install')

    def test_failed_hold_prefix_opens_and_is_marked_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=supported('DEV','Q','CLOSE_KEEP')['samples'][:112]
            rows[-1]['native_contact_events'][-1]['native_grasp']=False
            a=SimpleNamespace(output=Path(tmp),spec=dict(source_sha='SYNTHETIC_FIXTURE'),
                observe=lambda:dict(step=5303),command_event=lambda e:None)
            result=choose(a,self.episode(),rows);self.assertEqual(result['command_program'],'CLOSE_RELEASE')
            d=json.loads((Path(tmp)/'SUPPORTED_POLICY_DECISION.private.json').read_text())
            self.assertFalse(d['decision_prefix_eligible'])

    def test_unqualified_timing_and_unknown_policy_rejected(self):
        for change in [dict(close_start_atom=64),dict(policy_decision_atom=120),dict(method='UNQUALIFIED')]:
            with self.assertRaises(PermissionError):contract(dict(self.episode(),**change))


if __name__=='__main__':unittest.main()
