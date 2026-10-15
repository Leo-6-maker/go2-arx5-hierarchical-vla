"""CPU-only dependency checks explicitly required by the adopted direction."""
import copy
import hashlib
import json
import unittest
from .task_capture_scope import (
    SCOPE, LEGACY_SCOPE, HOLD_FAIL, HARD_GATES, DEMO_GATES, EXECUTOR_KEYS,
    historical_hold_classification, assess_capture, admit_success_demo,
)


class CaptureScopeTests(unittest.TestCase):
    def setUp(self):
        pins = {k: hashlib.sha256(k.encode()).hexdigest() for k in EXECUTOR_KEYS}
        self.args = dict(scope=SCOPE, stage='pilot', checks=dict.fromkeys(HARD_GATES, 'PASS'),
            resource=dict(authorizes_execution=True, lease_status='ACTIVE', budget_reserved=True,
                          start=10., new_work_cutoff=20., deadline=25.),
            teacher_executor=pins, model_executor=copy.deepcopy(pins),
            historical_hold_verdict=HOLD_FAIL, now=15.)

    def test_historical_hold_does_not_block_task_pilot(self):
        out = assess_capture(**self.args)
        self.assertEqual(out['blocked'], [])
        self.assertEqual(out['history'], 'DIAGNOSTIC_ONLY_HISTORICAL_WBC_HOLD')
        self.assertFalse(out['dispatch_performed'])
        self.assertFalse(out['resource_authorization_granted'])

    def test_each_missing_unknown_or_failed_hard_gate_blocks(self):
        for gate in HARD_GATES:
            for value in ('UNKNOWN', 'FAIL', True, None):
                with self.subTest(gate=gate, value=value):
                    args = copy.deepcopy(self.args)
                    args['checks'][gate] = value
                    self.assertIn(gate, assess_capture(**args)['blocked'])

    def test_failed_task_not_success_data(self):
        checks = dict.fromkeys(DEMO_GATES, 'PASS')
        self.assertFalse(admit_success_demo(checks=checks, task_success=False)['admit_success_demo'])
        self.assertFalse(admit_success_demo(checks=checks, task_success=1)['admit_success_demo'])

    def test_success_can_coexist_with_tracking_diagnostic_fail(self):
        checks = dict.fromkeys(DEMO_GATES, 'PASS')
        checks['hold_quality'] = 'FAIL'
        checks['hold_SO3_RMS_rad'] = .07
        self.assertTrue(admit_success_demo(checks=checks, task_success=True)['admit_success_demo'])
        for gate in DEMO_GATES:
            bad = dict(checks, **{gate: 'FAIL'})
            self.assertFalse(admit_success_demo(checks=bad, task_success=True)['admit_success_demo'])

    def test_expired_future_nonfinite_or_missing_resource_blocks(self):
        for now in (9., 20., 25., float('nan'), float('inf'), True):
            self.assertTrue(assess_capture(**dict(self.args, now=now))['blocked'])
        for key, value in (('authorizes_execution', False), ('lease_status', 'CLOSED'),
                           ('budget_reserved', False), ('start', None), ('deadline', 19.)):
            args = copy.deepcopy(self.args)
            args['resource'][key] = value
            self.assertTrue(assess_capture(**args)['blocked'])

    def test_same_frozen_teacher_model_executor_required(self):
        for key in EXECUTOR_KEYS:
            args = copy.deepcopy(self.args)
            args['model_executor'][key] = 'f' * 64
            self.assertTrue(assess_capture(**args)['blocked'])
        args = copy.deepcopy(self.args)
        args['teacher_executor'] = args['model_executor'] = dict.fromkeys(EXECUTOR_KEYS, 'UNKNOWN')
        self.assertTrue(assess_capture(**args)['blocked'])

    def test_production_requires_task_pilot_pilot_does_not(self):
        args = dict(self.args, stage='production')
        self.assertIn('task_pilot', assess_capture(**args)['blocked'])
        args['checks'] = dict(args['checks'], task_pilot='PASS')
        self.assertEqual(assess_capture(**args)['blocked'], [])

    def test_history_immutable_and_old_scope_stays_failed(self):
        old = dict(status=HOLD_FAIL, SO3_RMS_rad=.0696509067107775)
        before = json.dumps(old, sort_keys=True).encode()
        self.assertEqual(historical_hold_classification(LEGACY_SCOPE, old['status']), HOLD_FAIL)
        self.assertTrue(assess_capture(**dict(self.args, scope=LEGACY_SCOPE))['blocked'])
        self.assertEqual(before, json.dumps(old, sort_keys=True).encode())

    def test_unknown_scope_and_unrelated_failure_not_reclassified(self):
        with self.assertRaises(ValueError):
            historical_hold_classification('UNKNOWN', HOLD_FAIL)
        with self.assertRaises(ValueError):
            historical_hold_classification(SCOPE, 'NATIVE_SAFETY_FAIL')


if __name__ == '__main__':
    unittest.main()
