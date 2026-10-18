import json
from pathlib import Path
import tempfile
import time
import unittest

from scripts.pi0294.pv1.core_ledger import Ledger
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.scope import EVAL_ROLE


def fixture(tmp):
    parent=Path(tmp)/'campaign';root=parent/'children/policy-01';root.mkdir(parents=True)
    def put(p,v):p.write_text(json.dumps(v))
    now=time.time();lease=parent/'LEASE.json'
    put(lease,dict(status='ACTIVE',run_uuid=root.name,authority_release=AUTHORITY))
    put(root/'OUTPUT_BINDING.private.json',dict(lease=str(lease)))
    put(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',dict(authority_release=AUTHORITY,T0_epoch=now-1,
        stop_new_physics_epoch=now+3600,hard_closeout_epoch=now+7200))
    put(root/'RUN_MANIFEST.json',dict(scope=EVAL_ROLE,authority=dict(protocol_ref=AUTHORITY),
        limits=dict(physics_ticks=32000,engine_starts=100,policy_episodes=1,CPU_s=3600,operation_deadline_epoch=now+3600)))
    put(root/'RESOURCE_LEDGER.json',dict(schema='PI0294_PV1_CORE_POLICY_LEDGER_V1',run_uuid=root.name,
        authority_release=AUTHORITY,unknown_effects=[],live_and_recovery_integrations=0,shadow_integrations=0,
        engine_starts=0,main_starts=0,Core_CPU_s=0.,reservations={}))
    return root,Ledger(root)


class CoreCounterContracts(unittest.TestCase):
    def test_full_live_reserve_and_actual_rejected_shadow_cost_not_fictitious_16000(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,l=fixture(tmp);l.charge('main_starts',1);l.charge('engine_starts',1)
            with l.steps('live_and_recovery_integrations',16000,'case') as live:
                with l.steps('shadow_integrations',200,'S1-0') as shadow:shadow['actual']=37
                live['actual']=0
            v=json.loads(l.path.read_text())
            self.assertEqual(v['shadow_integrations'],37)
            self.assertEqual(v['live_and_recovery_integrations'],0)
            self.assertEqual(v['reservations'],{})
            with self.assertRaises(PermissionError):l.charge('main_starts',1)

    def test_uncertain_native_count_keeps_reservation_and_blocks_further_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,l=fixture(tmp)
            with self.assertRaises(RuntimeError):
                with l.steps('shadow_integrations',200,'S1-0'):
                    raise RuntimeError('UNKNOWN_NATIVE_STEP_COUNT')
            v=json.loads(l.path.read_text())
            self.assertIn('S1-0',v['reservations'])
            self.assertTrue(v['unknown_effects'])
            with self.assertRaises(PermissionError):l.charge('engine_starts',1)
            l.cpu(.25)
            self.assertEqual(json.loads(l.path.read_text())['Core_CPU_s'],.25)

    def test_core_cannot_charge_a_neural_generation_or_ignore_full_tick_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,l=fixture(tmp)
            with self.assertRaises(ValueError):l.charge('generation_queries',1)
            with self.assertRaises(PermissionError):
                with l.steps('live_and_recovery_integrations',32001,'excess'):self.fail('NOT_ADMITTED')
