import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.resources import TOTAL
from scripts.pi0294.pv1.neural_ledger import Ledger,check


def value():
    return dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',authority_release=AUTHORITY,
        campaign_run_uuid='ORIGINAL_CAMPAIGN',CPU_s=0.,unknown_effects=[],revision=0,
        used={k:0 for k in TOTAL},reserved={k:0 for k in TOTAL},
        optimizer_updates=0,generation_queries=0,model_loads=0,PBS_submissions=[],pending_updates=[],training_runs={})


class NeuralLedgerContracts(unittest.TestCase):
    def test_read_only_closeout_preserves_STOP_and_unknowns_without_admitting_new_effects(self):
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.pi0294.pv1.neural_ledger.time.time',return_value=1000):
            parent=Path(tmp)/'campaign';root=parent/'children/policy-01';root.mkdir(parents=True)
            p=parent/'HOPPER_RESOURCE_LEDGER.json';p.write_text(json.dumps(value()))
            (parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json').write_text(json.dumps(dict(run_uuid='ORIGINAL_CAMPAIGN',
                T0_epoch=900,stop_new_physics_epoch=1100,hard_closeout_epoch=1200)))
            lease=root/'lease.json';lease.write_text(json.dumps(dict(status='ACTIVE',run_uuid=root.name,
                authority_release=AUTHORITY,deadline_epoch=1200)))
            (root/'OUTPUT_BINDING.private.json').write_text(json.dumps(dict(lease=str(lease))))
            stop=root/'STOP.json';stop.write_bytes(b'PRESERVED_ENGINEERING_STOP')
            ledger=Ledger(root)
            with self.assertRaises(PermissionError):ledger.guard()
            ledger.guard(closeout=True)
            v=json.loads(p.read_text());v['unknown_effects']=['PRESERVED'];p.write_text(json.dumps(v))
            ledger.guard(closeout=True)
            with self.assertRaises(PermissionError):ledger.charge('generation_queries',1)
            with self.assertRaises(PermissionError):ledger.guard(closeout='yes')
            self.assertEqual(stop.read_bytes(),b'PRESERVED_ENGINEERING_STOP')
            self.assertEqual(json.loads(p.read_text())['unknown_effects'],['PRESERVED'])

    def test_native_queries_and_optimizer_updates_share_original_campaign_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'campaign';root=parent/'children/policy-01';root.mkdir(parents=True)
            p=parent/'HOPPER_RESOURCE_LEDGER.json';p.write_text(json.dumps(value()))
            (parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json').write_text(json.dumps(dict(run_uuid='ORIGINAL_CAMPAIGN')))
            ledger=Ledger(root)
            ledger.charge('generation_queries',3)
            with ledger.transaction() as v:v['optimizer_updates']+=1
            v=json.loads(p.read_text())
            self.assertEqual(v['used']['neural_queries'],3)
            self.assertEqual(v['used']['optimizer_updates'],1)
            self.assertEqual(v['revision'],2)
            ledger.cpu(.5)
            self.assertEqual(json.loads(p.read_text())['CPU_s'],.5)
            before=p.read_bytes()
            with self.assertRaises(PermissionError):ledger.charge('generation_queries',39000)
            self.assertEqual(before,p.read_bytes())
            # Closing cost remains recordable after a fault, but no further draw is allowed.
            v=json.loads(p.read_text());v['unknown_effects']=['PRESERVED_UNKNOWN'];p.write_text(json.dumps(v))
            ledger.cpu(.25)
            self.assertEqual(json.loads(p.read_text())['CPU_s'],.75)
            before=p.read_bytes()
            with self.assertRaises(PermissionError):ledger.charge('generation_queries',1)
            self.assertEqual(before,p.read_bytes())

    def test_unknown_effects_desynchronized_counts_and_final_flag_fail_closed(self):
        for mutate in ('unknown','counter','CPU','PBS','final'):
            v=value();args={}
            if mutate=='unknown':v['unknown_effects']=['UNKNOWN_NATIVE_EFFECT']
            if mutate=='counter':v['optimizer_updates']=1
            if mutate=='CPU':v['CPU_s']=True
            if mutate=='PBS':v['used']['PBS_submissions']=1
            if mutate=='final':args['final']='pretend'
            with self.subTest(mutate=mutate):
                with self.assertRaises(PermissionError):check(v,**args)
