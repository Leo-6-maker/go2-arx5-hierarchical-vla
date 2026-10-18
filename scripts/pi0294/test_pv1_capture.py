"""CPU-only plan, accounting and saved-negative contracts, no simulation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.pi0294.pv1.capture import Meter, capture_cards
from scripts.pi0294.pv1.negative_audit import audit
from scripts.pi0294.pv1.plan import original_group
from scripts.pi0294.fd1 import sha


class CaptureTests(unittest.TestCase):
    def test_only_train_two_fixed_references_counterbalanced(self):
        groups=[original_group('TRAIN','F1',i,0) for i in range(2)]
        c=capture_cards(groups,'CLEARANCE_PAIR')
        self.assertEqual([(r['task'],r['nominal_clearance_m']) for r in c],
            [('T1',.004),('T2',.004),('T2',.012),('T1',.004),('T2',.012),('T2',.004)])
        self.assertEqual(len(capture_cards(groups,'COVERAGE_FIRST')),4)
        for task in ['DEV','LOCKBOX']:
            with self.assertRaises(PermissionError):
                capture_cards([original_group(task,'F0',0,0)],'CLEARANCE_PAIR')
        with self.assertRaises(PermissionError):capture_cards(groups,'HEIGHT_SEARCH')

    def test_original_native_exception_still_charges_actual_tick(self):
        meter=object.__new__(Meter)
        meter.guard=lambda:None
        meter.m=dict(limits=dict(physics_ticks=32000))
        meter.ledger=dict(unknown_effects=[])
        meter.ticks=meter.live_ticks=meter.shadow_ticks=0
        engine=SimpleNamespace(steps=0)
        def step(e,*args):e.steps+=1;raise RuntimeError('ORIGINAL_NATIVE_SAFETY')
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL_NATIVE_SAFETY'):
            meter.step(engine,step,None,None,True)
        self.assertEqual((meter.ticks,meter.live_ticks,meter.shadow_ticks),(1,1,0))
        meter.ticks=32000
        with self.assertRaises(PermissionError):meter.step(engine,lambda *a:self.fail(),None,None,False)

    def test_unknown_native_step_count_cannot_be_silently_retried(self):
        meter=object.__new__(Meter)
        meter.guard=lambda:None
        meter.m=dict(limits=dict(physics_ticks=32000))
        meter.ledger=dict(unknown_effects=[])
        meter.ticks=meter.live_ticks=meter.shadow_ticks=0
        engine=SimpleNamespace(steps=0)
        def step(e,*args):e.steps+=2
        with self.assertRaisesRegex(RuntimeError,'UNKNOWN_NATIVE_STEP'):
            meter.step(engine,step,None,None,True)
        self.assertTrue(meter.ledger['unknown_effects'])

    def test_zero_live_s1_rejection_is_negative_not_demonstration_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'episodes/ZERO';folder.mkdir(parents=True)
            def put(name,value):(folder/name).write_text(json.dumps(value))
            put('samples.json',[]);put('references.json',[])
            put('S1_PREFLIGHTS.json',[dict(decision='REJECT',sequence=0)])
            for c in ['body_rgb','wrist_rgb']:
                (folder/c).mkdir();(folder/c/'0000.png').write_bytes(b'FIXTURE_FRAME_IDENTITY_ONLY')
            np.savez(folder/'highrate.npz',rows=np.zeros((0,44)),external_force_max=np.zeros((0,2)))
            put('WORKER_RECEIPT.json',dict(source_sha='a'*40,physics_steps=0,samples_sha256=sha(folder/'samples.json')))
            put('FILE_MANIFEST.json',[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha(p))
                for p in folder.rglob('*') if p.is_file()])
            result=dict(status='FAILED_PRESERVED',physics_steps=0,source_sha='a'*40,rgb_pairs=1,
                error='PermissionError: PRE_EXECUTION_SAFETY_REJECT')
            value=audit(root,dict(id='ZERO'),result,dict(pre_execution_rejections=1))
            self.assertFalse(value['admitted'])
            self.assertFalse(value['complete'])
            self.assertEqual(value['rejected_prefix_live_ticks'],0)
            self.assertTrue(value['valid_scientific_negative_not_engineering_replacement'])
            self.assertEqual(value['engine_starts'],0)
