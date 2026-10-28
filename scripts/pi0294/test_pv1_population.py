"""Isolated fake static qualifier, never constructs a native model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import population
from scripts.pi0294 import scene
from scripts.pi0294.pv1.plan import original_group
from scripts.pi0294.scene_cards import digest


class PopulationTests(unittest.TestCase):
    def exercise(self, first_fails=False, unknown_error=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in ['plans','evidence']:(root/n).mkdir()
            (root/'plans/EXCLUSIONS.json').write_text('[]')
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(status='RESERVED',unknown_effects=[],
                CPU_reserved_s=7200,engine_starts_reserved=8192)))
            groups=[original_group('TRAIN','F1',0,i) for i in range(2)]
            pool=dict(slots=[dict(candidates=groups)])
            calls=[]
            def qualify(engine,render,guard):
                guard();calls.append(engine.card)
                if unknown_error:raise RuntimeError('UNEXPLAINED_NATIVE_DEPENDENCY')
                fail=first_fails and engine.card['layout_candidate_index']==0
                return dict(status='STATIC_FAIL' if fail else 'STATIC_PASS_PENDING_LIVE_CONTACT',
                    geometry_passed=not fail,field_of_view=dict(passed=not fail),
                    integration_steps=0,task_success_claim=False,object_pose_modified=False)
            def engine(card,*args):
                return SimpleNamespace(card=card,close=lambda:None)
            native=SimpleNamespace(mj_step=lambda:None,mj_step1=lambda:None,mj_step2=lambda:None)
            manifest=dict(limits=dict(operation_deadline_epoch=10**12))
            with patch.dict('sys.modules',mujoco=native), \
                 patch.object(population,'validate',return_value=(manifest,dict(asset_root='FIXTURE_ONLY'))), \
                 patch.object(population,'candidate_pool',return_value=copy.deepcopy(pool)), \
                 patch.object(population,'slots',return_value=[1]), \
                 patch.object(population,'evaluation_cards',return_value=[]), \
                 patch.object(population,'validate_groups',return_value=dict(groups=1)), \
                 patch.object(population.os,'sched_setaffinity'), \
                 patch.object(population.resource,'setrlimit'), \
                 patch.object(population.signal,'signal'),patch.object(population.signal,'alarm'), \
                 patch.object(scene,'make_engine',side_effect=engine), \
                 patch.object(scene,'qualify_engine',side_effect=qualify):
                if unknown_error:
                    with self.assertRaisesRegex(RuntimeError,'UNEXPLAINED'):
                        population.run(root)
                else:population.run(root)
            ledger=json.loads((root/'RESOURCE_LEDGER.json').read_text())
            result=None
            p=root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'
            if p.exists():result=json.loads(p.read_text())
            return ledger,result,calls

    def test_static_only_seal_retains_fixed_task_pair_and_no_success_filter(self):
        ledger,result,calls=self.exercise()
        self.assertEqual([c['task'] for c in calls],['T1','T2'])
        self.assertEqual(ledger['engine_starts'],2)
        self.assertEqual(ledger['integrations'],0)
        self.assertEqual(ledger['CPU_reserved_s'],0)
        self.assertFalse(result['evaluation_success_screening'])
        self.assertEqual(len(result['static_receipts']),2)

    def test_static_rejection_is_preserved_and_uses_only_frozen_next_candidate(self):
        ledger,result,calls=self.exercise(first_fails=True)
        self.assertEqual(ledger['engine_starts'],4)
        self.assertEqual([c['layout_candidate_index'] for c in calls],[0,0,1,1])
        self.assertEqual(result['groups'][0]['layout_candidate_index'],1)
        self.assertEqual(len(result['rejected_candidates']),1)
        self.assertEqual(len(result['static_receipts']),4)

    def test_unexplained_failure_not_retried_as_static_rejection(self):
        ledger,result,calls=self.exercise(unknown_error=True)
        self.assertEqual(len(calls),1)
        self.assertEqual(ledger['engine_starts'],1)
        self.assertEqual(ledger['status'],'FAILED_PRESERVED')
        self.assertIsNone(result)
