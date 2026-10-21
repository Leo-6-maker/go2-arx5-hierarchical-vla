"""Exercise actual durable journals and callbacks without a model or GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown
from scripts.pi0294.pv1.recipe import begin_update,commit_update
from scripts.pi0294.pv1.training import commit_completed_update
from scripts.pi0294.scene_cards import digest
from scripts.pi0292.durable import read,write
from .a40_native_ledger import Ledger
from .test_a40_allocation import fixture


class NativeLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'native-source';self.root.mkdir()
        r=json.loads(fixture()['request_bytes']);self.request=r
        directory=self.root/'pbs/alloc-13';directory.mkdir(parents=True);write(directory/'REQUEST.json',r)
        plans=self.root/'plans';plans.mkdir()
        self.recipe=dict(schema='PI0294_PV1_PRIMARY_LEARNING_RECIPE_V1',arm='A_CONTROL',replication_of_local_recipe=False)
        write(plans/'PRIMARY_LEARNING_RECIPES.json',dict(recipes=dict(A_CONTROL=self.recipe)))
        self.ledger=Ledger(self.root,dict(job_id='17.stdct-mgmt-02',request_sha256='d'*64,submit_count=13))
        self.ledger.initialize();self.journal=Journal(self.root/'reservations/probe-generations',self.root.name)

    def generate(self,index,*,fail=False):
        def operation():
            self.ledger.charge('generation_queries',1)
            if fail:raise RuntimeError('SYNTHETIC_QUERY_LOSS')
            return dict(fixture=index)
        return self.journal.once('draw-'+str(index),dict(index=index),operation)

    def update(self,before):
        with self.ledger.transaction() as v:
            value=begin_update(v,arm='A_CONTROL',local_step=before,request_id=self.request['request_id'],recipe=self.recipe)
            v.clear();v.update(value)
        with self.ledger.transaction() as v:
            value=commit_update(v,arm='A_CONTROL',local_step_after=before+1,request_id=self.request['request_id'])
            v.clear();v.update(value)

    def qualify(self):
        self.ledger.charge('model_loads',1)
        for i in range(17):self.generate(i)

    def test_no_update_before_real_seventeen_completed_journal_operations(self):
        with self.assertRaises(PermissionError):self.update(0)
        self.qualify()
        for i in range(32):self.update(i)
        s=self.ledger.summary();self.assertEqual(s['optimizer_updates_committed'],32)
        self.assertEqual(s['query_attempts'],17);self.assertEqual(s['pending_updates'],[])
        self.assertEqual(s['unknown_query_intents'],[])
        self.assertEqual(s['new_physics'],0)

    def test_update_1025_requires_original_twelve_additional_query_journals(self):
        self.qualify()
        with self.ledger.transaction() as v:
            v['optimizer_updates']=1024
            v['training_runs']['A_CONTROL']=dict(recipe_sha256=digest(self.recipe),last_native_step=1024)
        with self.assertRaises(PermissionError):self.update(1024)
        for i in range(17,29):self.generate(i)
        self.update(1024)
        self.assertEqual(self.ledger.summary()['last_native_step'],1025)
        with self.assertRaises(PermissionError):self.generate(29)

    def test_unresolved_generation_is_not_repeated_or_hidden_by_next_id(self):
        self.ledger.charge('model_loads',1)
        with self.assertRaises(RuntimeError):self.generate(0,fail=True)
        with self.assertRaises(MutationOutcomeUnknown):self.generate(0)
        s=self.ledger.summary();self.assertEqual(s['query_attempts'],1)
        self.assertEqual(s['unknown_query_intents'],[0])
        with self.assertRaises(PermissionError):self.generate(1)
        with self.assertRaises(PermissionError):self.ledger.summary()

    def test_duplicate_completed_draw_reuses_it_without_a_new_charge(self):
        self.ledger.charge('model_loads',1);a=self.generate(0);b=self.generate(0)
        self.assertEqual(a,b);self.assertEqual(self.ledger.summary()['query_attempts'],1)

    def test_terminal_summary_refuses_corrupted_completed_query_bytes(self):
        self.ledger.charge('model_loads',1);self.generate(0)
        path=self.journal.path('draw-0');v=read(path);v['result']=dict(changed=True);write(path,v)
        with self.assertRaises(PermissionError):self.ledger.summary()

    def test_pending_optimizer_intent_preserved_and_cannot_be_advanced_again(self):
        self.qualify()
        with self.ledger.transaction() as v:
            value=begin_update(v,arm='A_CONTROL',local_step=0,request_id=self.request['request_id'],recipe=self.recipe)
            v.clear();v.update(value)
        self.assertEqual(len(self.ledger.summary()['pending_updates']),1)
        with self.assertRaises(PermissionError):self.update(0)

    def test_cuda_completion_must_succeed_before_native_commit(self):
        self.qualify()
        with self.ledger.transaction() as v:
            value=begin_update(v,arm='A_CONTROL',local_step=0,request_id=self.request['request_id'],recipe=self.recipe)
            v.clear();v.update(value)
        def failed_synchronization():raise RuntimeError('SYNTHETIC_ASYNC_CUDA_FAILURE')
        with self.assertRaises(RuntimeError):
            commit_completed_update(self.ledger,failed_synchronization,arm='A_CONTROL',local_step_after=1,request_id=self.request['request_id'])
        self.assertEqual(self.ledger.summary()['optimizer_updates_committed'],0)
        self.assertEqual(len(self.ledger.summary()['pending_updates']),1)
        commit_completed_update(self.ledger,lambda:None,arm='A_CONTROL',local_step_after=1,request_id=self.request['request_id'])
        self.assertEqual(self.ledger.summary()['optimizer_updates_committed'],1)

    def test_second_initializer_and_counter_tampering_refused(self):
        with self.assertRaises(FileExistsError):self.ledger.initialize()
        for field,value in [('generation_queries',30),('model_loads',5),('CPU_s',float('nan')),('job_id','other')]:
            v=read(self.ledger.path);v[field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):self.ledger.check(v)


if __name__=='__main__':unittest.main()
