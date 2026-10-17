import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,call

from .mp1 import cold_boundary as cold
from .native_dataset import sha
from .test_mp1_behavior_matrix import plan,receipt


class ColdBoundary(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/getattr(self,'parent_name','parent')/'children/source-pair-01';self.root.mkdir(parents=True)
        self.put(self.root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json',dict(T0_epoch=1792178085))
        self.put(self.root/'plans/SOURCE_EVALUATION_PLAN.json',plan())
        self.hashes={n:'a'*64 for n in cold.LEARNING_FILES}
        self.put(self.root/'capsules/old/SOURCE_SEAL.json',dict(source_sha=cold.OLD_SOURCE,files=self.hashes))
        self.put(self.root/'capsules/new/SOURCE_SEAL.json',dict(source_sha='b'*40,files=self.hashes))
        self.old=dict(schema='RUN_MANIFEST_V1',run_uuid=self.root.name,owner='EXISTING_LOCAL_PC',scope='MP1_SOURCE_SHARED_STUDY',
            authority=dict(protocol_ref=cold.AUTH),limits=dict(original_MP1_T0_epoch=1792178085,GPU_cap=43200),data=dict(bank='v3.0'),interfaces=dict(stats='fixed'),
            execution=dict(profile='hopper-neural',uid=1,hostnames=['fixture'],sealed_inputs={'plans/SOURCE_EVALUATION_PLAN.json':sha(self.root/'plans/SOURCE_EVALUATION_PLAN.json')}),
            code=dict(vla_commit=cold.OLD_SOURCE,capsule_relative='capsules/old/code',source_seal_relative='capsules/old/SOURCE_SEAL.json',source_seal_sha256=sha(self.root/'capsules/old/SOURCE_SEAL.json'),official_gr00t_commit='original',unchanged_numerical_files={'frozen':'a'*64}))
        self.put(self.root/cold.HISTORY,self.old)
        stop=dict(run_uuid=self.root.name,session='S-CORR-512-TRAIN4-ALLOCATION-03',reason='POISONED_FILE_RPC_UNKNOWN_EFFECT_NO_NEXT_CASE_OR_MODEL',native_safety_stop=False,scientific_policy_failure=False)
        self.put(self.root/'RELAY_ENGINEERING_STOP.json',stop)
        self.assertEqual(sha(self.root/'RELAY_ENGINEERING_STOP.json'),cold.STOP_SHA)
        self.past=dict(run_uuid=self.root.name,GPU_allocated_s=23728,GPU_reserved_s=0,pending_updates=0,optimizer_updates=2560,generation_queries=1277,
            PBS_submissions=[dict(job_id=j,status='NATIVE_TERMINAL',GPU_conservative_charge_s=n) for j,n in [('672080.hopper-m-02',11795),('672471.hopper-m-02',11521),('672829.hopper-m-02',412)]])
        self.put(self.root/cold.LEDGER,self.past)
        self.proof=dict(status='PASS_KNOWN_PARTIAL_PREFIX_AND_ORPHAN_QUERY_CLOSED_NOT_A_TASK_RESULT',valid_policy_outcome=False,rerun_authority=False,old_STOP_markers_modified=False,new_model_queries=0,new_physics=0,
            all_local_mutation_journal_entries_complete=1926,core=dict(executed_ACKs=26,integrations=5200),
            hopper=dict(native_draws=26,orphan_sequence=26,orphan_request_sha256='0b9a988b9053c972da640d966855c2a6d4e81b4a7bb6187c8dc4e08a68a91ab1',old_STOP_sha256=cold.STOP_SHA,all_array_hashes_pass=True))
        self.put(self.root/cold.EFFECTS,self.proof)
        self.rows=[]
        for name,spec in plan()['panels'].items():
            if name.startswith('S-BC'):self.rows.extend(receipt(c) for c in spec['cards'])
        corr=plan()['panels']['S-CORR-512-TRAIN4']['cards']
        self.rows.extend(receipt(c) for c in corr[:2]);self.rows.extend([receipt(corr[2],'ENGINEERING_INCOMPLETE'),receipt(dict(corr[2],id=corr[2]['id']+'-REPLACEMENT-01'),'ENGINEERING_INCOMPLETE')])
        self.put(self.root/cold.OUTCOMES,self.rows)
        self.binding=dict(schema='MP1_SAME_ROOT_COLD_ALLOC04_BOUNDARY_V1',authority_release=cold.AUTH,run_uuid=self.root.name,parent_run_uuid=self.root.parent.parent.name,original_T0_epoch=1792178085,
            target_allocation_number=4,new_source_sha='b'*40,prior_manifest_sha256=sha(self.root/cold.HISTORY),old_STOP_sha256=cold.STOP_SHA,new_budget_or_clock=False,new_attempt_rights=False,old_STOP_removed=False)
        self.put(self.root/cold.BINDING,self.binding)
        self.manifest=copy.deepcopy(self.old);self.manifest['code'].update(vla_commit='b'*40,capsule_relative='capsules/new/code',source_seal_relative='capsules/new/SOURCE_SEAL.json',source_seal_sha256=sha(self.root/'capsules/new/SOURCE_SEAL.json'),unchanged_learning_files=self.hashes)
        self.manifest['execution']['cold_engineering_binding_relative']=cold.BINDING;self.reseal()

    def put(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():p.chmod(0o600)
        p.write_text(json.dumps(value,indent=2)+'\n');p.chmod(0o400)

    def reseal(self):
        names=[cold.BINDING,cold.HISTORY,cold.LEDGER,cold.EFFECTS,cold.OUTCOMES,'RELAY_ENGINEERING_STOP.json']
        self.manifest['execution']['sealed_inputs']=dict(self.old['execution']['sealed_inputs'],**{n:sha(self.root/n) for n in names})
        self.put(self.root/'RUN_MANIFEST.json',self.manifest)

    def test_exact_boundary_allows_only_fourth_and_retains_stop(self):
        old=(self.root/'RELAY_ENGINEERING_STOP.json').read_bytes()
        self.assertEqual(cold.validate(self.root),self.binding);self.assertFalse(cold.blocked(self.root,4));self.assertTrue(cold.blocked(self.root,3))
        self.assertEqual(cold.stop_target(self.root,4),self.root/cold.NEW_STOP);self.assertEqual((self.root/'RELAY_ENGINEERING_STOP.json').read_bytes(),old)

    def test_new_engineering_stop_blocks_every_model(self):
        self.put(self.root/cold.NEW_STOP,{'stopped':True});self.assertTrue(cold.blocked(self.root,4));self.assertTrue(cold.Guard(self.root,4).blocked())

    def test_native_safety_stops_cannot_be_resolved(self):
        for name in ['SOURCE_PHYSICS_STOP.json','SOURCE_SAFETY_STOP_FROM_CORE.json']:
            self.put(self.root/name,{'stopped':True})
            with self.assertRaises(PermissionError):cold.validate(self.root)
            (self.root/name).unlink()

    def test_missing_unsealed_or_changed_stop_not_ignored(self):
        p=self.root/'RELAY_ENGINEERING_STOP.json';p.unlink()
        with self.assertRaises(PermissionError):cold.blocked(self.root,4)

    def test_changed_evidence_fails_even_after_validated_guard(self):
        guard=cold.Guard(self.root,4);self.assertFalse(guard.blocked())
        self.put(self.root/cold.EFFECTS,dict(self.proof,new_physics=1))
        with self.assertRaises(PermissionError):guard.blocked()

    def test_cannot_reset_prior_budget_or_attempt_rights(self):
        for key,value in [('target_allocation_number',5),('new_budget_or_clock',True),('new_attempt_rights',True),('old_STOP_removed',True),('original_T0_epoch',1)]:
            with self.subTest(key=key):
                old=self.binding[key];self.binding[key]=value;self.put(self.root/cold.BINDING,self.binding);self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.binding[key]=old

    def test_prior_costs_and_closed_native_jobs_are_exact(self):
        for key,value in [('GPU_allocated_s',0),('GPU_reserved_s',1),('pending_updates',1),('optimizer_updates',0),('generation_queries',0)]:
            with self.subTest(key=key):
                old=self.past[key];self.past[key]=value;self.put(self.root/cold.LEDGER,self.past);self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.past[key]=old

    def test_effects_cannot_be_declared_policy_result_or_rerun_permission(self):
        for key,value in [('valid_policy_outcome',True),('rerun_authority',True),('new_physics',1),('all_local_mutation_journal_entries_complete',0)]:
            with self.subTest(key=key):
                old=self.proof[key];self.proof[key]=value;self.put(self.root/cold.EFFECTS,self.proof);self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.proof[key]=old

    def test_exhausted_condition_is_not_reopened_by_omitting_first_attempt(self):
        self.put(self.root/cold.OUTCOMES,self.rows[:-1]);self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_native_faults_and_changed_scientific_identity_rejected(self):
        self.rows[0]['live_native_failures']=1;self.put(self.root/cold.OUTCOMES,self.rows);self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)
        self.rows[0]['live_native_failures']=0;self.put(self.root/cold.OUTCOMES,self.rows);self.reseal()
        self.manifest['data']={'bank':'v3.1'};self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_upstream_and_learning_bytes_cannot_change(self):
        self.manifest['code']['official_gr00t_commit']='other';self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)
        self.manifest['code']['official_gr00t_commit']='original';self.manifest['code']['unchanged_learning_files']={};self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_old_scope_with_no_boundary_remains_stopped(self):
        self.put(self.root/'RUN_MANIFEST.json',self.old)
        self.assertIsNone(cold.validate(self.root));self.assertTrue(cold.blocked(self.root,4));self.assertTrue(cold.Guard(self.root,4).blocked())

    def test_qualifying_proposed_manifest_does_not_activate_or_clear_old_scope(self):
        proposed=copy.deepcopy(self.manifest);self.put(self.root/'RUN_MANIFEST.json',self.old)
        self.assertEqual(cold.validate(self.root,proposed_manifest=proposed),self.binding)
        self.assertTrue(cold.blocked(self.root,4));self.assertEqual(json.loads((self.root/'RUN_MANIFEST.json').read_text()),self.old)

    def test_cold_manifest_refresh_is_readonly_parent_then_leaf(self):
        before=(self.root/'RUN_MANIFEST.json').read_bytes();sync=Mock()
        cold.synchronize_current_manifest(self.root,synchronize=sync)
        self.assertEqual(sync.call_args_list,[call(self.root),call(self.root/'RUN_MANIFEST.json')])
        self.assertEqual(before,(self.root/'RUN_MANIFEST.json').read_bytes())


if __name__=='__main__':unittest.main()
