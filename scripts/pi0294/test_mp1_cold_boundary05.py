import copy
import json
from pathlib import Path
import unittest

from .mp1 import cold_boundary as cold
from .mp1 import cold_boundary05 as fifth
from .mp1.recovery_config import limits
from .native_dataset import sha
from . import test_mp1_cold_boundary as fixtures
from .test_mp1_behavior_matrix import plan,receipt


class FifthBoundary(unittest.TestCase):
    def setUp(self):
        f=fixtures.ColdBoundary('runTest');f.setUp();self.addCleanup(f.temp.cleanup)
        self.f=f;self.root=f.root;self.put=f.put
        f.manifest['code']['vla_commit']=fifth.OLD_SOURCE
        f.binding['new_source_sha']=fifth.OLD_SOURCE
        f.put(f.root/cold.BINDING,f.binding);f.reseal()
        self.old=copy.deepcopy(f.manifest)
        self.put(self.root/fifth.HISTORY,self.old)
        stop=dict(run_uuid=self.root.name,session='S-CORR-512-TRAIN4-ALLOCATION-04',
            reason='POISONED_FILE_RPC_UNKNOWN_EFFECT_NO_NEXT_CASE_OR_MODEL',native_safety_stop=False,scientific_policy_failure=False)
        self.put(self.root/cold.NEW_STOP,stop)
        self.assertEqual(sha(self.root/cold.NEW_STOP),fifth.STOP04_SHA)
        self.past=copy.deepcopy(f.past);self.past.update(GPU_allocated_s=24119,generation_queries=1299)
        self.past['PBS_submissions'].append(dict(job_id='672907.hopper-m-02',status='NATIVE_TERMINAL',GPU_conservative_charge_s=391))
        self.put(self.root/fifth.LEDGER,self.past)
        self.rows=copy.deepcopy(f.rows);self.rows.append(receipt(plan()['panels']['S-CORR-512-TRAIN4']['cards'][3],'ENGINEERING_INCOMPLETE'))
        self.put(self.root/fifth.OUTCOMES,self.rows)
        card=Path(__file__).resolve().parents[2]/'reports/PI0294/design/SOURCE_PAIR_ADDITIONAL_PBS05_AUTHORIZATION_20261018.json'
        p=self.root/fifth.AUTHORIZATION;p.parent.mkdir(exist_ok=True);p.write_bytes(card.read_bytes());p.chmod(0o400)
        public=json.loads((card.parent.parent/'20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_TERMINAL_AND_CLOCK_REPAIR.json').read_text())
        self.proof=dict(status='PASS_SOURCE04_KNOWN_PARTIAL_EFFECTS_CLOSED_NOT_POLICY_RESULT',valid_policy_outcome=False,
            rerun_authority=False,old_and_allocation04_STOP_markers_modified=False,new_model_queries=0,new_physics=0,
            all_local_mutation_journal_entries_complete=1970,
            core=dict(executed_ACKs=21,integrations=4200,distinct_snapshots=22,raw_camera_hashes=44),
            hopper=dict(native_draws=22,executed_draws=21,uninstalled_draws=1,all_array_hashes_and_raw_actions_pass=True,
                both_STOP_bytes_preserved=True,uninstalled=public['effect_closure']['uninstalled']))
        self.put(self.root/fifth.EFFECTS,self.proof)
        self.put(self.root/'capsules/fifth/SOURCE_SEAL.json',dict(source_sha='c'*40,files=f.hashes))
        self.m=copy.deepcopy(self.old)
        self.m['code'].update(vla_commit='c'*40,capsule_relative='capsules/fifth/code',source_seal_relative='capsules/fifth/SOURCE_SEAL.json',
            source_seal_sha256=sha(self.root/'capsules/fifth/SOURCE_SEAL.json'))
        self.q=dict(schema='MP1_SAME_ROOT_COLD_ALLOC05_BOUNDARY_V1',authority_release=cold.AUTH,run_uuid=self.root.name,
            parent_run_uuid='parent',original_T0_epoch=1792178085,target_allocation_number=5,effective_source_PBS_cap=5,
            new_source_sha='c'*40,prior_manifest_sha256=sha(self.root/fifth.HISTORY),authorization_card_sha256=fifth.CARD_SHA,
            new_budget_or_clock=False,new_attempt_rights=False,old_STOP_removed=False,automatic_sixth_submission=False)
        self.put(self.root/fifth.BINDING,self.q);self.reseal()

    def reseal(self):
        self.m['execution']['cold_engineering_binding_relative']=fifth.BINDING
        self.m['execution']['sealed_inputs']=dict(self.old['execution']['sealed_inputs'],**{n:sha(self.root/n) for n in fifth.REQUIRED})
        self.put(self.root/'RUN_MANIFEST.json',self.m)

    def test_only_fifth_admitted_while_both_stops_and_old_scopes_remain(self):
        before=[(self.root/n).read_bytes() for n in ['RELAY_ENGINEERING_STOP.json',cold.NEW_STOP]]
        self.assertEqual(cold.validate(self.root),self.q)
        self.assertFalse(cold.Guard(self.root,5).blocked())
        self.assertFalse(cold.blocked(self.root,5))
        for n in [1,2,3,4,6]:self.assertTrue(cold.blocked(self.root,n))
        self.assertEqual(cold.stop_target(self.root,5),self.root/fifth.NEW_STOP)
        self.assertEqual(before,[(self.root/n).read_bytes() for n in ['RELAY_ENGINEERING_STOP.json',cold.NEW_STOP]])

    def test_cap_is_five_only_with_exact_sealed_cold_authority_and_gpu_unchanged(self):
        self.assertEqual(limits({})['PBS'],4)
        actual=limits({},root=self.root);self.assertEqual(actual['PBS'],5);self.assertEqual(actual['GPU_s'],43200)
        self.put(self.root/fifth.AUTHORIZATION,{'human':'not actual'});self.reseal()
        with self.assertRaises(PermissionError):limits({},root=self.root)

    def test_no_new_budget_clock_sixth_attempt_or_source_namespace(self):
        for key,value in [('effective_source_PBS_cap',6),('target_allocation_number',6),('new_budget_or_clock',True),
                ('new_attempt_rights',True),('automatic_sixth_submission',True),('original_T0_epoch',1)]:
            with self.subTest(key=key):
                old=self.q[key];self.q[key]=value;self.put(self.root/fifth.BINDING,self.q);self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.q[key]=old

    def test_four_original_costs_cannot_be_reset_or_leave_pending_updates(self):
        for key,value in [('GPU_allocated_s',0),('GPU_reserved_s',1),('pending_updates',1),('optimizer_updates',0),('generation_queries',0)]:
            with self.subTest(key=key):
                old=self.past[key];self.past[key]=value;self.put(self.root/fifth.LEDGER,self.past);self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.past[key]=old

    def test_uninstalled_action_cannot_be_claimed_executed_or_hide_incomplete_journal(self):
        for key,value in [('valid_policy_outcome',True),('rerun_authority',True),('new_physics',1),('all_local_mutation_journal_entries_complete',1969)]:
            with self.subTest(key=key):
                old=self.proof[key];self.proof[key]=value;self.put(self.root/fifth.EFFECTS,self.proof);self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.proof[key]=old

    def test_new_stop_blocks_and_native_safety_is_never_resolved(self):
        self.put(self.root/fifth.NEW_STOP,dict(stopped=True));self.assertTrue(cold.blocked(self.root,5))
        for name in ['SOURCE_PHYSICS_STOP.json','SOURCE_SAFETY_STOP_FROM_CORE.json']:
            self.put(self.root/name,dict(stopped=True))
            with self.assertRaises(PermissionError):cold.validate(self.root)
            (self.root/name).unlink()

    def test_old_fourth_evidence_still_guarded_after_admission(self):
        g=cold.Guard(self.root,5);self.assertFalse(g.blocked())
        self.put(self.root/cold.EFFECTS,dict(self.f.proof,new_physics=1))
        with self.assertRaises(PermissionError):g.blocked()

    def test_cannot_drop_f1_t1_exhaustion_or_impute_native_safety(self):
        self.put(self.root/fifth.OUTCOMES,self.rows[:-1]);self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)
        self.rows[0]['live_native_failures']=1;self.put(self.root/fifth.OUTCOMES,self.rows);self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_changed_science_and_worker_learning_bytes_rejected(self):
        worker=copy.deepcopy(self.m);worker['code']['unchanged_learning_files']={}
        with self.assertRaises(PermissionError):cold.validate(self.root,worker)
        self.m['data']['bank']='v3.1';self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_missing_fourth_stop_not_silently_cleared(self):
        (self.root/cold.NEW_STOP).unlink()
        with self.assertRaises(OSError):cold.blocked(self.root,5)


if __name__=='__main__':unittest.main()
