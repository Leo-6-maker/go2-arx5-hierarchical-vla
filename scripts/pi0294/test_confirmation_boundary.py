from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1 import confirmation_boundary as c,cold_boundary


class ConfirmationBoundaryTests(unittest.TestCase):
    def put(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():p.chmod(0o600)
        p.write_text(json.dumps(value,indent=2)+'\n');p.chmod(0o400)

    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)/c.n1.PARENT_RUN/'children/source-confirm-01';root.mkdir(parents=True)
        self.put(root/c.OLD_STOP,dict(native_safety_stop=False,scientific_policy_failure=False))
        self.put(root/'plans/SOURCE_EVALUATION_PLAN.json',dict(frozen='unit-test-only'))
        self.put(root/'old/SOURCE_SEAL.json',dict(files={}))
        old=dict(run_uuid=root.name,scope='MP1_SINGLE_SOURCE_CONFIRMATION_NO_TRAINING',
            code=dict(vla_commit=c.OLD_SOURCE,capsule_relative='old/code',source_seal_relative='old/SOURCE_SEAL.json',
                source_seal_sha256=sha(root/'old/SOURCE_SEAL.json'),unchanged_learning_files={'fixed':'same'}),
            execution=dict(profile='4080-simulation',confirmation_phase='POLICY_FROZEN',
                confirmation_n1_binding=dict(release=c.n1.RELEASE,adoption_delta_sha256=c.n1.ADOPTED_DELTA_SHA),
                sealed_inputs={'plans/SOURCE_EVALUATION_PLAN.json':sha(root/'plans/SOURCE_EVALUATION_PLAN.json')}))
        self.put(root/c.PRIOR,old)
        self.put(root/c.LEDGER,dict(run_uuid=root.name,GPU_allocated_s=8456,GPU_reserved_s=0,
            generation_queries=930,optimizer_updates=0,pending_updates=[],CPU_reserved_s=0,
            PBS_submissions=[dict(request_id='alloc-01',status='NATIVE_TERMINAL',job_id='673388.hopper-m-02',GPU_conservative_charge_s=8456)]))
        self.put(root/c.NATIVE,dict(job_id='673388.hopper-m-02',native=dict(job_state='F',Exit_status=1,
            project='CFP04-CF-007',Job_Owner='e1192548@fixture',resources_used=dict(walltime='02:20:53'),
            stime='Sun Oct 18 00:00:00 2026',obittime='Sun Oct 18 02:20:56 2026')))
        self.put(root/c.PROCESSES,dict(all_recorded_processes_terminal=True,unknown_effects=[],records=[dict(terminal_verified=True)]))
        self.put(root/c.DIAGNOSTIC,dict(status='ACTUAL_CPU_ROUTE_DIAGNOSIS_AND_CONTROLLED_MECHANISM_REPAIR_PASS_NOT_NEW_NATIVE_QUALIFICATION',
            actual_nominal80=dict(cycles=80,RPCs=160,all_actual_recorded_byte_checksums_match=True),
            controlled_written_fsync_comparison=dict(actual_requests=2,AFTER=dict(session_poisoned=False,response_parsed=True)),
            resources=dict(new_GPU_PBS=0,new_model_queries=0,new_physics=0,new_training_updates=0)))
        self.put(root/c.CLOSURE,dict(prior_allocations=1,prior_GPU_s=8456,prior_generation_queries=930,
            unknown_effects=[],all_recorded_processes_terminal=True,rows=[dict(live_native_failures=0) for _ in range(20)],
            core=dict(actual_executed_prefixes=917),hopper=dict(cumulative_generation_queries=930)))
        constants={k:sha(root/p) for k,p in [('STOP_SHA',c.OLD_STOP),('PLAN_SHA','plans/SOURCE_EVALUATION_PLAN.json'),('CLOSURE_SHA',c.CLOSURE)]}
        for key,value in constants.items():
            context=patch.object(c,key,value);context.start();self.addCleanup(context.stop)
        q=dict(schema=c.SCHEMA,authority_release=c.n1.RELEASE,run_uuid=root.name,parent_run_uuid=c.n1.PARENT_RUN,
            original_MP1_T0_epoch=c.n1.ORIGINAL_T0,N1_T0_epoch=1792264081,target_allocation_number=2,
            prior_manifest_sha256=sha(root/c.PRIOR),new_source_sha='a'*40,old_STOP_sha256=constants['STOP_SHA'],
            new_budget_or_clock=False,new_attempt_rights=False,old_STOP_removed=False,component_native_version_upper_bound=6)
        self.put(root/c.BINDING,q)
        m=deepcopy(old);m['code'].update(vla_commit='a'*40,capsule_relative='new/code',source_seal_relative='new/SOURCE_SEAL.json',source_seal_sha256='b'*64)
        m['execution'][c.KEY]=c.BINDING
        m['execution']['sealed_inputs'].update({n:sha(root/n) for n in c.NAMES})
        self.put(root/'RUN_MANIFEST.json',m)
        return root,m,q

    def test_exact_last_slot_keeps_original_stop_and_routes_only_new_stop(self):
        root,m,q=self.fixture();self.assertEqual(c.validate(root),q)
        self.assertFalse(c.Guard(root,2).blocked());self.assertTrue(c.Guard(root,1).blocked())
        self.assertFalse(cold_boundary.blocked(root,2))
        self.assertEqual(cold_boundary.stop_target(root,2),root/c.NEW_STOP)
        self.assertEqual(sha(root/c.OLD_STOP),c.STOP_SHA)
        with self.assertRaises(PermissionError):cold_boundary.stop_target(root,1)
        self.put(root/c.NEW_STOP,dict(reason='new epoch poisoned'))
        self.assertTrue(c.Guard(root,2).blocked());self.assertEqual(sha(root/c.OLD_STOP),c.STOP_SHA)

    def test_no_binding_does_not_ignore_original_stop(self):
        root,m,q=self.fixture();del m['execution'][c.KEY];self.put(root/'RUN_MANIFEST.json',m)
        self.assertIsNone(c.validate(root));self.assertTrue(c.Guard(root,2).blocked())

    def test_native_safety_unknown_effects_pending_cost_and_data_changes_fail_closed(self):
        changes=[(c.PROCESSES,'unknown_effects',['unknown']),
            (c.PROCESSES,'all_recorded_processes_terminal',False),
            (c.LEDGER,'GPU_allocated_s',0),(c.LEDGER,'GPU_reserved_s',1),
            (c.LEDGER,'generation_queries',929),(c.LEDGER,'optimizer_updates',1),
            (c.LEDGER,'pending_updates',[{}]),(c.DIAGNOSTIC,'status','CPU_TESTS_ONLY')]
        for name,key,value in changes:
            with self.subTest(name=name,key=key):
                root,m,q=self.fixture();v=json.loads((root/name).read_text());v[key]=value
                self.put(root/name,v);m['execution']['sealed_inputs'][name]=sha(root/name)
                with self.assertRaises(PermissionError):c.validate(root,m)
        root,m,q=self.fixture();self.put(root/'SOURCE_PHYSICS_STOP.json',dict(stop=True))
        with self.assertRaises(PermissionError):c.validate(root,m)
        root,m,q=self.fixture();m['code']['unchanged_learning_files']={}
        with self.assertRaises(PermissionError):c.validate(root,m)

    def test_no_third_slot_new_clock_attempt_right_or_seventh_component_version(self):
        for key,value in [('target_allocation_number',3),('new_budget_or_clock',True),
                ('new_attempt_rights',True),('old_STOP_removed',True),('component_native_version_upper_bound',7),
                ('component_native_version_upper_bound',True),('N1_T0_epoch',1792264082)]:
            with self.subTest(key=key,value=value):
                root,m,q=self.fixture();q[key]=value;self.put(root/c.BINDING,q)
                m['execution']['sealed_inputs'][c.BINDING]=sha(root/c.BINDING)
                with self.assertRaises(PermissionError):c.validate(root,m)

    def test_immutable_binding_change_after_admission_stops_guard(self):
        root,m,q=self.fixture();guard=c.Guard(root,2)
        (root/c.PROCESSES).chmod(0o600)
        with self.assertRaises(PermissionError):guard.blocked()

    def test_validation_snapshot_race_cannot_cache_a_changed_old_stop(self):
        root,m,q=self.fixture();original=c.validate
        def race(root,manifest):
            value=original(root,manifest);(root/c.OLD_STOP).chmod(0o600);return value
        with patch.object(c,'validate',side_effect=race):
            with self.assertRaisesRegex(PermissionError,'CHANGED_DURING_ADMISSION'):c.Guard(root,2)


if __name__=='__main__':unittest.main()
