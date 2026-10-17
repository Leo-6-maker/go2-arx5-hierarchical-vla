from copy import deepcopy
from datetime import datetime, timedelta
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from .mp1 import cold_boundary as cold
from .mp1 import cold_boundary05 as fifth
from .mp1 import n1_boundary as boundary
from .mp1 import n1_limits as n1
from .mp1.recovery_config import limits
from .mp1.plans import session_spec
from .mp1.allocation import request_contract
from .native_dataset import sha
from . import test_mp1_cold_boundary as fourth_fixture
from . import test_mp1_cold_boundary05 as fifth_fixture
from . import test_mp1_n1_limits as authority_fixture
from .test_mp1_behavior_matrix import plan, receipt


class N1BoundaryTests(unittest.TestCase):
    def setUp(self):
        with patch.object(fourth_fixture.ColdBoundary, 'parent_name', n1.PARENT_RUN, create=True):
            f = fifth_fixture.FifthBoundary('runTest');f.setUp()
        self.addCleanup(f.f.temp.cleanup)
        self.root, self.put = f.root, f.put
        (self.root/'evidence').mkdir(exist_ok=True)
        authority = authority_fixture.N1LimitsTests('runTest');authority.setUp()
        self.addCleanup(authority.tmp.cleanup)
        self.put(self.root.parent.parent / 'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json', authority.original)
        for name in n1.FILES:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((authority.root/name).read_bytes());p.chmod(0o400)
        parent_delta=self.root.parent.parent/n1.FIRST_ADOPTION
        parent_delta.parent.mkdir(parents=True)
        parent_delta.write_bytes((self.root/n1.DELTA).read_bytes());parent_delta.chmod(0o400)
        self.old=deepcopy(f.m)
        self.put(self.root/boundary.TRANSPORT_PLAN,dict(scope='CPU_TEST_ONLY_SYNTHETIC_RECORDED_REFERENCE'))
        self.past=deepcopy(f.past)
        self.past['PBS_submissions'].append(dict(job_id='673008.hopper-m-02',status='NATIVE_TERMINAL',GPU_conservative_charge_s=200))
        self.past.update(GPU_allocated_s=24319,generation_queries=1330)
        for i,row in enumerate(self.past['PBS_submissions'],1):row['request_id']='alloc-'+str(i).zfill(2)
        self.rows=deepcopy(f.rows)
        card=plan()['panels']['S-CORR-512-TRAIN4']['cards'][3]
        self.rows.append(receipt(dict(card,id=card['id']+'-REPLACEMENT-01')))
        self.processes=dict(status='PASS_PRIOR_OWNERS_AND_CONSUMERS_TERMINAL',rows=[
            dict(role=r,pid=2**30,proc_start='1',terminal_verified=True) for r in ['core','hopper','local_owner']])
        self.build(6)

    def native(self):
        values=[]
        for row in self.past['PBS_submissions']:
            seconds=row['GPU_conservative_charge_s'];start=datetime(2026,10,18)
            h,rem=divmod(seconds,3600);m,s=divmod(rem,60)
            values.append(dict(job_id=row['job_id'],native=dict(job_state='F',project='CFP04-CF-007',
                Job_Owner='e1192548@fixture',stime=start.strftime('%a %b %d %H:%M:%S %Y'),
                obittime=(start+timedelta(seconds=seconds)).strftime('%a %b %d %H:%M:%S %Y'),
                resources_used=dict(walltime=f'{h:02}:{m:02}:{s:02}'))))
        return values

    def build(self,number):
        self.number=number;self.names=boundary.paths(number)
        self.native_rows=self.native()
        self.effects=dict(status='PASS_ALL_PRIOR_SOURCE_EFFECTS_CLOSED',pending_queries=0,pending_installs=0,
            pending_starts=0,pending_optimizer_updates=0,mutation_journal_count=1970,
            core_reservations_closed=True,core_CPU_reserved_s=0,
            completed_mutation_journal_count=1970,cumulative_native_generation_queries=self.past['generation_queries'],
            native_draws_all_classified=True,all_action_arrays_verified=True,
            prior_outcome_ids=sorted(r['attempt_id'] for r in self.rows),new_model_queries=0,new_physics=0)
        for path,value in zip(self.names[1:],[self.old,self.past,self.rows,self.native_rows,self.effects,self.processes]):
            self.put(self.root/path,value)
        seal=f'capsules/n1-{number}/SOURCE_SEAL.json'
        self.put(self.root/seal,dict(source_sha='d'*40,files=self.old['code']['unchanged_learning_files']))
        self.m=deepcopy(self.old)
        self.m['code'].update(vla_commit='d'*40,capsule_relative=f'capsules/n1-{number}/code',
            source_seal_relative=seal,source_seal_sha256=sha(self.root/seal))
        self.m['execution'].update(cold_engineering_binding_relative=self.names[0],
            n1_authority_binding=dict(release=n1.RELEASE,adoption_delta_sha256=sha(self.root/n1.DELTA)))
        self.q=dict(schema=boundary.SCHEMA,authority_release=n1.RELEASE,parent_authority_release=cold.AUTH,
            run_uuid=self.root.name,parent_run_uuid=n1.PARENT_RUN,original_T0_epoch=n1.ORIGINAL_T0,
            target_allocation_number=number,effective_source_PBS_cap=12,new_source_sha='d'*40,
            prior_manifest_sha256=sha(self.root/self.names[1]),adoption_delta_sha256=sha(self.root/n1.DELTA),
            new_attempt_rights=False,old_STOP_removed=False,
            preserved_STOP_hashes={p.name:sha(p) for p in self.root.glob('RELAY_ENGINEERING_STOP*.json')})
        self.reseal()

    def reseal(self):
        for path,value in zip(self.names,[self.q,self.old,self.past,self.rows,self.native_rows,self.effects,self.processes]):
            self.put(self.root/path,value)
        self.m['execution']['sealed_inputs']=dict(self.old['execution']['sealed_inputs'],
            **{p:sha(self.root/p) for p in self.names+n1.FILES+list(self.q['preserved_STOP_hashes'])+[boundary.TRANSPORT_PLAN]})
        self.put(self.root/'RUN_MANIFEST.json',self.m)

    def request(self):
        created=1792264200
        queue='pbs/fixture/NATIVE_QUEUE_LIMIT.json'
        self.put(self.root/queue,dict(queue='auto',destination='small',observed_epoch=created-1,maximum_wall_s=28800))
        old_lease=self.root.parent.parent/'old-e1-lease.json'
        self.put(old_lease,dict(run_uuid='20261016T150228Z-localpc-e1',status='CLOSED'))
        self.put(self.root/'plans/OLD_E1_BINDING.private.json',dict(lease=str(old_lease)))
        request=dict(run_uuid=self.root.name,source_sha=self.m['code']['vla_commit'],run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),
            mode='MP1_FIXED_SOURCE_PAIR',numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',submit_count=self.number,
            walltime_s=28800,prior_GPU_allocated_s=self.past['GPU_allocated_s'],created_epoch=created,
            queue_limit_relative=queue,queue_limit_sha256=sha(self.root/queue),
            sealed_inputs=dict(self.m['execution']['sealed_inputs'],**{queue:sha(self.root/queue)}))
        ledger=dict(self.past,GPU_reserved_s=28800,PBS_submissions=self.past['PBS_submissions']+[{}])
        return request,ledger

    def test_single_generic_chain_reaches_twelve_without_numbered_exceptions(self):
        for number in range(6,13):
            with self.subTest(number=number):
                if number>6:
                    self.old=deepcopy(self.m)
                    self.past['PBS_submissions'].append(dict(request_id='alloc-'+str(number-1).zfill(2),
                        job_id=str(673008+number)+'.hopper-m-02',status='NATIVE_TERMINAL',GPU_conservative_charge_s=200))
                    self.past['GPU_allocated_s']+=200
                    self.build(number)
                self.assertEqual(cold.validate(self.root)['target_allocation_number'],number)
                self.assertFalse(cold.Guard(self.root,number).blocked())
                self.assertTrue(cold.blocked(self.root,number-1))
                self.assertEqual(cold.stop_target(self.root,number).name,f'RELAY_ENGINEERING_STOP_ALLOCATION{number:02}.json')
                r,l=self.request();request_contract(self.root,r,self.m,l,r['created_epoch']+1)
        with self.assertRaises(PermissionError):boundary.paths(13)
        self.assertIsNone(boundary.target('plans/N1/alloc-\u0660\u0666/COLD_BOUNDARY.json'))

    def test_native_caps_and_all_session_numbers_need_n1(self):
        self.assertEqual(limits({},root=self.root)['GPU_s'],129600)
        self.assertEqual(limits({},root=self.root)['PBS'],12)
        for number in range(1,13):
            label=f'S-BC-512-TRAIN4-ALLOCATION-{number:02}'
            self.assertEqual(session_spec(plan(),label,root=self.root)['checkpoint_step'],512)
        self.m['execution'].pop('n1_authority_binding');self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)
        with self.assertRaises(PermissionError):session_spec(plan(),'S-BC-512-TRAIN4-ALLOCATION-06',root=self.root)

    def test_transport_reference_is_required_sealed_and_stable_across_cold_resumes(self):
        path=self.root/boundary.TRANSPORT_PLAN
        self.assertEqual(cold.validate(self.root)['target_allocation_number'],6)
        path.chmod(0o600)
        with self.assertRaisesRegex(PermissionError,'SEALED_PRIOR'):cold.validate(self.root)
        path.chmod(0o400)
        self.m['execution']['sealed_inputs'].pop(boundary.TRANSPORT_PLAN)
        self.put(self.root/'RUN_MANIFEST.json',self.m)
        with self.assertRaisesRegex(PermissionError,'SEALED_PRIOR'):cold.validate(self.root)
        self.reseal();self.old=deepcopy(self.m)
        self.past['PBS_submissions'].append(dict(request_id='alloc-06',job_id='673014.hopper-m-02',
            status='NATIVE_TERMINAL',GPU_conservative_charge_s=200))
        self.past['GPU_allocated_s']+=200
        self.build(7)
        self.put(path,dict(scope='DIFFERENT_REFERENCE'))
        self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_live_or_unknown_prior_effects_cannot_be_fenced(self):
        for field,value in [('pending_installs',1),('native_draws_all_classified',False),('completed_mutation_journal_count',1969)]:
            with self.subTest(field=field):
                original=self.effects[field];self.effects[field]=value;self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.effects[field]=original
        self.processes['rows'][1].update(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21])
        self.reseal()
        with self.assertRaisesRegex(PermissionError,'STILL_LIVE'):cold.validate(self.root)

    def test_cost_reset_native_identity_and_pending_update_rejected(self):
        for key,value in [('GPU_allocated_s',0),('generation_queries',0),('pending_updates',1)]:
            with self.subTest(key=key):
                original=self.past[key];self.past[key]=value;self.reseal()
                with self.assertRaises(PermissionError):cold.validate(self.root)
                self.past[key]=original
        self.native_rows[-1]['native']['project']='OTHER';self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_old_policy_receipt_cannot_change_or_replay(self):
        self.rows[0]['status']='VALID_TASK_FAIL';self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)
        self.rows[0]['status']='SUCCESS'
        extra=deepcopy(self.rows[0]);extra['attempt_id']+='-REPLACEMENT-01';self.rows.append(extra)
        self.effects['prior_outcome_ids']=sorted(r['attempt_id'] for r in self.rows);self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_new_stop_preserves_prior_stops_and_safety_is_never_cleared(self):
        guard=cold.Guard(self.root,6)
        self.put(cold.stop_target(self.root,6),dict(stopped=True))
        self.assertTrue(guard.blocked())
        self.put(self.root/'SOURCE_PHYSICS_STOP.json',dict(stopped=True))
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_changed_birth_or_learning_bytes_rejected(self):
        self.m['data']['bank']='v3.1';self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)
        self.m['data']['bank']='v3.0'
        self.m['code']['unchanged_learning_files']={};self.reseal()
        with self.assertRaises(PermissionError):cold.validate(self.root)

    def test_native_request_smaller_site_limit_fence_and_cutoff(self):
        r,l=self.request()
        for changed in [dict(r,submit_count=7),dict(r,walltime_s=28801),dict(r,created_epoch=1792372081)]:
            with self.subTest(request=changed['submit_count']),self.assertRaises(PermissionError):
                request_contract(self.root,changed,self.m,l,1792264201)
        self.put(self.root/r['queue_limit_relative'],dict(queue='auto',destination='small',observed_epoch=r['created_epoch']-1,maximum_wall_s=14400))
        r['queue_limit_sha256']=sha(self.root/r['queue_limit_relative']);r['sealed_inputs'][r['queue_limit_relative']]=r['queue_limit_sha256']
        with self.assertRaises(PermissionError):request_contract(self.root,r,self.m,l,1792264201)

    def test_extended_cpu_and_queries_keep_original_update_cap(self):
        from .mp1.science import Ledger,CAPS,CPU_CAPS
        self.put(self.root/'MP1_STAGE_BINDING.json',dict(schema='MP1_SOURCE_SHARED_STAGE_V1',stage_caps=CAPS,CPU_core_s=CPU_CAPS))
        self.put(self.root/'HOPPER_RESOURCE_LEDGER.json',dict(self.past,CPU_s=180))
        ledger=Ledger(self.root,'hopper')
        self.assertEqual(ledger.cpu_caps,dict(core=648000,hopper=518400))
        self.assertEqual(ledger.caps['optimizer_updates'],4096)
        ledger.charge('generation_queries',12000-self.past['generation_queries'])
        with self.assertRaises(PermissionError):ledger.charge('generation_queries',1)

    def test_duplicate_native_owner_and_duplicate_settlement_do_not_add_cost(self):
        from scripts.pi0292.durable import OwnerLock
        from .mp1.science import CAPS,CPU_CAPS
        from .mp1 import pbs_owner
        owner=OwnerLock(self.root/'NEURAL_OWNER.lock')
        try:
            with self.assertRaises(BlockingIOError):OwnerLock(self.root/'NEURAL_OWNER.lock')
        finally:owner.close()
        self.put(self.root/'MP1_STAGE_BINDING.json',dict(schema='MP1_SOURCE_SHARED_STAGE_V1',stage_caps=CAPS,CPU_core_s=CPU_CAPS))
        self.put(self.root/'HOPPER_RESOURCE_LEDGER.json',dict(self.past,CPU_s=180))
        records={v['job_id']:v['native'] for v in self.native_rows}
        with patch.object(pbs_owner,'validate_runtime'),patch.object(pbs_owner,'native',side_effect=records.__getitem__),patch.dict(os.environ,USER='e1192548'):
            first=pbs_owner.settle(self.root)
            second=pbs_owner.settle(self.root)
        self.assertEqual(first['GPU_conservative_s'],24319)
        self.assertEqual(first,second)

    def test_real_queue_limit_reader_obeys_smaller_native_cap(self):
        import json
        from .mp1.pbs_owner import native_queue_limit
        auto=dict(queue_type='Route',route_destinations='small,medium',enabled='True',started='True')
        small=dict(queue_type='Execution',enabled='True',started='True',
            resources_min=dict(ngpus=1),resources_max=dict(ngpus=2,walltime='04:00:00'))
        def query(argv,**kwargs):return json.dumps({'Queue':{argv[-1]:auto if argv[-1]=='auto' else small}})
        self.assertEqual(native_queue_limit(query)['maximum_wall_s'],14400)
        small['resources_max']['walltime']='144:00:00'
        self.assertEqual(native_queue_limit(query)['maximum_wall_s'],518400)
        small['resources_min']['ngpus']=3
        with self.assertRaises(PermissionError):native_queue_limit(query)


if __name__=='__main__':unittest.main()
