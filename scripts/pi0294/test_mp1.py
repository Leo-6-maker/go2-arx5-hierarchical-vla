"""CPU-only MP1 contracts; synthetic rows never qualify scientific data."""
from collections import Counter
from types import SimpleNamespace
import unittest
from pathlib import Path
import subprocess
import tempfile
import json

from .mp1.dataset import SourceSampler


def dataset(tasks=('T1',)):
    rows=[]
    for task in ['T1','T2']:
        for episode in range(96 if task=='T1' else 30):
            rows.extend((dict(id=task+'-'+str(episode),task=task,split='TRAIN'),a) for a in range(3))
    original=len(rows)
    for task in tasks:
        for episode in range(2):
            rows.extend((dict(id='C-'+task+'-'+str(episode),task=task,split='TRAIN',source_student_episode='student-'+str(episode)),a) for a in range(5))
    return SimpleNamespace(rows=rows,original_rows=original,manifest_sha='fixture-only')


class MP1Tests(unittest.TestCase):
    def test_native_source_state_does_not_accept_weights_only_or_false_step(self):
        from .mp1.neural import checkpoint_state
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'training/S-BC/checkpoint-512';folder.mkdir(parents=True)
            p=folder/'CHECKPOINT_COMPLETE.json'
            self.assertEqual(checkpoint_state(root,'S-BC'),(0,None))
            p.write_text(json.dumps(dict(full_resume=False,global_step=512)))
            with self.assertRaises(PermissionError):checkpoint_state(root,'S-BC')
            p.write_text(json.dumps(dict(full_resume=True,global_step=511)))
            with self.assertRaises(PermissionError):checkpoint_state(root,'S-BC')
            p.write_text(json.dumps(dict(full_resume=True,global_step=512)))
            self.assertEqual(checkpoint_state(root,'S-BC'),(512,folder))

    def test_native_request_requires_old_E1_closed_and_preserves_queued_clock(self):
        from .mp1.allocation import request_contract
        from .native_dataset import sha
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'parent';root=parent/'children/source';(root/'plans').mkdir(parents=True)
            (parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json').write_text(json.dumps(dict(T0_epoch=100,
                new_submission_and_candidate_cutoff_epoch=200,new_experiment_cutoff_epoch=300)))
            (root/'RUN_MANIFEST.json').write_text('{}');lease=parent/'old-lease.json'
            lease.write_text(json.dumps(dict(status='ACTIVE',run_uuid='20261016T150228Z-localpc-e1')))
            (root/'plans/OLD_E1_BINDING.private.json').write_text(json.dumps(dict(lease=str(lease))))
            r=dict(run_uuid='source',source_sha='fixture',run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),mode='MP1_FIXED_SOURCE_PAIR',
                numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',submit_count=1,walltime_s=14400,prior_GPU_allocated_s=0,created_epoch=190,sealed_inputs={})
            ledger=dict(GPU_reserved_s=14400,PBS_submissions=[{}],GPU_allocated_s=0);m=dict(code=dict(vla_commit='fixture'))
            with self.assertRaises(PermissionError):request_contract(root,r,m,ledger,250)
            lease.write_text(json.dumps(dict(status='CLOSED',run_uuid='20261016T150228Z-localpc-e1')))
            request_contract(root,r,m,ledger,250)
            for bad in [dict(r,created_epoch=200),dict(r,submit_count=5),dict(r,prior_GPU_allocated_s=43200),dict(r,mode='E2')]:
                with self.assertRaises(PermissionError):request_contract(root,bad,m,ledger,250)
            with self.assertRaises(PermissionError):request_contract(root,r,m,ledger,300)

    def test_fixed_development_panels_preserve_valid_failure_and_single_replacement(self):
        from .mp1.plans import evaluation_plan,remaining_panel,session_spec
        cards=[dict(id=str(i)+'-'+t,task=t,split='TRAIN' if i<2 else 'DEV',scene_group='F0/'+str(i)) for i in range(6) for t in ['T1','T2']]
        plan=evaluation_plan(cards,'a'*64);self.assertEqual(plan['regular_starts'],32)
        spec=session_spec(plan,'S-BC-512-TRAIN4-ALLOCATION-01');self.assertEqual(len(spec['cards']),4)
        for bad in ['S-BC-512-TRAIN4','S-BC-512-TRAIN4-ALLOCATION-06','OTHER-ALLOCATION-01']:
            with self.assertRaises(PermissionError):session_spec(plan,bad)
        card=spec['cards'][0]
        row=dict(attempt_id=card['id'],model=card['mp1_model_label'],scene_group=card['scene_group'],task=card['task'],
            status='VALID_TASK_FAIL',failure_stage='S1_PRE_EXECUTION_REJECT',pipeline_trustworthy=True,live_native_failures=0)
        self.assertEqual(len(remaining_panel([row],spec)),3)
        interrupted=dict(row,status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,failure_stage='ENGINEERING_OR_UNKNOWN')
        replacement=remaining_panel([interrupted],spec)[0];self.assertTrue(replacement['id'].endswith('-REPLACEMENT-01'))
        repaired=dict(row,attempt_id=replacement['id'])
        self.assertEqual(len(remaining_panel([repaired,interrupted],spec)),3)
        with self.assertRaises(PermissionError):remaining_panel([row,repaired],spec)
        with self.assertRaises(PermissionError):remaining_panel([dict(row,status='ACTUAL_SAFETY_FAIL',live_native_failures=1)],spec)

    def test_prevalidated_file_endpoint_retains_mutation_journal(self):
        from scripts.pi0294.e1.rpc import call
        from scripts.pi0294.e1.journal import MutationOutcomeUnknown
        class Endpoint:
            role='hopper'
            count=0
            def hopper(self,**args):self.count+=1;return dict(file_intent_accepted=True)
        with tempfile.TemporaryDirectory() as d:
            endpoint=Endpoint();endpoint.root=Path(d);args=dict(operation='query',query=dict(body=dict(query_uuid='same-query')))
            call(d,'hopper','fixture',args,endpoint=endpoint);call(d,'hopper','fixture',args,endpoint=endpoint)
            self.assertEqual(endpoint.count,1)
            endpoint.hopper=lambda **kw:(_ for _ in ()).throw(ConnectionError('unknown-effect'))
            args=dict(operation='query',query=dict(body=dict(query_uuid='unknown-query')))
            with self.assertRaises(ConnectionError):call(d,'hopper','fixture',args,endpoint=endpoint)
            with self.assertRaises(MutationOutcomeUnknown):call(d,'hopper','fixture',args,endpoint=endpoint)

    def test_finite_stdio_rpc_never_replays_duplicate_or_exposes_teacher(self):
        from scripts.pi0294.relay_file_rpc import execute_request
        calls=[];seen=set()
        invoke=lambda args:calls.append(args) or dict(accepted=True)
        guard=lambda args:None
        packet=dict(rpc_id='one',args=dict(operation='query',episode='fixture'))
        self.assertTrue(execute_request(packet,invoke,guard,seen)['ok']);self.assertEqual(len(calls),1)
        with self.assertRaises(PermissionError):execute_request(packet,invoke,guard,seen)
        with self.assertRaises(PermissionError):execute_request(dict(rpc_id='two',args=dict(operation='teacher')),invoke,guard,seen)
        with self.assertRaises(ValueError):execute_request(dict(rpc_id='three',args={},extra=True),invoke,guard,seen)
        self.assertEqual(len(calls),1)

    def test_source_study_cannot_consume_Oracle_or_excess_update_slots(self):
        from .mp1.science import CAPS,CPU_CAPS,Ledger
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'MP1_STAGE_BINDING.json').write_text(json.dumps(dict(stage_caps=CAPS,CPU_core_s=CPU_CAPS)))
            (root/'RUN_MANIFEST.json').write_text('{}')
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(main_starts=0,reservations={},shadow_integrations=0)))
            ledger=Ledger(root,'core');ledger.charge('main_starts',64)
            with self.assertRaises(PermissionError):ledger.charge('main_starts',1)
            with self.assertRaises(PermissionError):ledger.charge('recovery_forks',1)
            with ledger.steps('shadow_integrations',200,'prefix') as meter:meter['actual']=200
            self.assertEqual(json.loads((root/'RESOURCE_LEDGER.json').read_text())['shadow_integrations'],200)
            with self.assertRaises(ValueError):
                with ledger.steps('shadow_integrations',200,'unknown') as meter:meter['actual']=None
            self.assertIn('unknown',json.loads((root/'RESOURCE_LEDGER.json').read_text())['reservations'])

    def test_source_recipe_is_new_seed_and_fixed_parent_not_E1_resume(self):
        from .mp1.train import validate_recipe
        recipe=dict(seed=20261017,max_steps=2048,warmup_updates=32,effective_batch=32,per_device_train_batch_size=2,gradient_accumulation_steps=16,
            learning_rate=1e-5,end_learning_rate=1e-6,new_optimizer=True,exact_resume=False,parent='AUDITED_M1_AT_1024_INFERENCE_WEIGHTS_ONLY',
            VLM_frozen=True,statistics_fit=False,gradient_checkpointing=False,bf16=True,tf32=True)
        validate_recipe(recipe,'S-BC')
        for bad in [dict(recipe,seed=20261016),dict(recipe,max_steps=4096),dict(recipe,exact_resume=True),dict(recipe,VLM_frozen=False)]:
            with self.assertRaises(PermissionError):validate_recipe(bad,'S-CORR')
        with self.assertRaises(PermissionError):validate_recipe(recipe,'S-EVENT')

    def test_ROS_setup_optional_variables_do_not_disable_later_guards(self):
        definition=Path('infrastructure/containers/vanda_wbc_jazzy.def').read_text()
        prefix=definition.split('%test\n',1)[1].split('    /opt/wbc-python/bin/python',1)[0]
        with tempfile.TemporaryDirectory() as d:
            setup=Path(d)/'setup.sh';setup.write_text('test -z "$AMENT_TRACE_SETUP_FILES"\nexport ROS_DISTRO=jazzy\n')
            script=prefix.replace('/opt/ros/jazzy/setup.sh',str(setup))
            good=subprocess.run(['/bin/sh','-c',script+'test "$ROS_DISTRO" = jazzy'],capture_output=True)
            self.assertEqual(good.returncode,0)
            unset=subprocess.run(['/bin/sh','-c',script+'echo "$MP1_POST_SETUP_UNDEFINED"'],capture_output=True)
            self.assertNotEqual(unset.returncode,0)
            setup.write_text('false\n')
            failed=subprocess.run(['/bin/sh','-c',script+'echo SHOULD_NOT_EXECUTE'],capture_output=True)
            self.assertNotEqual(failed.returncode,0);self.assertNotIn(b'SHOULD_NOT_EXECUTE',failed.stdout)

    def test_task_balance_precedes_T1_only_correction(self):
        d=dataset();s=SourceSampler(d,stop=64);seen=Counter()
        for cursor,index in s:
            seen[d.rows[index][0]['task']]+=1
            seen['correction' if index>=d.original_rows else 'original']+=1
        self.assertEqual(seen,Counter(T1=32,T2=32,correction=16,original=48))
        self.assertEqual(s.state_dict(64)['overall_correction_fraction'],.25)
        self.assertEqual({t:len(v) for t,v in s.original.items()},dict(T1=96,T2=30))

    def test_both_qualified_tasks_receive_half_within_task(self):
        d=dataset(('T1','T2'));s=SourceSampler(d,stop=64)
        self.assertEqual(sum(i>=d.original_rows for c,i in s),32)
        self.assertEqual(s.state_dict(64)['overall_correction_fraction'],.5)

    def test_BC_and_exact_resume_order(self):
        for tasks in [(),('T1',),('T1','T2')]:
            d=dataset(tasks);full=list(SourceSampler(d,stop=96))
            self.assertEqual(full[32:],list(SourceSampler(d,start=32,stop=96)))
            self.assertEqual(SourceSampler(d,stop=96).seed,20261017)
            if not tasks:self.assertFalse(any(i>=d.original_rows for c,i in full))

    def test_DEV_or_missing_original_episode_is_rejected(self):
        d=dataset();d.rows[0][0]['split']='DEV'
        with self.assertRaises(ValueError):SourceSampler(d)
        d=dataset();d.rows=d.rows[3:];d.original_rows-=3
        with self.assertRaises(ValueError):SourceSampler(d)

    def test_out_of_scope_cursor_and_single_correction_episode_fail(self):
        d=dataset();s=SourceSampler(d,stop=32)
        for cursor in [-1,32]:
            with self.assertRaises(IndexError):s.index_at(cursor)
        d.rows=d.rows[:-5]
        with self.assertRaises(ValueError):SourceSampler(d)
        d=dataset()
        for row,a in d.rows[d.original_rows:]:row['source_student_episode']='same-parent'
        with self.assertRaises(ValueError):SourceSampler(d)


if __name__=='__main__':unittest.main()
