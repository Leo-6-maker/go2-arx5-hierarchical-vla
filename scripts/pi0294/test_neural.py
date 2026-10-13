"""CPU-only sampling/allocation/cache contracts, not neural qualification."""
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile
import sys
from types import SimpleNamespace

import numpy as np

from .native_dataset import ByteLRU
from .sampling import BalancedSampler, PHASES
from .neural import recipe_for,benchmark_plan,smoke_plan
from .native_train import MeasuredLoader,qualify_cpu
from .local_backbone import local_model_name
from .neural import smoke_output_root,retire_redundant_smoke
from .allocation_admission import remaining_peak,GIB
from .native_dataset import sha


def fixture():
    rows=[]
    for task,origin in [('T1','legacy'),('T1','new'),('T2','new')]:
        for family in ('F0','F1'):
            for episode in range(2):
                record=dict(id=f'{task}-{origin}-{family}-{episode}',task=task,origin=origin,family=family,
                            stage_anchors={p:list(range(i*2,i*2+2)) for i,p in enumerate(PHASES)})
                rows.extend((record,anchor) for anchor in range(10))
    return SimpleNamespace(rows=rows,manifest_sha='UNIT_FIXTURE_ONLY')


class NeuralContracts(unittest.TestCase):
    def test_local_backbone_requires_exact_snapshot_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/'existing-snapshot';snapshot.mkdir()
            alias=root/'model_inputs/nvidia/Cosmos-Reason2-2B'
            alias.parent.mkdir(parents=True)
            with self.assertRaises(PermissionError):local_model_name(root,{'cosmos':str(snapshot)})
            alias.symlink_to(snapshot,target_is_directory=True)
            self.assertEqual(local_model_name(root,{'cosmos':str(snapshot)}),str(alias))
            with self.assertRaises(PermissionError):local_model_name(root,{'cosmos':str(root)})

    def test_smoke_retry_requires_known_zero_update_terminal_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proof=root/'retry.json'
            value=dict(status='KNOWN_TERMINAL_ZERO_UPDATE_ENGINEERING_RETRY',run_uuid=root.name,
                       actual_optimizer_updates=0,prior_job_terminal=True)
            proof.write_text(json.dumps(value))
            request=dict(smoke_output_relative='smoke_attempts/02',engineering_retry_receipt=str(proof),
                         engineering_retry_receipt_sha256=sha(proof))
            self.assertEqual(smoke_output_root(root,request),root/'smoke_attempts/02')
            with self.assertRaises(PermissionError):smoke_output_root(root,dict(request,smoke_output_relative='../escape'))
            value['actual_optimizer_updates']=1;proof.write_text(json.dumps(value));request['engineering_retry_receipt_sha256']=sha(proof)
            with self.assertRaises(PermissionError):smoke_output_root(root,request)

    def test_remaining_peak_does_not_double_count_existing_checkpoints(self):
        request=dict(predicted_total_owned_peak_bytes=160*GIB,predicted_remaining_peak_bytes=110*GIB)
        self.assertEqual(remaining_peak(request,71*GIB),89*GIB)
        for total in (70*GIB,181*GIB):
            with self.assertRaises(PermissionError):remaining_peak(dict(request,predicted_total_owned_peak_bytes=total),71*GIB)

    def test_six_update_recovery_forbids_repeating_first_half(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proof=root/'retry.json'
            value=dict(status='KNOWN_TERMINAL_SIX_UPDATE_SMOKE_RESUME_ONLY',run_uuid=root.name,
                       actual_optimizer_updates=6,prior_job_terminal=True)
            proof.write_text(json.dumps(value))
            request=dict(smoke_output_relative='smoke_attempts/02',engineering_retry_receipt=str(proof),
                         engineering_retry_receipt_sha256=sha(proof),mode='smoke-recovery-benchmark',
                         allowed_modes=['smoke-recovery-benchmark','smoke-resume','benchmark-all','benchmark'])
            self.assertEqual(smoke_output_root(root,request),root/'smoke_attempts/02')
            with self.assertRaises(PermissionError):smoke_output_root(root,dict(request,allowed_modes=['smoke-first']))

    def test_only_redundant_generated_smoke_states_retired_after_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence').mkdir();smoke=root/'smoke_attempts/02'
            comparison=smoke/'evidence/comparison.json';comparison.parent.mkdir(parents=True)
            comparison.write_text(json.dumps(dict(status='FAIL_PRESERVED')))
            with self.assertRaises(PermissionError):retire_redundant_smoke(root,smoke,comparison)
            comparison.write_text(json.dumps(dict(status='PASS_FIXED_RESUME_EQUIVALENCE')))
            for name in ('smoke_continuous/checkpoint-2','smoke_continuous/checkpoint-4','smoke_split/checkpoint-2','smoke_split/checkpoint-4'):
                path=smoke/'training'/name;path.mkdir(parents=True)
                (path/'state').write_bytes(b'unit fixture only')
                (path/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(dict(global_step=int(name[-1]),required_files={'state':sha(path/'state')})))
            retire_redundant_smoke(root,smoke,comparison)
            self.assertTrue((smoke/'training/smoke_split/checkpoint-4/state').exists())
            self.assertFalse((smoke/'training/smoke_continuous/checkpoint-4').exists())
            actual=json.loads((root/'evidence/REDUNDANT_SMOKE_RETIREMENT_ACTUAL.json').read_text())
            self.assertFalse(actual['historical_assets_deleted'])

    def test_task_and_old_exposure(self):
        data=fixture();sampler=BalancedSampler(data,stop=600)
        samples=[data.rows[i][0] for _,i in sampler]
        self.assertEqual(sum(r['task']=='T1' for r in samples),300)
        self.assertEqual(sum(r['origin']=='legacy' for r in samples),100)

    def test_resume_cursor_independent_of_prefetch(self):
        data=fixture()
        full=list(BalancedSampler(data,stop=320))
        restored=list(BalancedSampler(data,start=96,stop=320))
        self.assertEqual(full[96:],restored)

    def test_actual_legacy_only_F0_does_not_distort_family_balance(self):
        data=fixture()
        data.rows=[item for item in data.rows if item[0]['origin']!='legacy' or item[0]['family']=='F0']
        sampler=BalancedSampler(data,stop=12000)
        samples=[data.rows[index][0] for _,index in sampler]
        for task in ('T1','T2'):
            task_rows=[row for row in samples if row['task']==task]
            self.assertEqual(len(task_rows),6000)
            self.assertLess(abs(sum(row['family']=='F0' for row in task_rows)-3000),180)
        self.assertLessEqual(sum(row['origin']=='legacy' for row in samples),2000)
        self.assertIn('FAMILY_FIRST',sampler.state_dict(32)['algorithm'])

    def test_previous_single_task_smoke_sampler_identity_is_preserved(self):
        data=fixture();data.rows=[item for item in data.rows if item[0]['task']=='T1' and item[0]['origin']=='legacy']
        sampler=BalancedSampler(data,stop=128,smoke=True)
        self.assertEqual(sampler.state_dict(128)['algorithm'],'PI0294_POSITION_TASK_FAMILY_EPISODE_STAGE_V1')

    def test_empty_phase_explicit_renormalization(self):
        data=fixture()
        for row,anchor in data.rows:
            row['stage_anchors']={p:[] for p in PHASES}
            row['stage_anchors']['hold']=list(range(10))
        sampler=BalancedSampler(data,stop=32)
        self.assertTrue(sampler.empty)
        self.assertEqual(len(list(sampler)),32)

    def test_formal_single_task_rejected(self):
        data=fixture();data.rows=[r for r in data.rows if r[0]['task']=='T1']
        with self.assertRaises(ValueError):
            BalancedSampler(data)

    def test_lru_bounded(self):
        cache=ByteLRU(8)
        cache.put('a',np.ones(4,dtype=np.uint8),4)
        cache.put('b',np.ones(5,dtype=np.uint8),5)
        self.assertIsNone(cache.get('a'))
        self.assertLessEqual(cache.bytes,8)
        cache.put('big',None,9)
        self.assertLessEqual(cache.bytes,8)

    def test_recipe_real_effective_batch(self):
        for micro in (2,4,8):
            for workers in (0,2,4):
                recipe=recipe_for('main',micro,workers)
                self.assertEqual(recipe['per_device_train_batch_size']*recipe['gradient_accumulation_steps'],32)
                self.assertEqual(recipe['action_loss'],'UNCHANGED_NATIVE_MASKED_FLOW_MATCHING')
                self.assertFalse(recipe['tune_llm'])
                self.assertFalse(recipe['tune_visual'])
                self.assertTrue(recipe['bf16'])

    def test_partial_qualification_naturally_drains_finite_iterator(self):
        loader=MeasuredLoader(range(32));iterator=iter(loader)
        consumed=[next(iterator) for _ in range(20)]
        drained=list(iterator)
        self.assertEqual(consumed,list(range(20)))
        self.assertEqual(drained,list(range(20,32)))
        self.assertEqual(loader.batches,32)

    def test_failed_cpu_attempt_preserves_original_error_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'qualification.json'
            with patch.object(sys.modules[qualify_cpu.__module__],'_qualify_cpu',side_effect=ValueError('PRIMARY')):
                with self.assertRaisesRegex(ValueError,'PRIMARY'):
                    qualify_cpu(SimpleNamespace(output=str(output)))
            receipt=json.loads(output.with_name(output.name+'.CPU_ATTEMPT.json').read_text())
            self.assertFalse(output.exists())
            self.assertEqual(receipt['status'],'FAILED')
            self.assertTrue(receipt['unreaped_children_unknown'])
            self.assertGreaterEqual(receipt['CPU_s'],0)

    def test_prior_attempt_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'qualification.json'
            output.with_name(output.name+'.CPU_ATTEMPT.json').touch()
            with patch.object(sys.modules[qualify_cpu.__module__],'_qualify_cpu') as execute:
                with self.assertRaises(FileExistsError):qualify_cpu(SimpleNamespace(output=str(output)))
                execute.assert_not_called()

    def test_main_scheduler_identical_across_conditional_extension(self):
        normal=recipe_for('main',2,0,4096)
        extension=recipe_for('main',2,0,6144)
        self.assertEqual(normal,extension)
        self.assertEqual(normal['scheduler_horizon_steps'],6144)
        self.assertEqual(normal['warmup_steps'],308)

    def test_benchmarks_exact_six_updates_fixed_order(self):
        qualification=dict(status='PASS_PI0294_ACTUAL_NATIVE_CPU_LOADER_0_2_4',
            actual_loader_throughput={f'{w}/{s}':dict(natural_iterator_exhaustion=True,
                consumed_samples_per_s=10+w) for w in (0,2,4) for s in (0,1)})
        plan=benchmark_plan(qualification)
        self.assertEqual([r['microbatch'] for r in plan],[2,4,8])
        self.assertEqual(sum(r['updates'] for r in plan),6)
        self.assertTrue(all(r['workers']==4 for r in plan))
        qualification['actual_loader_throughput']['4/1']['natural_iterator_exhaustion']=False
        with self.assertRaises(PermissionError):benchmark_plan(qualification)

    def test_one_allocation_smoke_exact_eight_update_order(self):
        self.assertEqual(smoke_plan(),[('smoke-continuous',4),('smoke-first',2),('smoke-resume',2)])
        self.assertEqual(sum(n for _,n in smoke_plan()),8)


if __name__=='__main__':
    unittest.main()
