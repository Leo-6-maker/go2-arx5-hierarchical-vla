"""Tiny full-state metadata fixtures; no model, native job or payload copy."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.pv1 import checkpoint_reuse as reuse,training
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.pv1.recipe import primary_recipe
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.test_mp1_transport_fixture import put


class CheckpointReuse(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.parent=Path(tmp.name)/'campaign';self.root=self.parent/'children/new';self.source=self.parent/'children/old'
        self.recipe=primary_recipe(old_recipe(),'COVERAGE_FIRST','A_CONTROL');self.key='A_CONTROL';self.step=1024
        self.relative='training/A_CONTROL/checkpoint-1024';self.checkpoint=self.source/self.relative
        self.dataset=SimpleNamespace(manifest_sha='FIXED_ORIGINAL_C1_NEW_BANK')
        self.dataset.make_sampler=lambda dataset,stop:SimpleNamespace(state_dict=lambda n:dict(cursor=n,stop=stop,seed=20261019))
        for name in ('config.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors','processor/config.json','statistics.json'):
            put(self.checkpoint/name,dict(CPU_fixture=name))
        self.stats=sha(self.checkpoint/'statistics.json')
        put(self.checkpoint/'trainer_state.json',dict(global_step=self.step))
        put(self.checkpoint/'SAMPLER_STATE.json',dict(recipe=self.recipe,
            sampler=self.dataset.make_sampler(self.dataset,4096*32).state_dict(self.step*32),
            manifest_sha256=self.dataset.manifest_sha,statistics_sha256=self.stats))
        self.complete=dict(full_resume=True,global_step=self.step,origin_source_sha='a'*40,
            origin_job_id='123.hopper-m-02',files={str(p.relative_to(self.checkpoint)):sha(p)
                for p in self.checkpoint.rglob('*') if p.is_file()})
        put(self.checkpoint/'CHECKPOINT_COMPLETE.json',self.complete)
        put(self.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',dict(authority_release=AUTHORITY,run_uuid='CAMPAIGN'))
        self.lease=self.parent/'source-lease.json'
        self.lease_value=dict(status='CLOSED',authority_release=AUTHORITY,run_uuid='old',all_registered_workers_terminal=True)
        put(self.lease,self.lease_value)
        self.manifests={}
        self.registry=self.parent/'registry.json';entries=[]
        put(self.parent/'OUTPUT_BINDING.private.json',dict(registry=str(self.registry)))
        for child,code in ((self.source,'a'*40),(self.root,'b'*40)):
            binding=child/'OUTPUT_BINDING.private.json'
            role=reuse.ROLE+'__'+child.name
            put(binding,dict(run_uuid=child.name,run_root=str(child),uid=os.getuid(),lease=str(self.lease),
                registry=str(self.registry),role=role))
            entries.append(dict(run_uuid=child.name,role=role,scope=reuse.ROLE,path=str(child)))
            for folder in ('worker_processes','aux_worker_processes'):(child/folder).mkdir()
            sealed_inputs={}
            for name,value in [('plans/FROZEN_SOURCE_PAIR_RECIPE.json',old_recipe()),
                    ('plans/PRIMARY_LEARNING_RECIPES.json',dict(recipes={self.key:self.recipe}))]:
                put(child/name,value,True);sealed_inputs[name]=sha(child/name)
            m=dict(run_uuid=child.name,scope=reuse.ROLE,authority=dict(protocol_ref=AUTHORITY),code=dict(vla_commit=code),
                execution=dict(output_binding_sha256=sha(binding),sealed_inputs=sealed_inputs))
            put(child/'RUN_MANIFEST.json',m);self.manifests[child]=m
        put(self.registry,dict(entries=entries))
        self.campaign=dict(campaign_run_uuid='CAMPAIGN',authority_release=AUTHORITY,unknown_effects=[],pending_updates=[],
            optimizer_updates=self.step,training_runs={self.key:dict(recipe_sha256=digest(self.recipe),last_native_step=self.step)},
            PBS_submissions=[dict(run_uuid='old',status='NATIVE_TERMINAL',job_id='123.hopper-m-02',training_run_key=self.key)])
        put(self.parent/'HOPPER_RESOURCE_LEDGER.json',self.campaign)
        self.plan=dict(schema='PI0294_PV1_CLOSED_SIBLING_CHECKPOINT_REUSE_V1',authority_release=AUTHORITY,
            campaign_run_uuid='CAMPAIGN',source_child_root=str(self.source),source_code_sha='a'*40,
            source_manifest_sha256=sha(self.source/'RUN_MANIFEST.json'),source_checkpoint_relative=self.relative,
            destination_checkpoint_relative=self.relative,source_checkpoint_manifest_sha256=sha(self.checkpoint/'CHECKPOINT_COMPLETE.json'),
            run_key=self.key,global_step=self.step,recipe_sha256=digest(self.recipe),statistics_sha256=self.stats,
            dataset_manifest_sha256=self.dataset.manifest_sha,copy_full_resume_state=True,additional_optimizer_updates=0)
        self.bind()

    def bind(self):
        put(self.root/reuse.PLAN,self.plan,True)
        self.manifests[self.root]['execution']['sealed_inputs'][reuse.PLAN]=sha(self.root/reuse.PLAN)
        put(self.root/'RUN_MANIFEST.json',self.manifests[self.root])

    def validate(self):
        with patch.object(reuse,'STATS',self.stats),patch.object(training,'STATS',self.stats):
            return reuse.validate(self.root,self.recipe,self.dataset)

    def test_full_state_copy_plan_preserves_source_counters_and_exact_sample_cursor_without_copy(self):
        ledger=self.parent/'HOPPER_RESOURCE_LEDGER.json';before=ledger.read_bytes()
        result=self.validate()
        self.assertEqual(result['status'],'PASS_CLOSED_SIBLING_FULL_RESUME_COPY_PLAN_NOT_COPIED')
        self.assertEqual(result['resume_sample_cursor'],32768)
        self.assertEqual(result['optimizer_updates_already_charged'],1024)
        self.assertEqual(result['additional_optimizer_updates'],0)
        self.assertFalse(result['actual_copy_executed']);self.assertFalse((self.root/self.relative).exists())
        self.assertEqual(ledger.read_bytes(),before)
        self.assertEqual(set(result['files']),set(self.complete['files'])|{'CHECKPOINT_COMPLETE.json'})
        self.assertEqual(result['copy_logical_bytes'],sum((self.checkpoint/n).stat().st_size for n in result['files']))
        self.assertEqual(result['source_code_sha'],'a'*40);self.assertEqual(result['destination_code_sha'],'b'*40)

    def test_unknown_pending_active_native_or_newer_committed_step_never_rolls_back(self):
        for kind in ('unknown','pending','active_job','newer_step','wrong_recipe','wrong_origin_job'):
            with self.subTest(kind=kind):
                v=deepcopy(self.campaign)
                if kind=='unknown':v['unknown_effects']=['UNKNOWN_ACTION']
                elif kind=='pending':v['pending_updates']=['UNKNOWN_OPTIMIZER']
                elif kind=='active_job':v['PBS_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
                elif kind=='newer_step':v['training_runs'][self.key]['last_native_step']=1025
                elif kind=='wrong_recipe':v['training_runs'][self.key]['recipe_sha256']='c'*64
                else:v['PBS_submissions'][0]['job_id']='999.hopper-m-02'
                put(self.parent/'HOPPER_RESOURCE_LEDGER.json',v)
                with self.assertRaises(PermissionError):self.validate()
        put(self.parent/'HOPPER_RESOURCE_LEDGER.json',self.campaign)

    def test_active_lease_live_worker_and_existing_partial_destination_preserved(self):
        put(self.lease,dict(self.lease_value,status='ACTIVE'))
        with self.assertRaises(PermissionError):self.validate()
        put(self.lease,self.lease_value)
        proc=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        path=self.source/'worker_processes/live.json';put(path,dict(pid=os.getpid(),proc_start=proc[19]))
        with self.assertRaises(PermissionError):self.validate()
        path.unlink()
        destination=self.root/self.relative;destination.mkdir(parents=True)
        put(destination/'UNIQUE_PARTIAL.json',dict(keep=True))
        with self.assertRaises(PermissionError):self.validate()
        self.assertEqual(read(destination/'UNIQUE_PARTIAL.json'),dict(keep=True))

    def test_missing_duplicate_wrong_registry_and_missing_process_inventory_rejected(self):
        original=read(self.registry)
        variants=[dict(entries=original['entries'][1:]),dict(entries=original['entries']*2)]
        wrong=deepcopy(original);wrong['entries'][0]['path']=str(self.parent/'OTHER')
        variants.append(wrong)
        for variant in variants:
            with self.subTest(variant=variant):
                put(self.registry,variant)
                with self.assertRaisesRegex(PermissionError,'CANONICAL_REGISTERED'):self.validate()
        put(self.registry,original)
        (self.source/'aux_worker_processes').rmdir()
        with self.assertRaisesRegex(PermissionError,'PROCESS_INVENTORY'):self.validate()

    def test_changed_payload_and_validly_rehashed_wrong_sampler_are_not_exact_resume(self):
        optimizer=self.checkpoint/'optimizer.pt';before=optimizer.read_bytes();optimizer.write_bytes(b'CHANGED')
        with self.assertRaises(ValueError):self.validate()
        optimizer.write_bytes(before)
        saved=read(self.checkpoint/'SAMPLER_STATE.json');saved['sampler']['cursor']+=32
        put(self.checkpoint/'SAMPLER_STATE.json',saved)
        self.complete['files']['SAMPLER_STATE.json']=sha(self.checkpoint/'SAMPLER_STATE.json')
        put(self.checkpoint/'CHECKPOINT_COMPLETE.json',self.complete)
        self.plan['source_checkpoint_manifest_sha256']=sha(self.checkpoint/'CHECKPOINT_COMPLETE.json');self.bind()
        with self.assertRaises(PermissionError):self.validate()

    def test_source_identity_wrong_arm_parent_escape_and_unsealed_plan_rejected(self):
        original=deepcopy(self.plan)
        for field,value in [('source_code_sha','c'*40),('run_key','B_COVERAGE'),
                ('source_child_root',str(self.parent/'outside')),('source_checkpoint_relative','../outside'),
                ('destination_checkpoint_relative','training/A_CONTROL/checkpoint-1025'),('additional_optimizer_updates',1024)]:
            with self.subTest(field=field):
                self.plan=deepcopy(original);self.plan[field]=value;self.bind()
                with self.assertRaises(PermissionError):self.validate()
        self.plan=original;self.bind();(self.root/reuse.PLAN).chmod(0o600)
        with self.assertRaises(PermissionError):self.validate()

    def test_symlink_checkpoint_member_binding_and_destination_components_are_rejected(self):
        optimizer=self.checkpoint/'optimizer.pt';original=optimizer.read_bytes();optimizer.unlink()
        other=self.parent/'KEEP_OPTIMIZER';other.write_bytes(original);optimizer.symlink_to(other)
        with self.assertRaises(PermissionError):self.validate()
        optimizer.unlink();optimizer.write_bytes(original)
        binding=self.source/'OUTPUT_BINDING.private.json';original=binding.read_bytes();binding.unlink()
        other=self.parent/'KEEP_BINDING';other.write_bytes(original);binding.symlink_to(other)
        with self.assertRaises(PermissionError):self.validate()
        binding.unlink();binding.write_bytes(original)
        target=self.root/'training';target.symlink_to(self.source/'training',target_is_directory=True)
        with self.assertRaises(PermissionError):self.validate()

    def test_inference_only_or_terminal4096_checkpoint_cannot_be_continuation_parent(self):
        for field,value in [('full_resume',False),('global_step',4096),('origin_source_sha','c'*40)]:
            with self.subTest(field=field):
                complete=dict(self.complete,**{field:value});put(self.checkpoint/'CHECKPOINT_COMPLETE.json',complete)
                self.plan['source_checkpoint_manifest_sha256']=sha(self.checkpoint/'CHECKPOINT_COMPLETE.json');self.bind()
                with self.assertRaises(PermissionError):self.validate()
