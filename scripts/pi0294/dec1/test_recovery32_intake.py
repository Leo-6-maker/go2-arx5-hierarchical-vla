"""Synthetic tiny full-state files only; no actual model, tensors or remote IO."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import training
from . import recovery32_intake as intake


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return path


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'closed';target=Path(tmp)/'cold';source.mkdir();target.mkdir()
        recipe=dict(seed=20261019);expected=deepcopy(intake.EXPECTED)
        cp=source/'training/A_CONTROL/checkpoint-32';cp.mkdir(parents=True)
        stat=put(cp/'statistics.json',dict(fixture='ORIGINAL_STATS'))
        sampler=lambda dataset,stop:SimpleNamespace(state_dict=lambda n:dict(cursor=n,seed=20261019,stop=stop))
        dataset=SimpleNamespace(manifest_sha=expected['manifest_sha256'],make_sampler=sampler)
        for name in ('config.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors','processor/config.json'):
            put(cp/name,dict(fixture='NOT_A_TENSOR'))
        put(cp/'trainer_state.json',dict(global_step=32))
        put(cp/'SAMPLER_STATE.json',dict(recipe=recipe,sampler=sampler(dataset,4096*32).state_dict(1024),
            manifest_sha256=dataset.manifest_sha,statistics_sha256=intake.sha(stat)))
        complete=dict(full_resume=True,global_step=32,origin_job_id=expected['job_id'],origin_source_sha=expected['source_sha'],
            files={p.relative_to(cp).as_posix():intake.sha(p) for p in cp.rglob('*') if p.is_file()})
        expected['complete_sha256']=intake.sha(put(cp/'CHECKPOINT_COMPLETE.json',complete))
        effects=dict(job_id=expected['job_id'],request_id=expected['request_id'],request_sha256=expected['request_sha256'],
            source_sha=expected['source_sha'],recipe_sha256=intake.digest(recipe),arm='A_CONTROL',start_native_step=0,
            last_native_step=32,optimizer_updates_committed=32,pending_updates=[],unknown_query_intents=[],effect_journal_complete=True)
        terminal=dict(status='CLOSED_FIRST32_COST_STOP_FULL_STATE_PRESERVED_POSTSAVE_ENGINEERING_FAILURE',
            job_id=expected['job_id'],recoverable_full_state=True,global_step=32,settlement=dict(unknown_effects=[]),effects=effects)
        expected['terminal_sha256']=intake.sha(put(source/'TERMINAL_RECONCILIATION.json',terminal))
        expected['lease_sha256']=intake.sha(put(source/'lease.json',dict(status='CLOSED',run_uuid=source.name,
            terminal_receipt_sha256=expected['terminal_sha256'])))
        expected['run_manifest_sha256']=intake.sha(put(source/'RUN_MANIFEST.json',dict(code=dict(vla_commit=expected['source_sha']))))
        put(source/'OUTPUT_BINDING.private.json',dict(run_root=str(source),uid=__import__('os').getuid(),
            run_manifest_sha256=expected['run_manifest_sha256'],lease=str(source/'lease.json')))
        with patch.object(intake,'EXPECTED',expected),patch.object(intake,'RECIPE',intake.digest(recipe)),\
                patch.object(intake,'R0',intake.sha(stat)),patch.object(training,'STATS',intake.sha(stat)):
            yield source,target,recipe,dataset,expected


def stage(source,target,proof):
    shutil.copytree(source/'training',target/'training')
    p=put(target/intake.PROOF,proof);p.chmod(0o400)
    put(target/'RUN_MANIFEST.json',dict(authority=dict(recovery_authorization_release=intake.RELEASE),
        execution=dict(sealed_inputs={intake.PROOF:intake.sha(p)})))


class Recovery32IntakeTests(unittest.TestCase):
    def test_full_read_and_cold_copy_reuse_mature_resume_without_mutation(self):
        with fixture() as (s,t,recipe,data,e):
            before={p:intake.sha(p) for p in s.rglob('*') if p.is_file()}
            proof=intake.inspect_closed_source(s,recipe=recipe,dataset=data)
            self.assertEqual(proof['actual_consumed_cursor'],1024)
            stage(s,t,proof)
            result=intake.verify_cold_import(t,recipe=recipe,dataset=data)
            self.assertEqual(result['global_step'],32)
            self.assertEqual(result['new_GPU_PBS_models_queries_updates_physics'],0)
            self.assertEqual(before,{p:intake.sha(p) for p in s.rglob('*') if p.is_file()})

    def test_changed_source_manifest_terminal_or_lease_is_never_a_resume(self):
        for name in ('RUN_MANIFEST.json','TERMINAL_RECONCILIATION.json','lease.json'):
            with self.subTest(name=name),fixture() as (s,t,recipe,data,e):
                (s/name).write_text('{}')
                with self.assertRaises(PermissionError):intake.inspect_closed_source(s,recipe=recipe,dataset=data)

    def test_corrupt_optimizer_wrong_recipe_data_or_cursor_rejected(self):
        for kind in ('optimizer','recipe','data','cursor'):
            with self.subTest(kind=kind),fixture() as (s,t,recipe,data,e):
                if kind=='optimizer':(s/'training/A_CONTROL/checkpoint-32/optimizer.pt').write_text('DAMAGED')
                elif kind=='recipe':recipe['seed']+=1
                elif kind=='data':data.manifest_sha='CHANGED'
                else:data.make_sampler=lambda dataset,stop:SimpleNamespace(state_dict=lambda n:dict(cursor=n+32,seed=20261019,stop=stop))
                with self.assertRaises((PermissionError,ValueError)):
                    intake.inspect_closed_source(s,recipe=recipe,dataset=data)

    def test_inference_parent_or_missing_full_state_never_zero_restarts(self):
        with fixture() as (s,t,recipe,data,e):
            (s/'training/A_CONTROL/checkpoint-32/optimizer.pt').unlink()
            with self.assertRaises(PermissionError):intake.inspect_closed_source(s,recipe=recipe,dataset=data)
            self.assertFalse((t/'training').exists())

    def test_manifest_must_bind_exact_readonly_proof_and_copied_bytes(self):
        for kind in ('unsealed','writable','proof','member'):
            with self.subTest(kind=kind),fixture() as (s,t,recipe,data,e):
                proof=intake.inspect_closed_source(s,recipe=recipe,dataset=data);stage(s,t,proof)
                if kind=='unsealed':put(t/'RUN_MANIFEST.json',{})
                elif kind=='writable':(t/intake.PROOF).chmod(0o600)
                elif kind=='proof':
                    p=t/intake.PROOF;p.chmod(0o600);proof['global_step']=1056;put(p,proof);p.chmod(0o400)
                    put(t/'RUN_MANIFEST.json',dict(authority=dict(recovery_authorization_release=intake.RELEASE),
                        execution=dict(sealed_inputs={intake.PROOF:intake.sha(p)})))
                else:(t/'training/A_CONTROL/checkpoint-32/rng_state.pth').write_text('CHANGED')
                with self.assertRaises((PermissionError,ValueError)):
                    intake.verify_cold_import(t,recipe=recipe,dataset=data)

    def test_symlink_escape_does_not_grant_approved_alias_exception(self):
        with fixture() as (s,t,recipe,data,e):
            cp=s/'training/A_CONTROL/checkpoint-32';p=cp/'optimizer.pt'
            outside=s/'outside.pt';shutil.copyfile(p,outside);p.unlink();p.symlink_to(outside)
            with self.assertRaises(PermissionError):intake.inspect_closed_source(s,recipe=recipe,dataset=data)

    def test_terminal_semantics_still_reject_unknown_or_unclosed_effects(self):
        for key,value in (('recoverable_full_state',False),('global_step',1024),
                ('settlement',dict(unknown_effects=['UNCERTAIN']))):
            with self.subTest(key=key),fixture() as (s,t,recipe,data,e):
                terminal=intake.read(s/'TERMINAL_RECONCILIATION.json');terminal[key]=value
                e['terminal_sha256']=intake.sha(put(s/'TERMINAL_RECONCILIATION.json',terminal))
                lease=intake.read(s/'lease.json');lease['terminal_receipt_sha256']=e['terminal_sha256']
                e['lease_sha256']=intake.sha(put(s/'lease.json',lease))
                with self.assertRaises(PermissionError):intake.inspect_closed_source(s,recipe=recipe,dataset=data)

    def test_resealed_absolute_member_path_is_not_a_valid_full_state(self):
        with fixture() as (s,t,recipe,data,e):
            cp=s/'training/A_CONTROL/checkpoint-32';complete=intake.read(cp/'CHECKPOINT_COMPLETE.json')
            complete['files'][str(cp/'optimizer.pt')]=complete['files'].pop('optimizer.pt')
            e['complete_sha256']=intake.sha(put(cp/'CHECKPOINT_COMPLETE.json',complete))
            with self.assertRaises(PermissionError):intake.inspect_closed_source(s,recipe=recipe,dataset=data)

    def test_allocated_import_copies_exact_state_links_only_frozen_and_never_retries(self):
        import os
        from . import a40_training_context as context,segmented_training as seg,checkpoint_storage as storage
        with fixture() as (s,t,recipe,data,e):
            proof=intake.inspect_closed_source(s,recipe=recipe,dataset=data)
            p=put(t/intake.PROOF,proof);p.chmod(0o400)
            (t/'control').mkdir()
            binding=put(t/'plans/RECOVERY32_SOURCE_BINDING.private.json',dict(root=str(s),checkpoint_complete_sha256=e['complete_sha256']))
            parent=t.parent/'parent';parent.mkdir();name='model.safetensors'
            shutil.copyfile(s/'training/A_CONTROL/checkpoint-32'/name,parent/name)
            for file,value in ((storage.POLICY,dict(members={name:proof['members'][name]})),
                    ('plans/PV1_TRAIN_INPUTS.private.json',dict(CORR_parent_checkpoint=str(parent)))):
                put(t/file,value).chmod(0o400)
            binding.chmod(0o400)
            put(t/'RUN_MANIFEST.json',dict(authority=dict(recovery_authorization_release=intake.RELEASE),
                execution=dict(sealed_inputs={intake.PROOF:intake.sha(p)})))
            directory=t/'pbs/alloc-16';put(directory/'REQUEST.json',dict(training_block=dict(arm='A_CONTROL'),additional_output_upper_bytes=70000000000))
            original=parent.joinpath(name).stat()
            with patch.dict(os.environ,dict(PBS_JOBID='fixture-no-real-GPU')),\
                    patch.object(context,'current_directory',return_value=directory),\
                    patch.object(seg,'limits',return_value=(32,2048)),\
                    patch.object(storage,'_parent_files',return_value={name:proof['members'][name]}):
                result=intake.stage_in_allocation(t,recipe=recipe,dataset=data,guard=lambda:None,allocation={})
                self.assertEqual(result['global_step'],32)
                copied=t/'training/A_CONTROL/checkpoint-32'
                self.assertEqual((copied/name).stat().st_ino,original.st_ino)
                self.assertNotEqual((copied/'optimizer.pt').stat().st_ino,(s/'training/A_CONTROL/checkpoint-32/optimizer.pt').stat().st_ino)
                self.assertEqual((parent/name).stat().st_mode,original.st_mode)
                with self.assertRaises(PermissionError):intake.stage_in_allocation(t,recipe=recipe,dataset=data,guard=lambda:None,allocation={})
