"""Small synthetic files only; no model, Trainer, allocation or native claim."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import checkpoint_storage as storage
from .authority import AUTHORITY
from .source_admission import VERSION

NAME='model-00001-of-00003.safetensors'


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return storage.sha(path)


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as d:
        parent=Path(d)/'protected-parent';root=Path(d)/'new';parent.mkdir();root.mkdir()
        header=json.dumps({'backbone.layer.weight':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode()
        payload=struct.pack('<Q',len(header))+header+struct.pack('<f',.25)
        (parent/NAME).write_bytes(payload);(parent/NAME).chmod(0o600)
        digest=storage.sha(parent/NAME)
        parent_sha=put(parent/'CHECKPOINT_COMPLETE.json',dict(files={NAME:digest},global_step=2048,full_resume=True))
        checkpoint=root/'training/A_CONTROL/checkpoint-1024';checkpoint.mkdir(parents=True)
        (checkpoint/NAME).write_bytes(payload)
        (checkpoint/'optimizer.pt').write_bytes(b'ACTUAL_FIXTURE_FULL_STATE_NOT_A_TORCH_MODEL')
        complete=put(checkpoint/'CHECKPOINT_COMPLETE.json',dict(full_resume=True,global_step=1024,
            files={NAME:digest,'optimizer.pt':storage.sha(checkpoint/'optimizer.pt')},
            origin_job_id='fixture-job',origin_source_sha='a'*40))
        policy=dict(schema='DEC1_EXACT_FROZEN_SHARD_SHARING_V1',authority_release=AUTHORITY,
            parent_checkpoint_manifest_sha256=parent_sha,checkpoint_retirement=False,parent_chmod=False,
            maximum_shared_shards=1,members={NAME:digest})
        ph=put(root/storage.POLICY,policy);(root/storage.POLICY).chmod(0o400)
        put(root/'RUN_MANIFEST.json',dict(code=dict(vla_commit='a'*40),
            authority=dict(training_admission_version=VERSION,amendments=[AUTHORITY]),
            execution=dict(sealed_inputs={storage.POLICY:ph})))
        put(root/'plans/PV1_TRAIN_INPUTS.private.json',dict(CORR_parent_checkpoint=str(parent)))
        put(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json',{'backbone.layer.weight':dict(requires_grad=False)})
        calls=[]
        with patch.object(storage,'PARENT',parent_sha),patch.dict(os.environ,{'PBS_JOBID':'fixture-job'}):
            yield root,parent,checkpoint,lambda:calls.append('guard'),calls,complete


class SharingTests(unittest.TestCase):
    def test_exact_new_duplicate_shares_without_changing_parent_or_full_state(self):
        with fixture() as (root,parent,checkpoint,guard,calls,complete):
            original=(parent/NAME).read_bytes();mode=(parent/NAME).stat().st_mode
            optimizer=(checkpoint/'optimizer.pt').read_bytes()
            result=storage.share_after_native_save(root,checkpoint,guard)
            self.assertEqual(result['status'],'EXACT_FROZEN_SHARDS_SHARED_FULL_STATE_UNCHANGED')
            self.assertTrue(os.path.samefile(parent/NAME,checkpoint/NAME))
            self.assertEqual((parent/NAME).read_bytes(),original)
            self.assertEqual((parent/NAME).stat().st_mode,mode)
            self.assertEqual(storage.sha(checkpoint/'CHECKPOINT_COMPLETE.json'),complete)
            self.assertEqual((checkpoint/'optimizer.pt').read_bytes(),optimizer)
            self.assertGreaterEqual(len(calls),3)
            before=(parent/NAME).stat().st_nlink
            self.assertEqual(storage.share_after_native_save(root,checkpoint,guard),result)
            self.assertEqual((parent/NAME).stat().st_nlink,before)
            self.assertFalse(result['checkpoint_retirement'])

    def test_mixed_or_unknown_tensor_never_replaces_even_identical_bytes(self):
        for inventory in ({'backbone.layer.weight':dict(requires_grad=True)},{}):
            with self.subTest(inventory=inventory),fixture() as (root,parent,checkpoint,guard,_,_):
                put(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json',inventory)
                with self.assertRaisesRegex(PermissionError,'KNOWN_FROZEN_BACKBONE'):
                    storage.share_after_native_save(root,checkpoint,guard)
                self.assertFalse(os.path.samefile(parent/NAME,checkpoint/NAME))

    def test_changed_parent_or_candidate_bytes_are_preserved_and_rejected(self):
        for which in ('parent','candidate'):
            with self.subTest(which=which),fixture() as (root,parent,checkpoint,guard,_,_):
                path=(parent if which=='parent' else checkpoint)/NAME
                path.write_bytes(path.read_bytes()+b'x')
                before=path.read_bytes()
                with self.assertRaisesRegex(PermissionError,'EXACT_COMPLETE_SAME_FILESYSTEM'):
                    storage.share_after_native_save(root,checkpoint,guard)
                self.assertEqual(path.read_bytes(),before)
                self.assertFalse(os.path.samefile(parent/NAME,checkpoint/NAME))

    def test_unsettled_intent_is_not_blindly_replayed(self):
        with fixture() as (root,parent,checkpoint,guard,_,_):
            put(root/'evidence/DEC1_CHECKPOINT_SHARING/A_CONTROL-checkpoint-1024.INTENT.json',dict(pending=True))
            with self.assertRaisesRegex(PermissionError,'READONLY_RECONCILIATION'):
                storage.share_after_native_save(root,checkpoint,guard)
            self.assertFalse(os.path.samefile(parent/NAME,checkpoint/NAME))

    def test_incomplete_result_file_is_not_terminal_evidence(self):
        with fixture() as (root,parent,checkpoint,guard,_,_):
            folder=root/'evidence/DEC1_CHECKPOINT_SHARING'
            put(folder/'A_CONTROL-checkpoint-1024.INTENT.json',dict(members={NAME:'a'*64},job_id='fixture-job'))
            put(folder/'A_CONTROL-checkpoint-1024.RESULT.json',dict(status='EXACT_FROZEN_SHARDS_SHARED_FULL_STATE_UNCHANGED'))
            with self.assertRaisesRegex(PermissionError,'INCOMPLETE_OR_MISMATCHED_RECEIPT'):
                storage.share_after_native_save(root,checkpoint,guard)

    def test_foreign_existing_link_symlink_and_missing_native_job_rejected(self):
        for failure in ('existing-link','symlink','no-job','foreign-job'):
            with self.subTest(failure=failure),fixture() as (root,parent,checkpoint,guard,_,_):
                if failure=='existing-link':os.link(checkpoint/NAME,root/'another-consumer')
                elif failure=='symlink':
                    (checkpoint/NAME).unlink();(checkpoint/NAME).symlink_to(parent/NAME)
                elif failure=='no-job':os.environ.pop('PBS_JOBID')
                else:os.environ['PBS_JOBID']='other'
                with self.assertRaises(PermissionError):storage.share_after_native_save(root,checkpoint,guard)

    def test_unsealed_or_old_authority_cannot_opt_in(self):
        for failure in ('policy-mutable','policy-hash','old-authority'):
            with self.subTest(failure=failure),fixture() as (root,parent,checkpoint,guard,_,_):
                if failure=='policy-mutable':(root/storage.POLICY).chmod(0o600)
                elif failure=='policy-hash':
                    (root/storage.POLICY).chmod(0o600);(root/storage.POLICY).write_text('{}');(root/storage.POLICY).chmod(0o400)
                else:
                    m=storage._read(root/'RUN_MANIFEST.json');m['authority']['amendments']=[];put(root/'RUN_MANIFEST.json',m)
                with self.assertRaises(PermissionError):storage.share_after_native_save(root,checkpoint,guard)

    def test_legacy_wrapper_returns_original_type_without_mutation(self):
        with fixture() as (root,parent,checkpoint,guard,_,_):
            m=storage._read(root/'RUN_MANIFEST.json');m['execution']['sealed_inputs']={};put(root/'RUN_MANIFEST.json',m)
            class Base:pass
            self.assertIs(storage.wrap_trainer(Base,root,guard),Base)
            self.assertEqual(storage.share_after_native_save(root,checkpoint,guard)['status'],'NOT_CONFIGURED_OR_LEGACY_SAVE_UNCHANGED')
            self.assertFalse(os.path.samefile(parent/NAME,checkpoint/NAME))

    def test_wrapper_runs_original_complete_save_before_any_sharing(self):
        with fixture() as (root,parent,checkpoint,guard,calls,_):
            template=checkpoint.with_name('unsaved-template');checkpoint.rename(template)
            class Base:
                def _save_checkpoint(self,model,trial):
                    calls.append(('original-save',model,trial));shutil.copytree(template,checkpoint)
            wrapped=storage.wrap_trainer(Base,root,guard)()
            wrapped.args=SimpleNamespace(output_dir=checkpoint.parent)
            wrapped.state=SimpleNamespace(global_step=1024)
            wrapped.consumed=1024*32
            wrapped._save_checkpoint('model','trial')
            self.assertEqual(calls[0],('original-save','model','trial'))
            self.assertTrue(os.path.samefile(parent/NAME,checkpoint/NAME))
            before=(parent/NAME).stat()
            wrapped._save_checkpoint('model','trial')
            self.assertEqual([c for c in calls if isinstance(c,tuple)],[('original-save','model','trial')])
            after=(parent/NAME).stat()
            self.assertEqual((before.st_ino,before.st_mtime_ns,before.st_mode),
                             (after.st_ino,after.st_mtime_ns,after.st_mode))

    def test_wrapper_preserves_partial_and_corrupt_existing_state_without_writer(self):
        for failure in ('partial','corrupt','cursor'):
            with self.subTest(failure=failure),fixture() as (root,parent,checkpoint,guard,calls,_):
                class Base:
                    def _save_checkpoint(self,*args):raise AssertionError('WRITER_MUST_NOT_RUN')
                wrapped=storage.wrap_trainer(Base,root,guard)()
                wrapped.args=SimpleNamespace(output_dir=checkpoint.parent)
                wrapped.state=SimpleNamespace(global_step=1024);wrapped.consumed=1024*32
                if failure=='partial':
                    value=storage._read(checkpoint/'CHECKPOINT_COMPLETE.json');value['full_resume']=False
                    put(checkpoint/'CHECKPOINT_COMPLETE.json',value)
                elif failure=='corrupt':(checkpoint/'optimizer.pt').write_bytes(b'corrupt')
                else:wrapped.consumed-=1
                with self.assertRaises(PermissionError):wrapped._save_checkpoint(None,None)

    def test_wrapper_rejects_output_escape_before_original_writer(self):
        with fixture() as (root,parent,checkpoint,guard,_,_):
            class Base:
                def _save_checkpoint(self,*args):raise AssertionError('WRITER_MUST_NOT_RUN')
            wrapped=storage.wrap_trainer(Base,root,guard)()
            wrapped.args=SimpleNamespace(output_dir=parent)
            wrapped.state=SimpleNamespace(global_step=1024);wrapped.consumed=1024*32
            with self.assertRaisesRegex(PermissionError,'BEFORE_NATIVE_WRITER'):
                wrapped._save_checkpoint(None,None)


if __name__=='__main__':unittest.main()
