"""Generated fixture state retirement only, never actual model deletion."""
from pathlib import Path
import tempfile
import unittest

from scripts.pi0292.durable import read,write
from .native_dataset import sha
from .checkpoint_session import retirement
from .neural import retire_redundant_smoke


class CheckpointBoundaryTests(unittest.TestCase):
    def test_only_exact_M512_after_full_HF_and_later_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence').mkdir()
            earlier=root/'training/main/checkpoint-512';later=root/'training/main/checkpoint-2048'
            earlier.mkdir(parents=True);later.mkdir()
            (earlier/'state').write_bytes(b'synthetic artifact, not a model')
            (later/'state').write_bytes(b'synthetic later full state')
            write(later/'CHECKPOINT_COMPLETE.json',dict(global_step=2048,
                status='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING',required_files={'state':sha(later/'state')}))
            receipt=dict(status='PASS_COMPLETE_PRIVATE_CHECKPOINT_FIXED_REVISION_READBACK',global_step=512,
                revision='a'*40,prefix='private-fixture',files={'state':dict(bytes=(earlier/'state').stat().st_size,sha256=sha(earlier/'state'))})
            with self.assertRaises(PermissionError):retirement(root,earlier,dict(receipt,status='FAILED'),later,expected_step=512)
            self.assertTrue(earlier.exists())
            with self.assertRaises(PermissionError):retirement(root,earlier,receipt,earlier,expected_step=512)
            result=retirement(root,earlier,receipt,later,expected_step=512)
            self.assertFalse(earlier.exists());self.assertTrue((later/'state').exists())
            self.assertFalse(result['failed_checkpoint_deleted']);self.assertFalse(result['historical_data_deleted'])

    def test_missing_full_state_or_unexpected_members_prevents_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence').mkdir()
            earlier=root/'training/main/checkpoint-512';later=root/'training/main/checkpoint-2048'
            earlier.mkdir(parents=True);later.mkdir()
            (earlier/'unexpected').touch();(later/'state').write_bytes(b'synthetic')
            write(later/'CHECKPOINT_COMPLETE.json',dict(global_step=2048,
                status='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING',required_files={'state':sha(later/'state')}))
            receipt=dict(status='PASS_COMPLETE_PRIVATE_CHECKPOINT_FIXED_REVISION_READBACK',global_step=512,files={})
            with self.assertRaises(ValueError):retirement(root,earlier,receipt,later,expected_step=512)
            self.assertTrue(earlier.exists())

    def test_minimal_qualified_smoke_keeps_final_restored_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence').mkdir();smoke=root/'smoke_attempts/new'
            proof=smoke/'evidence/RESUME_EQUIVALENCE.json';proof.parent.mkdir(parents=True)
            write(proof,dict(status='PASS_FIXED_RESUME_EQUIVALENCE'))
            for relative in ('smoke_continuous/checkpoint-4','smoke_split/checkpoint-2','smoke_split/checkpoint-4'):
                folder=smoke/'training'/relative;folder.mkdir(parents=True)
                (folder/'state').write_bytes(b'fixture')
                write(folder/'CHECKPOINT_COMPLETE.json',dict(global_step=int(relative[-1]),required_files={'state':sha(folder/'state')}))
            retire_redundant_smoke(root,smoke,proof,minimal=True)
            self.assertTrue((smoke/'training/smoke_split/checkpoint-4/state').exists())
            self.assertFalse((smoke/'training/smoke_continuous/checkpoint-4').exists())
            self.assertEqual(len(read(root/'evidence/REDUNDANT_SMOKE_RETIREMENT_ACTUAL.json')['rows']),2)


if __name__=='__main__':unittest.main()
