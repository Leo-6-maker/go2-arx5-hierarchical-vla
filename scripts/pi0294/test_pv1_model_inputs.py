"""No weights or neural imports; copied aliases must fail before PBS work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1.model_inputs import preflight,prepare_alias
from scripts.pi0294.pv1 import pbs_owner


class ModelInputPreflight(unittest.TestCase):
    def test_cold_binder_links_existing_snapshot_without_copying_or_repairing_old_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'fresh';root.mkdir();snapshot=Path(d)/'pinned';snapshot.mkdir()
            (snapshot/'config.json').write_text('{}');(snapshot/'tensor-fixture.bin').write_bytes(b'not weights')
            report=prepare_alias(root,dict(cosmos=str(snapshot)))
            self.assertEqual(report['new_download_bytes'],0)
            alias=root/'model_inputs/nvidia/Cosmos-Reason2-2B'
            self.assertTrue(alias.is_symlink());self.assertEqual(alias.resolve(),snapshot)
            with self.assertRaises(FileExistsError):prepare_alias(root,dict(cosmos=str(snapshot)))
            other=Path(d)/'active';other.mkdir();(other/'RUN_MANIFEST.json').write_text('{}')
            with self.assertRaises(PermissionError):prepare_alias(other,dict(cosmos=str(snapshot)))
            self.assertFalse((other/'model_inputs').exists())

    def test_missing_copied_or_wrong_snapshot_alias_is_not_equivalent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';snapshot=Path(d)/'pinned';snapshot.mkdir()
            (snapshot/'config.json').write_text('{}');bindings=dict(cosmos=str(snapshot))
            alias=root/'model_inputs/nvidia/Cosmos-Reason2-2B';alias.parent.mkdir(parents=True)
            with self.assertRaises(PermissionError):preflight(root,bindings)
            alias.mkdir();(alias/'config.json').write_text('{}')
            with self.assertRaises(PermissionError):preflight(root,bindings)
            (alias/'config.json').unlink();alias.rmdir();alias.symlink_to(snapshot)
            value=preflight(root,bindings);self.assertEqual(value['new_model_loads'],0)
            self.assertEqual(value['exact_snapshot_resolved'],str(snapshot))
            wrong=Path(d)/'wrong';wrong.mkdir();(wrong/'config.json').write_text('{}')
            with self.assertRaises(PermissionError):preflight(root,dict(cosmos=str(wrong)))

    def test_missing_alias_rejects_prepare_and_submit_before_native_or_ledger_effects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snapshot=root/'pinned';snapshot.mkdir();(snapshot/'config.json').write_text('{}')
            (root/'model_inputs/nvidia').mkdir(parents=True)
            with patch.object(pbs_owner,'validate',return_value=({},dict(cosmos=str(snapshot)),{},None,None)),\
                    patch.object(pbs_owner,'bank_preflight',return_value={'CPU_fixture':True}),\
                    patch.object(pbs_owner,'project') as project,patch.object(pbs_owner.subprocess,'run') as native:
                for operation in (lambda:pbs_owner.prepare(root,stage='B0_DEV',project_name='CFP04-CF-007'),
                        lambda:pbs_owner.submit(root,1)):
                    with self.assertRaisesRegex(PermissionError,'EXACT_PINNED_LOCAL_COSMOS_ALIAS'):operation()
                native.assert_not_called();project.assert_not_called()
                self.assertFalse((root/'pbs').exists());self.assertFalse((root/'PV1_QSUB_OWNER.lock').exists())


if __name__=='__main__':unittest.main()
