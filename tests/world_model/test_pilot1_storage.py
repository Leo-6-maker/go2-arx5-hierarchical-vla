import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from go2arx5_vla.world_model.pilot1_storage import local_path,qualified_block_path
from go2arx5_vla.world_model.storage import digest


class PilotLocalDisk(unittest.TestCase):
    def test_new_disk_proof_does_not_override_root_disk_or_fabricated_probe(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)/'probe';root.mkdir();p=root/'RESULT';p.write_text(json.dumps(dict(root=str(root),status='PASSED_NODE_LOCAL_HDD_IO',same_root_device=False,independent_lock_conflict_exit=73,atomic_rename_and_directory_fsync=True,roundtrip_sha256='x',expected_sha256='x',device=root.stat().st_dev)))
            # This directory is on the ordinary root filesystem. A claimed PASS cannot turn it into another disk.
            if root.stat().st_dev==Path('/').stat().st_dev:
                with self.assertRaises(PermissionError):qualified_block_path(root,p,digest(p))
            with self.assertRaises(PermissionError):qualified_block_path(root,p,'wrong-hash')
            v=json.loads(p.read_text());v['atomic_rename_and_directory_fsync']=False;p.write_text(json.dumps(v))
            with self.assertRaises(PermissionError):qualified_block_path(root,p,digest(p))

    def test_legacy_node_local_success_keeps_existing_validation(self):
        with patch('scripts.pi0294.e1.smoke.node_local',return_value=Path('/verified-local'))as validate:
            self.assertEqual(local_path({},'/input'),Path('/verified-local'));validate.assert_called_once_with('/input')

    def test_legacy_rejection_needs_the_explicit_measured_proof(self):
        with patch('scripts.pi0294.e1.smoke.node_local',side_effect=PermissionError('network')):
            with self.assertRaises(KeyError):local_path({},'/network')


if __name__=='__main__':unittest.main()
