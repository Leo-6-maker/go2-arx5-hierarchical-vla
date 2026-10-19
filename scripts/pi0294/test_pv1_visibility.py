import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import visibility


class VisibilityContracts(unittest.TestCase):
    def test_mature_server_sync_then_enumeration_then_exact_direct_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new.json';events=[]
            def synchronize(directory):
                events.append(('sync',directory));p.write_text('{"same_packet":true}')
            direct=visibility.read_available
            def read(path):events.append(('read',path));return direct(path)
            with patch.object(visibility,'force_server_attributes',side_effect=synchronize):
                with patch.object(visibility,'read_available',side_effect=read):
                    self.assertEqual(visibility.read_visible(p),dict(same_packet=True))
            self.assertEqual(events,[('sync',p.parent),('read',p)])

    def test_missing_packet_stays_missing_without_mutation_or_sleep(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'absent.json'
            with patch.object(visibility,'force_server_attributes') as sync:
                self.assertIsNone(visibility.read_visible(p));sync.assert_called_once_with(p.parent)
            self.assertEqual(list(p.parent.iterdir()),[])

    def test_sync_failure_is_not_suppressed_or_called_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(visibility,'force_server_attributes',side_effect=OSError('SERVER_READ_FAILED')):
                with self.assertRaisesRegex(OSError,'SERVER_READ_FAILED'):visibility.refresh(tmp)

    def test_real_syscall_reads_metadata_only_and_never_follows_directory_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'packet.json';p.write_text(json.dumps(dict(exact='bytes')))
            before=p.read_bytes();self.assertEqual(visibility.read_visible(p),dict(exact='bytes'))
            self.assertEqual(p.read_bytes(),before)
            alias=Path(tmp)/'alias';alias.symlink_to(tmp,target_is_directory=True)
            with self.assertRaises(PermissionError):visibility.refresh(alias)

    def test_directory_scan_bound_covers_exact_current_dev_panel_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(visibility,'MAX_ENTRIES',2),patch.object(visibility,'force_server_attributes'):
                for i in range(3):(Path(tmp)/str(i)).touch()
                with self.assertRaisesRegex(PermissionError,'24_CONDITION_PACKET_DIRECTORY_BOUND'):visibility.refresh(tmp)
