import json
from pathlib import Path
import tempfile
import unittest
from .mp1.directory_visibility_audit import observe,force_server_attributes


class DirectoryVisibilityTests(unittest.TestCase):
    def test_real_statx_metadata_read_does_not_change_packet(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'packet';f.write_bytes(b'fixture');before=(f.read_bytes(),f.stat().st_mtime_ns)
            force_server_attributes(p)
            self.assertEqual((f.read_bytes(),f.stat().st_mtime_ns),before)
            with self.assertRaises(FileNotFoundError):force_server_attributes(p/'missing')

    def test_pending_predeclared_directory_is_not_created(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'future';now=[0.]
            def sleep(dt):now[0]+=dt
            result=observe([p],duration_s=.4,clock=lambda:now[0],sleep=sleep,allow_pending=True,force_sync=True)
            self.assertFalse(p.exists());self.assertEqual(result['count_change_samples'][0]['counts'],[None])

    def test_observer_preserves_packet_bytes_and_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);packet=p/'packet.json';packet.write_text(json.dumps(dict(action='fixture')))
            before=(packet.read_bytes(),packet.stat().st_mtime_ns);now=[0.]
            def sleep(dt):now[0]+=dt
            result=observe([p],duration_s=.6,clock=lambda:now[0],sleep=sleep)
            self.assertEqual((packet.read_bytes(),packet.stat().st_mtime_ns),before)
            self.assertEqual(result['scans'],3);self.assertEqual(result['packets_modified'],0)
            self.assertTrue(result['potential_NFS_lookup_cache_observer_effect'])
            self.assertFalse(result['unassisted_transport_qualification'])

    def test_scope_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'link').symlink_to(p,target_is_directory=True)
            with self.assertRaises(PermissionError):observe([p/'link'],duration_s=1)
            with self.assertRaises(PermissionError):observe([p],duration_s=1801)
            with self.assertRaises(PermissionError):observe([p],duration_s=1,interval_s=.1)
