from pathlib import Path
import tempfile
import unittest

from .mp1.relay_qualification import filesystem_probe


class FilesystemQualificationTests(unittest.TestCase):
    def test_real_fixture_durability_stays_synthetic_and_bounded(self):
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture';v=filesystem_probe(p,lambda:calls.append(1),cycles=3)
            self.assertEqual(v['cycles'],3);self.assertEqual(len(calls),3)
            self.assertEqual(v['new_model_queries'],v['new_physical_integrations'],0)
            self.assertFalse(v['full_32s_closed_loop_qualification'])
            self.assertFalse(v['cross_compute_node_NFS_visibility_tested'])
            self.assertEqual(len(v['summary']),6)
            self.assertTrue(all(x['samples']==3 for x in v['summary'].values()))
            with self.assertRaises(FileExistsError):filesystem_probe(p,lambda:None,cycles=1)

    def test_bound_and_guard_failure_are_not_silent_success(self):
        with tempfile.TemporaryDirectory() as d:
            for cycles in [0,97,True]:
                with self.assertRaises(ValueError):filesystem_probe(Path(d)/'bad',lambda:None,cycles=cycles)
            def stopped():raise PermissionError('STOP')
            with self.assertRaises(PermissionError):filesystem_probe(Path(d)/'stop',stopped,cycles=1)
