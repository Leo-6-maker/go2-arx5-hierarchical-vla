import tempfile
import unittest
from pathlib import Path

from scripts.pi0294.mp1.session_visibility import observe


class SessionVisibilityTest(unittest.TestCase):
    def test_existing_two_directories_read_only_until_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / 'inbox', root / 'outbox']
            for p in paths:
                p.mkdir()
                (p / 'fixed.json').write_bytes(b'untouched')
            now = [100.]
            calls = []
            def sleep(seconds):
                self.assertEqual(seconds, .2)
                now[0] += seconds
            value = observe(paths, lambda: len(calls) == 4, 110.,
                            clock=lambda: now[0], sleep=sleep,
                            synchronize=lambda p: calls.append(p))
            self.assertEqual(value['status'], 'TERMINAL_MARKER_OBSERVED')
            self.assertEqual(value['scans'], 2)
            self.assertEqual(calls, paths * 2)
            for p in paths:
                self.assertEqual(list(p.iterdir()), [p / 'fixed.json'])
                self.assertEqual((p / 'fixed.json').read_bytes(), b'untouched')

    def test_deadline_stops_without_requesting_model_or_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / n for n in ['inbox', 'outbox']]
            for p in paths:
                p.mkdir()
            now = [10.]
            def sleep(seconds):
                now[0] += seconds
            value = observe(paths, lambda: False, 10.5,
                            clock=lambda: now[0], sleep=sleep,
                            synchronize=lambda p: None)
            self.assertEqual(value['status'], 'ORIGINAL_SESSION_DEADLINE_OBSERVED')
            self.assertEqual(value['scans'], 3)

    def test_no_new_directory_and_no_symlink_or_unbounded_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'inbox', root / 'outbox'
            a.mkdir()
            with self.assertRaises(PermissionError):
                observe([a, b], lambda: False, 11, clock=lambda: 10)
            self.assertFalse(b.exists())
            b.symlink_to(a, target_is_directory=True)
            with self.assertRaises(PermissionError):
                observe([a, b], lambda: True, 11, clock=lambda: 10)
            b.unlink()
            b.mkdir()
            for end in [10, 7211]:
                with self.subTest(end=end), self.assertRaises(PermissionError):
                    observe([a, b], lambda: True, end, clock=lambda: 10)


if __name__ == '__main__':
    unittest.main()
