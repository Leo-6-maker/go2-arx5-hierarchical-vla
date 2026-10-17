import json
import tempfile
import unittest
from pathlib import Path

from scripts.pi0294.mp1.video_delivery import audited_movies, CHILDREN
from scripts.pi0294.native_dataset import sha


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    for child_id, count in zip(CHILDREN, [2, 2, 6]):
        child = root / 'children' / child_id
        lease = child / 'lease.json'
        put(lease, dict(status='CLOSED'))
        put(child / 'OUTPUT_BINDING.private.json', dict(lease=str(lease)))
        put(child / 'RESOURCE_LEDGER.json', dict(reservations={}, CPU_reserved_s=0))
        for i in range(count):
            name = child_id + '-attempt-' + str(i)
            decision = dict(status='ENGINEERING_INCOMPLETE' if i == 0 else 'SUCCESS',
                            task='T1', scene_group='TRAIN/F0/000')
            put(child / 'episodes' / name / 'MP1_ADJUDICATION.json', decision)
            folder = child / 'evidence/audits' / name
            folder.mkdir(parents=True)
            movie = folder / 'actual-body-wrist.mp4'
            movie.write_bytes(b'unit-test-fixture-not-a-real-qualified-video')
            put(folder / 'AUDIT.private.json', dict(attempt_id=name,
                status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT', adjudication=decision,
                video=dict(sha256=sha(movie), bytes=movie.stat().st_size,
                           decoded_frames=1, actual_paired_frames=1,
                           padding_or_interpolation=False, native_duration_s=0 if i == 0 else 32)))


class RecoveryVideoTest(unittest.TestCase):
    def test_exact_ten_read_only_hash_checked_movies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            before = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            rows, files, sources = audited_movies(root)
            self.assertEqual(len(rows), 10)
            self.assertEqual(len(files), 10)
            self.assertEqual(len(sources), 10)
            self.assertEqual(sum(not r['task_trajectory'] for r in rows), 3)
            self.assertFalse(any('/tmp/' in json.dumps(r) for r in rows))
            self.assertEqual(before, {str(p): sha(p) for p in root.rglob('*') if p.is_file()})

    def test_active_lease_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            child = root / 'children' / CHILDREN[0]
            put(child / 'lease.json', dict(status='ACTIVE'))
            with self.assertRaises(PermissionError):
                audited_movies(root)
            put(child / 'lease.json', dict(status='CLOSED'))
            movie = next(child.glob('evidence/audits/*/actual-body-wrist.mp4'))
            movie.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                audited_movies(root)

    def test_symlink_and_unqualified_audit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            child = root / 'children' / CHILDREN[0]
            audit = next(child.glob('evidence/audits/*/AUDIT.private.json'))
            value = json.loads(audit.read_text())
            value['status'] = 'ENGINEERING_AUDIT_INCOMPLETE'
            put(audit, value)
            with self.assertRaises(PermissionError):
                audited_movies(root)
            value['status'] = 'PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
            put(audit, value)
            movie = audit.parent / 'actual-body-wrist.mp4'
            movie.unlink()
            external = root / 'outside.mp4'
            external.write_bytes(b'other')
            movie.symlink_to(external)
            with self.assertRaises(PermissionError):
                audited_movies(root)


if __name__ == '__main__':
    unittest.main()
