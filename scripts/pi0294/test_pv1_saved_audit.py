import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.fd1 import sha
from scripts.pi0294.pv1.saved_audit import raw_view


class SavedAuditTests(unittest.TestCase):
    def test_original_failure_is_not_aliased_or_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';source.mkdir()
            raw=source/'samples.json';raw.write_text('[]\n')
            (source/'WORKER_RECEIPT.json').write_text('{}')
            (source/'AUDIT.json').write_text('{"status":"AUDIT_FAIL_PRESERVED"}')
            (source/'FILE_MANIFEST.json').write_text(json.dumps([dict(path='samples.json',bytes=raw.stat().st_size,sha256=sha(raw))]))
            before=sha(source/'AUDIT.json')
            view=root/'new-view'
            receipt=raw_view(source,view,sha(source/'FILE_MANIFEST.json'))
            self.assertTrue(receipt['original_derived_audit_not_linked'])
            self.assertFalse((view/'AUDIT.json').exists())
            (view/'AUDIT.json').write_text('{"status":"PASS_NEW_READ_SIDE_AUDIT"}')
            self.assertEqual(sha(source/'AUDIT.json'),before)
            self.assertEqual(sha(view/'samples.json'),sha(source/'samples.json'))
            with self.assertRaises(FileExistsError):raw_view(source,view,sha(source/'FILE_MANIFEST.json'))

    def test_changed_original_member_is_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';source.mkdir()
            raw=source/'samples.json';raw.write_text('[]')
            (source/'FILE_MANIFEST.json').write_text(json.dumps([dict(path='samples.json',bytes=2,sha256='0'*64)]))
            with self.assertRaises(ValueError):raw_view(source,root/'new',sha(source/'FILE_MANIFEST.json'))
