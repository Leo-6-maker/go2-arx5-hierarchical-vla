"""Projection fixtures only; no real receipt or dataset qualification."""
import json
from pathlib import Path
import tempfile
import unittest

from project_public_receipts import project,publish


class ProjectionTests(unittest.TestCase):
    def test_uint64_identity_and_path_redaction(self):
        value = dict(seed=2**64-1,raw_root='/var/cache/private/episode',sha256='a'*64,admitted=False)
        result = project(value)
        self.assertEqual(int(result['seed']),value['seed'])
        self.assertEqual(result['sha256'],value['sha256'])
        self.assertNotIn('/var/cache/',result['raw_root'])
        self.assertFalse(result['admitted'])

    def test_jsonl_denominator_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source,output = root/'private',root/'public'
            source.mkdir()
            (source/'ledger.jsonl').write_text(''.join(json.dumps(dict(seed=2**63+i,id=i))+'\n' for i in range(80)))
            publish(source,output,['ledger.jsonl=ATTEMPT_LEDGER.jsonl'])
            rows = [json.loads(line) for line in (output/'ATTEMPT_LEDGER.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),80)
            self.assertEqual(int(rows[-1]['seed']),2**63+79)
            with self.assertRaises(FileExistsError):
                publish(source,output,['ledger.jsonl=ATTEMPT_LEDGER.jsonl'])


if __name__ == '__main__':
    unittest.main()
