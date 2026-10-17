import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .relay_io import PhaseTrace,read_available,MAX_FILE_BYTES


class RelayIOTests(unittest.TestCase):
    def test_direct_read_never_uses_negative_stat_cache(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'packet'
            self.assertIsNone(read_available(p));p.write_text('{"ok":true}')
            with patch.object(Path,'is_file',side_effect=AssertionError('NO_PRE_OPEN_STAT')):
                self.assertEqual(read_available(p),{'ok':True})

    def test_malformed_large_and_symlink_are_not_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'packet';p.write_text('partial')
            with self.assertRaises(ValueError):read_available(p)
            p.write_bytes(b'x'*(MAX_FILE_BYTES+1))
            with self.assertRaises(ValueError):read_available(p)
            alias=Path(d)/'alias';alias.symlink_to(p)
            with self.assertRaises(OSError):read_available(alias)
            with self.assertRaises(ValueError):read_available(Path(d))

    def test_trace_exposes_order_and_stall_phase_not_payload_or_error_text(self):
        out=io.StringIO();clock=iter([1.,1.25,2.,23.])
        trace=PhaseTrace(out,clock=lambda:next(clock))
        with trace.phase('guard'):pass
        with self.assertRaises(RuntimeError):
            with trace.phase('accounting'):raise RuntimeError('/home/private/token')
        rows=[json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual([r['event'] for r in rows],['FILE_RPC_PHASE_BEGIN','FILE_RPC_PHASE_END']*2)
        self.assertEqual(rows[1]['elapsed_s'],.25);self.assertEqual(rows[-1]['elapsed_s'],21.)
        self.assertEqual(rows[-1]['status'],'FAILED_OR_UNKNOWN')
        self.assertNotIn('private',out.getvalue())
        with self.assertRaises(ValueError):
            with trace.phase('action'):pass
