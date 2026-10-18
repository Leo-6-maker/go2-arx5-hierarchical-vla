import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .mp1.file_rpc_journal import Attempt,append
from .relay_io import PhaseTrace


class FileRPCJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.session=SimpleNamespace(base=Path(self.temp.name),role='hopper',label='TEST-CPU',failed=True,
            process=SimpleNamespace(poll=lambda:None))

    def attempt(self):return Attempt(self.session,dict(operation='query'),b'{"sealed":"fixture"}\n','once',20.)

    def rows(self):
        return [json.loads(x) for x in (self.session.base/'file-rpc-TEST-CPU-hopper.attempts.private.jsonl').read_text().splitlines()]

    def test_prewrite_receipt_precedes_wire_and_unknown_failure_always_poisons(self):
        with self.assertRaises(TimeoutError):
            with self.attempt() as attempt:
                self.assertEqual(self.rows()[0]['event'],'REQUEST_INTENT_BEFORE_WIRE')
                attempt.written();self.session.failed=False
                raise TimeoutError('ORIGINAL_20S_FILE_RPC_TIMEOUT_UNKNOWN_EFFECT_NO_RETRY')
        rows=self.rows();self.assertEqual(len(rows),3);self.assertTrue(self.session.failed)
        self.assertEqual(rows[-1]['last_completed_phase'],'LOCAL_WRITE_RETURNED_NOT_REMOTE_ACK')
        self.assertTrue(rows[-1]['unknown_remote_effect']);self.assertFalse(rows[-1]['effect_retried'])
        self.assertNotIn('sealed',json.dumps(rows));self.assertEqual(rows[0]['call_watchdog_s'],20)

    def test_known_parsed_response_is_distinct_from_an_unknown_action_effect(self):
        with self.assertRaises(IOError):
            with self.attempt() as attempt:
                attempt.written();attempt.response(b'{"ok":true}',dict(ok=True));self.session.failed=False
                raise IOError('later local telemetry failure')
        self.assertTrue(self.session.failed);self.assertFalse(self.rows()[-1]['unknown_remote_effect'])
        self.assertTrue(self.rows()[-1]['response_received'])

    def test_journal_rejects_symlink_and_public_permissions_before_any_wire(self):
        target=self.session.base/'target';target.write_text('untouched')
        path=self.session.base/'linked';path.symlink_to(target)
        with self.assertRaises(OSError):append(path,dict(event='fixture'))
        self.assertEqual(target.read_text(),'untouched')
        path=self.session.base/'public';path.write_text('');path.chmod(0o644)
        with self.assertRaises(PermissionError):append(path,dict(event='fixture'))
        self.assertEqual(path.read_text(),'')

    def test_native_phase_identity_never_logs_packet_content_and_clears_on_bad_envelope(self):
        stream=io.StringIO();trace=PhaseTrace(stream)
        nonce='00000000-0000-4000-8000-000000000001'
        trace.bind_rpc(dict(rpc_id=nonce,args=dict(operation='response',payload='private-not-logged')))
        with trace.phase('invoke'):pass
        rows=[json.loads(x) for x in stream.getvalue().splitlines()]
        self.assertTrue(all(x['rpc_id']==nonce and x['operation']=='response' for x in rows))
        self.assertNotIn('private-not-logged',stream.getvalue())
        trace.bind_rpc(dict(rpc_id='not-an-opaque-request-id',args=dict(operation='query')))
        self.assertNotIn('rpc_id',trace.context);self.assertEqual(len(trace.context['rpc_id_sha256']),64)
        trace.bind_rpc(dict(other='invalid'));self.assertEqual(trace.context,{})


if __name__=='__main__':unittest.main()
