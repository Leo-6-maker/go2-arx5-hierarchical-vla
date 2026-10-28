import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .mp1.file_rpc_journal import Attempt,append
from .mp1 import file_rpc_journal as journal
from .relay_io import PhaseTrace
from .e1.stdio import FrameReader


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

    def test_prewrite_persistence_latency_is_visible_and_written_progress_does_not_fsync(self):
        clock=[0.]
        def sync(fd):clock[0]+=21.
        with patch.object(journal.os,'fsync',side_effect=sync),patch.object(
                journal.time,'monotonic',side_effect=lambda:clock[0]):
            with self.assertRaises(TimeoutError):
                with self.attempt() as attempt:
                    attempt.written()
                    self.assertEqual(clock[0],21.)
                    self.assertEqual(len(self.rows()),1)
                    raise TimeoutError('ORIGINAL_20S_FILE_RPC_TIMEOUT_UNKNOWN_EFFECT_NO_RETRY')
        rows=self.rows();timing=rows[-1]['previous_receipt_persistence']
        self.assertEqual(timing['event'],'REQUEST_INTENT_BEFORE_WIRE')
        self.assertEqual(timing['fsync_elapsed_s'],21.)
        self.assertEqual(timing['fsync_ended_monotonic']-timing['fsync_started_monotonic'],21.)
        self.assertEqual(rows[-1]['call_watchdog_s'],20)
        self.assertTrue(rows[-1]['unknown_remote_effect']);self.assertTrue(self.session.failed)
        self.assertEqual(len(rows),3)

    def test_written_is_persisted_with_response_before_success_can_return(self):
        with patch.object(journal.os,'fsync',wraps=journal.os.fsync) as sync:
            with self.attempt() as attempt:
                self.assertEqual(sync.call_count,1)
                attempt.written();self.assertEqual(sync.call_count,1)
                with self.assertRaises(PermissionError):attempt.written()
                attempt.response(b'{"ok":true}',dict(ok=True));self.assertEqual(sync.call_count,2)
            rows=self.rows()
            self.assertEqual([r['event'] for r in rows],['REQUEST_INTENT_BEFORE_WIRE',
                'LOCAL_WRITE_RETURNED_NOT_REMOTE_ACK','RESPONSE_PARSED'])
            self.assertEqual(attempt.persistence['batch_events'],['LOCAL_WRITE_RETURNED_NOT_REMOTE_ACK','RESPONSE_PARSED'])

    def test_crash_before_reply_retains_durable_unknown_intent_and_no_success_claim(self):
        attempt=self.attempt();attempt.written()
        self.assertEqual([r['event'] for r in self.rows()],['REQUEST_INTENT_BEFORE_WIRE'])
        self.assertTrue(self.session.failed)
        self.assertFalse(attempt.received)
        self.assertEqual(attempt.pending_written['event'],'LOCAL_WRITE_RETURNED_NOT_REMOTE_ACK')

    def test_parsed_response_then_slow_persistence_keeps_effect_known_but_session_poisoned(self):
        with self.assertRaises(TimeoutError):
            with self.attempt() as attempt:
                attempt.written();attempt.response(b'{"ok":true}',dict(ok=True))
                raise TimeoutError('ORIGINAL_20S_FILE_RPC_TIMEOUT_UNKNOWN_EFFECT_NO_RETRY')
        final=self.rows()[-1]
        self.assertEqual(final['previous_receipt_persistence']['event'],'RESPONSE_PARSED')
        self.assertFalse(final['unknown_remote_effect']);self.assertTrue(self.session.failed)

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

    def test_unbuffered_frame_reader_consumes_prefetched_second_frame_without_select_sleep(self):
        source,sink=os.pipe();self.addCleanup(os.close,source);self.addCleanup(os.close,sink)
        os.write(sink,b'{"rpc_id":"one"}\n{"rpc_id":"two"}\n')
        reader=FrameReader(source,1024)
        self.assertEqual(reader.read(1),b'{"rpc_id":"one"}\n')
        with patch('scripts.pi0294.e1.stdio.select.select',side_effect=AssertionError('buffered frame must not poll fd')):
            self.assertEqual(reader.read(1),b'{"rpc_id":"two"}\n')

    def test_unbuffered_frame_reader_preserves_timeout_eof_and_size_contract(self):
        source,sink=os.pipe();self.addCleanup(os.close,source)
        reader=FrameReader(source,8)
        self.assertIsNone(reader.read(0))
        os.write(sink,b'123456789\n');os.close(sink)
        with self.assertRaisesRegex(ValueError,'BOUNDED_ONE_JSON_LINE_REQUIRED'):reader.read(1)

    def test_unbuffered_frame_reader_joins_partial_reads_until_deadline(self):
        reader=FrameReader(17,1024)
        with patch('scripts.pi0294.e1.stdio.select.select',side_effect=[([17],[],[]),([17],[],[])]), \
             patch('scripts.pi0294.e1.stdio.os.read',side_effect=[b'{"rpc_id":',b'"split"}\n']):
            self.assertEqual(reader.read(1),b'{"rpc_id":"split"}\n')

    def test_unbuffered_frame_reader_exposes_incomplete_frame_at_eof(self):
        reader=FrameReader(17,1024)
        with patch('scripts.pi0294.e1.stdio.select.select',side_effect=[([17],[],[]),([17],[],[])]), \
             patch('scripts.pi0294.e1.stdio.os.read',side_effect=[b'{"rpc_id":',b'']):
            self.assertEqual(reader.read(1),b'{"rpc_id":')


if __name__=='__main__':unittest.main()
