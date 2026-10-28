from pathlib import Path
import errno
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read,write
from .mp1 import transport_fixture_faults as faults
from .mp1.owner_heartbeat import HeartbeatLost
from .native_dataset import sha


class TransportFaultPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.folder=self.root/'isolated';self.folder.mkdir()
        write(self.root/'OWNER_HEARTBEAT.json',dict(run_uuid='production',observed_epoch=time.time()))
        self.original=sha(self.root/'OWNER_HEARTBEAT.json')

    def tearDown(self):self.assertEqual(sha(self.root/'OWNER_HEARTBEAT.json'),self.original)

    def test_real_concurrent_atomic_replaces_are_read_after_the_clock_without_writing_production(self):
        value=faults.concurrent_replace(self.folder,'fixture-run')
        self.assertEqual(value['status'],'PASS');self.assertGreater(value['actual_replaces'],1)
        self.assertEqual(value['post_read_samples'],40)

    def test_isolated_concurrent_reader_outlasts_default_six_read_visibility_window(self):
        path=self.folder/'heartbeat.json';write(path,dict(run_uuid='fixture-run',observed_epoch=time.time()))
        attempts=[]
        def delayed_visibility(candidate):
            attempts.append(candidate)
            if len(attempts)<=6:raise FileNotFoundError(errno.ENOENT,'fixture rename not visible',candidate)
            return read(candidate)
        value=faults._read_concurrent_heartbeat(path,reader=delayed_visibility,sleep=lambda _:None)
        self.assertEqual(value['run_uuid'],'fixture-run');self.assertEqual(len(attempts),7)

    def test_isolated_concurrent_reader_does_not_retry_unclassified_errors(self):
        calls=[]
        def denied(candidate):
            calls.append(candidate);raise PermissionError(errno.EACCES,'denied',candidate)
        with self.assertRaises(PermissionError):
            faults._read_concurrent_heartbeat(self.folder/'heartbeat.json',reader=denied,sleep=lambda _:None)
        self.assertEqual(len(calls),1)

    def test_classified_estale_retry_is_bounded_and_only_in_fixture(self):
        value=faults.metadata_failure(self.folder)
        self.assertEqual(value['attempts'],2);self.assertEqual(value['injected_errno'],116)
        self.assertFalse(value['production_read_fault_injected']);self.assertLess(value['wall_s'],120)

    def test_real_held_lock_does_not_block_production_reply_primitive(self):
        value=faults.statistics_lock(self.folder)
        self.assertTrue(value['reply_while_lock_held']);self.assertEqual(value['actual_lock_holder_exit'],0)
        self.assertEqual(value['settlement_after_unlock']['RPC_calls'],1)
        self.assertEqual(read(self.folder/'statistics_lock/COUNTERS.json')['RPC_calls'],1)

    def test_exited_real_fixture_writer_does_not_renew_and_cannot_be_called_expired_early(self):
        path=faults.start_exit_heartbeat(self.folder,'fixture-run')
        self.assertEqual(read(path/'WRITER_TERMINAL.json')['returncode'],0)
        with self.assertRaisesRegex(PermissionError,'TRULY_AGE'):
            faults.finish(self.folder,'fixture-run')
        self.assertEqual(sha(path/'OWNER_HEARTBEAT.json'),read(path/'WRITER_TERMINAL.json')['heartbeat_sha256'])

    def test_actual_bad_fixture_inputs_keep_original_strict_watchdog(self):
        for name,run,epoch,expected in [('future','fixture-run',time.time()+30,HeartbeatLost),
                                      ('expired','fixture-run',time.time()-301,HeartbeatLost),
                                      ('wrong','other',time.time(),PermissionError)]:
            path=self.folder/name;path.mkdir();write(path/'OWNER_HEARTBEAT.json',dict(run_uuid=run,observed_epoch=epoch))
            value=faults.heartbeat_fault(path,'fixture-run',expected)
            self.assertTrue(value['failed_closed']);self.assertEqual(value['unknown_effects_resent'],0)


if __name__=='__main__':unittest.main()
