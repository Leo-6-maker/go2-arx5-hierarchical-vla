import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,call

from scripts.pi0294.mp1.owner_heartbeat import read_live_owner


class OwnerHeartbeat(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.path=self.root/'OWNER_HEARTBEAT.json';self.path.write_text(json.dumps(dict(run_uuid='source',observed_epoch=900)))
        self.original=self.path.read_bytes()

    def value(self,epoch,run='source'):return dict(run_uuid=run,observed_epoch=epoch)

    def test_fresh_normal_read_does_not_touch_metadata_or_write_heartbeat(self):
        sync=Mock();proof=read_live_owner(self.root,'source',now=1000,synchronize=sync)
        self.assertEqual(proof['cached_age_s'],100);self.assertFalse(proof['server_revalidation']);sync.assert_not_called()
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_server_revalidates_parent_then_file_only_when_cached_age_is_expired(self):
        reader=Mock(side_effect=[self.value(600),self.value(999)]);sync=Mock()
        proof=read_live_owner(self.root,'source',now=1000,reader=reader,synchronize=sync)
        self.assertEqual(sync.call_args_list,[call(self.root),call(self.path)])
        self.assertEqual(reader.call_args_list,[call(self.path),call(self.path)])
        self.assertEqual(proof['cached_age_s'],400);self.assertEqual(proof['revalidated_age_s'],1)
        self.assertEqual(proof['watchdog_s'],300);self.assertFalse(proof['heartbeat_written'])
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_original_300_seconds_boundary_not_extended(self):
        for age in [0,300]:
            with self.subTest(age=age):
                sync=Mock();read_live_owner(self.root,'source',now=1000,reader=Mock(return_value=self.value(1000-age)),synchronize=sync);sync.assert_not_called()
        for age in [300.001,301,1000]:
            with self.subTest(age=age):
                with self.assertRaises(TimeoutError):read_live_owner(self.root,'source',now=1000,reader=Mock(return_value=self.value(1000-age)),synchronize=Mock())

    def test_true_owner_loss_still_fails_after_exactly_two_reads(self):
        reader=Mock(return_value=self.value(600))
        with self.assertRaisesRegex(TimeoutError,'HEARTBEAT_LOST_AFTER_SERVER_REVALIDATION'):
            read_live_owner(self.root,'source',now=1000,reader=reader,synchronize=Mock())
        self.assertEqual(reader.call_count,2)

    def test_wrong_run_never_becomes_an_accepted_heartbeat(self):
        for sequence in [[self.value(999,'other')],[self.value(600),self.value(999,'other')]]:
            with self.subTest(sequence=sequence):
                with self.assertRaises(PermissionError):read_live_owner(self.root,'source',now=1000,reader=Mock(side_effect=sequence),synchronize=Mock())

    def test_future_timestamp_is_not_clock_skew_tolerance(self):
        with self.assertRaises(TimeoutError):read_live_owner(self.root,'source',now=1000,reader=Mock(return_value=self.value(1001)),synchronize=Mock())

    def test_invalid_timestamps_fail_closed(self):
        for epoch in [True,None,'900',float('nan'),float('inf')]:
            with self.subTest(epoch=epoch):
                with self.assertRaises(ValueError):read_live_owner(self.root,'source',now=1000,reader=Mock(return_value=self.value(epoch)),synchronize=Mock())

    def test_server_error_is_not_interpreted_as_valid_owner(self):
        with self.assertRaises(OSError):read_live_owner(self.root,'source',now=1000,reader=Mock(return_value=self.value(600)),synchronize=Mock(side_effect=OSError('server unavailable')))

    def test_no_symlink_heartbeat_or_parent(self):
        elsewhere=self.root/'saved.json';self.path.rename(elsewhere);self.path.symlink_to(elsewhere)
        with self.assertRaises(PermissionError):read_live_owner(self.root,'source',now=1000)
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(PermissionError):read_live_owner(alias,'source',now=1000)

    def test_forced_compute_admission_revalidates_even_a_fresh_cached_read(self):
        sync=Mock();proof=read_live_owner(self.root,'source',now=1000,synchronize=sync,force_revalidate=True)
        self.assertTrue(proof['server_revalidation']);self.assertEqual(sync.call_args_list,[call(self.root),call(self.path)])

    def test_expired_compute_admission_keeps_actual_read_values_in_error(self):
        from .mp1.owner_heartbeat import HeartbeatLost
        with self.assertRaises(HeartbeatLost) as caught:
            read_live_owner(self.root,'source',now=1500,synchronize=Mock(),force_revalidate=True)
        self.assertEqual(caught.exception.diagnostics['heartbeat']['observed_epoch'],900)
        self.assertEqual(caught.exception.diagnostics['watchdog_s'],300)
        self.assertFalse(caught.exception.diagnostics['heartbeat_written'])


if __name__=='__main__':unittest.main()
