import subprocess
import unittest
from unittest.mock import Mock

from scripts.pi0294.mp1.owner_boundary import sync_metadata


class OwnerBoundary(unittest.TestCase):
    def test_transient_read_failure_preserves_heartbeat_and_retries_same_idempotent_sync(self):
        sync=Mock(side_effect=[RuntimeError('transport read'),{'settled':True}]);heartbeat=Mock();sleep=Mock()
        self.assertEqual(sync_metadata(sync,heartbeat,sleep=sleep),{'settled':True})
        self.assertEqual(sync.call_count,2);self.assertEqual(heartbeat.call_count,3);sleep.assert_called_once_with(5)

    def test_finite_attempt_limit_never_swallows_failure(self):
        sync=Mock(side_effect=subprocess.TimeoutExpired('readonly metadata',15));heartbeat=Mock()
        with self.assertRaises(subprocess.TimeoutExpired):sync_metadata(sync,heartbeat,sleep=Mock())
        self.assertEqual(sync.call_count,3)

    def test_integrity_failure_not_retried(self):
        for exc in [PermissionError('scope'),AssertionError('ledger'),ValueError('unknown')]:
            with self.subTest(error=type(exc).__name__):
                sync=Mock(side_effect=exc)
                with self.assertRaises(type(exc)):sync_metadata(sync,Mock(),sleep=Mock())
                self.assertEqual(sync.call_count,1)

    def test_time_limit_preserves_original_error(self):
        sync=Mock(side_effect=RuntimeError('read'));clock=Mock(side_effect=[0,121])
        with self.assertRaisesRegex(RuntimeError,'read'):sync_metadata(sync,Mock(),clock=clock,sleep=Mock())
        self.assertEqual(sync.call_count,1)

    def test_heartbeat_failure_is_not_interpreted_as_sync_success(self):
        sync=Mock()
        with self.assertRaises(RuntimeError):sync_metadata(sync,Mock(side_effect=RuntimeError('heartbeat')))
        sync.assert_not_called()

    def test_no_unbounded_recovery_parameters(self):
        for args in [dict(attempts=4),dict(attempts=0),dict(attempts=True),dict(maximum_wait_s=300),dict(maximum_wait_s=0)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):sync_metadata(Mock(),Mock(),**args)


if __name__=='__main__':unittest.main()
