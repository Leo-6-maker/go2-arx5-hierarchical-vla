import errno
import subprocess
import signal
import sys
import time
import unittest
from unittest.mock import Mock

from .mp1.read_only_metadata import read_only


class ReadOnlyMetadataRecoveryTests(unittest.TestCase):
    def test_exact_observed_remote_stale_handle_is_retried_without_heartbeat_writer(self):
        operation=Mock(side_effect=[RuntimeError('hopper:readonly:OSError: [Errno 116] Stale file handle'),{'state':'R'}])
        self.assertEqual(read_only(operation,kind='native_job_status',sleep=Mock()),{'state':'R'})
        self.assertEqual(operation.call_count,2)

    def test_local_stale_handle_and_timeout_are_bounded(self):
        for error in [OSError(errno.ESTALE,'Stale file handle'),subprocess.TimeoutExpired('readonly',15)]:
            operation=Mock(side_effect=error)
            with self.subTest(error=type(error).__name__),self.assertRaises((RuntimeError,subprocess.TimeoutExpired)):
                read_only(operation,kind='registered_resource_status',sleep=Mock())
            self.assertEqual(operation.call_count,3)

    def test_remote_permission_integrity_unknown_and_unclassified_failures_never_retry(self):
        for message in ['PermissionError: auth','ValueError: stats','AssertionError: wrong job','JSONDecodeError',
                        'MUTATION_OUTCOME_UNKNOWN','other remote failure',
                        'OSError: [Errno 116] Stale file handle then PermissionError: authority']:
            operation=Mock(side_effect=RuntimeError(message))
            with self.subTest(message=message),self.assertRaises(PermissionError):
                read_only(operation,kind='closed_episode_metadata',sleep=Mock())
            self.assertEqual(operation.call_count,1)

    def test_mutation_kinds_rejected_before_any_call(self):
        for kind in ['query','install','start','qsub','action','settle_unknown']:
            operation=Mock()
            with self.subTest(kind=kind),self.assertRaises(PermissionError):read_only(operation,kind=kind)
            operation.assert_not_called()

    def test_original_attempt_and_time_bounds_cannot_expand(self):
        for bounds in [dict(attempts=4),dict(maximum_wait_s=121)]:
            operation=Mock()
            with self.assertRaises(ValueError):read_only(operation,kind='native_job_status',**bounds)
            operation.assert_not_called()

    def test_window_includes_a_blocked_read_and_restores_original_signal_handler(self):
        before=signal.getsignal(signal.SIGALRM);started=time.monotonic()
        def operation():subprocess.run([sys.executable,'-c','import time;time.sleep(5)'],timeout=120,check=True)
        with self.assertRaises(subprocess.TimeoutExpired):read_only(operation,kind='native_job_status',maximum_wait_s=.05)
        self.assertLess(time.monotonic()-started,1)
        self.assertEqual(signal.getsignal(signal.SIGALRM),before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)


if __name__=='__main__':unittest.main()
