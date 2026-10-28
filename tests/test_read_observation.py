import subprocess
import unittest

from scripts.operations.read_observation import observe


class ReadObservationTests(unittest.TestCase):
    def test_transient_dns_repolls_same_read_not_restart(self):
        calls = []
        sleeps = []

        def read():
            calls.append('same-job-read')
            if len(calls) < 3:
                raise RuntimeError('ssh: Could not resolve hostname: Name or service not known')
            return {'job_state': 'Q'}

        self.assertEqual(observe(read, sleep=sleeps.append), {'job_state': 'Q'})
        self.assertEqual(calls, ['same-job-read'] * 3)
        self.assertEqual(sleeps, [15, 45])

    def test_exhausted_observation_is_not_terminal_state(self):
        calls = []

        def read():
            calls.append(1)
            raise TimeoutError('SSH read timed out')

        with self.assertRaises(TimeoutError):
            observe(read, sleep=lambda _: None)
        self.assertEqual(len(calls), 3)

    def test_subprocess_timeout_retries_only_the_read(self):
        calls = []
        sleeps = []

        def read():
            calls.append('same-process-status')
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(['ssh', 'site', 'read-only-status'], 110)
            return {'active': True}

        self.assertEqual(observe(read, sleep=sleeps.append), {'active': True})
        self.assertEqual(calls, ['same-process-status'] * 2)
        self.assertEqual(sleeps, [15])

    def test_exhausted_subprocess_timeout_preserves_unknown_observation(self):
        calls = []

        def read():
            calls.append(1)
            raise subprocess.TimeoutExpired(['ssh', 'site', 'read-only-status'], 110)

        with self.assertRaises(subprocess.TimeoutExpired):
            observe(read, sleep=lambda _: None)
        self.assertEqual(len(calls), 3)

    def test_integrity_or_permission_error_is_not_retried(self):
        for error in (PermissionError('native authority'), RuntimeError('unknown effect'),
                      ValueError('bad JSON'), RuntimeError('lease mismatch')):
            calls = []

            def read():
                calls.append(1)
                raise error

            with self.assertRaises(type(error)):
                observe(read, sleep=lambda _: self.fail('unexpected retry'))
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
