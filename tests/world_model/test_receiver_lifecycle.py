import unittest
from scripts.wm02.receiver_lifecycle import spin_until_stopped


class ReceiverLifecycle(unittest.TestCase):
    def test_signal_waits_for_inflight_callback(self):
        events = []
        class Executor:
            stopped = False
            def spin_once(self, timeout_sec):
                events.append('take_message_with_live_context')
                self.stopped = True
                events.append('callback_completed')
        executor = Executor()
        spin_until_stopped(executor, lambda: True, lambda: executor.stopped)
        self.assertEqual(events, ['take_message_with_live_context', 'callback_completed'])

    def test_live_callback_error_remains_failure(self):
        class Executor:
            def spin_once(self, timeout_sec):
                raise RuntimeError('actual callback failure')
        with self.assertRaisesRegex(RuntimeError, 'actual callback failure'):
            spin_until_stopped(Executor(), lambda: True, lambda: False)
