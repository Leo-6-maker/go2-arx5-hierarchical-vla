"""Fault tests use no physics, GPU or neural dependencies."""
import errno
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .durable import MessageGate, OwnerLock, read, write
from .resource_guard import Budget


class DurableTest(unittest.TestCase):
    def test_message_rejections_do_not_advance(self):
        gate = MessageGate('run', 'episode')
        packet = dict(run='run', episode='episode', sequence=0, expires=10,
                      payload='actual observation', sha256=hashlib.sha256(b'actual observation').hexdigest())
        for mutation in ({'expires':0}, {'sequence':1}, {'run':'old'}, {'sha256':'corrupt'}):
            with self.assertRaises(ValueError):
                gate.accept({**packet, **mutation}, now=1)
            self.assertEqual(gate.sequence, 0)
        gate.accept(packet, now=1)
        with self.assertRaises(ValueError):
            gate.accept(packet, now=1)
        self.assertEqual(gate.sequence, 1)

    def test_lock_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = OwnerLock(Path(folder)/'lock')
            with self.assertRaises(BlockingIOError):
                OwnerLock(Path(folder)/'lock')
            lock.close()

    def test_failed_atomic_write_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.json'
            write(path, {'status':'COMMITTED'})
            with patch('os.replace', side_effect=OSError(errno.ENOSPC, 'disk full')):
                with self.assertRaises(OSError):
                    write(path, {'status':'UNCOMMITTED'})
            self.assertEqual(read(path), {'status':'COMMITTED'})
            self.assertEqual(len(list(Path(folder).glob('*.pending.*'))), 1)

    def test_counter_deadline_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write(root/'ADOPTION.json',dict(new_work_cutoff_epoch=200,run_uuid='test'))
            write(root/'ATTEMPT_PLAN.json',dict(attempts={'D291000':{}}))
            write(root/'RESOURCE_LEDGER.json',dict(starts=[],physics_steps=864000,pairs=0,CPU_s=0,CPU_upper_s=0))
            with patch('time.time',return_value=100):
                b=Budget(root)
                with self.assertRaisesRegex(RuntimeError,'COUNT_RESERVATION'):
                    b.start('D291000','source')
                with self.assertRaisesRegex(RuntimeError,'UNREGISTERED'):
                    b.start('T291000','source')
                b.value['starts']=[dict(id='D291000',status='TERMINAL')]
                with self.assertRaisesRegex(RuntimeError,'ALREADY_STARTED'):
                    b.start('D291000','source')
            with patch('time.time',return_value=201):
                with self.assertRaisesRegex(RuntimeError,'CUTOFF'):
                    b.guard()
            b.lock.close()


if __name__ == '__main__':
    unittest.main()
