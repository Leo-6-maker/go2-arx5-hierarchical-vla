import errno
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from .mp1 import recovery_neural
from .relay_file_rpc import ensure_model_not_terminal

control_read=sys.modules[recovery_neural.read.__module__]


class RecoveryControlReadTests(unittest.TestCase):
    def test_transient_file_table_pressure_retries_only_same_read_not_rpc_mutation(self):
        for code in (errno.ENFILE,errno.EMFILE):
            with patch.object(control_read,'original_read',side_effect=[OSError(code,'file table'),{'exact':'readback'}]) as read,\
                    patch.object(control_read.time,'sleep') as sleep:
                self.assertEqual(control_read.read('fixed-read-only-binding'),{'exact':'readback'})
                self.assertEqual(read.call_args_list,[unittest.mock.call('fixed-read-only-binding')]*2)
                sleep.assert_called_once_with(.1)
            with patch.object(control_read,'original_read',side_effect=OSError(code,'persistent')) as read,\
                    patch.object(control_read.time,'sleep') as sleep:
                with self.assertRaises(OSError):control_read.read('fixed-read-only-binding')
                self.assertEqual(read.call_count,6);self.assertEqual(sleep.call_count,5)

    def test_recovery_uses_existing_bounded_control_reader(self):
        self.assertEqual(recovery_neural.read.__module__,'scripts.pi0294.e1.control_read')
        fresh={'run_uuid':'owned','observed_epoch':123}
        with patch.object(control_read,'original_read',side_effect=[FileNotFoundError(errno.ENOENT,'rename'),fresh]) as read:
            with patch.object(control_read.time,'sleep') as sleep:
                self.assertEqual(recovery_neural.read('heartbeat'),fresh)
        self.assertEqual(read.call_count,2);sleep.assert_called_once_with(.1)

    def test_persistent_missing_stale_and_corrupt_are_not_accepted(self):
        for error in [FileNotFoundError(errno.ENOENT,'missing'),OSError(errno.ESTALE,'stale')]:
            with patch.object(control_read,'original_read',side_effect=error) as read:
                with patch.object(control_read.time,'sleep'):
                    with self.assertRaises(OSError):recovery_neural.read('heartbeat')
            self.assertEqual(read.call_count,6)
        for error in [PermissionError(errno.EACCES,'denied'),ValueError('corrupt')]:
            with patch.object(control_read,'original_read',side_effect=error) as read:
                with self.assertRaises(type(error)):recovery_neural.read('heartbeat')
            self.assertEqual(read.call_count,1)

    def test_terminal_marker_stops_ready_but_dead_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'pbs/alloc-01').mkdir(parents=True);(root/'pbs/alloc-02').mkdir()
            state=root/'pbs/MODEL_STATE.json';state.write_text(json.dumps(dict(label='panel',job_id='new')))
            (root/'pbs/alloc-01/PBS_PROCESS_FINAL.json').write_text(json.dumps(dict(job_id='old')))
            ensure_model_not_terminal(root,'panel')
            with self.assertRaises(PermissionError):ensure_model_not_terminal(root,'wrong')
            (root/'pbs/alloc-02/PBS_PROCESS_FINAL.json').write_text(json.dumps(dict(job_id='new')))
            with self.assertRaisesRegex(PermissionError,'ALREADY_TERMINAL'):ensure_model_not_terminal(root,'panel')

    def test_audit_uses_only_explicit_existing_adjudication_names(self):
        from .e1.audit import episode
        from .e1.zero_step_audit import audit
        with self.assertRaises(PermissionError):episode('/unused','case',adjudication_name='../other')
        with self.assertRaises(PermissionError):audit('/unused','case',None,adjudication_name='../other')
