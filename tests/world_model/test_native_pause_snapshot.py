import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter


class TracePauseSnapshot(unittest.TestCase):
    def test_transient_replacement_gap_retries_only_reads_then_exactly_seals(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);ack=r/'pause_ack.json';summary=r/'init.trace.summary';summary.write_text(json.dumps(dict(sealed_at_explicit_pause=True)))
            good=dict(step=4103,time_s=8.206,running=False,clock_paused=True);calls=[]
            def spin(seconds,predicate):
                calls.append(seconds)
                if len(calls)==1:
                    self.assertFalse(predicate()) # absent during replacement
                    ack.write_text('{');self.assertFalse(predicate()) # incomplete JSON
                    ack.write_text(json.dumps(good))
                return predicate()
            a=SimpleNamespace(output=r,observe=lambda:dict(step=4103,time_s=8.206),spin=spin,trace_sealed=False)
            TargetSequentialAdapter.seal_trace(a,4103)
            self.assertTrue(a.trace_sealed);self.assertEqual(calls,[3,3]);self.assertEqual(json.loads((r/'trace_seal_request').read_text())['step'],4103)
            self.assertFalse((r/'trace_seal_request.pending').exists());self.assertFalse((r/'step_quota').exists())

    def test_wrong_step_running_or_clock_never_seals_or_advances(self):
        for change in [dict(step=4102),dict(running=True),dict(clock_paused=False),dict(time_s=8.)]:
            with self.subTest(change=change),tempfile.TemporaryDirectory()as tmp:
                r=Path(tmp);ack=dict(step=4103,time_s=8.206,running=False,clock_paused=True);ack.update(change);(r/'pause_ack.json').write_text(json.dumps(ack))
                a=SimpleNamespace(output=r,observe=lambda:dict(step=4103,time_s=8.206),spin=lambda seconds,f:f(),trace_sealed=False)
                with self.assertRaisesRegex(RuntimeError,'NATIVE_PAUSE_NOT_CONFIRMED'):TargetSequentialAdapter.seal_trace(a,4103)
                self.assertFalse((r/'trace_seal_request').exists());self.assertFalse((r/'step_quota').exists())

    def test_unreadable_timeout_is_preserved_failure(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);a=SimpleNamespace(output=r,observe=lambda:dict(step=4103,time_s=8.206),spin=lambda seconds,f:f(),trace_sealed=False)
            with self.assertRaisesRegex(RuntimeError,'SNAPSHOT_NOT_READABLE'):TargetSequentialAdapter.seal_trace(a,4103)
            self.assertFalse(a.trace_sealed)


if __name__=='__main__':unittest.main()
