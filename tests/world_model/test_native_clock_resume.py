import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter


class ClockResume(unittest.TestCase):
    def test_TTL_keeps_registered_init_windows_and_only_actual_TTL_task_boundaries(self):
        with tempfile.TemporaryDirectory()as root:
            a=TargetSequentialAdapter(dict(phase='g2-standing-ttl-v2',max_ticks=2706,startup_pause_stride_ticks=25),root)
            a.resume_clock_once=Mock();a.spin=Mock(return_value=True);a.observe=Mock(return_value=dict(step=25,time_s=.05))
            a.advance_to(25);a.resume_clock_once.assert_called_once_with(25)
            self.assertEqual((Path(root)/'step_quota').read_text(),'25\n')
            # Ordinary 50ms task pauses remain disallowed in the TTL diagnostic.
            with self.assertRaisesRegex(PermissionError,'EXACT_REGISTERED_NATIVE_PAUSE'):a.advance_to(2528)
            a.ttl_boundaries=[2702,2703,2704,2706];a.observe.return_value=dict(step=2702,time_s=5.404)
            a.advance_to(2702);self.assertEqual((Path(root)/'step_quota').read_text(),'2702\n')
            with self.assertRaisesRegex(PermissionError,'EXACT_REGISTERED_NATIVE_PAUSE'):a.advance_to(2707)
            a.spec.pop('startup_pause_stride_ticks')
            with self.assertRaisesRegex(PermissionError,'EXACT_REGISTERED_NATIVE_PAUSE'):a.advance_to(25)

    def adapter(self,root,counts):
        keys=SimpleNamespace(get_subscription_count=Mock(side_effect=counts))
        def spin(seconds,predicate):
            self.assertEqual(seconds,3)
            return any(predicate()for _ in counts)
        return SimpleNamespace(output=Path(root),keys=keys,spin=spin,publish=Mock())

    def test_waits_for_real_match_and_publishes_toggle_only_once(self):
        with tempfile.TemporaryDirectory()as root:
            a=self.adapter(root,[0,0,1]);TargetSequentialAdapter.resume_clock_once(a,25)
            a.publish.assert_called_once_with(a.keys,'ENTER')
            record=json.loads((Path(root)/'CLOCK_RESUME_REQUESTS.jsonl').read_text())
            self.assertEqual(record['requested_step'],25)
            self.assertEqual(record['matched_subscriptions'],1)
            self.assertIn('SEPARATELY',record['consumption_evidence'])

    def test_unmatched_topic_never_publishes_and_preserves_failure(self):
        with tempfile.TemporaryDirectory()as root:
            a=self.adapter(root,[0,0])
            with self.assertRaisesRegex(RuntimeError,'SUBSCRIBER_NOT_MATCHED'):
                TargetSequentialAdapter.resume_clock_once(a,25)
            a.publish.assert_not_called()
            self.assertFalse((Path(root)/'CLOCK_RESUME_REQUESTS.jsonl').exists())

    def test_ambiguous_subscribers_reject_instead_of_broadcasting(self):
        with tempfile.TemporaryDirectory()as root:
            a=self.adapter(root,[2])
            with self.assertRaisesRegex(RuntimeError,'MULTIPLE_SUBSCRIBERS'):
                TargetSequentialAdapter.resume_clock_once(a,25)
            a.publish.assert_not_called()


if __name__=='__main__':unittest.main()
