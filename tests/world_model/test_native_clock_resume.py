import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter


class ClockResume(unittest.TestCase):
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
