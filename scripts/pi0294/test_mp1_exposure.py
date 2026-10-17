import json
from pathlib import Path
import tempfile
import unittest

from .mp1.dataset import SourceSampler
from .mp1.exposure import actual_exposure, read_committed_trace
from .test_mp1 import dataset


def trace(d, start=0, stop=64):
    d.identity = lambda i: dict(episode_id=d.rows[i][0]['id'], split='TRAIN', anchor=d.rows[i][1])
    return [dict(cursor=c, index=i, **d.identity(i)) for c, i in SourceSampler(d, start=start, stop=stop)]


class ExposureTest(unittest.TestCase):
    def test_actual_correction_lineage_and_equal_task_exposure(self):
        d = dataset()
        for row, anchor in d.rows[:d.original_rows]:
            row['stage_anchors'] = dict(approach=[0], hold=[1, 2])
        rows = trace(d)
        result = actual_exposure(d, rows, 0, 64)
        self.assertEqual(result['task_samples'], dict(T1=32, T2=32))
        self.assertEqual(result['origin_samples'], dict(original=48, correction=16))
        self.assertEqual(result['correction_sample_visits'], 16)
        self.assertEqual(result['distinct_correction_student_parents'], 2)
        actors = result['current_state_actor_visits']
        self.assertEqual(sum(actors.values()), 64)
        self.assertGreater(actors['EXPERT_AFTER_CORRECTION_ANCHOR'], 0)
        self.assertEqual(sum(result['manifest_phase_visits'].values()), 64)
        self.assertFalse(result['forecast_used_as_actual_consumption'])

    def test_changed_actual_trace_and_unknown_correction_parent_rejected(self):
        d = dataset()
        rows = trace(d)
        changed = [dict(x) for x in rows]
        changed[0]['anchor'] += 1
        with self.assertRaises(ValueError):
            actual_exposure(d, changed, 0, 64)
        corrected = next(i for c, i in SourceSampler(d, stop=64) if i >= d.original_rows)
        d.rows[corrected][0]['source_student_episode'] = None
        with self.assertRaises(ValueError):
            actual_exposure(d, rows, 0, 64)

    def test_completed_prefix_ignores_only_uncommitted_partial_line(self):
        d = dataset(())
        rows = trace(d)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / 'worker-1.jsonl'
            p.write_text(''.join(json.dumps(x) + '\n' for x in rows) + '{"cursor":64')
            self.assertEqual(read_committed_trace([root], 0, 64), rows)
            p.write_text(''.join(json.dumps(x) + '\n' for x in rows[:-1]))
            with self.assertRaises(ValueError):
                read_committed_trace([root], 0, 64)
            p.write_text(''.join(json.dumps(x) + '\n' for x in rows + [rows[0]]))
            with self.assertRaises(ValueError):
                read_committed_trace([root], 0, 64)

    def test_resume_partitions_and_unknown_phase_remain_explicit(self):
        d = dataset(())
        first = trace(d, 0, 32)
        second = trace(d, 32, 64)
        a = actual_exposure(d, first, 0, 32)
        b = actual_exposure(d, second, 32, 64)
        self.assertEqual(a['actual_sample_visits'] + b['actual_sample_visits'], 64)
        self.assertEqual(a['distinct_correction_student_parents'], 0)
        self.assertTrue(all('NOT_RECORDED' in k for k in a['manifest_phase_visits']))


if __name__ == '__main__':
    unittest.main()
