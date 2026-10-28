import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .mp1.dataset import SourceSampler
from .mp1.exposure import actual_exposure, read_committed_trace, read_checkpoint_trace, matched_pair_exposure
from .test_mp1 import dataset


def trace(d, start=0, stop=64):
    d.identity = lambda i: dict(episode_id=d.rows[i][0]['id'], split='TRAIN', anchor=d.rows[i][1])
    return [dict(cursor=c, index=i, **d.identity(i)) for c, i in SourceSampler(d, start=start, stop=stop)]


class ExposureTest(unittest.TestCase):
    def paired_fixture(self,step=512):
        a,b=dataset(()),dataset()
        for d in [a,b]:d.original_manifest_sha='original';d.statistics_sha='stats'
        a.correction=None;b.correction=SimpleNamespace(manifest_sha='qualified-correction')
        return a,trace(a,stop=step*32),b,trace(b,stop=step*32)

    def test_actual_matched_pair_joins_all_T2_and_retained_T1_cursors(self):
        a,ar,b,br=self.paired_fixture();result=matched_pair_exposure(a,ar,b,br,512)
        self.assertEqual(result['matched_T2_cursor_count'],8192)
        self.assertEqual(result['matched_retained_original_T1_cursor_count'],4096)
        self.assertEqual(result['actual_T1_correction_cursor_count'],4096)
        self.assertEqual(result['corrected']['distinct_correction_student_parents'],2)
        self.assertFalse(result['causal_correction_gain_proven'])
        self.assertFalse(result['training_or_sampler_modified'])

    def test_paired_actual_trace_tamper_or_missing_row_is_not_expected_exposure(self):
        a,ar,b,br=self.paired_fixture()
        bad=[dict(r) for r in br];bad[1]['anchor']+=1
        with self.assertRaisesRegex(ValueError,'ACTUAL_SAMPLE'):matched_pair_exposure(a,ar,b,bad,512)
        with self.assertRaisesRegex(ValueError,'RANGE_COUNT'):matched_pair_exposure(a,ar,b,br[:-1],512)
        b.rows[0][0]['new_unmatched_metadata']='changed'
        with self.assertRaisesRegex(PermissionError,'SAME_ORIGINAL'):matched_pair_exposure(a,ar,b,br,512)

    def test_pair_rejects_wrong_scope_stats_missing_or_T2_correction(self):
        a,ar,b,br=self.paired_fixture()
        for step in [True,1,511,1024,4096]:
            with self.assertRaises(PermissionError):matched_pair_exposure(a,ar,b,br,step)
        b.statistics_sha='different'
        with self.assertRaises(PermissionError):matched_pair_exposure(a,ar,b,br,512)
        b.statistics_sha=a.statistics_sha;b.correction=None
        with self.assertRaises(PermissionError):matched_pair_exposure(a,ar,b,br,512)
        b.correction=SimpleNamespace(manifest_sha='qualified-correction');b.rows[-1][0]['task']='T2'
        with self.assertRaises(PermissionError):matched_pair_exposure(a,ar,b,br,512)

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

    def checkpoint_fixture(self, root):
        checkpoint = root / 'checkpoint-2'
        checkpoint.mkdir()
        sampler = checkpoint / 'SAMPLER_STATE.json'
        sampler.write_text(json.dumps(dict(sampler=dict(cursor=64))))
        (checkpoint / 'CHECKPOINT_COMPLETE.json').write_text(json.dumps(dict(
            global_step=2, full_resume=True,
            files={'SAMPLER_STATE.json': hashlib.sha256(sampler.read_bytes()).hexdigest()})))
        d = dataset(())
        rows = trace(d)
        for name, start, end in [('session-start', 0, 1), ('session-resume-1', 1, 2)]:
            folder = root / name
            (folder / 'worker_trace').mkdir(parents=True)
            (folder / 'TRAINING_RUN.json').write_text(json.dumps(dict(
                resumed_from_step=start, global_step=end,
                actual_updates_this_attempt=end-start, consumed_samples=end*32)))
            # This tail belongs to prefetch, not to this session's committed work.
            committed = rows[start*32:end*32]
            tail = rows[end*32:end*32+2]
            (folder / 'worker_trace/worker-1.jsonl').write_text(
                ''.join(json.dumps(x) + '\n' for x in committed + tail))
        return rows

    def test_checkpoint_resume_excludes_prefetch_and_unfinished_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.checkpoint_fixture(root)
            (root / 'session-resume-2/worker_trace').mkdir(parents=True)
            (root / 'session-resume-2/worker_trace/worker-1.jsonl').write_text('{unfinished')
            rows, parts = read_checkpoint_trace(root, 2)
            self.assertEqual(rows, expected)
            self.assertEqual([(p['start_step'], p['end_step']) for p in parts], [(0, 1), (1, 2)])
            self.assertTrue(all(len(p['proof_sha256']) == 64 for p in parts))

    def test_checkpoint_resume_missing_partition_or_wrong_cursor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint_fixture(root)
            proof = root / 'session-resume-1/TRAINING_RUN.json'
            saved = proof.read_text()
            proof.unlink()
            with self.assertRaisesRegex(ValueError, 'MISSING_COMPLETED'):
                read_checkpoint_trace(root, 2)
            proof.write_text(saved)
            sampler = root / 'checkpoint-2/SAMPLER_STATE.json'
            sampler.write_text(json.dumps(dict(sampler=dict(cursor=65))))
            with self.assertRaisesRegex(ValueError, 'SAMPLER_HASH_AND_CURSOR'):
                read_checkpoint_trace(root, 2)

    def test_checkpoint_resume_duplicate_partition_symlink_and_step_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint_fixture(root)
            duplicate = root / 'session-copy'
            duplicate.mkdir()
            (duplicate / 'TRAINING_RUN.json').write_text((root / 'session-start/TRAINING_RUN.json').read_text())
            with self.assertRaisesRegex(ValueError, 'SESSION_DIRECTORY'):
                read_checkpoint_trace(root, 2)
            (duplicate / 'TRAINING_RUN.json').unlink()
            proof = root / 'session-resume-1/TRAINING_RUN.json'
            saved = proof.read_text()
            proof.unlink()
            (root / 'saved-proof.json').write_text(saved)
            proof.symlink_to(root / 'saved-proof.json')
            with self.assertRaises(PermissionError):
                read_checkpoint_trace(root, 2)
            for step in [True, 0, -1, 2049, 1.5]:
                with self.subTest(step=step), self.assertRaises(ValueError):
                    read_checkpoint_trace(root, step)

    def test_checkpoint_resume_overlap_false_counts_and_trace_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint_fixture(root)
            proof = root / 'session-start/TRAINING_RUN.json'
            saved = json.loads(proof.read_text())
            proof.write_text(json.dumps(dict(saved, global_step=2,
                                            actual_updates_this_attempt=2, consumed_samples=64)))
            # Complete the wider first trace, so rejection is the overlap itself.
            trace_path = root / 'session-start/worker_trace/worker-1.jsonl'
            trace_path.write_text(''.join(json.dumps(x) + '\n' for x in trace(dataset(()))))
            with self.assertRaisesRegex(ValueError, 'GAP_OR_OVERLAP'):
                read_checkpoint_trace(root, 2)
            proof.write_text(json.dumps(dict(saved, consumed_samples=33)))
            with self.assertRaisesRegex(ValueError, 'CHECKPOINT_PARTITION'):
                read_checkpoint_trace(root, 2)
            proof.write_text(json.dumps(saved))
            original = trace_path.read_text()
            trace_path.unlink()
            (root / 'saved-trace.jsonl').write_text(original)
            trace_path.symlink_to(root / 'saved-trace.jsonl')
            with self.assertRaisesRegex(PermissionError, 'TRACE_MEMBER_PATH_ESCAPE'):
                read_checkpoint_trace(root, 2)


if __name__ == '__main__':
    unittest.main()
