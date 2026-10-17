"""Read-only actual sample accounting; manifests annotate, never replace traces."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .dataset import SourceSampler


def read_committed_trace(directories, start, stop):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 65536:
        raise ValueError('BOUNDED_NONEMPTY_COMMITTED_SAMPLE_RANGE')
    rows = {}
    for directory in directories:
        for path in sorted(Path(directory).glob('worker-*.jsonl')):
            with path.open() as stream:
                for line in stream:
                    if not line.endswith('\n'):
                        # A live writer may have an unfinished prefetched row.
                        # It cannot prove any requested committed cursor.
                        continue
                    row = json.loads(line)
                    cursor = row['cursor']
                    if type(cursor) is not int or cursor < 0:
                        raise ValueError('EXACT_NONNEGATIVE_TRACE_CURSOR')
                    if start <= cursor < stop:
                        if cursor in rows:
                            raise ValueError('DUPLICATE_COMMITTED_CONSUMPTION_CURSOR')
                        rows[cursor] = row
    if sorted(rows) != list(range(start, stop)):
        raise ValueError('MISSING_COMMITTED_CONSUMPTION_CURSOR')
    return [rows[c] for c in range(start, stop)]


def actual_exposure(dataset, rows, start, stop):
    sampler = SourceSampler(dataset, start=start, stop=stop)
    if len(rows) != stop - start:
        raise ValueError('ACTUAL_TRACE_RANGE_COUNT')
    tasks, origins, episodes, parents, phases, actors, families = (Counter() for _ in range(7))
    correction_visits = 0
    phase_maps = {}
    for actual, (cursor, index) in zip(rows, sampler):
        expected = dict(cursor=cursor, index=index, **dataset.identity(index))
        if actual != expected:
            raise ValueError('ACTUAL_SAMPLE_IDENTITY_OR_FIXED_SAMPLER_MISMATCH')
        row, anchor = dataset.rows[index]
        if row['split'] != 'TRAIN':
            raise PermissionError('TRAIN_ONLY_EXPOSURE_AUDIT')
        corrected = index >= dataset.original_rows
        if corrected:
            if row['task'] != 'T1' or not row.get('source_student_episode'):
                raise PermissionError('FIXED_T1_ONLY_C1_STUDENT_LINEAGE')
            correction_visits += 1
            parents[row['source_student_episode']] += 1
            actor = 'STUDENT_AT_CORRECTION_ANCHOR' if anchor == 0 else 'EXPERT_AFTER_CORRECTION_ANCHOR'
            phase = 'NOT_RECORDED_IN_CORRECTION_MANIFEST'
        else:
            actor = 'ORIGINAL_SUCCESS_EXPERT_DEMONSTRATION'
            if row['id'] not in phase_maps:
                mapping = {}
                for name, anchors in row.get('stage_anchors', {}).items():
                    for value in anchors:
                        if value in mapping:
                            raise ValueError('MANIFEST_PHASE_ANCHOR_OVERLAP')
                        mapping[value] = name
                phase_maps[row['id']] = mapping
            phase = phase_maps[row['id']].get(anchor, 'NOT_RECORDED_IN_ORIGINAL_MANIFEST')
        tasks[row['task']] += 1
        origins['correction' if corrected else 'original'] += 1
        episodes[row['id']] += 1
        families[row.get('family', 'UNKNOWN')] += 1
        phases[row['task'] + ':' + phase] += 1
        actors[actor] += 1
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(status='PASS_ACTUAL_COMMITTED_TRACE_FIXED_SAMPLER_AND_LINEAGE',
        start_cursor=start, stop_cursor=stop, actual_sample_visits=stop-start,
        trace_semantic_sha256=digest, task_samples=dict(tasks), origin_samples=dict(origins),
        correction_sample_visits=correction_visits, episode_visits=dict(sorted(episodes.items())),
        distinct_episodes=len(episodes), correction_student_parent_visits=dict(sorted(parents.items())),
        distinct_correction_student_parents=len(parents), manifest_phase_visits=dict(sorted(phases.items())),
        current_state_actor_visits=dict(actors), family_visits=dict(families),
        manifest_phase_is_raw_reconstructed_event=False,
        correction_future_expert_states_are_all_student_states=False,
        sample_visits_or_overlapping_windows_are_independent_demonstrations=False,
        forecast_used_as_actual_consumption=False, new_models=0, new_physics=0)
