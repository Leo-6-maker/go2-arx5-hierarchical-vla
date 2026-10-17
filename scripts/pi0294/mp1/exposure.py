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
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise PermissionError('EXACT_EXISTING_TRACE_DIRECTORY')
        for path in sorted(directory.glob('worker-*.jsonl')):
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise PermissionError('NO_TRACE_MEMBER_PATH_ESCAPE')
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


def read_checkpoint_trace(directory, step):
    """Join completed resume partitions, not prefetched tails or live sessions."""
    if type(step) is not int or not 0 < step <= 2048:
        raise ValueError('BOUNDED_CHECKPOINT_STEP')
    root = Path(directory)
    checkpoint = root / ('checkpoint-' + str(step))
    snapshots = {}
    def read(path):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise PermissionError('NO_TRAINING_TRACE_PATH_ESCAPE')
        raw = path.read_bytes()
        snapshots[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    complete = read(checkpoint / 'CHECKPOINT_COMPLETE.json')
    if (type(complete['global_step']) is not int or complete['global_step'] != step
            or complete['full_resume'] is not True):
        raise ValueError('COMPLETE_CHECKPOINT_STEP_REQUIRED')
    sampler_path = checkpoint / 'SAMPLER_STATE.json'
    sampler = read(sampler_path)
    if (hashlib.sha256(sampler_path.read_bytes()).hexdigest() != complete['files']['SAMPLER_STATE.json']
            or sampler['sampler']['cursor'] != step * 32):
        raise ValueError('CHECKPOINT_COMMITTED_SAMPLER_HASH_AND_CURSOR')
    parts = []
    for proof_path in sorted(root.glob('session-*/TRAINING_RUN.json')):
        proof = read(proof_path)
        start, end = proof['resumed_from_step'], proof['global_step']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= 2048:
            raise ValueError('EXACT_COMPLETED_SESSION_STEP_RANGE')
        if start >= step:
            continue
        if (end > step or type(proof['actual_updates_this_attempt']) is not int
                or type(proof['consumed_samples']) is not int
                or proof['actual_updates_this_attempt'] != end - start
                or proof['consumed_samples'] != end * 32):
            raise ValueError('EXACT_COMPLETED_CHECKPOINT_PARTITION')
        expected = 'session-start' if start == 0 else 'session-resume-' + str(start)
        if proof_path.parent.name != expected:
            raise ValueError('EXACT_COMPLETED_SESSION_DIRECTORY')
        parts.append((start, end, proof_path))
    rows, partitions, cursor = [], [], 0
    for start, end, proof_path in sorted(parts):
        if start != cursor:
            raise ValueError('COMMITTED_RESUME_PARTITION_GAP_OR_OVERLAP')
        traces = proof_path.parent / 'worker_trace'
        if traces.is_symlink() or not traces.resolve().is_relative_to(root.resolve()):
            raise PermissionError('NO_TRAINING_TRACE_PATH_ESCAPE')
        rows.extend(read_committed_trace([traces], start * 32, end * 32))
        partitions.append(dict(start_step=start, end_step=end,
            proof_sha256=snapshots[proof_path]))
        cursor = end
    if cursor != step:
        raise ValueError('MISSING_COMPLETED_CHECKPOINT_PARTITION')
    if any(path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, digest in snapshots.items()):
        raise ValueError('CHECKPOINT_OR_SESSION_PROOF_CHANGED_DURING_AUDIT')
    return rows, partitions


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


def matched_pair_exposure(baseline, baseline_rows, corrected, corrected_rows, step):
    """Compare consumed identities, not expected sampler counts or forecasts."""
    if type(step) is not int or step not in [512,2048]:
        raise PermissionError('ONLY_FIXED_SOURCE_PAIR_512_OR2048_CONSUMPTION')
    if (baseline.correction is not None or corrected.correction is None
            or baseline.original_manifest_sha!=corrected.original_manifest_sha
            or baseline.statistics_sha!=corrected.statistics_sha
            or baseline.original_rows!=corrected.original_rows
            or baseline.rows!=corrected.rows[:corrected.original_rows]
            or {row['task'] for row,_ in corrected.rows[corrected.original_rows:]}!={'T1'}):
        raise PermissionError('SAME_ORIGINAL_BANK_STATS_AND_T1_ONLY_CORRECTION')
    stop=step*32
    a=actual_exposure(baseline,baseline_rows,0,stop)
    b=actual_exposure(corrected,corrected_rows,0,stop)
    if (a['task_samples']!={'T1':stop//2,'T2':stop//2} or b['task_samples']!=a['task_samples']
            or a['correction_sample_visits']!=0 or b['correction_sample_visits']!=stop//4
            or b['distinct_correction_student_parents']<2):
        raise ValueError('ACTUAL_MATCHED_TASK_EXPOSURE_AND_INDEPENDENT_CORRECTION_PARENTS')
    t2=[];original_t1=[];correction_t1=[]
    for first,second in zip(baseline_rows,corrected_rows):
        cursor=first['cursor'];row,anchor=corrected.rows[second['index']]
        is_correction=second['index']>=corrected.original_rows
        if row['task']=='T2':
            if first!=second or is_correction:
                raise ValueError('EVERY_ACTUAL_T2_CURSOR_AND_SAMPLE_IDENTITY_MUST_MATCH')
            t2.append(first)
        elif not is_correction:
            if first!=second:
                raise ValueError('RETAINED_ORIGINAL_T1_CURSOR_AND_SAMPLE_IDENTITY_MUST_MATCH')
            original_t1.append(first)
        else:
            if row['task']!='T1' or cursor%4!=2:
                raise ValueError('ONLY_DECLARED_T1_CORRECTION_CURSOR_REPLACEMENT')
            correction_t1.append(second)
    if (len(t2)!=stop//2 or len(original_t1)!=stop//4
            or len(correction_t1)!=stop//4):
        raise ValueError('EXACT_MATCHED_T2_RETAINED_T1_AND_CORRECTION_VISIT_COUNTS')
    digest=lambda rows:hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(status='PASS_ACTUAL_MATCHED_SOURCE_PAIR_TASKS_T2_AND_RETAINED_T1_CURSORS',
        checkpoint_step=step,stop_cursor=stop,original_manifest_sha256=baseline.original_manifest_sha,
        original_statistics_sha256=baseline.statistics_sha,correction_manifest_sha256=corrected.correction.manifest_sha,
        baseline=a,corrected=b,matched_T2_cursor_count=len(t2),matched_retained_original_T1_cursor_count=len(original_t1),
        actual_T1_correction_cursor_count=len(correction_t1),
        matched_T2_trace_semantic_sha256=digest(t2),matched_retained_T1_trace_semantic_sha256=digest(original_t1),
        actual_T1_correction_trace_semantic_sha256=digest(correction_t1),
        sample_visits_are_independent_demonstrations=False,
        parent_initialization_training_recipe_and_closed_loop_outcomes_are_separate_proofs=True,
        original_task_exposure_confounded_by_3_to1=False,new_model_queries=0,new_physics=0,
        training_or_sampler_modified=False,causal_correction_gain_proven=False)
