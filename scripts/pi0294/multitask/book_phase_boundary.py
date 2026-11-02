"""CPU preparation for PR317's D2-only phase/boundary sampling pair.

This does not admit a training recipe or launch a model. Phase and contact
metadata only select TRAIN indices; they never enter observations or labels.
"""
from collections import defaultdict

from .book_bc_dagger import population_rows
from .book_phase_sampling import SLOTS, STAGES, digest, stream_sequence

SCHEMA = 'BOOK_D2_PHASE_BOUNDARY_INDEX_V1'
LATE = 'new_late_takeover_first16'
CONTROL_SLOTS = (*SLOTS, (LATE, 0))
BOUNDARY_SLOTS = (('uniform', 14), *SLOTS[1:], (LATE, 2))


def pools(rows, index, *, index_sha256, dataset_sha256, statistics_sha256, population_sha256):
    if (digest(index) != index_sha256 or index.get('schema') != SCHEMA
            or index['dataset_manifest_sha256'] != dataset_sha256
            or index['statistics_sha256'] != statistics_sha256
            or index['TRAIN_population_sha256'] != population_sha256):
        raise PermissionError('SEALED_D2_TRAIN_PHASE_INDEX_REQUIRED')
    episodes = index['episodes']
    entries = population_rows('D2', [dict(id=e['id'], windows=e['windows']) for e in episodes])
    late_ids = {e['id'] for e in entries if e['id'].startswith('BOOK_LATE_RECOVERY_')}
    if len(entries) != 28 or len(late_ids) != 7 or sum(e['windows'] for e in entries) != 15932:
        raise ValueError('EXACT_ADMITTED_D2_28_ENTRIES_15932_WINDOWS')
    metadata = {}
    for e in episodes:
        name, n = e['id'], e['windows']
        source, stages, closed = (e[k] for k in ('source_observation_index','stages','prefix8_closed'))
        boundary = set(e['source_planning_indices'])
        late = name in late_ids
        if (len(source) != n or len(stages) != n or len(closed) != n
                or source != list(range(source[0], source[0]+n))
                or not set(stages) <= STAGES or any(type(c) is not bool for c in closed)
                or e['expert_suffix'] != ('EXPERT_SUFFIX' in name)
                or e['new_late_takeover'] != late):
            raise ValueError('ACTUAL_SOURCE_PHASE_AND_HANDOVER_ALIGNMENT')
        for anchor in range(n):
            key = name, anchor
            if key in metadata:
                raise ValueError('DUPLICATE_TRAIN_PHASE_WINDOW')
            metadata[key] = (stages[anchor], closed[anchor], source[anchor] in boundary,
                             e['expert_suffix'] and not late, late)
    groups = {name: defaultdict(list) for name, _ in CONTROL_SLOTS}
    seen = set()
    for i, (row, anchor) in enumerate(rows):
        if row['split'] != 'TRAIN' or row['task'] != 'BOOK':
            raise PermissionError('BOOK_TRAIN_ONLY_NO_DEV_TEST')
        key = row['id'], anchor
        if key in seen or key not in metadata:
            raise ValueError('EXACT_UNIQUE_NUMERIC_WINDOW_JOIN')
        seen.add(key)
        stage, closed, boundary, early, late = metadata[key]
        def add(name):
            groups[name][row['id']].append(i)
        add('uniform')
        if stage in ('approach', 'align'):
            add('approach_align')
            if early:
                add('early_recovery')
        if stage in ('close', 'verify_grasp'):
            add('close_verify')
        if stage in ('lift', 'preinsert', 'insert') and closed:
            add('carry_boundary' if boundary else 'carry_other')
        if stage in ('release', 'withdraw'):
            add(stage)
        if late and anchor < 16:
            add(LATE)
    if seen != set(metadata):
        raise ValueError('EXACT_ALL_ADMITTED_D2_WINDOWS')
    if any(not g or any(not values for values in g.values()) for g in groups.values()):
        raise ValueError('DECLARED_STREAM_MUST_BE_NONEMPTY_NO_SILENT_FALLBACK')
    if sum(map(len, groups[LATE].values())) != 112:
        raise ValueError('EXACT_SEVEN_TAKEOVERS_FIRST16_NOT_FIRST8')
    return {name: dict(sorted(group.items())) for name, group in groups.items()}


def sequence(rows, index, *, boundary, seed, updates=4096, return_streams=False, **pins):
    if type(boundary) is not bool or updates != 4096:
        raise ValueError('MATCHED_FINITE4096_PAIR_ONLY')
    groups = pools(rows, index, **pins)
    slots = BOUNDARY_SLOTS if boundary else CONTROL_SLOTS
    return stream_sequence(groups, slots, seed, updates, return_streams=return_streams)
