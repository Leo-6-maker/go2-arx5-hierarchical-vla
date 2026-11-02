"""PR317's fixed-D2 phase/boundary recipes, using the existing Trainer.

Recipes do not grant native capacity or launch models. Phase/contact metadata
only select TRAIN indices; they never enter observations or action labels.
"""
from collections import defaultdict

from .book_bc_dagger import population_rows
from .book_phase_sampling import SLOTS, STAGES, digest, stream_sequence

SCHEMA = 'BOOK_D2_PHASE_BOUNDARY_INDEX_V1'
PLAN_SCHEMA = 'BOOK_D2_PHASE_BOUNDARY4096_V1'
LATE = 'new_late_takeover_first16'
CONTROL_SLOTS = (*SLOTS, (LATE, 0))
BOUNDARY_SLOTS = (('uniform', 14), *SLOTS[1:], (LATE, 2))


def plan(variant, dataset_sha256, statistics_sha256, population_sha256,
         index_sha256, entries, *, steps=4096):
    import re
    from .book_bc_dagger import plan as data_plan
    if variant not in ('CONTROL', 'BOUNDARY'):
        raise ValueError('TWO_PREDECLARED_PHASE_BOUNDARY_ARMS_ONLY')
    if type(index_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', index_sha256):
        raise ValueError('EXACT_SAVED_D2_PHASE_INDEX_PIN')
    p = data_plan('D2',dataset_sha256,statistics_sha256,population_sha256,entries,steps=steps)
    if p['TRAIN_episodes']!=28 or p['TRAIN_windows']!=15932:
        raise ValueError('EXACT_ADMITTED_D2_NOT_D3_UNION')
    p.update(schema=PLAN_SCHEMA, pb_variant=variant, phase_index_schema=SCHEMA,
        phase_index_sha256=index_sha256,
        phase_slots_per_batch=dict(BOUNDARY_SLOTS if variant=='BOUNDARY' else CONTROL_SLOTS),
        sampling='DETERMINISTIC_EPISODE_BALANCED_PHASE_BOUNDARY_V1',
        phase_metadata_student_input=False,
        purpose='Fixed D2 phase control versus two uniform slots transferred to seven takeover first16 windows. Same b248, common statistics, M32, loss, continuous decoder and4096 schedule. Qualification is not a formal model.')
    return p


def is_plan(p):
    if p.get('schema')!=PLAN_SCHEMA:
        return False
    try:
        return p==plan(p['pb_variant'],p['dataset_manifest_sha256'],p['statistics_sha256'],
            p['TRAIN_population_sha256'],p['phase_index_sha256'],p['TRAIN_entries'],steps=p['max_steps'])
    except (KeyError,TypeError,ValueError,PermissionError):
        return False


def validate_population(manifest,population,p):
    from .book_bc_dagger import plan as data_plan, validate_population as validate_data
    if not is_plan(p):
        raise PermissionError('EXACT_FROZEN_PHASE_BOUNDARY_RECIPE')
    baseline=data_plan('D2',p['dataset_manifest_sha256'],p['statistics_sha256'],
        p['TRAIN_population_sha256'],p['TRAIN_entries'],steps=p['max_steps'])
    return validate_data(manifest,population,baseline)


def recipe_sequence(rows,index,p):
    if not is_plan(p):
        raise PermissionError('EXACT_FROZEN_PHASE_BOUNDARY_RECIPE')
    # The native10 probe uses the exact first320 draws of the4096 recipe;
    # it does not invent a shorter cosine or resample a qualification population.
    return sequence(rows,index,boundary=p['pb_variant']=='BOUNDARY',seed=p['seed'],
        index_sha256=p['phase_index_sha256'],dataset_sha256=p['dataset_manifest_sha256'],
        statistics_sha256=p['statistics_sha256'],population_sha256=p['TRAIN_population_sha256'])


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
