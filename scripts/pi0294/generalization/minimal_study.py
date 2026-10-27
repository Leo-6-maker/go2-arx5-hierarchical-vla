"""Pure paired-parent and cursor contract; no dispatch or scientific execution."""
from collections import Counter, defaultdict
import hashlib
import json

UPSTREAM = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
R0 = 'fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7'
SCHEMA = 'ARX5_MINIMAL_SIX_PAIRED_GEOMETRY_TWELVE_PARENT_V1'
FAMILIES = ('F0', 'F1', 'F2')


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def paired_panel(manifest_bytes, *, expected_manifest_sha256, seed=20261019):
    need(hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha256,
         'EXACT_SAVED_ORIGINAL_MANIFEST_BYTES_REQUIRED')
    m = json.loads(manifest_bytes)
    need(m['schema'] == 'PI0294_FROZEN_MULTITASK_NATIVE_V1' and m['frozen'] is True
         and m['fixture'] is False and m['official_source_sha'] == UPSTREAM
         and m['statistics_sha256'] == R0 and m['RAW_DATA_READY'] is True,
         'ORIGINAL_FROZEN_DATA_R0_AND_OFFICIAL_SOURCE')
    need(type(seed) is int and seed >= 0, 'EXACT_PANEL_SELECTION_SEED')
    rows = m['episodes']
    need(len({r['id'] for r in rows}) == len(rows), 'NO_DUPLICATE_PARENT_ID')
    grouped = defaultdict(list)
    for r in rows:
        if (r['split'] == 'TRAIN' and r['origin'] == 'new' and r['first_attempt'] is True
                and r['id'].startswith('PI0294-TRAIN-') and r['windows'] == 601):
            grouped[r['group']].append(r)
    families = defaultdict(list)
    dev_ids = {r['id'] for r in rows if r['split'] != 'TRAIN'}
    dev_hashes = {r['sha256'] for r in rows if r['split'] != 'TRAIN'}
    dev_groups = {r['group'] for r in rows if r['split'] != 'TRAIN'}
    for group, pair in grouped.items():
        if len(pair) != 2 or {r['task'] for r in pair} != {'T1', 'T2'}:
            continue
        need(len({r['family'] for r in pair}) == 1, 'PAIRED_GEOMETRY_FAMILY_MATCH')
        family = pair[0]['family']
        need(family in FAMILIES, 'ORIGINAL_THREE_GEOMETRY_FAMILIES_ONLY')
        need(group not in dev_groups and all(r['id'] not in dev_ids and r['sha256'] not in dev_hashes
                                            for r in pair), 'NO_DEV_LINEAGE_OR_PAYLOAD_OVERLAP')
        families[family].append((group, sorted(pair, key=lambda r: r['task'])))
    need(set(families) == set(FAMILIES), 'THREE_FAMILIES_WITH_PAIRED_SUPPORT')
    for family in families:
        families[family].sort(key=lambda p: hashlib.sha256(f'{seed}:panel:{p[0]}'.encode()).digest())
    chosen = []
    # The actual original126 pool has F0/F1/F2 (not GEN1's four layouts).
    # Predeclared round-robin selects TWO paired geometries per family.
    for index in range(2):
        for family in FAMILIES:
            need(index < len(families[family]), 'ENOUGH_COMPLETE_PAIRS_FOR_FIXED_SELECTION_RULE')
            chosen.append(families[family][index])
            if len(chosen) == 6:
                break
        if len(chosen) == 6:
            break
    keys = ('id', 'group', 'task', 'family', 'windows', 'sha256', 'raw_FILE_MANIFEST_sha256',
            'samples_sha256', 'physics_sha256', 'source_sha')
    parents = [{key: r[key] for key in keys} for _, pair in chosen for r in pair]
    need(len({r['sha256'] for r in parents}) == 12, 'TWELVE_DISTINCT_SAVED_PARENT_PAYLOADS')
    return dict(schema=SCHEMA, original_manifest_sha256=expected_manifest_sha256, seed=seed,
                selection='SHA256_RANK_WITHIN_FAMILY_THEN_TWO_F0_F1_F2_ROUNDS',
                statistics_sha256=R0, statistics_refit=False, parents=parents,
                independent_geometry_count=6, parents_per_task={'T1': 6, 'T2': 6},
                overlapping_windows=sum(r['windows'] for r in parents),
                runtime_payload_revalidation='REQUIRED_BEFORE_NATIVE_MODEL_RELEASE',
                native_bank_qualified=False, native_adopted=False)


def paired_cursors(panel, *, stop=32768):
    need(panel['schema'] == SCHEMA and panel['overlapping_windows'] == 7212
         and len(panel['parents']) == 12 and type(stop) is int and 0 < stop <= 32768,
         'FIXED_TWELVE_PARENT_CURSOR_RANGE')
    windows = [(r, anchor) for r in panel['parents'] for anchor in range(r['windows'])]
    cursor = 0
    for epoch in range((stop + len(windows) - 1) // len(windows)):
        ordered = sorted(windows, key=lambda w: hashlib.sha256(
            f"{panel['seed']}:cursor:{epoch}:{w[0]['id']}:{w[1]}".encode()).digest())
        for row, anchor in ordered:
            if cursor >= stop:
                return
            yield dict(cursor=cursor, parent=row['id'], group=row['group'], task=row['task'], anchor=anchor)
            cursor += 1


def stream_summary(panel):
    rows = list(paired_cursors(panel))
    return dict(schema='ARX5_MATCHED_CURSOR_STREAM_SUMMARY_V1', panel_sha256=digest(panel),
                seed=panel['seed'], cursor_order='SHA256_WINDOW_PERMUTATIONS_NO_ARM_INPUT',
                scheduler_horizon=1024, fixed_pilot_step=512, batch=32,
                pilot_cursor=16384, conditional_terminal_cursor=32768,
                pilot_stream_sha256=digest(rows[:16384]), full_stream_sha256=digest(rows),
                pilot_exposure=dict(sorted(Counter(r['parent'] for r in rows[:16384]).items())),
                full_exposure=dict(sorted(Counter(r['parent'] for r in rows).items())),
                first_batch=rows[:32], consumed_by_native_training=0,
                optimizer_or_sampler_resume_reset_allowed=False)
