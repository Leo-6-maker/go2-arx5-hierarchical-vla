"""Finite offline candidates and transitive lineage audit, not a native population seal."""
from collections import Counter
from copy import deepcopy
import hashlib
import math

from scripts.pi0294.scene_cards import near_duplicate_report, normalise_layouts
from .spec import LAYOUT_VERSION, canonical, digest, require, seed, sha256, validate, validate_layout

STRATA = ('+Y/NEAR', '+Y/FAR', '-Y/NEAR', '-Y/FAR')
HISTORY_FAMILIES = ('legacy_training', 'legacy_confirmation', 'PV1_all_splits',
                    'COV1_all_versions', 'C1_parents_and_recovery', 'actual_training_and_replay')
HOLDOUTS = (('+Y', 'FAR', 'V2'), ('-Y', 'NEAR', 'V1'))


def _uniform(seed_value, field):
    bits = int.from_bytes(hashlib.sha256(canonical([seed_value, field])).digest()[:8], 'big') >> 11
    return bits / 2**53


def candidate_layout(master_seed, slot_id, stratum, candidate_index):
    require(stratum in STRATA, 'FIXED_FOUR_LAYOUT_STRATA')
    require(type(candidate_index) is int and 0 <= candidate_index < 32, 'FINITE_32_CANDIDATES_PER_SLOT')
    stream = seed(master_seed, 'geometry', slot_id+'/'+str(candidate_index))
    x = .435+.03*_uniform(stream, 'source_x')
    center = -.012+.024*_uniform(stream, 'center_y')
    dx = -.006+.012*_uniform(stream, 'goal_dx')
    direction, distance_bin = stratum.split('/')
    # Leave a declared 1e-12m endpoint margin for FAR to avoid rounding into NEAR.
    low = .16 if distance_bin == 'NEAR' else .18+1e-12
    high = .18 if distance_bin == 'NEAR' else .20
    gap = low+(high-low)*_uniform(stream, 'Y_gap')
    sign = 1 if direction == '+Y' else -1
    layout = dict(generator_version=LAYOUT_VERSION, source_xy_m=[x, center-sign*gap/2],
                  goal_xy_m=[x+dx, center+sign*gap/2], cube_yaw_rad=-.08+.16*_uniform(stream, 'yaw'),
                  direction=direction, distance_bin=distance_bin)
    validate_layout(layout)
    return dict(slot_id=slot_id, candidate_index=candidate_index, geometry_seed=stream, layout=layout)


def candidate_pool(slots, *, geometry_master_seed, candidates_per_slot=32):
    require(type(candidates_per_slot) is int and 1 <= candidates_per_slot <= 32, 'FINITE_CANDIDATE_COUNT')
    require(type(slots) is list and 0 < len(slots) <= 128, 'FINITE_SLOT_LIST')
    require(all(type(s) is dict and set(s) == {'slot_id', 'stratum'} for s in slots), 'EXACT_SLOT_FIELDS')
    names = [s['slot_id'] for s in slots]
    require(all(isinstance(n, str) and 0 < len(n) <= 200 for n in names)
            and len(set(names)) == len(names), 'UNIQUE_SLOT_IDENTITIES')
    rows = [candidate_layout(geometry_master_seed, s['slot_id'], s['stratum'], i)
            for s in sorted(slots, key=lambda s:s['slot_id']) for i in range(candidates_per_slot)]
    return dict(schema='GEN1_FINITE_OFFLINE_CANDIDATE_POOL_V1', candidates=rows,
                generator=LAYOUT_VERSION, geometry_master_seed=geometry_master_seed,
                candidates_per_slot=candidates_per_slot, pool_sha256=digest(rows),
                status='UNQUALIFIED_METADATA_POOL_NOT_EXECUTION_ORDER',
                outcome_dependent_redraw=False, native_execution_authorized=False)


def _group(spec):
    layout = spec['layout']
    return dict(group_id=spec['geometry_id'], split=spec['study_split'],
                source_xy=layout['source_xy_m'], goal_xy=layout['goal_xy_m'], cube_yaw=layout['cube_yaw_rad'])


def audit_splits(specs, *, historical_groups=(), historical_source_sha256=None):
    """Audit supplied metadata; source completeness must be independently established.

    Source hashes identify the caller's coverage claims, not proof that all historical
    exposures were supplied. This report never grants a new TEST, export or native run.
    """
    require(type(specs) is list and 0 < len(specs) <= 1024, 'BOUNDED_SPEC_LIST')
    for spec in specs:validate(spec)
    require(len({s['design_hash'] for s in specs}) == len(specs), 'NO_DUPLICATE_VARIANT_RECORDS')
    sources = deepcopy(historical_source_sha256 or {})
    require(type(sources) is dict and set(sources) <= set(HISTORY_FAMILIES)
            and all(type(v) is list and v and all(sha256(h) for h in v) for v in sources.values()),
            'EXPLICIT_HISTORY_SOURCE_HASHES')
    raw_history = list(historical_groups)
    for row in raw_history:
        g = row.get('card', row)
        require(all(type(g.get(k)) is list and len(g[k]) == 2 for k in ('source_xy', 'goal_xy'))
                and all(type(x) in (int, float) and math.isfinite(x)
                        for x in [*g['source_xy'], *g['goal_xy'], g.get('cube_yaw')]),
                'FINITE_HISTORY_GEOMETRY')
    history = normalise_layouts(raw_history)
    require(len(history) <= 20000, 'BOUNDED_HISTORY_GEOMETRY_INDEX')
    for g in history:
        require(len(g['source_xy']) == len(g['goal_xy']) == 2
                and all(type(x) in (int, float) and math.isfinite(x)
                        for x in [*g['source_xy'], *g['goal_xy'], g['cube_yaw']]), 'FINITE_HISTORY_GEOMETRY')
    groups = [_group(s) for s in specs]
    all_groups = groups + history
    parent = list(range(len(all_groups)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]];i = parent[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        if a != b:parent[max(a,b)] = min(a,b)
    edges = []
    for i, s in enumerate(specs):
        for j in range(i):
            reason = None
            if s['lineage_id'] == specs[j]['lineage_id']:reason = 'SHARED_PARENT_LINEAGE'
            elif s['geometry_id'] == specs[j]['geometry_id']:reason = 'SAME_GEOMETRY'
            elif near_duplicate_report([groups[i]], [groups[j]])['conflicts']:reason = 'ORIGINAL_2MM_2DEG_NEAR_DUPLICATE'
            if reason:
                union(i,j);edges.append(dict(left=s['design_hash'], right=specs[j]['design_hash'], reason=reason))
    # Historical nodes also participate: A--old-parent--B can cross a split even
    # when A and B themselves are farther apart than the original near bound.
    # A source-XY grid only narrows comparisons; the original full rule decides.
    buckets, identities = {}, {}
    for i, g in enumerate(all_groups):
        key = tuple(math.floor(x/.002) for x in g['source_xy'])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((key[0]+dx, key[1]+dy), ()):
                    if i < len(specs) and j < len(specs):continue
                    if near_duplicate_report([g], [all_groups[j]])['conflicts']:union(i, j)
        identity = specs[i]['lineage_id'] if i < len(specs) else raw_history[i-len(specs)].get('lineage_id', g['group_id'])
        if identity in identities:union(i, identities[identity])
        else:identities[identity] = i
        buckets.setdefault(key, []).append(i)
    components = {}
    for i, s in enumerate(specs):components.setdefault(root(i), []).append(s)
    component_rows = sorted([dict(members=sorted(s['design_hash'] for s in own),
                                  splits=sorted({s['study_split'] for s in own}),
                                  lineages=sorted({s['lineage_id'] for s in own}))
                             for own in components.values()], key=lambda c:c['members'])
    collisions = near_duplicate_report(groups, history)['conflicts']
    historical_collisions = [c for c in collisions if c['legacy']]
    held_out = [s['design_hash'] for s in specs
                if s['study_split'] != 'G_TEST'
                and (s['layout']['direction'], s['layout']['distance_bin'], s['visual_profile']) in HOLDOUTS]
    nearest = []
    train = [g for g in groups if g['split'] == 'G_TRAIN']
    for s, g in zip(specs, groups):
        if s['study_split'] == 'G_TRAIN':continue
        distances = [max(math.dist(g['source_xy'], t['source_xy']),
                         math.dist(g['goal_xy'], t['goal_xy'])) for t in train]
        nearest.append(dict(design_hash=s['design_hash'], split=s['study_split'],
                            min_max_source_goal_XY_m=min(distances) if distances else None,
                            not_an_OOD_or_independence_proof=True))
    missing = sorted(set(HISTORY_FAMILIES)-set(sources))
    return dict(schema='GEN1_METADATA_SPLIT_AUDIT_V1',
                status='EXCLUSION_INDEX_INCOMPLETE' if missing else 'HISTORY_COVERAGE_CLAIMS_REQUIRE_INDEPENDENT_READBACK',
                supplied_history_records=len(history), supplied_history_source_sha256=sources,
                missing_history_families=missing, history_completeness_qualified=False,
                lineage_components=component_rows, connection_edges=edges,
                cross_split_components=[c for c in component_rows if len(c['splits'])>1],
                historical_geometry_collisions=historical_collisions,
                proposed_holdout_leaks_into_TRAIN_or_DEV=sorted(held_out),
                unseen_combination_claim_qualified=False,
                actual_full_training_factor_exposure_checked=False,
                nearest_TRAIN_layout=sorted(nearest, key=lambda r:r['design_hash']),
                factor_counts=dict(sorted(Counter('/'.join((s['study_split'], s['layout']['direction'],
                                                           s['layout']['distance_bin'], s['visual_profile']))
                                                   for s in specs).items())),
                native_execution_authorized=False, new_TEST_qualified=False, training_export_authorized=False)


def require_separated_metadata(report):
    """A report is inspectable even on failure; consumers must call this gate."""
    require(report['schema'] == 'GEN1_METADATA_SPLIT_AUDIT_V1', 'SPLIT_REPORT_SCHEMA')
    require(not report['cross_split_components'], 'CROSS_SPLIT_LINEAGE')
    require(not report['proposed_holdout_leaks_into_TRAIN_or_DEV'], 'PROPOSED_HOLDOUT_LEAK')
    require(not any(c['split'] in ('G_DEV', 'G_TEST') for c in report['historical_geometry_collisions']),
            'DEV_TEST_HISTORICAL_GEOMETRY_COLLISION')
    return 'SUPPLIED_METADATA_SEPARATED_NOT_FULL_HISTORY_OR_NATIVE_QUALIFICATION'


def select_metadata(pool, historical_groups):
    """First metadata-separated candidate per slot, no engine/outcome screening.

    Exhaustion is preserved. This does not grant static qualification or dispatch.
    All candidates, including rejected candidates, remain in the original pool.
    """
    require(pool['schema'] == 'GEN1_FINITE_OFFLINE_CANDIDATE_POOL_V1'
            and pool['pool_sha256'] == digest(pool['candidates']), 'EXACT_FINITE_POOL')
    prior = normalise_layouts(historical_groups)
    chosen, rejected, exhausted = [], [], []
    slots = sorted({r['slot_id'] for r in pool['candidates']})
    for identity in slots:
        rows = sorted((r for r in pool['candidates'] if r['slot_id'] == identity),
                      key=lambda r:r['candidate_index'])
        for row in rows:
            layout = row['layout']
            validate_layout(layout)
            group = dict(group_id=identity, split=identity.split('/')[1],
                         source_xy=layout['source_xy_m'], goal_xy=layout['goal_xy_m'],
                         cube_yaw=layout['cube_yaw_rad'])
            require(group['split'] in ('G_DEV', 'G_TEST'), 'NEW_METADATA_DEV_TEST_ONLY')
            conflicts = near_duplicate_report([group], prior)['conflicts']
            if conflicts:
                rejected.append(dict(slot_id=identity, candidate_index=row['candidate_index'], conflicts=conflicts))
            else:
                chosen.append(deepcopy(row));prior.append(group);break
        else:exhausted.append(identity)
    return dict(selected=chosen, rejected=rejected, exhausted_slots=exhausted,
                status='METADATA_ONLY_NOT_STATIC_OR_NATIVE_QUALIFIED')
