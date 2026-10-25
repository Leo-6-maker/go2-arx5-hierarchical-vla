"""Build inspectable CPU metadata from caller-supplied, hash-bound old indexes.

No command-line dispatcher is introduced. The caller saves TEST-containing output
in existing private storage; a public report contains only hashes and counts.
"""
from copy import deepcopy
from . import population as P, spec as S
from .validation import common_home_preparation


def build_metadata(config, *, train_groups, train_source_sha256, history, history_sources):
    require = S.require
    require(len(train_groups) == 12 and len({g['group_id'] for g in train_groups}) == 12,
            'EXACT_12_PRESERVED_TRAIN_GEOMETRIES')
    seed = config['seeds']
    rows = [S.import_cov1_train(g, source_population_sha256=train_source_sha256,
                              physics_profile_sha256=config['physics_profile_sha256'],
                              visual_master_seed=seed['visual'], reset_master_seed=seed['reset'])
            for g in sorted(train_groups,key=lambda g:g['group_id'])]
    pool = P.candidate_pool(config['new_slots'], geometry_master_seed=seed['geometry'],
                            candidates_per_slot=config['candidates_per_slot'])
    selection = P.select_metadata(pool, history)
    for candidate in selection['selected']:
        identity = candidate['slot_id']
        rows.append(S.make_spec(candidate['layout'], split=identity.split('/')[1], lineage_id=identity,
                                source_population_sha256=pool['pool_sha256'],
                                physics_profile_sha256=config['physics_profile_sha256'],
                                seeds=dict(geometry=candidate['geometry_seed'],
                                           visual=S.seed(seed['visual'],'visual',identity),
                                           reset=S.seed(seed['reset'],'reset',identity))))
    # Keep proposed factorial combinations separate from the V0 geometry manifest.
    # The all-history exposure audit has not established that any combination is unseen.
    combinations=[]
    for row in rows:
        for visual in config['visual_profiles']:
            key=(row['layout']['direction'],row['layout']['distance_bin'],visual)
            combinations.append(dict(geometry_id=row['geometry_id'],split=row['study_split'],
                                     direction=key[0],distance_bin=key[1],visual=visual,
                                     proposed_allowed=row['study_split']=='G_TEST' or key not in P.HOLDOUTS,
                                     unseen_qualified=False))
    report = P.audit_splits(rows, historical_groups=history, historical_source_sha256=history_sources)
    # Preserve rather than suppress a failed separation report.
    try:report['supplied_metadata_separation'] = P.require_separated_metadata(report)
    except ValueError as exc:report['supplied_metadata_separation'] = str(exc)
    return dict(manifest=rows, pool=pool, selection=selection, split_report=report,
                combination_table=combinations, reset_profile_report=common_home_preparation(),
                status='CPU_METADATA_PREPARATION_NOT_NATIVE_OR_FULL_EXCLUSION_QUALIFICATION',
                new_models=0, new_integrations=0, new_demonstrations=0)


def preview_design(train_specs):
    """Predeclare six TRAIN geometries before seeing any render, keeping failures.

    Original V1 NEAR candidates are retained even where the original scoring gate
    rejects them. Static preview is not a Teacher or successful-data qualification.
    The V1 preview images are diagnostics, not training examples or unseen evidence.
    """
    counts={'+Y/NEAR':2,'+Y/FAR':1,'-Y/NEAR':1,'-Y/FAR':2}
    selected=[]
    for stratum,count in counts.items():
        own=sorted((s for s in train_specs if s['study_split']=='G_TRAIN'
                    and '/'.join((s['layout']['direction'],s['layout']['distance_bin']))==stratum),
                   key=lambda s:s['lineage_id'])
        S.require(len(own)>=count,'SIX_PREVIEW_GEOMETRIES_IN_TRAIN')
        selected.extend(deepcopy(own[:count]))
    return dict(schema='GEN1_SIX_TRAIN_PREVIEW_DESIGN_V1',geometry_ids=[s['geometry_id'] for s in selected],
                lineage_ids=[s['lineage_id'] for s in selected],visuals=['V0','V1'],
                cameras=['body_rgb','wrist_rgb'],expected_initial_images=24,
                actual_images=0,actual_constructors=0,actual_integration_steps=0,
                status='NOT_RUN_REQUIRES_LEGAL_ISOLATED_NATIVE_SCOPE',
                native_execution_authorized=False,Teacher_pilot_authorized=False,
                images_admitted_to_training=False)
