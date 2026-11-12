"""Fixed SLA development inputs; no outcome-dependent population selection."""
import hashlib
import json

# Eight locations strictly inside the existing near-workspace support. None
# equals an existing TRAIN/DEV/qualification/recovery offset. Actual initial
# geometry identity must also be checked against the site's sealed catalogs.
DEV_OFFSETS = {f'SLADEV{i+1:02d}': (x, y) for i, (x, y) in enumerate(
    (x, y) for x in (-.0027, -.0009, .0009, .0027) for y in (-.0024, .0024))}
TRAIN_ORDER = ('TRAIN01', 'TRAIN07', 'TRAIN13', 'TRAIN25')
METHODS = ('S1_TARGET500_PLUS_ENDPOINT_BASIS', 'S2_TARGET500_PLUS_ENDPOINT_BASIS')


def initial_geometry(geometry):
    """Exclude explanatory role strings from the physical input identity."""
    keys = ('table_center_xyz_m', 'table_half_size_m', 'object_initial_xyz_m',
            'cube_edge_m', 'cube_mass_kg')
    return {k: geometry[k] for k in keys}


def geometry_sha(geometry):
    return hashlib.sha256(json.dumps(initial_geometry(geometry), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def declaration():
    from .task_learning_scene_v1 import geometry, scene_id
    from .task_learning_grasp_v1 import score_contract
    rows = []
    for i, layout in enumerate(DEV_OFFSETS):
        for restart in (1, 2):
            seed = 202611120000 + i*100 + restart*20
            rows.append(dict(layout=layout, restart=restart, parent=scene_id(layout),
                             split='SLA_DEV', geometry=initial_geometry(geometry(layout)),
                             initial_geometry_sha256=geometry_sha(geometry(layout)),
                             query_seeds=list(range(seed, seed+15))))
    return dict(schema='SLA_FIXED_DEV16_INPUTS_V1', status='DECLARED_BEFORE_CANDIDATE_OUTPUTS',
                population=rows, requested_per_method=16, methods=list(METHODS),
                method_order='S1_THEN_S2_FIXED_ENDPOINTS_NO_CHECKPOINT_SELECTION',
                same_seeds_do_not_imply_cross_model_same_noise=True,
                independent_from_old_DEV_TEST=True, old_TEST_scores_read=False,
                native_start_step=5703, physics_dt_s=.002, maximum_queries_per_condition=15,
                score_contract=score_contract(), WBC_changes=False, Teacher_fallback=False,
                clipping=False, retries=False, optimizer_updates=0, physical_requests=0)
