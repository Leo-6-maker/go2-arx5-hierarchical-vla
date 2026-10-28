"""CPU preparation adapters. No renderer, collector, model or native admission."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.pi0291_arx5_data.config import PARAMETERS
from .spec import VISUALS, digest, require, validate


def physics_profile():
    # Parameter binding is distinct from the later compiled-model/asset audit.
    return dict(schema='GEN1_ORIGINAL_ARX5_PARAMETER_BINDING_V1',
                parameters=deepcopy(PARAMETERS), compiled_asset_qualification='NOT_MEASURED')


def load_config(path):
    """JSON subset of YAML: no custom YAML tags or dependency/environment change."""
    value = json.loads(Path(path).read_bytes())
    require(set(value) == {'schema', 'seeds', 'new_layout_profile', 'visual_profiles',
                          'reset_profiles', 'camera_profile', 'codec', 'physics_profile_sha256',
                          'candidates_per_slot', 'new_slots', 'proposed_holdouts', 'native_execution_authorized'},
            'EXACT_CONFIG_FIELDS')
    from .spec import LAYOUT_VERSION
    require(value['schema'] == 'GEN1_METADATA_CONFIG_V1'
            and value['new_layout_profile'] == LAYOUT_VERSION, 'CONFIG_PROFILE')
    require(value['visual_profiles'] == VISUALS, 'FIXED_TABLE_ONLY_PROFILES')
    require(value['reset_profiles'] == ['LEGACY_ABOVE_SOURCE', 'GEN_COMMON_HOME'], 'EXPLICIT_RESET_VERSIONS')
    require(value['camera_profile'] == 'ORIGINAL_DUAL_RGB'
            and value['codec'] == 'R0_EEF10_H40_PREFIX8', 'ORIGINAL_CAMERA_AND_CODEC')
    require(value['physics_profile_sha256'] == digest(physics_profile()), 'ORIGINAL_PHYSICS_PARAMETERS')
    require(type(value['seeds']) is dict and set(value['seeds']) == {'geometry','visual','reset'}
            and all(type(v) is int and 0 <= v < 2**64 for v in value['seeds'].values()), 'CONFIG_SEEDS')
    from .population import candidate_pool, HOLDOUTS
    from .population import STRATA
    expected=[dict(slot_id=f'GEN1/{split}/{stratum}/{i:02d}',stratum=stratum)
              for split in ('G_DEV','G_TEST') for stratum in STRATA for i in range(3)]
    require(sorted(value['new_slots'],key=lambda s:s['slot_id']) == sorted(expected,key=lambda s:s['slot_id']),
            'EXACT_12_DEV_12_TEST_METADATA_SLOTS')
    require(value['proposed_holdouts'] == [list(x) for x in HOLDOUTS], 'DECLARED_HOLDOUTS')
    require(value['native_execution_authorized'] is False, 'NO_NATIVE_AUTHORITY_FROM_CONFIG')
    candidate_pool(value['new_slots'], geometry_master_seed=value['seeds']['geometry'],
                   candidates_per_slot=value['candidates_per_slot'])
    return value


def common_home_preparation():
    """Three fixed TRAIN-only candidates, never selected using source/task/test.

    The rotation is the original downward convention. No card gets initial_tcp
    until original IK/FOV, the approach path and a valid scope are established.
    """
    from scripts.pi0291_arx5_data.scene import DOWNWARD_ROTATION
    candidates = []
    for i, y in enumerate((-.04, 0., .04)):
        transform = [[0.,0.,0.,0.] for _ in range(4)]
        for r in range(3):
            transform[r][:3] = DOWNWARD_ROTATION[r].tolist()
            transform[r][3] = [.45, y, PARAMETERS['table_top']+.10][r]
        transform[3][3] = 1.
        candidates.append(dict(id=f'GEN_COMMON_HOME_V1/H{i}', initial_tcp=transform))
    return dict(schema='GEN1_COMMON_HOME_PREPARATION_V1', status='COMMON_HOME_UNQUALIFIED',
                independent_reset_intervention=True, candidates=candidates,
                selection_rule='FIRST_COMMON_TRAIN_STATIC_IK_COLLISION_FOV_AND_APPROACH_PATH_PASS',
                selection_uses_TEST=False, selected=None, native_execution_authorized=False,
                actual_reset_state_digest=None, camera_calibration_digest=None, initial_RGB_sha256=None,
                dynamic_teacher='NOT_RUN', visual_localisation_generalisation_proven=False)


def require_training_export(spec):
    """An explicit fail-closed seam; metadata alone cannot call the old exporter."""
    validate(spec)
    require(spec['study_split'] == 'G_TRAIN', 'DEV_TEST_NEVER_TRAINING_EXPORT')
    raise PermissionError('GEN1_NATIVE_RAW_CODEC_SPLIT_ADMISSION_NOT_IMPLEMENTED')
