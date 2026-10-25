"""Strict metadata-only scene specification; design hashes are not reset evidence."""
from copy import deepcopy
import hashlib
import json
import math
import re

from scripts.pi0294.scene_cards import INSTRUCTIONS, task_card

SCHEMA = 'GEN1_SCENE_SPEC_V1'
SPLITS = ('G_TRAIN', 'G_DEV', 'G_TEST')
VISUALS = {'V0': [.69, .70, .66, 1.], 'V1': [.72, .65, .58, 1.],
           'V2': [.55, .62, .70, 1.]}
LAYOUT_VERSION = 'GEN1_LEGACY_SMALL_LAYOUT_V1'
QUALIFICATION = dict(metadata='SCHEMA_ONLY', static_ik_collision='NOT_RUN',
                     actual_dual_rgb='NOT_RUN', dynamic_teacher='NOT_RUN',
                     raw_and_codec_audit='NOT_RUN', actual_reset='NOT_MEASURED')
FIELDS = {'schema', 'study_split', 'geometry_id', 'lineage_id',
          'source_population_sha256', 'layout', 'visual_profile', 'reset_profile',
          'camera_profile', 'tasks', 'physics_profile_sha256', 'codec', 'seeds',
          'qualification', 'design_hash', 'origin', 'scene_variant_id'}
LAYOUT_FIELDS = {'generator_version', 'source_xy_m', 'goal_xy_m', 'cube_yaw_rad',
                 'direction', 'distance_bin'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError('GEN1_'+reason)


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def seed(master, stream, identity):
    require(type(master) is int and 0 <= master < 2**64, 'UINT64_MASTER_SEED')
    require(stream in ('geometry', 'visual', 'reset'), 'NAMED_INDEPENDENT_SEED_STREAM')
    require(isinstance(identity, str) and 0 < len(identity) <= 256, 'SEED_IDENTITY')
    return int.from_bytes(hashlib.sha256(canonical([SCHEMA, master, stream, identity])).digest()[:8], 'big')


def validate_layout(layout, origin=None):
    require(type(layout) is dict and set(layout) == LAYOUT_FIELDS, 'EXACT_LAYOUT_FIELDS_AND_UNITS')
    for key in ('source_xy_m', 'goal_xy_m'):
        require(type(layout[key]) is list and len(layout[key]) == 2
                and all(number(x) for x in layout[key]), 'FINITE_PLANAR_METRES')
    source, goal = layout['source_xy_m'], layout['goal_xy_m']
    yaw = layout['cube_yaw_rad']
    require(number(yaw), 'FINITE_YAW_RADIANS')
    if layout['generator_version'] in ('COV1_V1', 'COV1_NEAR_015_018'):
        from scripts.pi0294.data_cov1.plan import validate_group, NEAR_REVISION
        require(type(origin) is dict and set(origin) == {'kind', 'group'}
                and origin['kind'] == 'PRESERVED_ORIGINAL_COV1_TRAIN', 'EXACT_ORIGINAL_COV1_GROUP')
        group = origin['group']
        validate_group(group)
        require(group['split'] == 'TRAIN', 'HISTORICAL_COV1_TRAIN_ONLY')
        version = 'COV1_NEAR_015_018' if group.get('geometry_revision') == NEAR_REVISION else 'COV1_V1'
        require(layout == dict(generator_version=version, source_xy_m=group['source_xy'],
                               goal_xy_m=group['goal_xy'], cube_yaw_rad=group['cube_yaw'],
                               direction=group['direction'], distance_bin=group['distance_bin']),
                'HISTORICAL_COORDINATES_REVISION_AND_BINS_UNCHANGED')
        return layout
    require(layout['generator_version'] == LAYOUT_VERSION and origin is None,
            'EXPLICIT_NEW_LAYOUT_PROFILE_WITHOUT_HISTORICAL_RELABELLING')
    require(number(yaw) and -.08 <= yaw <= .08, 'FROZEN_SMALL_YAW_RADIANS')
    gap = abs(goal[1]-source[1])
    require(.435 <= source[0] <= .465 and abs(goal[0]-source[0]) <= .006+1e-12
            and abs((source[1]+goal[1])/2) <= .012+1e-12
            and .16-1e-12 <= gap <= .20+1e-12, 'EXPLICIT_SMALL_LAYOUT_ENVELOPE')
    require(.15 <= math.dist(source, goal) <= .22, 'ORIGINAL_SCORE_DISTANCE_ENVELOPE')
    direction = '+Y' if goal[1] > source[1] else '-Y'
    # The signed Y gap, not Euclidean distance, defines this new profile's bin.
    distance_bin = 'NEAR' if gap <= .18 else 'FAR'
    require(layout['direction'] == direction and layout['distance_bin'] == distance_bin,
            'DERIVED_DIRECTION_AND_Y_GAP_BIN_NEAR_INCLUDES_018')
    return layout


def geometry_id(layout, origin=None):
    validate_layout(layout, origin)
    # Deliberately exclude appearance, task, split and seed from geometry identity.
    return 'GEN1/GEOM/'+digest({k: layout[k] for k in ('source_xy_m', 'goal_xy_m', 'cube_yaw_rad')})


def validate(spec):
    require(type(spec) is dict and set(spec) == FIELDS, 'EXACT_SPEC_FIELDS')
    require(spec['schema'] == SCHEMA and spec['study_split'] in SPLITS, 'SCHEMA_AND_STUDY_SPLIT')
    validate_layout(spec['layout'], spec['origin'])
    require(spec['geometry_id'] == geometry_id(spec['layout'], spec['origin']), 'CONTENT_BOUND_GEOMETRY_ID')
    require(spec['scene_variant_id'] == variant_id(spec), 'CONTENT_BOUND_SCENE_VARIANT_ID')
    require(isinstance(spec['lineage_id'], str) and 0 < len(spec['lineage_id']) <= 256,
            'EXPLICIT_LINEAGE_ID')
    if spec['origin'] is not None:
        require(spec['study_split'] == 'G_TRAIN'
                and spec['lineage_id'] == spec['origin']['group']['group_id'],
                'PRESERVED_TRAIN_LINEAGE_CANNOT_BECOME_DEV_OR_TEST')
    require(sha256(spec['source_population_sha256']) and sha256(spec['physics_profile_sha256']),
            'SOURCE_AND_FROZEN_PHYSICS_SHA256')
    require(spec['visual_profile'] in VISUALS, 'TABLE_COLOR_WHITELIST_ONLY')
    require(spec['reset_profile'] in ('LEGACY_ABOVE_SOURCE', 'GEN_COMMON_HOME'), 'RESET_PROFILE')
    require(spec['camera_profile'] == 'ORIGINAL_DUAL_RGB' and spec['codec'] == 'R0_EEF10_H40_PREFIX8',
            'FIXED_CAMERA_AND_CODEC')
    require(spec['tasks'] == ['T1', 'T2'], 'SAME_SCENE_BOTH_ORIGINAL_TASKS')
    require(type(spec['seeds']) is dict and set(spec['seeds']) == {'geometry', 'visual', 'reset'}
            and all(type(v) is int and 0 <= v < 2**64 for v in spec['seeds'].values()), 'UINT64_SEED_SUBSTREAMS')
    require(spec['qualification'] == QUALIFICATION, 'METADATA_CANNOT_CLAIM_NATIVE_QUALIFICATION')
    body = {k: v for k, v in spec.items() if k != 'design_hash'}
    require(spec['design_hash'] == digest(body), 'EXACT_DESIGN_HASH_NOT_RESET_DIGEST')
    return spec


def variant_id(spec):
    return 'GEN1/VARIANT/'+digest({k:spec[k] for k in
                                  ('geometry_id','visual_profile','reset_profile','camera_profile')})


def make_spec(layout, *, split, lineage_id, source_population_sha256,
              physics_profile_sha256, seeds, visual='V0', reset='LEGACY_ABOVE_SOURCE', origin=None):
    result = dict(schema=SCHEMA, study_split=split, geometry_id=geometry_id(layout, origin),
                  lineage_id=lineage_id, source_population_sha256=source_population_sha256,
                  layout=deepcopy(layout), visual_profile=visual, reset_profile=reset,
                  camera_profile='ORIGINAL_DUAL_RGB', tasks=['T1', 'T2'],
                  physics_profile_sha256=physics_profile_sha256, codec='R0_EEF10_H40_PREFIX8',
                  seeds=deepcopy(seeds), qualification=deepcopy(QUALIFICATION), origin=deepcopy(origin))
    result['scene_variant_id'] = variant_id(result)
    result['design_hash'] = digest(result)
    validate(result)
    return result


def import_cov1_train(group, *, source_population_sha256, physics_profile_sha256,
                     visual_master_seed, reset_master_seed, visual='V0'):
    """Retain a saved TRAIN geometry's version; no static/dynamic status is inherited.

    Applying GEN visual profiles is a new design variant, not an assertion that the
    original COV1 AP appearance has been reproduced or never seen in training.
    """
    from scripts.pi0294.data_cov1.plan import validate_group, NEAR_REVISION
    validate_group(group)
    identity = group['group_id']
    version = 'COV1_NEAR_015_018' if group.get('geometry_revision') == NEAR_REVISION else 'COV1_V1'
    layout = dict(generator_version=version, source_xy_m=group['source_xy'],
                  goal_xy_m=group['goal_xy'], cube_yaw_rad=group['cube_yaw'],
                  direction=group['direction'], distance_bin=group['distance_bin'])
    return make_spec(layout, split='G_TRAIN', lineage_id=identity,
                     source_population_sha256=source_population_sha256,
                     physics_profile_sha256=physics_profile_sha256,
                     seeds=dict(geometry=group['seed'], visual=seed(visual_master_seed, 'visual', identity),
                                reset=seed(reset_master_seed, 'reset', identity)), visual=visual,
                     origin=dict(kind='PRESERVED_ORIGINAL_COV1_TRAIN', group=deepcopy(group)))


def design_cards(spec):
    """Original task text/card conversion for design only, never native admission.

    COMMON_HOME has no actual calibration yet, so no guessed transform is emitted.
    These cards retain G_* identities rejected by the original PV1 native scope.
    No policy-input encoder is provided by this metadata module.
    """
    validate(spec)
    require(spec['reset_profile'] == 'LEGACY_ABOVE_SOURCE', 'COMMON_HOME_UNQUALIFIED')
    layout = spec['layout']
    group = dict(group_id=spec['scene_variant_id'],
                 scene_group=spec['scene_variant_id'], split=spec['study_split'],
                 family='GEN1', source_xy=deepcopy(layout['source_xy_m']),
                 goal_xy=deepcopy(layout['goal_xy_m']), cube_yaw=layout['cube_yaw_rad'],
                 visual=dict(table_rgba=deepcopy(VISUALS[spec['visual_profile']]),
                             floor_rgba=[.72, .74, .76, 1.], light_scale=1.,
                             body_camera_translation=[0., 0., 0.], body_camera_rpy=[0., 0., 0.],
                             wrist_camera_translation=[0., 0., 0.], wrist_camera_rpy=[0., 0., 0.]),
                 qualification='DESIGN_ONLY_NOT_NATIVE', native_execution_authorized=False)
    cards = [task_card(group, task) for task in spec['tasks']]
    require(all(c['instruction'] == INSTRUCTIONS[c['task']] for c in cards), 'ORIGINAL_TASK_LANGUAGE')
    return cards
