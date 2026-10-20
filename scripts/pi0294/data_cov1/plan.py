"""Finite crossed-factor candidates, never Teacher-success-driven redraws."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.pi0294.scene_cards import digest, normalise_layouts, scene_group, task_card

PLAN_RELEASE = 'd4d59c39215daff061c19be3c5f514a015cab8eb'
CARD = 'reports/PI0294/design/DATA_COV1_STAGE_GATES_AND_PROPOSED_BUDGET_20261020.json'
CARD_SHA256 = 'f4fe904917008494c35fd79d1db198ad3180d464f6aa0d29b8c94af7d0bd26e1'
NAMESPACE = 'PI0294/DATA_COV1'
PROFILES = ('AP0','AP1','AP2')
DIRECTIONS = ('+Y','-Y')
DISTANCES = dict(NEAR=(.12,.15), FAR=(.19,.22))
NEAR_REVISION = 'NEAR_015_018_USER_AMENDMENT_V2'
AMENDMENT_CARD = 'reports/PI0294/design/DATA_COV1_NEAR_GEOMETRY_AMENDMENT_20261020.json'
AMENDMENT_CARD_SHA256 = 'cc1c95edb38ad8669451e3ea4e2ba97460371d2dd906d1bc5cfc99cb921effff'
STAGE_REPLICATES = dict(P24=2,P72=6,P144=12)


def protocol_card():
    path=Path(__file__).resolve().parents[3]/CARD
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CARD_SHA256:
        raise PermissionError('DATA_COV1_FIXED_PROTOCOL_CARD_CHANGED')
    return json.loads(raw)


def slot_id(split,appearance,direction,distance,replicate):
    maximum=12 if split=='TRAIN' else 2
    if (split not in ('TRAIN','DEV','LOCKBOX') or appearance not in PROFILES
            or direction not in DIRECTIONS or distance not in DISTANCES
            or type(replicate) is not int or not 0<=replicate<maximum):
        raise PermissionError('DATA_COV1_EXACT_FINITE_SLOT')
    return f'{NAMESPACE}/{split}/{appearance}/{direction}/{distance}/{replicate:03d}'


def slots(split='TRAIN'):
    return [dict(split=split,appearance_profile=ap,direction=direction,distance_bin=distance,replicate=rep)
        for rep in range(12 if split=='TRAIN' else 2)
        for ap in PROFILES for direction in DIRECTIONS for distance in DISTANCES]


def candidate(split,appearance_profile,direction,distance_bin,replicate,index,geometry_revision='V1'):
    identity=slot_id(split,appearance_profile,direction,distance_bin,replicate)
    if type(index) is not int or not 0<=index<32:
        raise PermissionError('DATA_COV1_MAXIMUM_32_PREFROZEN_STATIC_CANDIDATES')
    if geometry_revision not in ('V1',NEAR_REVISION):
        raise PermissionError('DATA_COV1_UNKNOWN_GEOMETRY_REVISION')
    seed_identity=f'DATA_COV1/V1/{split}/{appearance_profile}/{direction}/{distance_bin}/{replicate}/{index}'
    seed=int.from_bytes(hashlib.sha256(seed_identity.encode()).digest()[:8],'big')
    rng=np.random.Generator(np.random.PCG64(seed))
    x=float(rng.uniform(.425,.475));cy=float(rng.uniform(-.025,.025))
    limits=(.15,.18) if geometry_revision==NEAR_REVISION and distance_bin=='NEAR' else DISTANCES[distance_bin]
    dx=float(rng.uniform(-.012,.012));d=float(rng.uniform(*limits))
    dy=(1 if direction=='+Y' else -1)*np.sqrt(d*d-dx*dx)
    visual=deepcopy(scene_group('PILOT','F0',0)['visual'])
    if appearance_profile=='AP1':
        visual.update(table_rgba=[.60,.62,.66,1.],floor_rgba=[.70,.68,.65,1.],light_scale=1.)
    elif appearance_profile=='AP2':
        visual.update(table_rgba=[*rng.uniform(.50,.76,3).tolist(),1.],
            floor_rgba=[*rng.uniform(.54,.78,3).tolist(),1.],light_scale=float(rng.uniform(.8,1.2)))
    group=dict(group_id=identity,scene_group=identity,split=split,family='DATA_COV1',
        appearance_profile=appearance_profile,direction=direction,distance_bin=distance_bin,replicate=replicate,
        seed=seed,seed_identity=seed_identity,layout_candidate_index=index,
        source_xy=[x,float(cy-dy/2)],goal_xy=[x+dx,float(cy+dy/2)],
        cube_yaw=float(rng.uniform(-.16,.16)),table_top_m=scene_group('PILOT','F0',0)['table_top_m'],
        visual=visual,euclidean_distance_m=d,legacy_family_not_redefined=True,
        qualification='NOT_STATIC_OR_DYNAMIC_QUALIFIED')
    if geometry_revision==NEAR_REVISION and distance_bin=='NEAR':
        group['geometry_revision']=NEAR_REVISION
    validate_group(group)
    return group


def validate_group(group):
    expected=slot_id(group['split'],group['appearance_profile'],group['direction'],group['distance_bin'],group['replicate'])
    if group['group_id']!=expected or group['scene_group']!=expected or group['family']!='DATA_COV1':
        raise PermissionError('DATA_COV1_NAMESPACE_NOT_LEGACY_PV1')
    index=group['layout_candidate_index']
    if type(index) is not int or not 0<=index<32:
        raise PermissionError('DATA_COV1_FIXED_CANDIDATE_INDEX')
    source,goal=np.asarray(group['source_xy']),np.asarray(group['goal_xy'])
    if source.shape!=(2,) or goal.shape!=(2,) or not np.isfinite(np.r_[source,goal,group['cube_yaw']]).all():
        raise ValueError('DATA_COV1_FINITE_EXACT_PLANAR_GEOMETRY')
    revision=group.get('geometry_revision','V1')
    if revision not in ('V1',NEAR_REVISION):raise PermissionError('DATA_COV1_UNKNOWN_GEOMETRY_REVISION')
    low,high=(.15,.18) if revision==NEAR_REVISION and group['distance_bin']=='NEAR' else DISTANCES[group['distance_bin']]
    distance=float(np.linalg.norm(goal-source))
    if (not low-1e-12<=distance<=high+1e-12 or abs(distance-group['euclidean_distance_m'])>1e-12
            or not .425<=source[0]<=.475 or abs(goal[0]-source[0])>.012+1e-12
            or abs(float((source[1]+goal[1])/2))>.025+1e-12 or abs(group['cube_yaw'])>.16
            or (goal[1]>source[1])!=(group['direction']=='+Y')):
        raise ValueError('DATA_COV1_AUTHORIZED_ENVELOPE_NOT_Y_PROJECTION_DISTANCE')
    if any(group['visual'][camera+suffix]!=[0.,0.,0.] for camera in ('body','wrist')
            for suffix in ('_camera_translation','_camera_rpy')):
        raise PermissionError('DATA_COV1_CAMERA_CALIBRATION_FROZEN')


def candidate_pool(exclusions):
    prior=normalise_layouts(exclusions)
    population=[dict(**s,candidates=[candidate(**s,index=i) for i in range(32)])
        for split in ('TRAIN','DEV','LOCKBOX') for s in slots(split)]
    return dict(schema='PI0294_DATA_COV1_FINITE_POOL_V1',protocol_release=PLAN_RELEASE,
        protocol_card_sha256=CARD_SHA256,slots=population,candidates_per_slot=32,
        TRAIN_slots=144,DEV_metadata_slots=24,LOCKBOX_metadata_slots=24,
        exclusion_population_sha256=digest(prior),exclusion_records=len(prior),
        future_evaluation_metadata_only=True,future_evaluation_static_or_dynamic_execution=False,
        Teacher_or_student_success_screening=False,integrations=0,
        status='PREFREEZE_DESIGN_ONLY_NOT_NATIVE_QUALIFICATION_OR_AUTHORITY')


def amend_near_pool(original,*,approval_ref,parent_pool_sha256):
    """Append a new TRAIN design; do not rewrite old pools or redraw from outcomes."""
    if (not approval_ref or len(parent_pool_sha256)!=64
            or original['schema']!='PI0294_DATA_COV1_FINITE_POOL_V1'):
        raise PermissionError('DATA_COV1_EXPLICIT_NEAR_AMENDMENT_AND_ORIGINAL_POOL')
    value=deepcopy(original)
    for slot in value['slots']:
        if slot['split']=='TRAIN' and slot['distance_bin']=='NEAR':
            keys={k:slot[k] for k in ('split','appearance_profile','direction','distance_bin','replicate')}
            slot['candidates']=[candidate(**keys,index=i,geometry_revision=NEAR_REVISION) for i in range(32)]
    value.update(geometry_revision=NEAR_REVISION,parent_pool_sha256=parent_pool_sha256,
        geometry_amendment_approval_ref=approval_ref,original_attempts_preserved=True,
        physical_replacements=0,original_T0_and_budgets_unchanged=True,
        future_evaluation_metadata_revision='ORIGINAL_V1_METADATA_ONLY_NOT_ADOPTED_FOR_EVALUATION')
    return value


def preflight_score_geometry(card):
    """Reject impossible scorer inputs before spending constructors or physics."""
    distance=float(np.linalg.norm(np.asarray(card['goal_xy'])-np.asarray(card['source_xy'])))
    if not np.isfinite(distance) or distance<.15:
        raise PermissionError('DATA_COV1_FROZEN_SCORER_REQUIRES_SOURCE_GOAL_AT_LEAST_015M')
    return distance


def stage_groups(groups,stage):
    if stage not in STAGE_REPLICATES:
        raise PermissionError('DATA_COV1_FIXED_CUMULATIVE_STAGE')
    own=[g for g in groups if g['split']=='TRAIN' and g['replicate']<STAGE_REPLICATES[stage]]
    expected={slot_id(**dict(split=s['split'],appearance=s['appearance_profile'],direction=s['direction'],
        distance=s['distance_bin'],replicate=s['replicate'])) for s in slots() if s['replicate']<STAGE_REPLICATES[stage]}
    if len(own)!=len(expected) or {g['group_id'] for g in own}!=expected:
        raise PermissionError('DATA_COV1_STAGE_NEEDS_ALL_PREFROZEN_SLOTS')
    for g in own:validate_group(g)
    return own


def capture_cards(groups,stage):
    own=stage_groups(groups,stage)
    first=[slot_id('TRAIN','AP0','-Y','NEAR',0),slot_id('TRAIN','AP2','-Y','FAR',0)]
    own=sorted(own,key=lambda g:(first.index(g['group_id']) if g['group_id'] in first else 2,
        g['replicate'],g['appearance_profile'],g['direction'],g['distance_bin']))
    cards=[]
    for group in own:
        for task in ('T1','T2'):
            card=task_card(group,task)
            card.update(id=card['id']+'-C04',nominal_clearance_m=.004,
                origin='DATA_COV1_ACTUAL_ORIGINAL_TEACHER_WITH_ORIGINAL_S1')
            cards.append(card)
    return cards
