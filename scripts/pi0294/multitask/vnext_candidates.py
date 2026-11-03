"""Twelve fixed prospective TRAIN candidates, separate from frozen shared59.

Cards are not native admission or demonstrations. A rejected candidate is not
replaced. Recovery prefixes remain student data until a real safe handover.
"""
from copy import deepcopy
import math
import numpy as np

VERSION='TRAIN_VNEXT_FIXED12_MARGIN_CONTACT_GEOMETRY_V1'
BOOK_VARIANTS=((-0.01,-1.),(-0.02,-1.),(-0.01,-2.),(-0.02,-2.))
DOOR_OFFSETS=((-0.008,-0.006),(-0.008,0.006),(0.008,-0.006),(0.008,0.006))


def book_geometry_card(index):
    from .libero_book_teacher import route_card,RELEASE_REVISION
    if type(index)is not int or not 1<=index<=4:raise ValueError('FIXED_FOUR_BOOK_MARGIN_CANDIDATES')
    dx,degrees=BOOK_VARIANTS[index-1];card=route_card(RELEASE_REVISION)
    angle=math.radians(degrees);rotation=np.array([[math.cos(angle),-math.sin(angle),0],
        [math.sin(angle),math.cos(angle),0],[0,0,1.]])
    native=np.array(card['initial_book_native_world']);native[:3,:3]=rotation@native[:3,:3];native[0,3]+=dx
    card['initial_book_native_world']=native.tolist();card['source_base_card']['source_xy'][0]+=dx
    card.update(id=f'VNEXT_BOOK_MARGIN_TRAIN{index-1:03}',study_split='TRAIN',
        vnext_book_geometry_index=index,source_geometry_set=VERSION,
        initial_world_yaw_delta_deg=degrees,initial_translation_world_m=[dx,0.,0.],
        candidate_static_or_native_qualification=False)
    return card


def door_geometry_card(index):
    from .cabinet_braked_turn import cards
    if type(index)is not int or not 1<=index<=4:raise ValueError('FIXED_FOUR_DOOR_GEOMETRY_CANDIDATES')
    card=deepcopy(cards()['right']);offset=DOOR_OFFSETS[index-1]
    for key in ('cabinet','door'):
        field='cabinet_world'if key=='cabinet'else'hinge_world'
        for axis,value in enumerate(offset):card[key][field][axis][3]+=value
    for axis,value in enumerate(offset):card['source_base_card']['goal_xy'][axis]+=value
    card.update(id=f'VNEXT_DOOR_GEOMETRY_TRAIN{index-1:03}',study_split='TRAIN',
        vnext_door_geometry_index=index,source_geometry_set=VERSION,
        fixture_translation_world_m=[*offset,0.],candidate_static_or_native_qualification=False)
    return card


def recovery_conditions():
    # These are new TRAIN rollouts, not the already collected eight parent5–12
    # late recoveries. No event means no suffix. A model binding and remaining
    # route static qualification must precede an actual handover.
    choices=(('GRASP_CONFIRMED',0.),('LIFT_CARRY',.03),('LIFT_CARRY',.05),('NEAR_SUPPORT',.08))
    return [dict(id=f'VNEXT_BOOK_CONTACT_TRAIN{i:03}',task='BOOK',split='TRAIN',
        TRAIN_parent=f'BOOK_SOURCE_SEED{i+1:03}',event=event,threshold_m=height,
        source_kind='PROSPECTIVE_ACTUAL_TRAIN_STUDENT_PREFIX',
        student_prefix_is_expert=False,DEV_or_TEST_states=False,
        minimum_observed_grasp_seconds=.3,latest_takeover_index=480,
        takeover_only_original_source_Prefix8_boundaries=True,
        requires_saved_full_state_and_controller=True,requires_remaining_route_static_qualification=True,
        native_adapter_bound=False,training_admitted=False)
        for i,(event,height)in enumerate(choices)]


def candidates():
    return ([dict(id=book_geometry_card(i)['id'],task='BOOK',kind='JOINT_MARGIN_GEOMETRY',split='TRAIN',
                  card=book_geometry_card(i))for i in range(1,5)]
            +[dict(r,kind='CONTACT_EVENT_RECOVERY')for r in recovery_conditions()]
            +[dict(id=door_geometry_card(i)['id'],task='DOOR',kind='INDEPENDENT_GEOMETRY',split='TRAIN',
                   card=door_geometry_card(i))for i in range(1,5)])


def specification():
    return dict(schema=VERSION,candidates=candidates(),maximum_first_physical_attempts=12,
        maximum_attempts_per_candidate=1,replacement_seed_after_rejection=False,
        geometry_order=['maximum_minimum_all_six_joint_margin','maximum_swept_clearance','fixed_candidate_id'],
        original_safety_score_controller_unchanged=True,raw_or_parent_statistics_changed=False,
        added_to_shared59_D2_or_PB=False,student_DEV_is_TRAIN=False,
        output_population='TRAIN_VNEXT',candidate_is_demonstration=False,
        physics_integrations=0,GR00T_model_loads=0,qualified_demonstrations=0)
