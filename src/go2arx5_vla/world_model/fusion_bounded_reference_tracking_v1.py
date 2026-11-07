"""Mechanical full-proposal screening, independent of semantic task success.

This new diagnostic can examine a bounded command with a failed task direction.
It never changes the strict short-task score, certifies joint safety, or grants
execution. Every original per-install certificate and consumption check remains.
"""
from .fusion_graded_short_skill_v1 import screen_proposal

TASK_CONDITION='BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'
TASK_ID='FUSION_BOUNDED_FULL_H40_MECHANICAL_TRACKING_AT34303_V1'


def domain_contract():
    return dict(schema='WM02_BOUNDED_FULL_H40_MECHANICAL_DOMAIN_V1',
        maximum_TCP_excursion_m=.025,maximum_TCP_rotation_rad=.1,
        minimum_finger_opening_m=.038,minimum_TCP_cube_distance_m=.14,
        raw_knots=40,geometry_scale=1.,task_direction_is_an_outcome_not_an_admission_gate=True,
        task_success_claim=False,native_safety_thresholds_changed=False)


def screen(program,cube):
    strict=screen_proposal(program,cube)
    domain=domain_contract()
    metrics=dict(strict['metrics'])
    metrics['maximum_TCP_rotation_rad']=float(program.evidence()['whole_desired_path']['max_anchor_rotation_rad'])
    predicates=dict(excursion=metrics['maximum_TCP_excursion_m']<=domain['maximum_TCP_excursion_m'],
        orientation=metrics['maximum_TCP_rotation_rad']<=domain['maximum_TCP_rotation_rad'],
        open=metrics['minimum_finger_opening_m']>=domain['minimum_finger_opening_m'],
        noncontact_geometry=metrics['minimum_desired_or_reference_cube_distance_m']>=domain['minimum_TCP_cube_distance_m'])
    return dict(schema='WM02_BOUNDED_FULL_H40_MECHANICAL_SCREEN_V1',task_id=TASK_ID,
        domain_contract=domain,metrics=metrics,predicates=predicates,
        strict_short_proposal_screen=strict,strict_short_task_success=False,
        eligible_for_original_native_certificate=all(predicates.values()),
        native_execution_authorized=False,native_safety_established=False,
        role='MECHANICAL_CONSUMPTION_QUALIFICATION_NOT_SEMANTIC_SKILL_OR_WM_BENEFIT')
