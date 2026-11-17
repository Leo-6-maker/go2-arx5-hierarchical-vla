"""Pure V6 target package contract; no learner or model construction."""
PLAN='ae3d3c0dceade437b0104a150cd0402fef650052'
RECIPE='SLA_B1_PARENT2000_FRESH_ADAM_FULL_GENERATED_H40_V6'
B1='b9ba974c374b3c3a09d7742004fc799337d5e7f1b9c9c7d290af066766e4706c'
SCHEMA='SLA_V6_B1_PARENT_FULL_H40_RGB_PACKAGE'


def contract(m):
    from .gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
    ORDER='INVERSE_FROZEN_QUANTILES_SIGMOID_PARAMETERS_REFERENCE_CONTINUED_ROBOT_TIME_QUINTIC'
    b=m['bindings']
    if(m['schema']!=SCHEMA or m['codec']!=VERSION or m['decode_order']!=ORDER
       or m['step']!=500 or m['cursor']!=16000 or m['parameter_slot_shift']!=SHIFT
       or m['parameter_slots']!=[[int(i)for i in group]for group in SLOTS]
       or b['recipe']!=RECIPE or b['goal_plan_sha']!=PLAN or b['B1_parent_package']['sha256']!=B1
       or b['parent_cumulative_updates']!=2000 or b['parent_cumulative_visits']!=64000
       or b['successor_cumulative_updates']!=2500 or b['successor_cumulative_visits']!=80000
       or b['full_H40_lambda']not in(0.,.1)or b['full_H40_margin']!=.2 or b['full_H40_scale']!=.25
       or b['full_sampling_steps']!=10 or b['aux_samples_per_update']!=4 or b['clean_proxy']is not False
       or b['inference_route']!='cached_actual' or m['previous_reference_required']is not True
       or any(m[k]is not False for k in('teacher_required','optimizer_required','training_cache_required','physical_execution_admitted','endpoint_basis_applied'))
       or m['statistics_sha256']!='f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e'):
        raise ValueError('EXPLICIT_V6_B1_PARENT_FULL_H40_PACKAGE_REQUIRED')
