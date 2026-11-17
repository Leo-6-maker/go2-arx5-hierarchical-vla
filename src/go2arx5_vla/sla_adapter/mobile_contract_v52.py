"""Pure V5.2 package identity; no learner, SciPy, JAX or model construction."""
from .gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
from .gripper_latent_v2 import VERSION as STATS_CODEC

PLAN='90d39196bdc345b977ee54ed64e750cf3561aa72'
PARENT='ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb'
RECIPE='SLA_MOBILE_SHIFT7_PARENT1500_FRESH_ADAM_ENDPOINT500_V52'
SCHEMA='SLA_MOBILE_SHIFT7_SUCCESSOR_RGB_PACKAGE_V52'
ORDER='INVERSE_FROZEN_QUANTILES_SIGMOID_PARAMETERS_REFERENCE_CONTINUED_ROBOT_TIME_QUINTIC'


def contract(m):
    b=m['bindings']
    if (m['schema']!=SCHEMA or m['codec']!=VERSION or m['decode_order']!=ORDER
            or m['step']!=500 or m['cursor']!=16000 or b['recipe']!=RECIPE
            or b['successor_parent_package']['sha256']!=PARENT
            or b['parent_cumulative_updates']!=1500 or b['parent_cumulative_visits']!=48000
            or b['successor_cumulative_updates']!=2000 or b['successor_cumulative_visits']!=64000
            or b['new_planned_updates']!=500 or b['new_planned_visits']!=16000
            or b['endpoint_lambda']not in(0.,.1)or b['endpoint_margin_m_s2']!=.2
            or b['endpoint_scale_m_s2']!=.25 or b['endpoint_tau_max']!=.5
            or b['goal_plan_sha']!=PLAN or not b['fresh_Adam']
            or b['codec_version']!=VERSION or b['statistics_codec_version']!=STATS_CODEC
            or b['inference_route']!='cached_actual' or m['parameter_slot_shift']!=SHIFT
            or m['parameter_slots']!=[[int(i)for i in g]for g in SLOTS]
            or m['previous_reference_required']is not True
            or any(m[k]is not False for k in ('teacher_required','optimizer_required','training_cache_required','physical_execution_admitted','endpoint_basis_applied'))):
        raise ValueError('EXPLICIT_FIXED_SHIFT7_SUCCESSOR_V52_LINEAGE_REQUIRED')
