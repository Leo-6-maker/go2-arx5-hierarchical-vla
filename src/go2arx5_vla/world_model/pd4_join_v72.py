"""Fresh TRAIN4 system comparison for the fixed JOIN10 adapter, never raw score."""
from .pd4_student_v71 import (KEY, ENDPOINT, QUERIES, LAYOUTS, policy_contract,
    panel_contract, configuration, native_scene_admission, validate_scope,
    recipe_config, audit_prefix_predecessor, admitted, validate_live, validate_gpu)

VERSION = 'SLA_PD4_MATCHED500_JOIN10_P10_STUDENT_V72'
PROTOCOL = 'SLA_PD4_MATCHED500_JOIN10_P10_NATIVE_EVALUATION_V72'
NATIVE_SCHEMA = VERSION+'_NATIVE_ACTUAL'
EXPERIMENT = VERSION+'_TRAIN4_NOT_DEV_TEST'
SCOPE = VERSION+'_SIM_ONLY'
GPU_SITE = 'CORE_SLA_PD4_JOIN10_RGB_V72'
NATIVE_SITE = 'CORE_SLA_PD4_JOIN10_NATIVE_V72'


def protocol():
    from .sla_dense_native_v5 import protocol as dense
    from .sla_reference_join_v72 import configuration as join
    return dict(dense(), stage=PROTOCOL,
                scene_family='PD4_MATCHED_TRAIN4_NEW_RESTARTS_NOT_GENERALIZATION',
                policy='SLA_MATCHED500_RAW_INTENT_FIXED_JOIN10_REFERENCE',
                reference_connection=join())


def validate_entry(spec):
    from .stable_task_scene_v1 import pinned
    from .sla_reference_join_v72 import VERSION as consumer_version,configuration as join
    proof=pinned(spec['reference_join_qualification'])
    if (spec.get('reference_join10_v72')!=join()or proof['source_sha']!=spec['source_sha']
            or proof['schema']!=consumer_version+'_SAVED_QUERY_INTERFACE_QUALIFICATION'
            or not proof['passed']or len(proof['rows'])!=15
            or sum(r['candidate_compiled']for r in proof['rows'])!=14
            or any(proof[k]!=0 for k in ('optimizer_updates','training_visits','model_loads',
                                        'model_generations','certifier_calls','physical_requests','physics_integrations'))):
        raise PermissionError('SAME_SOURCE_SAVED15_INTERFACE_QUALIFICATION_REQUIRED')
