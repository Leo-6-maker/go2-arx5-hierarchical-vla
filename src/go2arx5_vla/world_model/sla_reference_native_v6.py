"""Bind corrected-label endpoints to the unchanged raw-H40 native consumer."""
from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from ..sla_adapter.reference_dataset_v6 import (
    DATASET_ID,RECIPE,OPERATION,AUDIT_SCHEMA,ORIGINAL_MANIFEST_SHA)
from ..sla_adapter.reference_labels_v6 import VERSION

METHOD='N20_REFERENCE_FM3K_RAW_P10'
ADMISSION='SLA_REFERENCE_N20_FM3K_CORE_RGB16_NATIVE_ADMISSION_V6'
SCHEMA='SLA_REFERENCE_N20_RAW_P10_TRAIN8_PANEL_V6'
INPUT_SCHEMA='SLA_REFERENCE_N20_RAW_P10_TRAIN8_GROUP_INPUT_V6'
DECLARATION='experiments/sla_go2_posttrain/reference20_train8_inputs_v6.json'
EXPERIMENT='SAME_N20_OBSERVATIONS_REFERENCE_LABEL3K_RAW_TRAIN8_NOT_D100'


def endpoint(training,owner):
    task.need(training['schema']=='SLA_N20_REFERENCE_LABEL_FINITE_FM3K_ACTUAL_V6'
        and training['operation']==OPERATION and training['complete3K'] is True
        and training['noncandidate'] is False and training['reference_label_semantics']==VERSION
        and training['same_original20_observations'] is True
        and training['original_3K_control_retained'] is True
        and training['D100_new'] is training['D300_new'] is False
        and owner['schema']=='SLA_CORE_FRESH_REFERENCE_N20_FM3K_RGB_ACTUAL_V6'
        and owner['passes'] is owner['complete3K'] is owner['fresh_RGB16_bit_exact'] is True
        and (owner['step'],owner['target_visits'],owner['source_visits'])==(3000,96000,0)
        and owner['fresh_complete_episodes']==20,
        'REFERENCE_NEW_LABEL_FULL3K_NOT_OLD_N20_OR_EXTRA_OLD_UPDATES')
    common=pinned(owner['common_initial_actual'])
    task.need(common['passes'] is common['bit_exact'] is True
        and common['arrays_compared']==87 and common['optimizer_updates']==0,
        'REFERENCE_ACTUAL_ALL87_COMMON_MODEL_ADAM_RNG_INITIAL_ARRAYS')


def labels(audit,bindings,manifest_sha256):
    task.need(audit['schema']==AUDIT_SCHEMA and audit['passes'] is True
        and (audit['read_windows'],audit['episodes'],audit['full_H40_envelope_pass'],
             audit['FP32_decoded_envelope_pass'])==(1220,20,1220,1220)
        and audit['original_manifest_sha256']==ORIGINAL_MANIFEST_SHA
        and audit['label_semantics']==bindings['reference_label_semantics']==VERSION
        and audit['original_observations_and_states_unchanged'] is True
        and audit['all40_target_knots_actually_consumed'] is True
        and all(audit[k]==0 for k in ('model_loads','optimizer_updates','physics_integrations','DEV_TEST_reads'))
        and audit['RGB_requalified'] is False
        and bindings['original_observation_manifest_sha256']==ORIGINAL_MANIFEST_SHA
        and bindings['current_anchor_equivalent_consumed_reference_labels'] is True
        and bindings['dense_manifest_sha256']==manifest_sha256
        and bindings['old_2EP_reused']==bindings['old_engineering16_reused']==0
        and bindings['D300_new'] is False,
        'REFERENCE_ALL1220_CONSUMED_COMMAND_AND_FP32_GATE_PROOFS')
