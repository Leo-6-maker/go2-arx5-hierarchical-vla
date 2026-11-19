"""Immutable model-package contracts for both GPU providers and native preflight."""
from .dense_g7_v6 import VERSION
from .gripper_latent_v2 import EPSILON
from .provider_v3 import DECODE_ORDER

C20_MANIFEST = '4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de'
C20_STATISTICS = 'f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e'

D4_ARMS = ('D4-A', 'D4-B')
UPDATES = 500
VISITS = UPDATES * 32
D4_RECIPE = 'SLA_D4_C20_G7_MATCHED_TRANSFER_500'
D4_SCHEMA = 'SLA_D4_C20_G7_RGB_PACKAGE_V1'

D5_PLAN = 'cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb'
D5_RECIPE = 'SLA_D5_FROM_HISTORICAL_B_STEP_ZERO_G7'
D5_SCHEMA = 'SLA_D5_C20_G7_RGB_PACKAGE_V1'
D5_ARMS = ('D5-W', 'D5-T')
D5_STEP_ZERO = 'f9134574a0bf357d64852fc6d626eafb2e90d4de5b068d7d59354aa0969f0eaa'

J_PLAN = 'cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb'
J_RECIPE = 'SLA_J_C20_B500_ROBODOJO_H50_MATCHED_500'
J_SCHEMA = 'SLA_J_C20_G7_RGB_PACKAGE_V1'
J_ARMS = ('J0', 'J1')
J_B_PACKAGE = 'db2f4c31165dd442b712f286e8589ee2f86ff91f968fdcef1cd7e12d8f74e290'
J_B_CHECKPOINT = '13f70f1f4e7d9f42e535ebced0e3a0540dfbc12c13bdbf0795d9dcc2ecceb793'
J_SOURCE_CHECKPOINT = 'c08e919b5151ccafdfd5b427e02710f1a8e9f9189ca227314f72e5ec0296ab6d'
J_SOURCE_MANIFEST = '7954a6637564ad39a88f3e173c47687d66c39c50bbaccd9a2ad5f1a361667ab5'
J_SOURCE_BANK = 'a23fb4e46f86c5ea6f1d17748b913bcdcd5565cc1d36c2c04f5508a047b396af'
J_SOURCE_STATS = 'a94e1567bb002621d5841150a8e677e38fa6d02b51af22e4ab6f301317ebd521'
J_SOURCE_SEED = 2026111907
J_SOURCE_BATCH = 8
J_SOURCE_WEIGHT = .25


def d4_package_contract(manifest, *, schema=D4_SCHEMA, recipe=D4_RECIPE, arms=D4_ARMS, source_arm=D4_ARMS[1]):
    bindings = manifest['bindings']
    if (manifest['schema'] != schema or manifest['codec'] != VERSION
            or manifest['decode_order'] != DECODE_ORDER or manifest['label_epsilon'] != EPSILON
            or (manifest['step'], manifest['cursor']) != (UPDATES, VISITS)
            or len(manifest['adapter_arrays']) != 28 or bindings['recipe'] != recipe
            or bindings['arm'] not in arms or bindings['codec_version'] != VERSION
            or bindings['inference_route'] != 'cached_actual'
            or bindings['dataset_manifest_sha256'] != C20_MANIFEST
            or manifest['statistics_sha256'] != C20_STATISTICS
            or bindings['derived_statistics_sha256'] != C20_STATISTICS
            or not bindings['C20_only'] or bindings['Teacher_episodes']
            or bindings['source_optimization_visits']
            or bool(bindings['source_initialization']) != (bindings['arm'] == source_arm)
            or manifest['previous_reference_required'] is not True
            or any(manifest[key] is not False for key in ('teacher_required', 'training_cache_required',
                'optimizer_required', 'state_gripper_transformed', 'physical_execution_admitted',
                'endpoint_basis_applied'))):
        raise ValueError('D4_EXACT_FIXED_C20_RGB_PACKAGE_REQUIRED')


def d5_package_contract(manifest):
    d4_package_contract(manifest, schema=D5_SCHEMA, recipe=D5_RECIPE, arms=D5_ARMS, source_arm=None)
    binding = manifest['bindings']
    if (binding['goal_plan_sha'] != D5_PLAN or binding['historical_B_step_zero_sha256'] != D5_STEP_ZERO
            or binding['physical_width_lambda'] != 1. or binding['physical_delta_lambda'] != (binding['arm'] == D5_ARMS[1])):
        raise ValueError('D5_ENDPOINT_REQUIRES_THE_PREDECLARED_HISTORICAL_START_AND_LOSS')


def joint_package_contract(value):
    d4_package_contract(value, schema=J_SCHEMA, recipe=J_RECIPE, arms=J_ARMS, source_arm=None)
    b = value['bindings']
    batch = J_SOURCE_BATCH if b['arm'] == 'J1' else 0
    if (b['goal_plan_sha'] != J_PLAN or b['B500_package_sha256'] != J_B_PACKAGE
            or b['B500_checkpoint_sha256'] != J_B_CHECKPOINT or b['S1000_checkpoint_sha256'] != J_SOURCE_CHECKPOINT
            or b['joint_source_batch'] != batch or b['joint_source_weight'] != (J_SOURCE_WEIGHT if batch else 0.)
            or b['joint_source_rng_seed'] != J_SOURCE_SEED or b['joint_source_MANIFEST_sha256'] != J_SOURCE_MANIFEST
            or b['joint_source_statistics_sha256'] != J_SOURCE_STATS
            or b['joint_source_visit_bank_sha256'] != J_SOURCE_BANK or len(b['joint_source_frozen_leaves']) != 8):
        raise ValueError('FIXED_J_ENDPOINT_CONTRACT_REQUIRED')
