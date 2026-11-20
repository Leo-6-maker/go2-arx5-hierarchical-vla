"""G7 package identity shared by GPU inference and the NumPy-only consumer."""
from .dense_g7_v6 import VERSION
from .gripper_latent_v2 import EPSILON
from .provider_v3 import DECODE_ORDER

C20_STATISTICS = 'f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e'
SCHEMA = 'SLA_V6_G7_TARGET_RGB_PACKAGE'
RECIPE = 'SLA_V6_G7_SHARED_TARGET_FM1000_FULL_H40'
NATIVE_VERSION = 'MFREE_MULTITASK_NATIVE_V6'
NATIVE_METHOD = VERSION
ARMS = ('M-BASE', 'M-SRC')


def contract(package):
    bindings = package['bindings']
    if (package['schema'] != SCHEMA or package['codec'] != VERSION or package['decode_order'] != DECODE_ORDER
            or package['step'] != 1000 or package['cursor'] != 32000 or package['label_epsilon'] != EPSILON
            or bindings['recipe'] != RECIPE or bindings['codec_version'] != VERSION or len(package['adapter_arrays']) != 28
            or bindings['arm'] not in ARMS or bindings['inference_route'] != 'cached_actual'
            or package['previous_reference_required'] is not True
            or any(package[key] is not False for key in ('teacher_required','training_cache_required','optimizer_required',
                    'state_gripper_transformed','physical_execution_admitted','endpoint_basis_applied'))):
        raise ValueError('EXACT_FIXED1000_TEACHER_FREE_G7_PACKAGE_REQUIRED')
