import pytest

from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON, VERSION
from go2arx5_vla.sla_adapter.provider_v2 import DECODE_ORDER, SCHEMA, contract


def manifest():
    return dict(schema=SCHEMA, codec=VERSION, decode_order=DECODE_ORDER, label_epsilon=EPSILON,
                bindings=dict(codec_version=VERSION), state_gripper_transformed=False,
                teacher_required=False, training_cache_required=False, optimizer_required=False)


def test_logit_package_has_distinct_version_and_single_decode_contract():
    contract(manifest())


@pytest.mark.parametrize('key,value', [('schema', 'SLA_ORDINARY_RGB_INFERENCE_PACKAGE_V1'),
                                      ('decode_order', 'SIGMOID_THEN_INVERSE_NORMALIZATION'),
                                      ('state_gripper_transformed', True), ('label_epsilon', 0.),
                                      ('teacher_required', True), ('optimizer_required', True)])
def test_incompatible_or_training_dependent_package_rejected(key, value):
    package = manifest(); package[key] = value
    with pytest.raises(ValueError):
        contract(package)
