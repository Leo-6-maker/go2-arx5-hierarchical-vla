import pytest
from go2arx5_vla.sla_adapter import provider_v2,provider_v3
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON,VERSION


def test_device_recipe_package_is_versioned_and_old_provider_rejects_it():
 m=dict(schema=provider_v3.SCHEMA,codec=VERSION,decode_order=provider_v3.DECODE_ORDER,label_epsilon=EPSILON,
  bindings=dict(codec_version=VERSION),state_gripper_transformed=False,teacher_required=False,
  training_cache_required=False,optimizer_required=False)
 assert provider_v3.SCHEMA!=provider_v2.SCHEMA
 provider_v3.contract(m)
 with pytest.raises(ValueError):provider_v2.contract(m)
 m['schema']=provider_v2.SCHEMA
 with pytest.raises(ValueError):provider_v3.contract(m)
