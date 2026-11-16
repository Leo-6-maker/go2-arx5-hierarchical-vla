import pytest
from go2arx5_vla.sla_adapter.provider_v3 import Provider as Legacy,SCHEMA as OLD_SCHEMA,DECODE_ORDER
from go2arx5_vla.sla_adapter.provider_time_v1 import Provider as Time,SCHEMA,ORDER
from go2arx5_vla.sla_adapter.gripper_time_v1 import VERSION
from go2arx5_vla.sla_adapter.gripper_latent_v2 import VERSION as OLD,EPSILON
from go2arx5_vla.sla_adapter.time_training_v1 import RECIPE


def test_time_and_legacy_packages_cannot_silently_select_each_others_decoder():
    legacy=dict(schema=OLD_SCHEMA,codec=OLD,decode_order=DECODE_ORDER,label_epsilon=EPSILON,
        bindings=dict(codec_version=OLD),state_gripper_transformed=False,teacher_required=False,
        optimizer_required=False,training_cache_required=False)
    timed=dict(schema=SCHEMA,codec=VERSION,decode_order=ORDER,step=500,cursor=16000,
        bindings=dict(recipe=RECIPE,codec_version=VERSION,statistics_codec_version=OLD,parent_step=1000,parent_visits=32000,new_planned_updates=500),
        previous_reference_required=True,teacher_required=False,optimizer_required=False,
        training_cache_required=False,physical_execution_admitted=False)
    a,b=Legacy.__new__(Legacy),Time.__new__(Time)
    a.validate_package_contract(legacy);b.validate_package_contract(timed)
    with pytest.raises(ValueError):a.validate_package_contract(timed)
    with pytest.raises(ValueError):b.validate_package_contract(legacy)
    with pytest.raises(ValueError):b.validate_package_contract(dict(timed,previous_reference_required=False))
