"""Diagnostic-reference identity seams, not actual native restore evidence."""
import copy
import pytest
from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
from go2arx5_vla.world_model import target_gr00t_core_v1 as core
from go2arx5_vla.world_model.task_learning_core_reference_v1 import identity
from test_target_gr00t_core_v1 import spec


def inputs():
    s=spec('RESTORE4','FULL_ACTION_SIDE',4)
    s.update(core_native_operation='FULL_ACTION_RESTORE_ONLY',generation_id='sealed-diagnostic4',
        parent={'path':'/original/corr'},bindings={'dataset':'/diagnostic4'},GPU_uuid='same-device')
    s['recipe']=profile.recipe('a'*64,'b'*64,mode='RESTORE4',variant='FULL_ACTION_SIDE',
        endpoint=4,memory_policy='CORE_FROZEN_BACKBONE_SWAP_NO_WEIGHT_CACHE_V5')
    c=dict(s['recipe'],**{k:s[k]for k in('parent','bindings','GPU_uuid')})
    h=dict(schema='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1',global_step=4,cursor=128,
        generation_id=s['generation_id'],parent_manifest_sha256=profile.PARENT,recipe=copy.deepcopy(s['recipe']))
    return s,h,c


def test_complete_original_diagnostic_identity_is_accepted_without_a_python_field_in_child_spec():
    s,h,c=inputs();identity(s,h,c)


@pytest.mark.parametrize('change',[{'global_step':2},{'cursor':64},{'generation_id':'other'},
    {'parent_manifest_sha256':'0'*64},{'schema':'PARTIAL_CHECKPOINT'}])
def test_partial_or_other_generation_reference_is_rejected(change):
    s,h,c=inputs()
    with pytest.raises(PermissionError,match='SEALED_COMPLETE'):identity(s,dict(h,**change),c)


@pytest.mark.parametrize('field',['dataset_sha256','statistics_sha256','memory_policy'])
def test_runtime_data_and_storage_policy_must_match(field):
    s,h,c=inputs();c[field]='changed'
    with pytest.raises(PermissionError,match='DIAGNOSTIC_ONLY'):identity(s,h,c)


def test_reference_cannot_be_consumed_by_legacy_or_formal_core_route():
    with pytest.raises(PermissionError,match='REFERENCE_ONLY'):
        core.full_restore_only({'sealed_continuous4_reference':{'path':'/old','sha256':'a'*64}})
    s,h,c=inputs();s['recipe']=dict(s['recipe'],mode='FORMAL',endpoint=2000)
    with pytest.raises(PermissionError):identity(s,h,c)
