"""New model-only route remains inside original scope and measured GPU bounds."""
import copy
import pytest
from scripts.wm02 import train_fusion_program_response as entry
from go2arx5_vla.world_model import fusion_scope

GPU='GPU-845348f1-1511-dc19-0877-cc97658e3edd'


def fixture(monkeypatch):
    calls=[];monkeypatch.setattr(fusion_scope,'validate',lambda s:calls.append(s))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',GPU)
    spec=dict(kind=entry.KIND,profile=entry.PROFILE,execution_site=entry.SITE,recipe=entry.RECIPE,
        physics=False,WBC_changes=False,device='cuda',Actor_updates=0,GR00T_updates=0,GPU_uuid=GPU,
        allocator_limit_bytes=1024**3,endpoint=200,arms=list(entry.ARMS),hidden=32,batch_size=64,
        learning_rate=.001,weight_decay=.0001,numeric_mode='FP32_TF32_OFF',sampler='SEEDED_GLOBAL_SHUFFLE',
        seeds=[367401],primary_seed=367401,data_roles=['TRAIN'])
    return spec,dict(uuid=GPU,free_bytes=4*1024**3),calls


def test_real_original_scope_still_required_and_finite_endpoint_is_not_fixed200(monkeypatch):
    spec,actual,calls=fixture(monkeypatch)
    fusion_scope.validate_gpu(spec,actual)
    spec['endpoint']=2000;fusion_scope.validate_gpu(spec,actual)
    assert len(calls)==2


@pytest.mark.parametrize('fault',['physics','actor','device','headroom','roles','arms','sampler','endpoint','primary'])
def test_no_typo_or_config_flag_grants_extra_compute(monkeypatch,fault):
    spec,actual,calls=fixture(monkeypatch)
    if fault=='physics':spec['physics']=True
    elif fault=='actor':spec['Actor_updates']=1
    elif fault=='device':actual['uuid']='WRONG_GPU'
    elif fault=='headroom':actual['free_bytes']=2*1024**3
    elif fault=='roles':spec['data_roles']=['TRAIN','DEV']
    elif fault=='arms':spec['arms']=['SCALED']
    elif fault=='sampler':spec['sampler']='GROUPED'
    elif fault=='endpoint':spec['endpoint']=0
    elif fault=='primary':spec['primary_seed']=367402
    with pytest.raises(PermissionError):fusion_scope.validate_gpu(spec,actual)
    assert not calls
