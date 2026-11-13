import copy
import pytest
from go2arx5_vla.sla_adapter.site_v2 import contract


def fixture():
 paths=dict(parent_params='/node/parent',dataset='/node/data',compilation_cache_dir='/node/cache')
 b=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host='node',uid=3,GPU_uuid='GPU-own',GPU_name='H200',
  GPU_headroom_bytes=10,allocator_limit_bytes=80,host_headroom_bytes=10,allocation_deadline_epoch=10000,
  paths=paths,allocation=dict(kind='PBS',job_id='1.server',project='project'))
 s=dict(GPU_uuid='GPU-own',max_walltime_s=7200,deadline_epoch=7000,max_gpu_hours=2,save_and_exit_on_limit=True,
  model_service_timeout_s=600,RPC_timeout_s=30,JAX_memory_fraction=.5,**paths)
 a=dict(host='node',uid=3,GPU_uuid='GPU-own',GPU_name='H200',CUDA_VISIBLE_DEVICES='GPU-own',GPU_free_bytes=10**12,
  GPU_total_bytes=100,host_available_bytes=100,paths=paths,cache_filesystem='ext4',PBS_JOBID='1.server',
  allocation_project='project',allocation_running=True)
 b['GPU_headroom_bytes']=1024**3
 return b,s,a


def test_actual_single_card_allocation_can_bind_finite_operation():
 b,s,a=fixture();contract(b,s,a,now=1)


@pytest.mark.parametrize('key,value',[('PBS_JOBID','another'),('allocation_project','other'),('allocation_running',False),
 ('cache_filesystem','nfs4'),('CUDA_VISIBLE_DEVICES','GPU-other')])
def test_wrong_owner_project_allocation_or_network_cache_rejected(key,value):
 b,s,a=fixture();a[key]=value
 with pytest.raises(PermissionError):contract(b,s,a,now=1)
