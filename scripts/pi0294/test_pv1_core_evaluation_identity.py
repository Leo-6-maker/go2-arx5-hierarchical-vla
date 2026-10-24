"""CPU OS parsers and strict physical role bindings; no GPU context created."""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from scripts.pi0294.pv1 import evaluation_core_identity as n,evaluation_core_accounting as c
from scripts.pi0294.pv1 import evaluation_platform as e,evaluation_stage as s

DEVICE='0, GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad, NVIDIA GeForce RTX 4080 SUPER, 00000000:02:00.0, 580.173.02, 32760, 481, 0\n1, GPU-845348f1-1511-dc19-0877-cc97658e3edd, NVIDIA GeForce RTX 4080 SUPER, 00000000:83:00.0, 580.173.02, 16376, 38, 0\n'
GPU='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'


def test_exact_native_32GIB_device_and_no_foreign_processes():
    v=n.parse_GPU_rows(DEVICE,'',GPU_uuid=GPU);assert v['index']==0 and v['total_bytes']==32760*1024**2
    assert n.parse_GPU_rows(DEVICE,GPU+', 123, python, 12000\n',GPU_uuid=GPU,own_pid=123)['GPU_uuid']==GPU
    with pytest.raises(PermissionError):n.parse_GPU_rows(DEVICE,'',GPU_uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd')
    with pytest.raises(PermissionError):n.parse_GPU_rows(DEVICE,GPU+', 124, python, 12000\n',GPU_uuid=GPU,own_pid=123)
    with pytest.raises(PermissionError):n.parse_GPU_rows(DEVICE,GPU+', 123, python, 12000\n',GPU_uuid=GPU)


@pytest.mark.parametrize('device,process',[
    ('',''),(DEVICE+DEVICE,''),(DEVICE.replace('RTX 4080 SUPER','A40'),''),
    (DEVICE.replace('32760','16376'),''),(DEVICE.replace('32760','N/A'),''),
    (DEVICE.replace('481, 0','40000, 0'),''),(DEVICE,GPU+', N/A, python, 1\n'),
    (DEVICE,'truncated_process_read')])
def test_missing_truncated_wrong_device_or_busy_reads_are_not_free_GPUs(device,process):
    with pytest.raises(PermissionError):n.parse_GPU_rows(device,process,GPU_uuid=GPU)


def test_real_cpu_memory_limits_not_advertised_metadata():
    assert n.kernel_limits('400000 100000','51539607552','32')['memory_max_bytes']==48*1024**3
    for cpu,memory,tasks in [('max 100000','51539607552','32'),('500000 100000','51539607552','32'),
        ('400000 100000','max','32'),('400000 100000','68719476736','32'),
        ('400000 100000','51539607552','max'),('400000 100000','51539607552','65')]:
        with pytest.raises(PermissionError):n.kernel_limits(cpu,memory,tasks)


def peers():
    inputs={s.PLAN:s.PLAN_SHA,s.CARD:s.CARD_SHA,'plans/V4_S2_GLOBAL_RESOURCE_AUTHORIZATION.json':c.scope.GRANT_SHA}
    gpu=dict(scope=c.ROLE,platform_evaluation_version=c.VERSION,execution=dict(profile=c.PROFILE,sealed_inputs=inputs))
    physical=dict(scope='PI0294_PV1_CORE_POLICY_EVALUATION_V1',evaluation_peer_platform=c.VERSION,
        execution=dict(profile='4080-simulation',sealed_inputs=inputs))
    return gpu,physical


def test_core_peers_use_real_process_handle_and_cannot_accept_PBS():
    gpu,physical=peers();handle=c.process_handle('12345678-1234-4234-8234-123456789abc',123,456)
    assert e.file_RPC_profile(gpu,'hopper')==c.PROFILE
    assert e.file_RPC_profile(physical,'core')=='4080-simulation'
    for role,m in [('hopper',gpu),('core',physical)]:
        pattern=e.fixture_job_pattern(m,role);assert re.fullmatch(pattern,handle)
        assert re.fullmatch(pattern,'1480000.stdct-mgmt-02') is None
        assert re.fullmatch(pattern,'1480000.hopper-m-02') is None
    for role,base,key,value in [('hopper',gpu,'scope',e.ROLE),('hopper',gpu,'scope','PI0294_PV1_SHARED_POLICY_HOPPER_V1'),
        ('core',physical,'platform_evaluation_version',c.VERSION),('core',physical,'scope',c.ROLE)]:
        m=deepcopy(base);m[key]=value
        with pytest.raises(PermissionError):e.fixture_job_pattern(m,role)
    bad=deepcopy(gpu);bad['execution']['sealed_inputs'][s.PLAN]='f'*64
    with pytest.raises(PermissionError):e.fixture_job_pattern(bad,'hopper')


def test_actual_captured_native_driver_and_kernel_limits():
    folder=Path(__file__).resolve().parents[4]/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01'
    gp=folder/'CORE_GPU_PHYSICAL_IDENTITIES.private.json';limits=folder/'ACTUAL_CORE_OS_LIMITS_CPU.private.json'
    if not gp.is_file() or not limits.is_file():pytest.skip('private original raw OS reads absent')
    raw=json.loads(gp.read_bytes())['value']['native_reads']
    assert raw['devices']['returncode']==raw['processes']['returncode']==0
    assert n.parse_GPU_rows(raw['devices']['stdout'],raw['processes']['stdout'],GPU_uuid=GPU)['total_bytes']>=30*1024**3
    result=json.loads(limits.read_bytes())['value'];assert result['returncode']==0
    r=result['value']['limits'];n.kernel_limits(r['cpu.max'],r['memory.max'],r['pids.max'])
    assert result['GPU_allocation'] is False
