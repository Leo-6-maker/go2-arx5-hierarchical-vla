"""Actual canonical Core bytes; no future model/process/source fixtures."""
import base64,gzip,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_core_d25_local_successor as f


@pytest.fixture(scope='module')
def actual_LOCAL():
    directory=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not directory:pytest.skip('actual canonical Core terminal LOCAL capture required')
    v=json.loads((Path(directory)/'CORE_TERMINAL_INVENTORY01_PREPARED_REFRESHED2.private.json').read_bytes())
    return json.loads(gzip.decompress(base64.b64decode(v['LOCAL_gzip_base64'])))


def test_actual_Core_context_consumes_own_table_without_scalar_GLOBAL_fabrication(actual_LOCAL):
    before=deepcopy(actual_LOCAL);marker=f.validated(actual_LOCAL)
    assert marker==actual_LOCAL[f.FIELD]and marker['replacement_request_id']=='3e774da8-1f75-4930-b732-3f0409f3d5a4'
    assert f.finite_control_scope(actual_LOCAL,marker['replacement_request_id'])['total_control_CPU_s']==9000
    assert actual_LOCAL==before and type(actual_LOCAL['CPU_s'])is dict and 'schema'not in actual_LOCAL


@pytest.mark.parametrize('change',['Hopper','CoreLab','local_PC','transfer','Core_output','local_output','other_campaign','wrong_type'])
def test_every_actual_LOCAL_invoice_remains_required(actual_LOCAL,change):
    v=deepcopy(actual_LOCAL);m=v[f.FIELD];b=m['financial_before']
    if change in('Hopper','CoreLab','local_PC'):v['CPU_s'][change]=b['local_CPU_s'][change]
    elif change=='transfer':v['network_transfer_accounted_bytes']=b['network_transfer_accounted_bytes']
    elif change=='Core_output':v['Core_native_GPU_current_storage_bytes']=b['Core_native_GPU_current_storage_bytes']
    elif change=='local_output':v['V5_CD_finite_control_local_storage_bytes'][m['replacement_request_id']]=0
    elif change=='other_campaign':v['run_uuid']='unrelated'
    else:v['CPU_s']=True
    with pytest.raises(PermissionError):f.validated(v)
