"""Actual saved terminal and current ledger bytes; no native dispatch."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_retry_custody as retry


@pytest.fixture
def evidence():
    root=Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    def latest(pattern):return json.loads(sorted(root.glob(pattern))[-1].read_bytes())
    pair=latest('D25_ORIGINAL_CORE_FRESH_[0-9]*private.json')
    native=json.loads(latest('FIRST512_FAILED_NATIVE_INDEPENDENT_AUDIT_READONLY_*private.json')['stdout'])
    closed=json.loads(latest('FIRST512_FAILED_FULL_CLOSURE_INDEPENDENT_READONLY_*private.json')['stdout'])
    return dict(global_bytes=gzip.decompress(base64.b64decode(pair['GLOBAL_gzip_base64'])),
        local_bytes=gzip.decompress(base64.b64decode(pair['LOCAL_gzip_base64'])),native_readback=native,closed_readback=closed)


def test_actual_closed_zero_effect_still_does_not_authorize_retry(evidence):
    result=retry.inspect(**evidence)
    assert result['job_id']=='1501631.stdct-mgmt-02' and result['retained_worker_allocated_bytes']>0
    assert result['gross_duplicate_charge_shortfalls']['Hopper_storage_bytes']>0
    assert not result['exact_reused_IO_credit_granted'] and not result['native_dispatch_admitted']


@pytest.mark.parametrize('change',['audit','inventory','receipt','unknown'])
def test_no_credit_from_changed_evidence_or_unknown(evidence,change):
    v=deepcopy(evidence)
    if change=='audit':v['native_readback']['audit']['new_model_loads']=1
    if change=='inventory':v['native_readback']['inventory']['worker_inventory_sha256']='0'*64
    if change=='receipt':v['closed_readback']['amounts']['GPU_s']+=1
    if change=='unknown':v['closed_readback']['unknown_effects']=['unknown']
    with pytest.raises(PermissionError):retry.inspect(**v)
