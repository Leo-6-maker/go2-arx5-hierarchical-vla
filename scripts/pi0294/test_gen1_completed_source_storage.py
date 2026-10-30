"""Actual saved terminal accounting, no new model, physics or remote writes."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_pilot_scope as scope


@pytest.fixture
def actual():
    path=os.environ.get('GEN1_COMPLETED_SOURCE_GLOBAL_CAPTURE')
    if not path:pytest.skip('actual complete1024 GLOBAL capture required')
    capture=json.loads(Path(path).read_bytes())
    return json.loads(gzip.decompress(base64.b64decode(capture['after_GLOBAL_gzip_base64'])))


def test_actual_closed_bank_removes_only_future_source_storage(actual):
    before=deepcopy(actual);proof=scope.completed_source_storage(actual)
    assert proof['request_id']=='99eb0b5b-5f6a-4483-a00a-513ecc343311'
    assert proof['future_Core_storage_bytes']==21541945344
    assert proof['historical_costs_and_storage_uppers_retained']and not proof['native_admission']
    assert actual==before


@pytest.mark.parametrize('change',['incomplete','updates','audit','bank','unknown','source-terminal'])
def test_no_storage_credit_for_incomplete_or_inconsistent_history(actual,change):
    from scripts.pi0294.generalization import amd_training_accounting as train
    g=deepcopy(actual);row=train.rows(g)[-1]
    if change=='incomplete':
        row['scientific_endpoint_complete']=False
        assert scope.completed_source_storage(g)is None
        return
    if change=='updates':row['committed_optimizer_updates']=1023
    elif change=='audit':row['full_checkpoint_audit_sha256']='0'*64
    elif change=='bank':row['bank_manifest_sha256']='0'*64
    else:
        source=g[train.authority.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']
        if change=='unknown':source['actual']['unknown_native_effects']=['unreconciled']
        else:source['status']='RESERVED_TEN_UNATTEMPTED_ORIGINAL_CONDITIONS'
    with pytest.raises(PermissionError):scope.completed_source_storage(g)
