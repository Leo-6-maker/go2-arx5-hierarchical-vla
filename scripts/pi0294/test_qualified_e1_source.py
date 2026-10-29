"""Captured closed1506193 integration; no HF IO or new neural execution.

The private native corpus is deliberately not committed. Missing corpus is a
skip, never evidence that this native closure path ran in generic CI.
"""
import base64
from copy import deepcopy
import json
from pathlib import Path
import pytest

from scripts.pi0294.generalization import minimal_execution as e
from scripts.pi0294.generalization import minimal_inference_transfer as transfer
from scripts.pi0294.generalization import minimal_qualified_e1_source as qualified


@pytest.fixture(scope='module')
def actual():
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027'
    p=root/'QUALIFIED_E1_CLOSED512_INFERENCE_TRANSFER_PREPARED.private.json'
    s=root/'SAVED512_F1506193_QUALIFIED_E1_SMALL_SOURCE_CAPTURE.private.json'
    if not p.is_file()or not s.is_file():
        pytest.skip('exact private closed1506193 qualified E1 corpus required')
    plan=json.loads(p.read_bytes());c=plan['private_restoration']['closed_evidence']
    return dict(gb=base64.b64decode(c['global_base64']),lease=base64.b64decode(c['lease_base64']),
        settled=base64.b64decode(c['settled_base64']),observation=c['owner_observation'],
        small=json.loads(s.read_bytes())['decoded']['small'],plan=plan)


def prepare(v,*,arm=qualified.ARM):
    return transfer.prepare(v['gb'],v['lease'],v['settled'],v['observation'],
        e.raw(v['small']),arm=arm)


def test_actual_qualified_E1_prepares_while_failed_E2_stays_incomplete(actual):
    plan=prepare(actual)
    assert plan==actual['plan']
    assert plan['public_plan']['checkpoint_step']==512
    assert plan['public_plan']['payload_bytes']==12576325049
    g=json.loads(actual['gb']);_,row=qualified.execution.basic(g)
    assert row['scientific_endpoint_complete']is False
    assert row['committed_optimizer_updates']==122
    assert plan['HF_IO_performed']is False


@pytest.mark.parametrize('field',['audit','pair_base64','native_readback'])
def test_no_missing_evidence_can_promote_the_partial_allocation(actual,field):
    v=deepcopy(actual);del v['observation'][qualified.KEY][field]
    with pytest.raises((PermissionError,KeyError)):prepare(v)


@pytest.mark.parametrize('change',['tensor_audit','pair','live_owner','witness','closed_lease','old_owner'])
def test_actual_evidence_tamper_is_rejected(actual,change):
    v=deepcopy(actual);ev=v['observation'][qualified.KEY]
    if change=='tensor_audit':ev['audit']['saved_states'][qualified.SID]['global_step']=2
    elif change=='pair':ev['pair_base64']=base64.b64encode(b'{}').decode()
    elif change=='live_owner':ev['native_readback']['old_owner_and_wrapper_terminal']=False
    elif change=='witness':ev['native_readback']['all_original_witness_stats_unchanged']=False
    elif change=='closed_lease':v['lease']=e.raw(dict(status='ACTIVE'))
    else:ev['native_readback']['files']['old_owner_OS1']['sha256']='0'*64
    with pytest.raises(PermissionError):prepare(v)


def test_partial_E2_never_uses_E1_delivery_path(actual):
    with pytest.raises(PermissionError,match='DOES_NOT_ADMIT_PARTIAL_E2'):
        prepare(actual,arm='E2_SHARED_ACTION_HEAD')


def test_complete_saved_E1_without_new_inference_receipt_is_not_enough(actual):
    v=deepcopy(actual);v['observation'].pop(qualified.KEY)
    with pytest.raises(PermissionError):prepare(v)
