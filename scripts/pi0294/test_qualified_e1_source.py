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


@pytest.mark.parametrize('legacy_home_alias',[False,True])
def test_own_source_role_binds_real_metadata_without_editing_legacy_parent(tmp_path,legacy_home_alias):
    parent=tmp_path/'parent';root=parent/'control/source-only';root.mkdir(parents=True)
    def put(p,v):p.write_bytes(e.raw(v))
    put(parent/'RUN_MANIFEST.json',dict(legacy_parent=True))
    pl=parent/'PARENT_LEASE.json';put(pl,dict(status='ACTIVE_CONTROL_PLANE_ONLY'))
    legacy_path=pl
    if legacy_home_alias:
        alias=tmp_path/'legacy-home';alias.symlink_to(parent,target_is_directory=True)
        legacy_path=alias/pl.name
    put(parent/'OUTPUT_BINDING.private.json',dict(lease=str(legacy_path)));old=pl.read_bytes()
    m=dict(candidate=qualified.ARM,profile='vanda-arx5-qualified-E1-model-only-IO',canonical_parent=str(parent),
        control_root=str(root),run_uuid=root.name,parent_manifest_sha256=e.sha((parent/'RUN_MANIFEST.json').read_bytes()))
    body=e.raw(m);(root/'RUN_MANIFEST.json').write_bytes(body);(root/'CONTROL_RUN_MANIFEST.json').write_bytes(body)
    lease=parent/'SOURCE_LEASE.json';registry=parent/'RUN_REGISTRY.json';env=parent/'ENV.json';role='CPU_SOURCE_ROLE_FIXTURE'
    put(lease,dict(status='ACTIVE',manifest_sha256=e.sha(body),run_uuid=root.name,role=role,CPU_upper_s=900))
    put(registry,dict(entries=[dict(role=role,path=str(root),run_uuid=root.name)]))
    put(env,dict(profiles=[dict(name=m['profile'],role=role,manifest=str(root/'RUN_MANIFEST.json'),model_or_optimizer_on_login=False)]))
    put(root/'OUTPUT_BINDING.private.json',dict(run_root=str(root),run_manifest_sha256=e.sha(body),lease=str(lease),registry=str(registry),environment=str(env),role=role))
    qualified.source_control_role(root,body,canonical_parent=parent)
    assert pl.read_bytes()==old and 'manifest_sha256'not in json.loads(old)
    put(lease,dict(status='CLOSED',manifest_sha256=e.sha(body),run_uuid=root.name,role=role,CPU_upper_s=900))
    with pytest.raises(PermissionError):qualified.source_control_role(root,body,canonical_parent=parent)
    assert pl.read_bytes()==old


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
