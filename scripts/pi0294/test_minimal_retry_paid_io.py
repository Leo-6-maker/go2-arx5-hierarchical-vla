"""Captured native inputs; candidate is not an allocation or credit."""
import json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.test_minimal_retry_custody import evidence
from scripts.pi0294.generalization import minimal_retry_paid_io as retry


@pytest.fixture
def packet(evidence):
    root=Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    wire=json.loads(sorted(root.glob('FIRST512_RETRY_BOTH_OLD_ROOTS_UNIQUE_INODE_READONLY_*private.json'))[-1].read_bytes())
    inv=json.loads(wire['stdout'])
    old=json.loads(json.loads(sorted(root.glob('FIRST512_NATIVE_COLD01_OLD_COLD_LEASE_INDEPENDENT_READONLY_*private.json'))[-1].read_bytes())['stdout'])
    return dict(**evidence,root_inventory=inv,old_cold_close_readback=old,new_request_id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        new_nonce='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',
        new_root='/test/children/arx5-minimal-raw-parent-first512-retry-01',now_epoch=inv['observed_epoch']+1)


def test_actual_candidate_preserves_gross_but_grants_nothing(packet):
    result=retry.candidate(**packet)
    assert result['shortfalls']=={} and result['old_costs_current_and_peak_unchanged']
    assert not result['actual_credit_granted'] and not result['native_registration_or_permission']
    assert not result['periodic_actual_physical_enforcement_implemented']


@pytest.mark.parametrize('change',['stale','too_large','model','old_request','same_nonce','wrong_root','old_lease'])
def test_unproven_or_cross_scope_credit_rejected(packet,change):
    p=deepcopy(packet)
    if change=='stale':p['now_epoch']+=121
    if change=='too_large':p['root_inventory']['unique_allocated_bytes']=2**30
    if change=='model':next(iter(p['root_inventory']['roots'].values()))['rows'].append({'path':'output/model.safetensors'})
    if change=='old_request':p['new_request_id']=p['native_readback']['native']['request_id']
    if change=='same_nonce':p['new_nonce']=p['native_readback']['native']['submission_nonce']
    if change=='wrong_root':p['new_root']='/different/root'
    if change=='old_lease':p['old_cold_close_readback']['status']='ACTIVE'
    with pytest.raises(PermissionError):retry.candidate(**p)
