"""Closed predecessor evidence must survive a future replacement unchanged."""
import json
from copy import deepcopy
import pytest
from scripts.pi0294.test_minimal_retry_custody import evidence
from scripts.pi0294.generalization import minimal_retry_history as history
from scripts.pi0294.generalization import minimal_execution as e


def test_capture_preserves_actual_old_terminal_without_mutation(evidence):
    before=deepcopy(evidence)
    packet=history.capture(**evidence)
    result=history.verify(json.loads(evidence['global_bytes']),packet)
    assert result['failed_request_id']==packet['original_row']['request_id'] and not result['native_admission']
    assert result['paid_output_bytes']==e.native_resources()['Hopper_storage_bytes']
    assert packet['original_row']['job_id']=='1501631.stdct-mgmt-02'
    assert evidence==before


@pytest.mark.parametrize('field',['row','marker','audit','inventory','cost','duplicate','old_row_change'])
def test_changed_history_is_rejected(evidence,field):
    g=json.loads(evidence['global_bytes']);packet=history.capture(**evidence)
    if field=='row':packet['original_row']['conservative_H200_s']+=1
    elif field=='marker':packet['original_marker']['terminal']['effects']['model_load_attempts']=1
    elif field=='audit':packet['independent_audit_sha256']='0'*64
    elif field=='inventory':packet['stable_worker_inventory_sha256']='0'*64
    elif field=='cost':packet['original_paid_output_bytes']-=1
    elif field=='duplicate':g['PBS_submissions'].append(deepcopy(packet['original_row']))
    else:
        next(r for r in g['PBS_submissions']if r.get('request_id')==packet['original_row']['request_id'])['conservative_CPU_s']-=1
    with pytest.raises(PermissionError):history.verify(g,packet)


def test_original_terminal_verified_when_future_package_is_active(evidence):
    packet=history.capture(**evidence);g=json.loads(evidence['global_bytes'])
    # A future package state is a CPU fixture, never an actual retry adoption.
    g[e.package.FIELD]['status']='MAIN_NATIVE_REGISTERED'
    before=deepcopy(g)
    result=history.verify(g,packet)
    assert not result['history_mutated'] and g==before
