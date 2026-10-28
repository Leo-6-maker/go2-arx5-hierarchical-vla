"""C/D24 CPU evidence fixtures reuse every original raw/causal audit check."""
from copy import deepcopy
import json

import pytest

from scripts.pi0294.test_pv1_baseline import fixtures,raw,groups
from scripts.pi0294.pv1 import baseline as b
from scripts.pi0294.dec1.authority import BANK

ENDPOINT='1'*64


def evidence(arm):
    result=[]
    for item in fixtures():
        v={k:json.loads(value)for k,value in item.items()}
        row=v['adjudication']
        if row['noise_bank']!=294101:continue
        identifier=arm+'-'+row['attempt_id'].removeprefix('B0_CORR-')
        row.update(attempt_id=identifier,model=arm)
        v['result']['id']=identifier;row['result_sha256']=b.hashlib.sha256(raw(v['result'])).hexdigest()
        v['audit'].update(attempt_id=identifier,adjudication=deepcopy(row))
        ready=v['model_ready'];ready['label']=arm+'-DEV-N294101'
        ready['identity'].update(label=arm,global_step=1024,checkpoint_manifest_sha256=ENDPOINT)
        v['causal_chain'].update(attempt_id=identifier,adjudication_sha256=b.hashlib.sha256(raw(row)).hexdigest(),
            identity=deepcopy(ready['identity']))
        result.append({k:raw(x)for k,x in v.items()})
    return result


@pytest.fixture
def population(monkeypatch):
    value=raw(dict(groups=groups()));monkeypatch.setattr(b,'POPULATION',b.hashlib.sha256(value).hexdigest());return value


@pytest.mark.parametrize('arm',['C50','D25'])
def test_all_twenty_four_valid_failures_stay_valid_without_success_gate_or_training_GO(population,arm):
    values=evidence(arm);v=b.partial_history(population,candidate=arm,evidence=values,checkpoint_manifest_sha256=ENDPOINT)
    assert v['valid_conditions']==24 and v['complete_DEV24']and v['successes']==dict(T1=0,T2=0)
    assert not v['training_or_selection_admission']and not v['native_or_resource_admission']
    assert len(v['all_actual_attempts'])==24 and len(v['remaining_panels'])==1
    for bad in (values+[values[0]],):
        with pytest.raises(PermissionError):b.partial_history(population,candidate=arm,evidence=bad,checkpoint_manifest_sha256=ENDPOINT)
    with pytest.raises(PermissionError):b.admission(population,candidate=arm,evidence=values,
        checkpoint_manifest_sha256=ENDPOINT,route='COVERAGE_FIRST',new_bank_manifest_sha256=BANK)


@pytest.mark.parametrize('case',range(8))
def test_foreign_step_panel_missing_causal_or_changed_native_safety_rejected(population,case):
    values=evidence('C50');item={k:json.loads(v)for k,v in values[0].items()}
    if case==0:item['model_ready']['identity']['global_step']=4096
    if case==1:item['model_ready']['identity']['checkpoint_manifest_sha256']='2'*64
    if case==2:item['model_ready']['native_pre_model_qualification']['transport_cycles']=79
    if case==3:item['adjudication']['live_native_failures']=1
    if case==4:item['adjudication']['noise_bank']=294102
    if case==5:item['audit']['checks']['raw_manifest_all_sizes_and_hashes']=False
    if case==6:item['causal_chain']['executed_ACKs']=79
    if case==7:item.pop('causal_chain')
    values[0]={k:raw(v)for k,v in item.items()}
    with pytest.raises(PermissionError):b.partial_history(population,candidate='C50',evidence=values,checkpoint_manifest_sha256=ENDPOINT)


def test_no_unbound_endpoint_or_adaptive_partial_subset_claims_complete(population):
    with pytest.raises(PermissionError):b.partial_history(population,candidate='C50',evidence=[])
    value=b.partial_history(population,candidate='C50',evidence=evidence('C50')[:12],checkpoint_manifest_sha256=ENDPOINT)
    assert value['valid_conditions']==12 and not value['complete_DEV24']and value['remaining_eligible_conditions']==12
