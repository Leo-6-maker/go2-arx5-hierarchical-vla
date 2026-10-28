"""Actual saved audit/closure inputs; future own counters are CPU fixtures."""
import json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.generalization import minimal_remaining_contract as r,minimal_execution as e

@pytest.fixture(scope='module')
def actual_prefix():
    directory=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not directory:pytest.skip('actual full independent prefix RESULT and closed receipt required')
    v=json.loads((Path(directory)/'PARTIAL_COMPLETION01_FULL_RESULT_INDEPENDENT_EXACT_BYTES.private.json').read_bytes())
    return e.raw(v['result']),e.raw(v['independent_lease_close'])

def test_exact_native_prefix_reused_without_reauditing_or_model_permission(actual_prefix):
    p=r.saved_prefix(*actual_prefix)
    assert p['completed_sessions']==[s['id']for s in e.native_sessions()[:2]]
    assert p['old_paid_effects']==dict(optimizer_updates=2,generation_queries=71,model_loads=5)
    assert p['native_admission']is False

def test_new_remaining_totals_do_not_recharge_old_effects_or_credit_interrupted_load(actual_prefix):
    r.saved_prefix(*actual_prefix)
    rows={s['id']:dict(optimizer_updates=s['own_optimizer_limit'],generation_queries=s['own_query_limit'],model_loads=s['own_model_load_limit'])for s in r.sessions()}
    v=r.completed_totals(rows)
    assert v['own_native']==dict(optimizer_updates=1026,generation_queries=76,model_loads=7)
    assert v['actual_study_lifetime']==dict(optimizer_updates=1028,generation_queries=147,model_loads=12)
    assert not v['formal_endpoint_OR_native_success_proven']and len(r.tickets())==9

@pytest.mark.parametrize('phase,own,logical_queries',[
    ('MODEL_RAW',(0,0,0),15),('FIRST_UPDATE',(0,0,1),15),('RESTORE_FULL',(2,1,1),16)])
def test_E2_fifteen_carried_zeros_stay_outside_own_query_debits(phase,own,logical_queries):
    s=r.sessions()[0];v=r.check_phase(s['id'],phase,dict(zip(r.KEYS,own)))
    assert v['logical_session_counters']['generation_queries']==logical_queries
    assert v['own_counters']['generation_queries']==own[1]and not v['model_or_optimizer_permission']

@pytest.mark.parametrize('kind',['old_E0','old_E1','old17_as_new','old_load_credit','formal_from_step2','tampered_prefix','partial_final'])
def test_no_replay_double_charge_or_incomplete_completion(actual_prefix,kind):
    s=r.sessions()[0]
    with pytest.raises((PermissionError,KeyError)):
        if kind=='old_E0':r.session(e.native_sessions()[0]['id'])
        elif kind=='old_E1':r.session(e.native_sessions()[1]['id'])
        elif kind=='old17_as_new':r.logical_counters(s['id'],dict(optimizer_updates=2,generation_queries=17,model_loads=3))
        elif kind=='old_load_credit':r.check_phase(s['id'],'MODEL_RAW',dict(optimizer_updates=0,generation_queries=0,model_loads=1))
        elif kind=='formal_from_step2':r.check_phase(r.sessions()[1]['id'],'FIRST_UPDATE',dict(optimizer_updates=2,generation_queries=0,model_loads=1))
        elif kind=='tampered_prefix':r.saved_prefix(actual_prefix[0]+b' ',actual_prefix[1])
        else:
            rows={x['id']:dict(optimizer_updates=x['own_optimizer_limit'],generation_queries=x['own_query_limit'],model_loads=x['own_model_load_limit'])for x in r.sessions()}
            rows[s['id']]['model_loads']=2;r.completed_totals(rows)
