"""Captured actual-ledger escrow arithmetic; never native or model dispatch."""
import base64
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_package as p, minimal_scope as s, minimal_adoption as a
from scripts.pi0294.pv1.neural_ledger import check
from scripts.pi0294.pv1.resources import pbs_limit
from scripts.pi0294.dec1 import source_cpu_once_io as once


@pytest.fixture
def captured():
    name=os.environ.get('ARX5_FIRST512_FORECAST_INPUTS')
    if not name:pytest.skip('Private actual cap-adoption capture explicitly required')
    root=Path(name);v=json.loads((root/'FIRST512_SEVEN_CAPS_INDEPENDENT_READBACK.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(v['GLOBAL_gzip_base64'],validate=True))
    lb=(root/'FIRST512_HISTORICAL_LOCAL_BYTES.private.json').read_bytes()
    assert a.sha(gb)=='d3e48b1148987b3b5279d258d39f10552606db3d27ab77e6d5ff9aa6bcd07ec4'
    return gb,lb


def reserved(captured):return p.reserve(*captured,now_epoch=1793094000.)


def test_complete_package_reserves_once_and_preserves_old_spent_rows_and_original_escrow(captured):
    gb,lb=captured;before=json.loads(gb);result=reserved(captured);after=result['proposed_global'];v=after[p.FIELD]
    assert set(after)-set(before)=={p.FIELD}
    assert all(after[k]==x for k,x in before.items()if k not in ('revision','reserved','CPU_reserved_s','storage_reserved_bytes'))
    assert after['revision']==before['revision']+1
    for key in p.TRACKED:assert after['reserved'][key]==before['reserved'][key]+v['original_resources'][key]
    assert after['CPU_reserved_s']==before['CPU_reserved_s']+v['original_resources']['Hopper_CPU_s']
    assert after['storage_reserved_bytes']==before['storage_reserved_bytes']+v['original_resources']['Hopper_storage_bytes']
    assert before['used']==after['used'] and before['PBS_submissions']==after['PBS_submissions']
    assert not result['actual_LOCAL_changed'] and not result['native_admission']
    assert result['new_requests']==result['new_GPU']==result['new_PBS']==0
    assert v['remaining_resources']['optimizer_updates']==1028
    assert not result['full_future_proof']['cap_changes_required']
    assert check(after,allow_unknown=True)==after


def test_reservation_credit_respects_higher_historical_Core_GPU_floor(captured):
    gb,lb=captured;before=s.first512_forecast(gb,lb);result=reserved(captured);after=result['full_future_proof']
    assert after['scope_adopted'] and after['LOCAL_is_historical']
    assert after['dimensions']['H200_s']['exact_first512_reservation_projection_credit']==91980
    assert after['dimensions']['H200_s']['exact_first512_reservation_projection_credit']<p.resources()['H200_s']
    for key in before['dimensions']:
        assert after['dimensions'][key]['projected']==pytest.approx(before['dimensions'][key]['projected'],rel=0,abs=1e-7)
    assert after['dimensions']['optimizer_updates']['projected']==15814


def test_session_plan_keeps_qualification_and_formal_counters_distinct():
    rows=p.sessions();assert len(rows)==5 and len({r['id']for r in rows})==5
    assert sum(r['optimizer_limit']for r in rows)==1028
    assert sum(r['query_limit']for r in rows)==147
    for arm in ('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):
        q,t=[r for r in rows if r['arm']==arm]
        assert (q['optimizer_limit'],q['sample_visits'],q['formal_candidate'])==(2,64,False)
        assert (t['optimizer_limit'],t['sample_visits'],t['formal_candidate'])==(512,16384,True)
        assert q['fresh_raw_parent']and t['fresh_raw_parent']


def test_no_duplicate_package_or_unapproved_cap_adoption(captured):
    result=reserved(captured)
    with pytest.raises(PermissionError,match='PACKAGE_ONCE'):
        p.reserve(a.raw(result['proposed_global']),captured[1],now_epoch=1793094001.)
    g=json.loads(captured[0]);g.pop(a.FIELD)
    with pytest.raises(PermissionError,match='PACKAGE_ONCE'):
        p.reserve(a.raw(g),captured[1],now_epoch=1793094001.)


@pytest.mark.parametrize('change',['GPU_hold','CPU_hold','storage_hold','recipe','session','fake_Core','fake_admission'])
def test_missing_holds_science_drift_or_fresh_Core_claim_refused(captured,change):
    g=reserved(captured)['proposed_global'];v=g[p.FIELD]
    if change=='GPU_hold':g['reserved']['H200_s']-=1
    elif change=='CPU_hold':g['CPU_reserved_s']-=1
    elif change=='storage_hold':g['storage_reserved_bytes']-=1
    elif change=='recipe':v['original_resources']['optimizer_updates']=2048
    elif change=='session':v['session_plan'][1]['formal_candidate']=True
    elif change=='fake_Core':v['actual_LOCAL_written']=True
    elif change=='fake_admission':v['native_admission']=True
    with pytest.raises(PermissionError):p.validate(g)


def test_exact_cap_overlay_is_not_new_authority_for_unmarked_or_tampered_ledgers(captured):
    old=json.loads(captured[0]);old.pop(a.FIELD)
    assert pbs_limit(old)==37
    assert a.global_limits(old,{'H200_s':121*3600})=={'H200_s':121*3600}
    current=json.loads(captured[0]);assert pbs_limit(current)==39
    current[a.FIELD]['caps']['PBS_submissions']=100
    with pytest.raises(PermissionError):pbs_limit(current)


def test_original_onceIO_temp_escrow_CAS_keeps_no_retry_on_existing_intent(captured,tmp_path):
    result=reserved(captured);target=tmp_path/'GLOBAL.json';target.write_bytes(captured[0])
    args=dict(expected_before_sha256=a.sha(captured[0]),proposed_bytes=a.raw(result['proposed_global']),
        request_id='CPU-FIRST512-WHOLE-PACKAGE',intent_path=tmp_path/'I.json',receipt_path=tmp_path/'R.json',lock_path=tmp_path/'L')
    once.commit(target,**args)
    assert once.reconcile(target,intent_path=args['intent_path'],receipt_path=args['receipt_path'])['receipt_binding_valid']
    with pytest.raises(FileExistsError):once.commit(target,**args)
