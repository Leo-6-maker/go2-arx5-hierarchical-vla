"""Actual closed bytes, future clock observations are explicit CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_terminal_inventory as n,minimal_execution as e


@pytest.fixture(scope='module')
def capture():
    path=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not path:pytest.skip('explicit actual proposed completion GLOBAL and historical LOCAL required')
    p=json.loads((Path(path)/'PARTIAL_COMPLETION01_PREPARED.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(p['historical_LOCAL_gzip_base64']))
    now=p['marker']['adopted_epoch']+1
    fresh=dict(LOCAL_sha256=e.sha(lb),GLOBAL_sha256=e.sha(gb),observed_epoch=now,new_mutations=0,CPU_fixture_only=True)
    return gb,lb,fresh,now


@pytest.fixture(scope='module')
def proposed(capture):
    gb,lb,fresh,now=capture
    return n.propose(gb,lb,fresh_readback=fresh,now_epoch=now)


def test_complete_terminal_inventory_and_paid_controls_preserve_all14_dimensions(capture,proposed):
    gb,lb,_,_=capture;g,l=json.loads(gb),json.loads(lb);value=proposed['proposed_LOCAL']
    assert len(value['PBS_submissions'])==38 and value['PBS_submissions']==g['PBS_submissions']
    assert value[n.concurrency.RETRY_LOCAL_MIRROR]==l[n.concurrency.RETRY_LOCAL_MIRROR]
    assert [v['updates']for v in proposed['terminal_rows']]==[0,0,2]
    assert [v['queries']for v in proposed['terminal_rows']]==[0,0,71]
    assert proposed['full_future_proof']['cap_changes_required']=={}
    assert proposed['new_budget']is proposed['native_admission']is False
    assert proposed['new_request_PBS_model_or_physics']==0
    assert value[n.FIELD]['actual_GLOBAL_already_paid_controls']is True


@pytest.mark.parametrize('kind',['stale','wrong_LOCAL','unknown','missing_pending_guard','changed_old_row','repeat'])
def test_missing_fresh_identity_or_old_unknown_never_becomes_catchup(capture,proposed,kind):
    gb,lb,old,now=capture;g,l=json.loads(gb),json.loads(lb);fresh=deepcopy(old)
    if kind=='stale':fresh['observed_epoch']=now-121
    elif kind=='wrong_LOCAL':fresh['LOCAL_sha256']='0'*64
    elif kind=='unknown':g['unknown_effects']=['PRESERVED']
    elif kind=='missing_pending_guard':l['pending_queries']=['PRESERVED']
    elif kind=='changed_old_row':l['PBS_submissions'][0]['job_id']='OTHER'
    elif kind=='repeat':l=proposed['proposed_LOCAL']
    with pytest.raises((PermissionError,ValueError)):n.propose(e.raw(g),e.raw(l),fresh_readback=fresh,now_epoch=now)
