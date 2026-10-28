"""Captured D25 closure; new process/request/source are explicit CPU fixtures."""
import base64
import gzip
import json
import os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_d25_continuation as d
from scripts.pi0294.pv1 import mix_core_accounting as c, mix_accounting as a, mix_stage_handoff as h

RID = 'd1111111-1111-4111-8111-111111111111'
NONCE = 'd2222222-2222-4222-8222-222222222222'


@pytest.fixture
def captured():
    root = Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    x = json.loads((root/'D25_ORIGINAL_CORE_ZERO_EFFECT_CLOSED_READONLY_1793161909237875897.private.json').read_bytes())
    return dict(global_bytes=gzip.decompress(base64.b64decode(x['Vanda']['GLOBAL_gzip_base64'])),
        local_bytes=gzip.decompress(base64.b64decode(x['Core']['LOCAL_gzip_base64'])), closed_readback=x['Core'],
        new_request_id=RID,new_nonce=NONCE,source_sha='1'*40,now_epoch=x['Core']['observed_epoch'])


def test_original_costs_and_all_future_preserved(captured):
    before=deepcopy(captured);out=d.propose(**captured)
    g,l=json.loads(captured['global_bytes']),json.loads(captured['local_bytes'])
    v,w=out['proposed_global'],out['proposed_LOCAL'];m=v[d.FIELD]
    assert captured==before and m['output_credit_bytes']==0
    assert v['used']==g['used'] and v['PBS_submissions']==g['PBS_submissions'] and v[c.FIELD]==g[c.FIELD]
    assert v['CPU_s']==g['CPU_s']+7200 and v['new_storage_current_bytes']==g['new_storage_current_bytes']+64*1024**2
    assert v['storage_reserved_bytes']==g['storage_reserved_bytes']+c.remaining_stage()['Hopper_storage_bytes']
    assert w['network_transfer_accounted_bytes']==l['network_transfer_accounted_bytes']+1024**3
    assert v[a.PACKAGE]['engineering_replacements_used']==g[a.PACKAGE]['engineering_replacements_used']==4
    assert v[a.PACKAGE]['completed_stages']==g[a.PACKAGE]['completed_stages']
    assert out['full_future_proof']['cap_changes_required']=={} and not out['native_registration']
    with pytest.raises(PermissionError):
        d.propose(**dict(captured,global_bytes=a.raw(v),local_bytes=a.raw(w)))


@pytest.mark.parametrize('change',['stale','LOCAL','lease','owner','history','effect','unknown','identity','window'])
def test_reject_uncertain_or_changed_terminal(captured,change):
    v=deepcopy(captured);cap=v['closed_readback']
    if change=='stale':v['now_epoch']+=121
    elif change=='LOCAL':cap['LOCAL_sha256']='0'*64
    elif change=='lease':cap['leases']['GPU']['value']['status']='ACTIVE'
    elif change=='owner':cap['files']['OWNER_RESULT.json']['value']['changed']=True
    elif change=='history':cap['files']['resident/INDEPENDENT_PANEL_HISTORY_AUDIT.json']['value']['history']['valid_conditions']=1
    elif change=='effect':cap['files']['resident/DUAL_FINANCIAL_TERMINAL_INTENT.json']['value']['effects']['query_attempts']=1
    elif change=='unknown':
        g=json.loads(v['global_bytes']);g['unknown_effects']=['CPU_FIXTURE_UNKNOWN'];v['global_bytes']=a.raw(g)
    elif change=='identity':v['new_request_id']=d.FAILED
    else:v['now_epoch']=1793537549.8133287
    with pytest.raises((PermissionError,ValueError)):d.propose(**v)


def test_full_D25_register_keeps_previous_physical_accounting(captured):
    out=d.propose(**captured);g,l=out['proposed_global'],out['proposed_LOCAL']
    root=Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    old=json.loads((root/'D25_ORIGINAL_CORE_PEERS_PREPARED.private.json').read_bytes())
    terminal={k:base64.b64decode(v)for k,v in old['terminal_capture_base64'].items()}
    proof=h.closed_endpoint(global_bytes=a.raw(g),arm='D25',**terminal)
    row=deepcopy(old['assembly']['record']);row.update(request_id=RID,submission_nonce=NONCE,
        source_sha='1'*40,child_run_uuid=d.RUN,created_epoch=captured['now_epoch'],training_handoff_sha256=a.sha(a.raw(proof)))
    physical=json.loads(base64.b64decode(old['assembly']['rows']['core']['manifest_bytes_base64']))
    physical['run_uuid']=d.RUN;physical['code']['vla_commit']='1'*40
    new=c.register(g,l,row,a.raw(physical),terminal_capture=terminal,now_epoch=captured['now_epoch'])
    v,w=new['proposed_global'],new['proposed_LOCAL']
    assert v[c.FIELD][d.FAILED]==g[c.FIELD][d.FAILED]
    assert v[c.FIELD][RID]['remaining_conditions']==24 and v[c.FIELD][RID]['query_limit']==2097
    assert w['V5_CD_historical_closed_Core_CPU_substages'][d.FAILED]==l['V5_CD_Core_CPU_substages'][c.key('D25')]
    assert w['V5_CD_Core_CPU_substages'][c.key('D25')]['request_id']==RID
    with pytest.raises(PermissionError):c.register(v,w,row,a.raw(physical),terminal_capture=terminal,now_epoch=captured['now_epoch'])


def test_wrong_replacement_is_not_another_D25_slot(captured):
    out=d.propose(**captured);g=out['proposed_global'];row=deepcopy(g[c.FIELD][d.FAILED]);row['request_id']='d3333333-3333-4333-8333-333333333333'
    g[c.FIELD][row['request_id']]=row
    with pytest.raises(PermissionError):d.validate(g)
