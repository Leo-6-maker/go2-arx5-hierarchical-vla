"""CPU handoff preparation, never a synthetic native adoption receipt."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
import pytest
from . import vanda_control_handoff as c

GRANT=Path(__file__).resolve().parents[3]/c.PLAN
def raw(v):return (json.dumps(v,indent=2)+'\n').encode()


def fixture():
    rows=[]
    for i in range(799):
        effect='READ_ONLY' if i<542 else 'MUTATION' if i<544 else 'HEARTBEAT'
        phase=(sorted(c.RELEASE_PHASES)[i-542] if effect=='MUTATION' else effect)
        intent=dict(operation_id=f'{i:032x}',identity=deepcopy(c.IDENTITY),effect=effect,phase=phase)
        rows.append(dict(intent=intent,resolved_status='READ_FAILED' if i==0 else 'CONFIRMED'))
    current=dict(status='CONFIRMED',intent=deepcopy(rows[-1]['intent']))
    catalog=dict(sha256=c.JOURNAL_CATALOG_SHA,files=1599,intents=799)
    h=dict(optimizer_updates=8744,generation_queries=1197,unknown_effects=[],pending_updates=[],
        used=dict(PBS_submissions=23),reserved=dict(H200_s=22560,optimizer_updates=2048,neural_queries=29),
        CPU_reserved_s=54000,storage_reserved_bytes=70000000000,
        PBS_submissions=[dict(status='NATIVE_TERMINAL',preserved_old_failure=i) for i in range(22)]+
            [dict(status='SUBMITTED_PENDING_NATIVE',job_id=c.failed.JOB,request_id=c.failed.REQUEST,native_platform='VANDA_A40')],
        historical_checkpoints_and_costs={'A':'closed','old_B_lost805':'paid','all_caps':'unchanged'})
    d=dict(optimizer_updates=8744,unknown_effects=[],new_submissions=[{'preserved':i} for i in range(12)])
    kwargs=dict(grant_bytes=GRANT.read_bytes(),owner_state=dict(MainPID='0',ActiveState='failed',ExecMainStatus='1'),
        stop=dict(epoch=1792755974.8454535,type='PermissionError',message='VANDA_PROJECT_TOKEN_EXPIRED',automatic_mutation_retry=False),
        current_rpc=current,controller_journal=rows,controller_catalog=catalog,now_epoch=1792759500.)
    return h,d,kwargs


def prepare(h,d,kw):
    hb,db=raw(h),raw(d)
    with patch.object(c,'GLOBAL_SHA',hashlib.sha256(hb).hexdigest()),patch.object(c,'DEC1_SHA',hashlib.sha256(db).hexdigest()):
        return c.prepare(hb,db,**kw)


def test_handoff_keeps_every_cost_cap_history_and_reservation_and_labels_origin_unavailable():
    h,d,kw=fixture();before=deepcopy((h,d,kw));nh,nd,marker=prepare(h,d,kw)
    assert (h,d,kw)==before
    assert {k:v for k,v in nh.items() if k!=c.FIELD}==h
    assert {k:v for k,v in nd.items() if k!=c.FIELD}==d
    assert marker['origin_is_fresh_native_read'] is False
    assert marker['old_parent_mutations_permitted'] is False
    assert marker['new_GPU_PBS_training_or_budget']==0
    assert marker['journal']['failed_reads_preserved']==1


@pytest.mark.parametrize('fault',['active_owner','unknown_write','extra_mutation','duplicate_release',
    'omitted_operation','wrong_catalog','wrong_request','other_Hopper_job','dropped_reserve',
    'refunded_paid_updates','unknown_global','expired','bool_clock','wrong_grant','replay'])
def test_missing_fence_or_changed_accounting_cannot_transfer_authority(fault):
    h,d,kw=fixture()
    if fault=='active_owner':kw['owner_state']['MainPID']='123'
    elif fault=='unknown_write':kw['controller_journal'][-1]['resolved_status']='UNKNOWN'
    elif fault=='extra_mutation':kw['controller_journal'][1]['intent'].update(effect='MUTATION',phase='NEW_HOPPER_WRITE')
    elif fault=='duplicate_release':kw['controller_journal'][543]['intent']['phase']=kw['controller_journal'][542]['intent']['phase']
    elif fault=='omitted_operation':kw['controller_journal'].pop(10)
    elif fault=='wrong_catalog':kw['controller_catalog']['sha256']='0'*64
    elif fault=='wrong_request':kw['controller_journal'][10]['intent']['identity']['job_id']='different'
    elif fault=='other_Hopper_job':h['PBS_submissions'][0].update(status='SUBMITTED_PENDING_NATIVE',native_platform='HOPPER_H200')
    elif fault=='dropped_reserve':h['reserved']['H200_s']=0
    elif fault=='refunded_paid_updates':h['optimizer_updates']=d['optimizer_updates']=7822
    elif fault=='unknown_global':h['unknown_effects']=[{}]
    elif fault=='expired':kw['now_epoch']=1792952996
    elif fault=='bool_clock':kw['now_epoch']=True
    elif fault=='wrong_grant':kw['grant_bytes']+=b' '
    else:h[c.FIELD]={}
    with pytest.raises(PermissionError):prepare(h,d,kw)


def test_exact_parent_byte_hashes_cannot_be_replaced_by_semantically_similar_json():
    h,d,kw=fixture()
    with pytest.raises(PermissionError):c.prepare(raw(h),raw(d),**kw)


def test_complete_real_file_catalog_retains_failed_reads_and_rejects_omission(tmp_path):
    _,_,kw=fixture()
    for row in kw['controller_journal']:
        intent=row['intent'];op=intent['operation_id']
        (tmp_path/(op+'-INTENT.json')).write_bytes(raw(dict(intent=intent,event='INTENT')))
        event='ERROR' if row['resolved_status']=='READ_FAILED' else 'CONFIRMED'
        (tmp_path/(op+'-'+event+'.json')).write_bytes(raw(dict(intent=intent,event=event,outcome='READ_FAILED' if event=='ERROR' else 'CONFIRMED')))
    (tmp_path/'RPC_EFFECT_STATE.json').write_bytes(raw(kw['current_rpc']))
    catalog=[(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(tmp_path.glob('*.json'))]
    digest=hashlib.sha256(json.dumps(catalog,separators=(',',':')).encode()).hexdigest()
    with patch.object(c,'JOURNAL_CATALOG_SHA',digest):
        rows,current,meta=c.read_journal(tmp_path)
        assert len(rows)==799 and meta['files']==1599
        assert sum(r['resolved_status']=='READ_FAILED' for r in rows)==1
        assert current==kw['current_rpc']
        (tmp_path/(rows[0]['intent']['operation_id']+'-ERROR.json')).unlink()
        with pytest.raises(PermissionError):c.read_journal(tmp_path)
