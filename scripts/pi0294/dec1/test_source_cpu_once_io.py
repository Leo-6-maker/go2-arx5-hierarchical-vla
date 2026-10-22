"""Real local metadata files; no native resource or external action."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_cpu_once_io as unit


@pytest.fixture
def state(tmp_path):
    path=tmp_path/'RESOURCE_LEDGER.json';old=b'{"old_cost":123.5,"unknown_effects":[]}\n'
    path.write_bytes(old);path.chmod(0o400)
    args=dict(expected_before_sha256=hashlib.sha256(old).hexdigest(),
        proposed_bytes=b'{"old_cost":123.5,"unknown_effects":[],"reserved":100}\n',request_id='once-01',
        intent_path=tmp_path/'INTENT.json',receipt_path=tmp_path/'RECEIPT.json',lock_path=tmp_path/'LOCK')
    return path,old,args


def test_exact_write_and_actual_receipt_readback_cannot_be_replayed(state):
    path,old,args=state;r=unit.commit(path,**args)
    assert r['native_dispatch_admitted'] is False and path.read_bytes()==args['proposed_bytes']
    assert json.loads(path.read_bytes())['old_cost']==123.5
    view=unit.reconcile(path,intent_path=args['intent_path'],receipt_path=args['receipt_path'])
    assert view['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT' and not view['automatic_retry_allowed']
    with pytest.raises(FileExistsError):unit.commit(path,**args)


def test_stale_snapshot_rejects_before_any_intent(state):
    path,old,args=state;args['expected_before_sha256']='0'*64
    with pytest.raises(PermissionError,match='BEFORE_INTENT'):unit.commit(path,**args)
    assert path.read_bytes()==old and not args['intent_path'].exists()


@pytest.mark.parametrize('fault',['replace_before','replace_after','receipt_IO','readback_IO'])
def test_lost_write_result_reconciles_actual_bytes_without_retry(state,monkeypatch,fault):
    path,old,args=state;replace=unit.os.replace;exclusive=unit._exclusive;read=unit._file
    if fault in ('replace_before','replace_after'):
        def fail(src,dst):
            if fault=='replace_after':replace(src,dst)
            raise OSError('LOST_REPLACE_RESULT')
        monkeypatch.setattr(unit.os,'replace',fail)
    if fault=='receipt_IO':
        def fail(p,b,mode=0o400):
            if Path(p)==args['receipt_path']:raise OSError('LOST_RECEIPT_IO')
            return exclusive(p,b,mode)
        monkeypatch.setattr(unit,'_exclusive',fail)
    if fault=='readback_IO':
        def fail(p):
            b=read(p)
            if Path(p)==path and b==args['proposed_bytes']:raise OSError('LOST_ACTUAL_READBACK')
            return b
        monkeypatch.setattr(unit,'_file',fail)
    with pytest.raises(OSError):unit.commit(path,**args)
    monkeypatch.setattr(unit.os,'replace',replace);monkeypatch.setattr(unit,'_exclusive',exclusive);monkeypatch.setattr(unit,'_file',read)
    before={p:p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    view=unit.reconcile(path,intent_path=args['intent_path'],receipt_path=args['receipt_path'])
    assert view['status']==('CURRENT_OLD_BYTES_NO_RETRY_AUTHORIZATION' if fault=='replace_before'
        else 'CURRENT_PROPOSED_BYTES_RECEIPT_UNCONFIRMED')
    assert before=={p:p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):unit.commit(path,**args)


def test_changed_actual_bytes_and_tampered_receipt_cannot_confirm(state):
    path,old,args=state;unit.commit(path,**args)
    receipt=json.loads(args['receipt_path'].read_bytes());receipt['request_id']='another'
    args['receipt_path'].chmod(0o600);args['receipt_path'].write_text(json.dumps(receipt));args['receipt_path'].chmod(0o400)
    assert unit.reconcile(path,intent_path=args['intent_path'],receipt_path=args['receipt_path'])['receipt_binding_valid'] is False
    path.chmod(0o600);path.write_text('{"different":true}');path.chmod(0o400)
    assert unit.reconcile(path,intent_path=args['intent_path'],receipt_path=args['receipt_path'])['status'].startswith('UNKNOWN')


@pytest.mark.parametrize('fault',['symlink','escaped_aux','same_path','unsafe_id','no_change','bad_json'])
def test_unsafe_or_invalid_local_IO_is_rejected_before_intent(state,tmp_path,fault):
    path,old,args=state
    if fault=='symlink':
        other=tmp_path/'ORIGINAL.json';path.rename(other);path.symlink_to(other)
    if fault=='escaped_aux':args['receipt_path']=tmp_path.parent/'elsewhere'
    if fault=='same_path':args['receipt_path']=path
    if fault=='unsafe_id':args['request_id']='../../not-safe'
    if fault=='no_change':args['proposed_bytes']=old
    if fault=='bad_json':args['proposed_bytes']=b'{"number":NaN}'
    with pytest.raises((PermissionError,ValueError)):unit.commit(path,**args)
    assert not args['intent_path'].exists()
