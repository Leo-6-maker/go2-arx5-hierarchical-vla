"""Captured canonical state; future adoption/freshness are explicit fixtures.

The temporary original Journal/onceIO test executes a real bounded CPU child,
not a model or GPU. No experiment-host or canonical write occurs in this file.
"""
import base64,gzip,json,os,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_d25_control_wire as w

@pytest.fixture(scope='module')
def captured():
    value=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not value:pytest.skip('Explicit actual registered D25/counter capture required')
    root=Path(value);p=json.loads((root/'D25_FINITE_LOCAL04_DISPATCH_PREPARED.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(p['LOCAL_gzip_base64']))
    cb=(root/'D25_FINITE_LOCAL03_CONTROL_IO/CONTROL_FINITE_WIRE_AND_CPU.json').read_bytes()
    now=time.time();f=dict(GLOBAL_sha256=w.a.sha(gb),LOCAL_sha256=w.a.sha(lb),request_id=w.source.REQUEST,
        request_sha256=p['assembly']['request_sha256'],native_owner_and_neural_units_not_started=True,
        new_mutations=0,observed_epoch=now,CPU_FIXTURE_NOT_NATIVE_READ=True)
    before=json.loads(cb);assert w.COUNTER_FIELD not in before
    out=w.propose(cb,gb,lb,fresh_native_join=f,now_epoch=now)
    return root,cb,gb,lb,f,now,out

def test_same_invoice_all_finance_rows_clocks_costs_and_protected_banks(captured):
    _,cb,gb,lb,_,_,p=captured;g,l=json.loads(gb),json.loads(lb);v,z=p['proposed_global'],p['proposed_LOCAL'];m=p['marker']
    assert sum(m['old_caps'].values())==sum(m['new_caps'].values())==64*1024**2
    assert m['new_caps'][w.RECEIVER]-m['old_caps'][w.RECEIVER]==65536
    assert m['old_caps'][w.DONOR]-m['new_caps'][w.DONOR]==65536
    assert all(m['old_caps'][k]==m['new_caps'][k]for k in w.PROTECTED)
    assert {k:x for k,x in v.items()if k not in(w.FIELD,'revision')}=={k:x for k,x in g.items()if k!='revision'}
    assert {k:x for k,x in z.items()if k!=w.FIELD}==l
    assert g[w.paid.FIELD]==v[w.paid.FIELD]and p['new_financial_resources']==0 and not p['native_admission']
    assert json.loads(cb)['wire_caps']==w.paid.WIRE

def changed_counter(captured):
    _,cb,_,_,_,_,p=captured
    return w.counter_after_native_dual_adoption(json.loads(cb),p['marker'],GLOBAL_sha256='1'*64,LOCAL_sha256='2'*64)

def test_old_history_and_unknown_upper_prefix_never_rewritten(captured):
    v=changed_counter(captured);old=json.loads(captured[1])
    assert v['wire_spent']==old['wire_spent']and v['CPU_spent']==old['CPU_spent']and v['operations']==old['operations']
    assert any(x.get('status')=='RESERVED_UNKNOWN_UNTIL_EXACT_REPLY'for x in old['operations'])
    assert w.effective_caps(v)==w.caps()
    with pytest.raises(PermissionError):w.counter_after_native_dual_adoption(v,captured[-1]['marker'],GLOBAL_sha256='1'*64,LOCAL_sha256='2'*64)

@pytest.mark.parametrize('kind',['protected','increase','wire_refund','CPU_refund','CPU_cap','old_operation'])
def test_amended_counter_tamper_is_rejected_without_repair(captured,kind):
    v=changed_counter(captured)
    if kind=='protected':v[w.COUNTER_FIELD]['new_caps']['protected_closeout']-=1
    elif kind=='increase':v['wire_caps'][w.RECEIVER]+=1
    elif kind=='wire_refund':v['wire_spent'][w.RECEIVER]-=1
    elif kind=='CPU_refund':v['CPU_spent']['local']-=1
    elif kind=='CPU_cap':v['CPU_upper']['core_cold']+=1
    else:v['operations'][0]['operation']='CHANGED_OLD_HISTORY'
    before=deepcopy(v)
    with pytest.raises(PermissionError):w.effective_caps(v)
    assert v==before

def test_unqualified_unknown_and_stale_fake_join_never_admit(captured):
    _,cb,gb,lb,f,now,_=captured;v=json.loads(cb)
    v['operations'].append(dict(operation='CPU_UNRESOLVED_WRITE',status='RESERVED_UNKNOWN_UNTIL_EXACT_REPLY'))
    with pytest.raises(PermissionError,match='UNQUALIFIED'):w.propose(w.a.raw(v),gb,lb,fresh_native_join=f,now_epoch=now)
    # Already adopted counter is refused before native/large-ledger work.
    with pytest.raises(PermissionError):w.propose(w.a.raw(changed_counter(captured)),gb,lb,fresh_native_join=f,now_epoch=now)

def test_same_original_controller_real_temp_onceIO_and_native_child(captured,tmp_path):
    root,*_=captured;original=(root/'unified_finite_control_io.py').read_text()
    body=w.controller_source(original);namespace={'__name__':'_CPU_original_ControlIO_cold_wire_policy','__file__':str(root/'unified_finite_control_io.py')}
    # Original management imports resolve from the explicit private input root.
    sys.path.insert(0,str(root))
    try:exec(compile(body,'<exact-original-controller-new-cold-cap-policy>','exec'),namespace)
    finally:sys.path.remove(str(root))
    v=changed_counter(captured);counter=tmp_path/'CONTROL_FINITE_WIRE_AND_CPU.json';counter.write_bytes(w.a.raw(v));counter.chmod(0o600)
    def transport(program,payload,limit):
        assert payload=={} and limit<=35
        return subprocess.run([sys.executable,'-B','-c',program],capture_output=True,text=True,timeout=limit)
    control=namespace['ControlIO'](tmp_path,v['identity'],transport={'core':transport})
    result=control.call('CPU_ONE_BOUND_READ','core',w.RECEIVER,"import json;print(json.dumps(dict(status='CPU_CHILD_ONLY',new_models=0)))",{},effect='READ_ONLY',CPU_upper=5,reply_upper=2048)
    assert result==dict(status='CPU_CHILD_ONLY',new_models=0)
    assert control.value()['operations'][-1]['status']=='CONFIRMED'
    assert control.value()['operations'][:len(v['operations'])]==v['operations']
    with pytest.raises(AssertionError,match='ALREADY_ATTEMPTED'):control.call('CPU_ONE_BOUND_READ','core',w.RECEIVER,'',{},effect='READ_ONLY',CPU_upper=5,reply_upper=2048)
    assert not list(tmp_path.glob('A40*'))
