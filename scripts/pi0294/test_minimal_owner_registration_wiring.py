"""Execute the real owner registration seam, without a model or scheduler."""
import json,time
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import minimal_native_owner as owner
from scripts.pi0294.generalization import minimal_e2only_execution as x


def test_real_registration_decodes_its_request_before_retry_selection(tmp_path,monkeypatch):
    old=tmp_path/'closed';old.mkdir();plans=tmp_path/'plans';plans.mkdir()
    files={'lease_bytes':('E2_ONLY_PREVIOUS_ZERO_LEASE.private.json',b'closed lease'),
        'settled_bytes':('E2_ONLY_PREVIOUS_ZERO_SETTLED.private.json',b'settled'),
        'finite_OS_bytes':('E2_ONLY_PREVIOUS_ZERO_FINITE_OS0.private.json',b'OS0')}
    for _,(name,b)in files.items():(plans/name).write_bytes(b)
    lease=old/'LEASE.json';lease.write_bytes(files['lease_bytes'][1])
    (old/'OUTPUT_BINDING.private.json').write_text(json.dumps({'lease':str(lease)}))
    p=old/'control/resident-owner';p.mkdir(parents=True);(p/'TERMINAL_SETTLED.private.json').write_bytes(files['settled_bytes'][1])
    final=old/'OS_EXIT.json';final.write_bytes(files['finite_OS_bytes'][1])
    gb=b'{"CPU_s": 123}';gp=tmp_path/'GLOBAL.json';gp.write_bytes(gb)
    request={'attempt_index':2,'request_id':'exact-fixture-request'};rb=json.dumps(request).encode();seen=[]
    # The test isolates provenance/I/O dependencies while executing the actual
    # owner method and its same-byte predecessor reads. Contract arithmetic is
    # covered separately by full-ledger tests.
    monkeypatch.setattr(x,'request',lambda r:seen.append(r))
    monkeypatch.setattr(x.authority,'predecessor',lambda g:({'terminal':{'native':{'all_native_workers_terminal':True}}},{}))
    monkeypatch.setattr(x,'closed_zero_binding',lambda g,**kw:dict(verified=True))
    monkeypatch.setattr(x,'register',lambda *a,**kw:kw)
    physical=SimpleNamespace(custody={'closed_zero_roots':[dict(root=str(old),finite_OS_path=str(final))]},inspect=lambda **kw:dict(observed_epoch=time.time()))
    o=object.__new__(owner.Owner);o.adapter=x;o.e2only_physical=physical;o.global_path=gp;o.root=tmp_path
    o.CONFIG={'outside_confirmation_base64':''};o.local=b'local';o.current=dict(global_ledger_sha256=x.e.sha(gb),view=dict(observed_epoch=time.time(),complete_account_queue={'fixture':True}))
    result=o.registration(gb,rb,b'capture')
    assert seen==[request]and result['request_bytes']==rb
    assert result['fresh_readback']['closed_zero_predecessor']=={'verified':True}
    assert result['closed_zero_files']=={k:b for k,(_,b)in files.items()}


def test_repaired_owner_preserves400_old_plus600_metadata_inside3600():
    r=dict(platform_training_version=x.VERSION,profile=x.PROFILE,pre_native_cold_generation=1,pre_native_owner_CPU_upper_s=2600)
    assert owner.control_CPU_limits(r)==(2590,2600)
    assert 400+600+owner.control_CPU_limits(r)[1]==3600
    with pytest.raises(PermissionError):owner.control_CPU_limits(dict(r,pre_native_owner_CPU_upper_s=3000))
    assert owner.control_CPU_limits(dict(platform_training_version=x.VERSION,profile=x.PROFILE))==(2990,3000)
