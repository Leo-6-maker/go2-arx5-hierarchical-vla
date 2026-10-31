import base64
from copy import deepcopy
import hashlib
import json
import pytest

from scripts.pi0294.multitask import night_scope as n,night_source_stage as stage
from scripts.pi0294.test_night_source_v7_carry import closed,policy,raw
from scripts.pi0294.test_night_source_v7_finite import profile


def successor(old,lease,terminal,destination):
    return n.development_account(raw(old),raw(lease),raw(terminal),policy(),
        destination=destination,now=n.STOP_NEW-1)


def test_compressed_parent_roundtrips_all_history_and_keeps_legacy_byte_compatibility():
    old,lease,terminal=closed();old['operations'][0]['preserved_evidence']='paid-history-'*50000
    original=raw(old);new=successor(old,lease,terminal,'/owned/compact-source')
    carry=new[n.V7_CARRY]
    assert n.closed_carry_bytes(carry,'account')==original
    assert n.closed_carry_bytes(carry,'lease')==raw(lease)
    assert len(carry['account_base64'])<len(original)/20
    legacy=deepcopy(new);legacy[n.V7_CARRY].pop('account_encoding')
    legacy[n.V7_CARRY]['account_base64']=base64.b64encode(original).decode()
    assert n.validate_account(legacy)==legacy
    for value in(new,legacy):
        moved=n.transfer_development_custody(original,raw(value),destination='/owned/compact-source')
        assert moved['charged_upper']==old['charged_upper']and moved['operations']==old['operations']
    assert raw(old)==original


def test_second_compressed_generation_carries_exact_prior_bytes_without_growth():
    old,lease,terminal=closed();old['operations'][0]['evidence']='A'*600000
    first=successor(old,lease,terminal,'/owned/first');first.update(native_adopted=True,custody='ONE_NATIVE_CORE_SOURCE_OWNER')
    lease=dict(lease,run_uuid='first');terminal=dict(terminal,closed_lease_sha256=hashlib.sha256(raw(lease)).hexdigest())
    second=successor(first,lease,terminal,'/owned/second')
    assert n.closed_carry_bytes(second[n.V7_CARRY],'account')==raw(first)
    # The operations intentionally remain full. The embedded encoding stays compact.
    assert len(second[n.V7_CARRY]['account_base64'])<len(raw(first))/20
    assert second['charged_upper']==first['charged_upper']and second['operations']==first['operations']
    bad=deepcopy(second);bad['charged_upper']['attempts']-=1
    with pytest.raises(PermissionError):n.validate_account(bad)
    bad=deepcopy(second);bad[n.V7_CARRY]['account_encoding']='UNKNOWN'
    with pytest.raises(PermissionError):n.validate_account(bad)


def test_native_stage_compares_decoded_exact_parent_preimage(tmp_path):
    old,lease,terminal=closed();new=successor(old,lease,terminal,'/owned/next')
    (tmp_path/'SOURCE_V7_FINITE_JOB.json').write_bytes(raw(profile(n.SUCCESSOR_DEADLINE+100)))
    (tmp_path/'SOURCE_V7_EXECUTION_POLICY.md').write_bytes(policy())
    runtime=n.for_root(tmp_path)
    for name,value in {'control/NIGHT_SOURCE_ACCOUNT.json':old,'control/NATIVE_SOURCE_LEASE.json':lease,
                       'outputs/SOURCE_CHAIN_TERMINAL.json':terminal}.items():
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw(value))
    code=stage.PROGRAM.split(" for key,name in dict(account='control/NIGHT_SOURCE_ACCOUNT.json'",1)[1]
    code="for key,name in dict(account='control/NIGHT_SOURCE_ACCOUNT.json'"+code.split('\n q=subprocess.run',1)[0]
    exec(compile(code,'actual_native_carry_preimage_check','exec'),dict(previous=tmp_path,carry=new[n.V7_CARRY],scope=runtime))
    (tmp_path/'control/NIGHT_SOURCE_ACCOUNT.json').write_bytes(raw(old)+b' ')
    with pytest.raises(AssertionError):
        exec(compile(code,'actual_native_carry_preimage_check','exec'),dict(previous=tmp_path,carry=new[n.V7_CARRY],scope=runtime))
