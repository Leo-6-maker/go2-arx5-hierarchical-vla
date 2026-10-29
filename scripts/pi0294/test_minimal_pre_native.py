"""One whole-inventory retry and preserved unsubmitted E2 preparations."""
import json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_retry_physical_guard as scan
from scripts.pi0294.generalization import minimal_e2only_pre_native as cold
from scripts.pi0294.generalization import minimal_e2only_execution as x,minimal_e2only_contract as c


def test_actual_scan_reenumerates_full_tree_after_atomic_cache_rename(tmp_path,monkeypatch):
    cache=tmp_path/'__pycache__';cache.mkdir()
    transient=cache/'heartbeat_recovery.cpython-39.pyc.23438967198064'
    transient.write_bytes(b'atomic bytecode bytes')
    included=tmp_path/'other.json';included.write_bytes(b'charge this too')
    original=Path.lstat;calls=[]
    def lstat(path,*args,**kw):
        if path==transient:
            calls.append(str(path));path.rename(cache/'heartbeat_recovery.cpython-39.pyc')
        return original(path,*args,**kw)
    monkeypatch.setattr(Path,'lstat',lstat)
    rows,_=scan._scan(tmp_path,permitted_links=set(),transient_bytecode_retry=True)
    assert len(calls)==1
    assert {v['path']for v in rows.values()}=={'.','__pycache__','__pycache__/heartbeat_recovery.cpython-39.pyc','other.json'}
    assert sum(v['allocated']for v in rows.values())==sum(original(p).st_blocks*512 for p in [tmp_path,cache,included,cache/'heartbeat_recovery.cpython-39.pyc'])


@pytest.mark.parametrize('relative,enabled',[('__pycache__/a.cpython-39.pyc.123',False),('data.json',True),('__pycache__/a.pyc',True)])
def test_other_missing_entries_or_strict_old_scans_fail(tmp_path,monkeypatch,relative,enabled):
    original=scan._scan_once;calls=[]
    def fail(*a,**k):
        calls.append(1);raise FileNotFoundError(2,'missing',str(tmp_path/relative))
    monkeypatch.setattr(scan,'_scan_once',fail)
    with pytest.raises(FileNotFoundError):scan._scan(tmp_path,permitted_links=set(),transient_bytecode_retry=enabled)
    assert len(calls)==1


def test_repeated_transient_failure_stops_after_one_complete_rescan(tmp_path,monkeypatch):
    calls=[]
    def fail(*a,**k):
        calls.append(1);raise FileNotFoundError(2,'missing',str(tmp_path/'__pycache__/a.cpython-39.pyc.123'))
    monkeypatch.setattr(scan,'_scan_once',fail)
    with pytest.raises(FileNotFoundError):scan._scan(tmp_path,permitted_links=set(),transient_bytecode_retry=True)
    assert len(calls)==2


def packet():
    return dict(schema=cold.SCHEMA,request_id=cold.OLD_REQUEST,run_uuid=cold.OLD_RUN,source_sha=cold.OLD_SOURCE,
        manifest_sha256=cold.OLD_MANIFEST,owner_OS_sha256=cold.OLD_OS,owner_stop_sha256=cold.OLD_STOP,
        closed_lease_sha256='a'*64,snapshot_sha256='b'*64,snapshot_identity_sha256='c'*64,
        native_requests=0,all_recorded_actors_terminal=True,lease_status='CLOSED',allocated_bytes=100000)


def test_preparation_retirement_only_removes_unused_forecast_keeps_payment():
    p=packet();paid=dict(request_id=p['request_id'],run_uuid=p['run_uuid'],source_sha=p['source_sha'],run_manifest_sha256=p['manifest_sha256'])
    g={x.PREPAID:{p['request_id']:paid},x.RETIRED:{p['request_id']:p},'PBS_submissions':[]}
    assert cold.retired(g)=={p['request_id']:p}
    for field,value in [('lease_status','ACTIVE'),('native_requests',1),('owner_OS_sha256','d'*64)]:
        bad=deepcopy(g);bad[x.RETIRED][p['request_id']][field]=value
        with pytest.raises(PermissionError):cold.retired(bad)
    bad=deepcopy(g);bad['PBS_submissions']=[dict(request_id=p['request_id'])]
    with pytest.raises(PermissionError):cold.retired(bad)
    assert g[x.PREPAID][p['request_id']]==paid


def test_one_cold_role_changes_no_scientific_fields():
    original=x.fixed_request_fields(2);replacement=x.fixed_request_fields(2,1)
    assert {k for k in original if original[k]!=replacement[k]}=={'run_uuid'}
    assert set(replacement)-set(original)=={'pre_native_cold_generation'}
    assert replacement['run_uuid']==c.run_name(2,1)
    for attempt,generation in [(1,1),(2,2),(3,1)]:
        with pytest.raises(PermissionError):c.run_name(attempt,generation)


def test_actual_paid_control_retirement_and_new_preparation_keep_one_native_forecast():
    import base64,gzip,time
    root=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not root:pytest.skip('private actual unsubmitted preparation capture required')
    source=json.loads((Path(root)/'E2ONLY_RETRY02_OWNER01_PAID_INDEPENDENT.private.json').read_bytes())
    gb=gzip.decompress(base64.b64decode(source['GLOBAL_gzip_base64']));g=json.loads(gb)
    lb=gzip.decompress(base64.b64decode(json.loads((Path(root)/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes())['LOCAL_gzip_base64']))
    p=packet();now=time.time();fresh=dict(GLOBAL_sha256=x.e.sha(gb),request_id=p['request_id'],native_requests=0,
        all_recorded_actors_terminal=True,closed_record_sha256=x.e.sha(x.e.raw(p)),observed_epoch=now)
    before=x.planned_retry(g);closed=cold.propose(g,p,fresh=fresh,now_epoch=now)
    assert closed[x.PREPAID]==g[x.PREPAID] and closed['CPU_s']==g['CPU_s']
    assert x.planned_retry(closed)==x.e._vector({})
    identity=dict(request_id='00000000-0000-4000-8000-000000000091',submission_nonce='00000000-0000-4000-8000-000000000092',source_sha='a'*40,run_uuid=c.run_name(2,1))
    fresh=dict(GLOBAL_sha256=x.e.sha(x.e.raw(closed)),retired_preparation_sha256=x.e.sha(x.e.raw(p)),
        no_existing_new_owner_or_dispatch=True,observed_epoch=now)
    result=cold.propose_payment(x.e.raw(closed),lb,identity=identity,plan_sha256='d'*64,fresh=fresh,now_epoch=now)
    paid=result['proposed_global'];assert x.planned_retry(paid)==before==x.native_resources()
    assert paid['CPU_s']==g['CPU_s']+x.CONTROL['Hopper_CPU_s']
    assert paid['PBS_submissions']==g['PBS_submissions'] and result['full_future_proof']['cap_changes_required']=={}
    assert paid[x.PREPAID][p['request_id']]==g[x.PREPAID][p['request_id']]
    with pytest.raises(PermissionError):x.require_paid_retry(paid,dict(identity,attempt_index=2))
    with pytest.raises(PermissionError):cold.propose_payment(x.e.raw(paid),lb,identity=identity,plan_sha256='d'*64,fresh=fresh,now_epoch=now)
