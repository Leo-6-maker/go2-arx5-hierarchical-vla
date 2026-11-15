import copy,json,threading,time
from pathlib import Path
import pytest
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import Ledger,CELLS,resumable_checkpoint

def carry():
    return [dict(id=f'old-{c}-{i}',cell=c,state='QUALIFIED',worker='old',physical_requests=1,native_integrations=1,result=dict(path=f'/sealed/{c}-{i}',sha256=f'{c}-{i}'))for c in CELLS for i in (1,2)]

def ledger_at(tmp_path):
    ledger=Ledger(tmp_path);ledger.create(carry(),deadline_epoch=time.time()+7200);ledger.seal_ready();return ledger

def test_concurrent_claims_and_exact_carryover(tmp_path):
    ledger=ledger_at(tmp_path);rows=[]
    def claim(i):rows.append(ledger.claim(f'worker-{i}'))
    threads=[threading.Thread(target=claim,args=(i,))for i in range(12)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert len(rows)==len({r['id']for r in rows})==12
    value=json.loads(ledger.path.read_bytes());assert len(value['attempts'])==32
    assert all(r['reset']>=10000 for r in rows)
    with pytest.raises(PermissionError):ledger.claim('worker-0')

def test_isolation_blocks_other_worker_and_preserves_failures(tmp_path):
    ledger=ledger_at(tmp_path);ledger.isolate('F1',dict(reason='contact'))
    for i in range(8):assert ledger.claim(str(i))['family']=='F2'
    assert json.loads((tmp_path/'ISOLATION.private.json').read_bytes())['F1']['reason']=='contact'

def test_caps_nested_snapshot_only_terminal_success(tmp_path):
    ledger=ledger_at(tmp_path)
    for i in range(80):
        row=ledger.claim('w')
        if row is None:ledger.seal_ready();row=ledger.claim('w')
        assert row is not None
        ledger.update(row['id'],state='QUALIFIED',result=dict(path='/sealed/'+row['id'],sha256=row['id']),physical_requests=1)
        ledger.seal_ready()
    value=json.loads(ledger.path.read_bytes())
    assert len(value['attempts'])==100
    assert [value['snapshots'][x]['actual_qualified']for x in ('C20','C60','C100')]==[20,60,100]
    assert ledger.claim('w')is None
    assert set(value['snapshots']['C20']['episode_ids'])<set(value['snapshots']['C60']['episode_ids'])<set(value['snapshots']['C100']['episode_ids'])

def test_total_attempts_include_import_and_partial(tmp_path):
    ledger=ledger_at(tmp_path)
    for i in range(100):
        r=ledger.claim('w');assert r
        ledger.update(r['id'],state='TASK_FAILED',physical_requests=1,native_integrations=9)
    ledger.seal_ready();v=json.loads(ledger.path.read_bytes())
    assert len(v['attempts'])==120
    assert v['snapshots']['C60']['actual_qualified']==20
    assert len(v['blocked_cells'])==10
    assert ledger.claim('w')is None

def test_unknown_no_relaunch_and_epoch_limits(tmp_path):
    ledger=ledger_at(tmp_path);r=ledger.claim('worker');ledger.update(r['id'],state='UNKNOWN')
    with pytest.raises(PermissionError):ledger.claim('worker')
    for i in range(3):ledger.start_epoch(100+i,str(i),'cgroup')
    with pytest.raises(PermissionError):ledger.start_epoch(103,'3','cgroup')

def test_resume_last_committed87_and_caps():
    complete=dict(schema='SLA_COMPLETE_PEFT_STATE_V1',arrays=[{}]*87,step=500,cursor=16000)
    assert resumable_checkpoint(complete,500,resumes=1)==500
    for last,resumes in ((501,1),(500,2)):
        with pytest.raises(PermissionError):resumable_checkpoint(complete,last,resumes=resumes)
    complete['arrays'].pop()
    with pytest.raises(PermissionError):resumable_checkpoint(complete,500,resumes=0)
