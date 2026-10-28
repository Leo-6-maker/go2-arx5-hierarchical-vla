"""Original once IO and RPC latch in temporary files, never SSH or native GO.

Only the remote host/account assertion is replaced for this local CPU fixture.
The actual captured 25-row budget/grant validators and write program stay real.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.pi0294.pv1 import evaluation_core_ledger as remote
from scripts.pi0294.pv1 import evaluation_core_accounting as a
from scripts.pi0294.dec1.owner_transport import Journal,UnknownMutation
from scripts.pi0294.test_pv1_core_evaluation_accounting import captured_global,row,process,NOW


@pytest.fixture
def local_remote(captured_global,tmp_path,monkeypatch):
    actor=os.getpid();start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    r=row();v=a.reserve(captured_global,r);p=process(r);p.update(pid=actor,proc_start=start)
    v=a.bind(v,request_id=r['request_id'],process=p,now_epoch=NOW)
    parent=tmp_path/a.scope.CAMPAIGN;parent.mkdir();ledger=parent/'HOPPER_RESOURCE_LEDGER.json'
    ledger.write_text(json.dumps(v,indent=2)+'\n')
    code=Path(__file__).resolve().parents[2]
    members=['scripts/pi0294/pv1/evaluation_core_accounting.py','scripts/pi0294/pv1/evaluation_core_ledger.py',
        'scripts/pi0294/pv1/neural_ledger.py','scripts/pi0294/dec1/source_cpu_once_io.py']
    seal=tmp_path/'SOURCE_SEAL.json';seal.write_text(json.dumps(dict(source_sha=r['source_sha'],files={
        n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in members})))
    root=tmp_path/r['child_run_uuid'];(root/'evidence').mkdir(parents=True);control=root/'test-rpc';control.mkdir()
    obj=remote.Ledger.__new__(remote.Ledger);obj.root=root;obj.actor_pid=actor;obj.actor_start=start
    obj.request=r;obj.request_sha=r['request_sha256'];obj.native_handle=v[a.FIELD][r['request_id']]['native_handle']
    obj.sequence=0;obj.path=root/'evidence/SNAPSHOT.json'
    obj.binding=dict(canonical_parent=str(parent),remote_code=str(code),remote_seal=str(seal),
        remote_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest())
    obj.journal=Journal(control,dict(run_uuid=root.name,job_id=obj.native_handle,source_sha=r['source_sha'],request_sha256=r['request_sha256']))
    actual='assert os.getuid()==22182 and os.environ.get(\'USER\')==\'e1192548\' and not os.environ.get(\'PBS_JOBID\')'
    assert actual in remote.REMOTE
    substitute='assert os.getuid()=='+str(os.getuid())+' and not os.environ.get(\'PBS_JOBID\')'
    monkeypatch.setattr(remote,'REMOTE',remote.REMOTE.replace(actual,substitute))
    obj._argv=lambda:[sys.executable,'-c',"import sys;exec(compile(sys.stdin.read(),'<temp-only-Core-global-program>','exec'))"]
    return obj,ledger


def test_remote_program_exact_serialization_and_safe_once_request_id():
    prefix=remote.REMOTE.split("b=P['binding']",1)[0];ns={};exec(prefix,ns)
    assert ns['raw']({})==b'{}\n'
    assert 'request_id=key,' in remote.REMOTE
    assert "request_id=P['effect_id']" not in remote.REMOTE


def test_same_real_IO_program_read_and_pre_effect_debit_without_PBS_change(local_remote):
    obj,path=local_remote;old=json.loads(path.read_bytes());view=obj.state();assert view==old
    updated=obj._rpc('DEBIT',kind='model_loads',count=1)
    assert updated['model_loads']==old['model_loads']+1
    assert updated['PBS_submissions']==old['PBS_submissions'] and updated['optimizer_updates']==11714
    assert json.loads(path.read_bytes())==updated
    assert obj.journal._state()['status']=='CONFIRMED'
    assert list(path.parent.glob('CORE-DEV-*.RECEIPT.private.json'))


def test_write_applied_but_transport_failed_stays_unknown_and_never_reissued(local_remote):
    obj,path=local_remote;old=json.loads(path.read_bytes())
    obj._argv=lambda:[sys.executable,'-c',"import sys;exec(compile(sys.stdin.read(),'<temp-only-Core-global-program>','exec'));sys.exit(7)"]
    with pytest.raises(Exception):obj._rpc('DEBIT',kind='model_loads',count=1)
    after=json.loads(path.read_bytes());assert after['model_loads']==old['model_loads']+1
    assert obj.journal._state()['status']=='UNKNOWN'
    logs=list(obj.journal.root.glob('TRANSPORT-*.private.json'));assert len(logs)==1
    assert json.loads(logs[0].read_bytes())['returncode']==7
    with pytest.raises(UnknownMutation):obj._rpc('DEBIT',kind='model_loads',count=1)
    assert json.loads(path.read_bytes())==after and len(list(obj.journal.root.glob('TRANSPORT-*.private.json')))==1


def test_forked_actor_cannot_issue_any_global_call(local_remote):
    obj,path=local_remote;old=path.read_bytes();obj.actor_pid+=1
    with pytest.raises(PermissionError):obj.state()
    assert path.read_bytes()==old
