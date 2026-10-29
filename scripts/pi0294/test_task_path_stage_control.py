import copy
import json
from pathlib import Path
import zlib

import pytest

from scripts.pi0294.dec1.owner_transport import Journal, UnknownMutation
from scripts.pi0294 import task_path_stage_control as unit


def fixture(tmp_path):
    identity=dict(run_uuid='run',job_id='S1_NATIVE_COLD_NO_PBS',source_sha='a'*40,request_sha256='b'*64)
    journal=Journal(tmp_path,identity,clock=lambda:110)
    with pytest.raises(UnknownMutation):
        journal.call('S1_COMPLETE_NATIVE_COLD_STAGE_ONCE','MUTATION',
                     lambda *args: (_ for _ in ()).throw(TimeoutError()))
    intent=json.loads(journal.state_path.read_text())['intent']
    proof=dict(schema='S1_COLD_STAGE_EXACT_NO_EFFECT_READBACK_V1',intent=intent,
        host='corelab-4080s',account_sha256='c'*64,registry_sha256='d'*64,
        absent_paths={n:True for n in unit.ABSENT},matching_stage_processes=[],read_only=True,observed_epoch=110)
    return journal,proof


def test_exact_reconciliation_retains_original_error(tmp_path):
    journal,proof=fixture(tmp_path)
    before={p.name:p.read_bytes()for p in tmp_path.glob('*-ERROR.json')}
    result=unit.reconcile_no_effect(journal,proof,account_sha256='c'*64,registry_sha256='d'*64)
    assert result['status']=='CONFIRMED_ZERO_EFFECT_NO_REPLAY'
    assert before=={p.name:p.read_bytes()for p in tmp_path.glob('*-ERROR.json')}
    assert json.loads(journal.state_path.read_text())['resolution']=='ACTUAL_NO_EFFECT_AFTER_TRANSPORT_TIMEOUT'
    with pytest.raises(PermissionError):
        unit.reconcile_no_effect(journal,proof,account_sha256='c'*64,registry_sha256='d'*64)


@pytest.mark.parametrize('fault',['live','files','account','registry','intent','stale','future','effect'])
def test_uncertain_or_mismatched_readback_keeps_unknown(tmp_path,fault):
    journal,proof=fixture(tmp_path);proof=copy.deepcopy(proof)
    if fault=='live':proof['matching_stage_processes']=[dict(pid=12,proc_start='3')]
    elif fault=='files':proof['absent_paths']['native-entry']=False
    elif fault in ('account','registry'):proof[fault+'_sha256']='e'*64
    elif fault=='intent':proof['intent']['operation_id']='other'
    elif fault=='stale':proof['observed_epoch']=-100
    elif fault=='future':proof['observed_epoch']=111
    elif fault=='effect':proof['read_only']=False
    with pytest.raises(PermissionError):
        unit.reconcile_no_effect(journal,proof,account_sha256='c'*64,registry_sha256='d'*64)
    assert json.loads(journal.state_path.read_text())['status']=='UNKNOWN'


def test_capsule_compression_and_nested_ssh_quoting(monkeypatch):
    seen={}
    def run(args,**kwargs):
        seen.update(args=args,**kwargs)
        return 'sent-once'
    monkeypatch.setattr(unit.subprocess,'run',run)
    payload=dict(text="quotes ' ; $HOME `echo dangerous`",body='repeat'*100000)
    assert unit.compressed_call('print(P)',payload)=='sent-once'
    assert json.loads(zlib.decompress(seen['input']))==dict(program='print(P)',payload=payload)
    assert len(seen['input'])<10000
    import shlex
    inner=shlex.split(seen['args'][-1]);assert inner[-2]=='gpu4080s'
    remote=shlex.split(inner[-1]);assert remote[:2]==['python3','-c']
    compile(remote[2],'<boot>','exec')
    assert payload['text']not in remote[2]and seen['timeout']==35
