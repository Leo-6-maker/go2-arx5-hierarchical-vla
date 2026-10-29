import ast
import copy
import json

import pytest

from scripts.pi0294.dec1.owner_transport import Journal,UnknownMutation
from scripts.pi0294 import task_path_entry_tail as unit


def fixture(tmp_path):
    identity=dict(run_uuid='run',job_id='S1_NATIVE_COMPRESSED_COLD_NO_PBS',source_sha='a'*40,request_sha256='b'*64)
    journal=Journal(tmp_path,identity,clock=lambda:110)
    with pytest.raises(UnknownMutation):
        journal.call('S1_NATIVE_COMPRESSED_COLD_CONTINUATION_ONCE','MUTATION',
            lambda *args: (_ for _ in ()).throw(TimeoutError()))
    intent=json.loads(journal.state_path.read_text())['intent']
    expected=dict(intent=intent,expected_files={'account':'paid'},registry_before_sha256='original',absent_tail=['adoption'])
    proof=dict(schema='S1_COLD_IMPORT_REJECTION_EXACT_PARTIAL_V1',intent=intent,observed_epoch=110,
        files=expected['expected_files'],registry_sha256='original',source_seal_verified=True,
        preparation_receipt=dict(status='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'),absent_tail=['adoption'],
        matching_processes=[],starts=0,integrations=0,RGB_pairs=0,read_only=True)
    transport=dict(returncode=1,stdout='',stderr="ModuleNotFoundError: No module named 'rclpy'")
    return journal,proof,expected,transport


def test_partial_readback_preserves_charge_and_error(tmp_path):
    j,p,e,t=fixture(tmp_path);old={x.name:x.read_bytes()for x in tmp_path.glob('*-ERROR.json')}
    result=unit.reconcile_partial_import_failure(j,p,e,t)
    assert result['status']=='KNOWN_PARTIAL_NO_REPLAY_NO_REFUND'
    assert old=={x.name:x.read_bytes()for x in tmp_path.glob('*-ERROR.json')}


@pytest.mark.parametrize('fault',['process','paid_file','source','exit','physics','stale'])
def test_partial_unknown_cannot_be_cleared_without_exact_evidence(tmp_path,fault):
    j,p,e,t=fixture(tmp_path);p=copy.deepcopy(p)
    if fault=='process':p['matching_processes']=[dict(pid=12)]
    elif fault=='paid_file':p['files']['account']='other'
    elif fault=='source':p['source_seal_verified']=False
    elif fault=='exit':t['returncode']=0
    elif fault=='physics':p['integrations']=1
    elif fault=='stale':p['observed_epoch']=-100
    with pytest.raises(PermissionError):unit.reconcile_partial_import_failure(j,p,e,t)
    assert json.loads(j.state_path.read_text())['status']=='UNKNOWN'


def test_tail_keeps_original_capsule_and_does_not_repeat_preparation_charge():
    tree=ast.parse(unit.TAIL_PROGRAM)
    raw=next(n for n in ast.walk(tree)if isinstance(n,ast.Lambda)and 'json.dumps' in ast.unparse(n))
    encode=eval(compile(ast.Expression(raw),'<raw>','eval'),dict(json=json))
    assert json.loads(encode({'proof':'exact'}))=={'proof':'exact'}
    assert 'source_tar_base64'not in unit.TAIL_PROGRAM
    assert "account['Core_CPU_upper_charged_s']+="not in unit.TAIL_PROGRAM
    assert "repr([str(code),str(code/'src'),*inputs['runtime_environment']['PYTHONPATH'].split(':')])"in unit.TAIL_PROGRAM
    command=unit.worker_bootstrap('/actual root', 'S1_SHORT_T1')
    compile(command,'<worker>','exec')
    assert "os.environ['PYTHONPATH']=p"in command
