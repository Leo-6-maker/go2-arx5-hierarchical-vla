"""Real temporary raw files and images; model/native observations are fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.e1 import audit as original
from scripts.pi0294.pv1 import audit as pv1


@pytest.fixture
def veto(tmp_path,monkeypatch):
    root=tmp_path/'registered-run';identifier='FIRST-REFERENCE-VETO'
    folder=root/'episodes'/identifier;folder.mkdir(parents=True)
    result=dict(id=identifier,physics_steps=0,source_sha='a'*40,status='FAILED_PRESERVED')
    write(folder/'RESULT.json',result)
    row=dict(attempt_id=identifier,status='VALID_TASK_FAIL',failure_stage='ORIGINAL_REFERENCE_GATE_REJECT',
             pipeline_trustworthy=True,main_integrations=0,result_sha256=sha(folder/'RESULT.json'))
    files={'CARD.json':dict(id=identifier),'PV1_ADJUDICATION.json':row,
           'WORKER_RECEIPT.json':dict(id=identifier,physics_steps=0,source_sha='a'*40,rgb_pairs=1),
           'MODEL_REJECT.json':dict(query_uuid='CPU_ONLY_QUERY',action_sha256='b'*64,
                                   original_gate=dict(decision='REJECT',issues=[dict(code='LINEAR_SPEED_REJECT')])),
           'samples.json':[],'references.json':[],
           'S1_RESULT.json':dict(shadow_invocations=0,pre_execution_rejections=0,live_native_failures=0)}
    for name,value in files.items():write(folder/name,value)
    np.savez(folder/'highrate.npz',rows=np.empty((0,5)))
    for name in ['body_rgb','wrist_rgb']:
        (folder/name).mkdir();Image.new('RGB',(256,256),(10,20,30)).save(folder/name/'0000.png')
    for name in ['intents','commands','acks']:(folder/'relay'/name).mkdir(parents=True)
    manifest=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha(p))
              for p in folder.rglob('*')if p.is_file()]
    write(folder/'FILE_MANIFEST.json',manifest)
    prior=root/'evidence/audits'/identifier/'AUDIT.private.json';prior.parent.mkdir(parents=True)
    write(prior,dict(status='ENGINEERING_AUDIT_INCOMPLETE',error='missing physics.npz'))
    primary=root/'evidence/PV1_AUDIT_RECEIPTS'/(identifier+'.json');primary.parent.mkdir()
    write(primary,read(prior))
    worker=root/'worker_processes'/(identifier+'.json');worker.parent.mkdir()
    write(worker,dict(pid=2**30,proc_start='1'))
    monkeypatch.setattr(original,'video',lambda folder,path,n:dict(actual_paired_frames=n,CPU_video_fixture=True))
    monkeypatch.setattr(pv1,'validate_runtime',lambda *a,**k:None)
    monkeypatch.setattr(pv1,'Ledger',lambda *a,**k:SimpleNamespace(guard=lambda **kw:None,cpu=lambda cost:None))
    return SimpleNamespace(root=root,identifier=identifier,folder=folder,row=row,prior=prior,primary=primary)


def test_first_reference_veto_uses_raw_zero_execution_without_synthesizing_physics(veto):
    v=veto;before=v.prior.read_bytes();primary=v.primary.read_bytes()
    result=pv1.run(v.root,v.identifier)
    assert result['status']=='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
    assert result['adjudication']==v.row and result['physical_integrations']==0
    assert result['score']is None and result['policy_outcome_requires_complete_causal_join']is True
    assert result['video']['logical_id']=='PV1_MAIN/'+v.identifier
    assert result['video']['initial_frame_only']is True and result['video']['task_trajectory_video']is False
    assert result['reference_veto_audit_revision']=='ZERO_REFERENCE_VETO_V2'
    assert result['prior_PV1_audit_receipt_sha256']==sha(v.primary)
    assert v.prior.read_bytes()==before and v.primary.read_bytes()==primary
    assert not(v.folder/'physics.npz').exists()and not(v.folder/'SCORE.json').exists()
    assert result==pv1.run(v.root,v.identifier)


@pytest.mark.parametrize('fault',['command','labels','highrate','missing_reject','gate_pass','nonzero_steps'])
def test_reference_veto_never_hides_execution_or_a_missing_reject(veto,fault):
    v=veto
    if fault=='command':write(v.folder/'relay/acks/000.json',dict(unexpected=True))
    if fault=='labels':write(v.folder/'samples.json',[dict(supervision='NOT_ALLOWED')])
    if fault=='highrate':np.savez(v.folder/'highrate.npz',rows=np.ones((1,5)))
    if fault=='missing_reject':(v.folder/'MODEL_REJECT.json').unlink()
    if fault=='gate_pass':
        value=read(v.folder/'MODEL_REJECT.json');value['original_gate']['decision']='PASS';write(v.folder/'MODEL_REJECT.json',value)
    if fault=='nonzero_steps':
        value=read(v.folder/'RESULT.json');value['physics_steps']=200;write(v.folder/'RESULT.json',value)
    value=original.episode(v.root,v.identifier,adjudication_name='PV1_ADJUDICATION.json')
    assert value['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'


def test_engineering_zero_step_keeps_its_non_policy_classification(veto):
    v=veto;row=dict(v.row,status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False)
    write(v.folder/'PV1_ADJUDICATION.json',row)
    value=original.episode(v.root,v.identifier,adjudication_name='PV1_ADJUDICATION.json')
    assert value['scientific_policy_outcome_valid']is False
    assert value['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
