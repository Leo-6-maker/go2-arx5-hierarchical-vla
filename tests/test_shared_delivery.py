import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import shutil
import pytest
from scripts.pi0294.multitask import shared_delivery as d,shared_checkpoint as cp,shared_training as t
from scripts.pi0294.night10.native_relay import sha,raw


def source(root,*,step=8192,closed=True):
    producer=root/'producer';checkpoint=producer/'training/checkpoint-8192';checkpoint.mkdir(parents=True)
    pins={}
    for name in ('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
            'statistics.json','TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json',
            'SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json',
            'RAW_PARENT_REFERENCE.json','M32_MEMORY_EXECUTION.json'):
        f=checkpoint/name;f.write_bytes(raw(cp.raw_reference())if name=='RAW_PARENT_REFERENCE.json'else b'{}\n');pins[name]=sha(f)
    p=t.recipe('1'*64,pins['statistics.json'],pins['TRAIN_POPULATION.json'],pins['TASK_DEV_POPULATION.json'])
    header=dict(schema=cp.SCHEMA,recipe=p,global_step=step,cursor=step*32,standalone_model=False,
        full_resume_with_required_parent=True,required_raw_parent_files_identity=cp.PARENT_ID,files=pins)
    (checkpoint/'CHECKPOINT_COMPLETE.json').write_bytes(raw(header))
    values={'RUN_MANIFEST.json':dict(recipe=p,science_run_id='one',run_uuid='native'),
        'RESULT.json':dict(status='COMPLETE_SHARED_RAW_FINITE_LEARNING',optimizer_updates=step,TRAIN_visits=step*32,error=None),
        'RELAY_CLOSED.json':dict(effects_closed=closed,producer_run_uuid='native'),
        'ACTUAL_DELTA_CHECKPOINT_RESTORE.json':dict(checkpoint_manifest_sha256=sha(checkpoint/'CHECKPOINT_COMPLETE.json'),
            model_reconstruction=True,optimizer_state_entries=42,optimizer_updates=step,scheduler_restored=True,
            RNG_restored=True,sampler_cursor=step*32,frozen_parent_unchanged=True)}
    values['control/LEASE.json']=dict(status='CLOSED'if closed else'ACTIVE',
        manifest_sha256=__import__('hashlib').sha256(raw(values['RUN_MANIFEST.json'])).hexdigest(),
        result_sha256=__import__('hashlib').sha256(raw(values['RESULT.json'])).hexdigest())
    for name,value in values.items():
        f=producer/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw(value))
    return producer,checkpoint


def test_export_exact8192_without_optimizer_and_preserves_full_source(tmp_path):
    producer,checkpoint=source(tmp_path);dest=tmp_path/'inference'
    proof=d.export(producer,dest,guard=lambda:None)
    value=d.validate(dest,manifest_sha256=proof['manifest_sha256'],checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'])
    assert value['global_step']==8192 and value['DEV_conditions']==68
    assert not(dest/'optimizer.pt').exists()and(checkpoint/'optimizer.pt').exists()
    assert (dest/'CHECKPOINT_COMPLETE.json').read_bytes()==(checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes()
    (dest/'target_parameters.pt').write_bytes(b'changed')
    with pytest.raises(PermissionError,match='CHANGED'):d.validate(dest,manifest_sha256=proof['manifest_sha256'],checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'])


@pytest.mark.parametrize('step,closed',[(2048,True),(8192,False)])
def test_live_or_intermediate_is_not_default_policy(tmp_path,step,closed):
    producer,checkpoint=source(tmp_path,step=step,closed=closed)
    with pytest.raises(PermissionError):d.export(producer,tmp_path/'rejected',guard=lambda:None)


def test_private_fixed_commit_round_trip_before_core_model_load(tmp_path,monkeypatch):
    producer,checkpoint=source(tmp_path);server=tmp_path/'server';server.mkdir()
    output=tmp_path/'output';output.mkdir();exported=d.export(producer,output/'package',guard=lambda:None)
    calls=[]
    class API:
        def repo_info(self,**kwargs):return SimpleNamespace(private=True,sha='b'*40)
        def upload_folder(self,**kwargs):
            calls.append(kwargs);shutil.copytree(kwargs['folder_path'],server/kwargs['path_in_repo'])
            return SimpleNamespace(oid='b'*40)
    route=dict(repo='own/private',prefix='one',token_path='EXISTING_ONLY')
    monkeypatch.setattr(d,'hf_api',lambda r:(API(),'TEST_ONLY_NOT_WRITTEN'))
    receipt=d.publish_hf(dict(root=str(output),hf_route=route,science_run_id='one'),exported)
    assert receipt['revision']=='b'*40 and len(calls)==1 and calls[0]['parent_commit']=='b'*40
    def fetch(**kwargs):
        assert kwargs['revision']=='b'*40
        return str(server/kwargs['filename'])
    import sys
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(hf_hub_download=fetch))
    monkeypatch.setitem(sys.modules,'huggingface_hub.errors',SimpleNamespace(EntryNotFoundError=FileNotFoundError))
    core=tmp_path/'core';core.mkdir()
    result=d.receive_hf(dict(root=str(core),hf_route=route,shared_science_run_id='one',
        shared_delivery_root=str(core/'policy'),dispatch_deadline_epoch=__import__('time').time()+180))
    assert result['status']=='COMPLETE_CORE_PRIVATE_SHARED8192_INPUT'
    assert (core/'policy/target_parameters.pt').is_file()and not(core/'policy/optimizer.pt').exists()
    assert not any(b'TEST_ONLY_NOT_WRITTEN'in f.read_bytes()for f in(core/'policy').rglob('*')if f.is_file())


def test_unknown_hf_write_keeps_one_intent_without_retry(tmp_path,monkeypatch):
    producer,checkpoint=source(tmp_path);output=tmp_path/'output';output.mkdir()
    exported=d.export(producer,output/'package',guard=lambda:None);calls=[]
    class API:
        def repo_info(self,**kwargs):return SimpleNamespace(private=True,sha='b'*40)
        def upload_folder(self,**kwargs):calls.append(kwargs);raise ConnectionError('unknown commit')
    monkeypatch.setattr(d,'hf_api',lambda r:(API(),None))
    with pytest.raises(ConnectionError):d.publish_hf(dict(root=str(output),hf_route=dict(repo='own/private',prefix='one'),science_run_id='one'),exported)
    assert len(calls)==1 and(output/'HF_PUBLISH_INTENT.json').exists()
    assert not(output/'HF_PUBLISH_COMPLETE.json').exists()
