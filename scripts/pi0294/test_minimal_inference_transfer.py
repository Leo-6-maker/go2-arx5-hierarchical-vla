"""CPU artifact fixtures; no new closed native endpoint or HF IO is claimed."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import pytest

from scripts.pi0294.generalization import minimal_inference_transfer as transfer
from scripts.pi0294.generalization import minimal_inference_export as export
from scripts.pi0294.generalization import minimal_execution as e


@pytest.fixture(params=e.training.ARMS)
def packet(monkeypatch,request):
    arm=request.param;raw=transfer.original.encode
    stats=raw({'fixture_statistics':True});monkeypatch.setattr(e,'R0',e.sha(stats))
    endpoint=dict(global_step=512,cursor=16384,scheduler_horizon=1024,full_resume=True,
        qualification_only=False,TRAIN_probes=12,language_probes=24,statistics_sha256=e.R0,
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,checkpoint_manifest_sha256='b'*64)
    reference=dict(schema=export.SCHEMA,arm=arm,global_step=512,cursor=16384,scheduler_horizon=1024,
        statistics_sha256=e.R0,raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,source_sha='c'*40,
        wrapped_checkpoint_manifest_sha256=endpoint['checkpoint_manifest_sha256'],
        full_resume_checkpoint_retained=True,ordinary_upstream_full_model=True,adapter_only_file=False)
    index={'weight_map':{'actual_fixture_weight':'model-00001-of-00001.safetensors'}}
    files={'config.json':raw({'model_name':'/scratch/fixture/Cosmos','action_horizon':40}),
        'model.safetensors.index.json':raw(index),'statistics.json':stats,
        'processor/processor_config.json':raw({'processor_kwargs':{'model_name':'/scratch/fixture/Cosmos'}}),
        'processor/statistics.json':stats,'processor/embodiment_id.json':raw({'NEW_EMBODIMENT':10}),
        'FULL_SOURCE_CHECKPOINT_REFERENCE.json':raw(reference),
        'model-00001-of-00001.safetensors':b'opaque CPU fixture, never parsed as a real model'}
    complete=dict(schema=export.SCHEMA,arm=arm,global_step=512,cursor=16384,scheduler_horizon=1024,
        full_resume=False,original_full_resume_retained=True,statistics_sha256=e.R0,
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
        wrapped_checkpoint_manifest_sha256=endpoint['checkpoint_manifest_sha256'],
        files={n:e.sha(b)for n,b in files.items()})
    complete_raw=raw(complete);endpoint['official_inference_manifest_sha256']=e.sha(complete_raw)
    root=Path('/fixture/inference')/arm
    all_files=dict(files,**{'CHECKPOINT_COMPLETE.json':complete_raw})
    proof=dict(GLOBAL_sha256='d'*64,source_sha='c'*40,endpoint=endpoint,audit=dict(streamed_witnesses=dict(
        all_paths_rechecked=True,all_unique_inodes_independently_rehashed=True,
        paths=[dict(path=str(root/n),bytes=len(b),sha256=e.sha(b))for n,b in all_files.items()])))
    small=dict(schema='ARX5_MINIMAL_CLOSED512_INFERENCE_SMALL_READBACK_V1',arm=arm,
        GLOBAL_sha256=proof['GLOBAL_sha256'],tensor_payloads_read=False,official_root=str(root),
        files={n:dict(base64=base64.b64encode(all_files[n]).decode(),bytes=len(all_files[n]),sha256=e.sha(all_files[n]))
            for n in transfer.SMALL+('CHECKPOINT_COMPLETE.json',)})
    return dict(arm=arm,complete=complete,index=index,reference=reference,endpoint=endpoint,
        files=files,proof=proof,small=small)


def isolated_prepare(monkeypatch,packet):
    # Only artifact construction is isolated here. This does not assert that
    # the actual running job passed the production closed_endpoint validator.
    monkeypatch.setattr(transfer,'closed_endpoint',lambda *a,**k:packet['proof'])
    return transfer.prepare(b'fixture',b'fixture',b'fixture',{},
        transfer.original.encode(packet['small']),arm=packet['arm'])


def test_ordinary512_artifact_roundtrip_restores_exact_two_private_configs(monkeypatch,packet,tmp_path):
    result=isolated_prepare(monkeypatch,packet);public=dict(packet['files'])
    for name,b in result['public_rewritten_metadata_base64'].items():public[name]=base64.b64decode(b)
    transfer.original.reject_private(result['public_plan'])
    # Exercise the existing stable file stream reader for an opaque shard.
    name='model-00001-of-00001.safetensors';p=tmp_path/name;p.write_bytes(public[name]);public[name]=p
    restored=transfer.restore_original(public,result['private_restoration'],arm=packet['arm'])
    assert {n:b.read_bytes()if isinstance(b,Path)else b for n,b in restored.items()}==packet['files']
    through_existing=transfer.original.restore_original(public,result['private_restoration'],candidate=packet['arm'])
    assert through_existing==restored
    assert result['resource_counters_changed']is False and result['HF_IO_performed']is False
    assert result['public_plan']['native_admission']is False
    assert set(result['private_restoration']['original_metadata_base64'])==set(transfer.original.REWRITTEN)


@pytest.mark.parametrize('change',['qualification','wrapped','wrong_arm','step1024','horizon512',
    'wrong_R0','missing_probe','wrong_wrapped_reference','optimizer','index_escape','missing_shard'])
def test_qualification_wrong_identity_or_non_inference_members_refused(packet,change):
    c=deepcopy(packet['complete']);i=deepcopy(packet['index']);r=deepcopy(packet['reference']);p=deepcopy(packet['endpoint'])
    if change=='qualification':c['global_step']=2;c['cursor']=64
    elif change=='wrapped':c['full_resume']=True
    elif change=='wrong_arm':r['arm']=next(x for x in e.training.ARMS if x!=packet['arm'])
    elif change=='step1024':c['global_step']=1024
    elif change=='horizon512':p['scheduler_horizon']=512
    elif change=='wrong_R0':c['statistics_sha256']='e'*64
    elif change=='missing_probe':p['TRAIN_probes']=11
    elif change=='wrong_wrapped_reference':r['wrapped_checkpoint_manifest_sha256']='e'*64
    elif change=='optimizer':c['files']['optimizer.pt']='e'*64
    elif change=='index_escape':i['weight_map']['actual_fixture_weight']='../model-00001-of-00001.safetensors'
    elif change=='missing_shard':c['files'].pop('model-00001-of-00001.safetensors')
    with pytest.raises(PermissionError):transfer.ordinary_members(c,i,r,arm=packet['arm'],endpoint=p)


@pytest.mark.parametrize('change',['metadata_byte','different_global','no_final_stream','duplicate_witness','wrong_source'])
def test_postclosure_metadata_must_join_same_independently_streamed_export(monkeypatch,packet,change):
    if change=='metadata_byte':packet['small']['files']['config.json']['base64']=base64.b64encode(b'{}').decode()
    elif change=='different_global':packet['small']['GLOBAL_sha256']='e'*64
    elif change=='no_final_stream':packet['proof']['audit']['streamed_witnesses']['all_unique_inodes_independently_rehashed']=False
    elif change=='duplicate_witness':packet['proof']['audit']['streamed_witnesses']['paths']*=2
    else:packet['proof']['source_sha']='e'*40
    with pytest.raises(PermissionError):isolated_prepare(monkeypatch,packet)


@pytest.mark.parametrize('change',['tensor_corruption','other_arm','missing_member','private_bundle_config'])
def test_downloaded_corruption_or_arm_swap_never_restores(monkeypatch,packet,change):
    result=isolated_prepare(monkeypatch,packet);public=dict(packet['files']);bundle=result['private_restoration'];arm=packet['arm']
    for name,b in result['public_rewritten_metadata_base64'].items():public[name]=base64.b64decode(b)
    if change=='tensor_corruption':public['model-00001-of-00001.safetensors']+=b'bad'
    elif change=='other_arm':arm=next(x for x in e.training.ARMS if x!=packet['arm'])
    elif change=='missing_member':public.pop('statistics.json')
    else:bundle['original_metadata_base64']['config.json']=base64.b64encode(b'{}').decode()
    with pytest.raises(PermissionError):transfer.restore_original(public,bundle,arm=arm)


@pytest.mark.parametrize('status,complete',[('BOUND',False),('NATIVE_TERMINAL',False)])
def test_running_and_failed_training_cannot_prepare_any_delivery(status,complete):
    g=dict(PBS_submissions=[dict(platform_training_version=transfer.remaining.VERSION,status=status,
        scientific_endpoint_complete=complete)],pending_updates=[],unknown_effects=[],
        **{transfer.remaining.FIELD:dict(status='REGISTERED')})
    with pytest.raises(PermissionError):transfer.closed_endpoint(transfer.original.encode(g),b'{}',b'{}',{},arm=e.training.ARMS[0])


@pytest.mark.parametrize('change',[None,'live_owner','live_wrapper','OS1','foreign_effects','foreign_owner',
    'wrong_lease','wrong_settlement','close_after_OS','nonfinite_clock'])
def test_closure_seam_delegates_original_full_validator_and_binds_own_OS0(monkeypatch,packet,change):
    # Isolate only the new settlement/lease/owner seam. Full GLOBAL history,
    # physical custody and terminal validation belong to the original suite;
    # this compact fixture is deliberately not a native closed GLOBAL.
    raw=transfer.original.encode;row=dict(platform_training_version=transfer.remaining.VERSION,
        status=e.STATES[2],scientific_endpoint_complete=True,request_id='fixture-request',
        request_sha256='a'*64,job_id='12345.stdct-mgmt-02',source_sha='c'*40,
        run_manifest_sha256='f'*64,run_uuid='fixture-run',settled_epoch=100.)
    terminal=dict(native={'fixture_native':True},effects={'fixture_effects':True},
        audit={'endpoints':{packet['arm']:packet['endpoint']}})
    g=dict(PBS_submissions=[row],pending_updates=[],unknown_effects=[],
        **{transfer.remaining.FIELD:dict(status='COMPLETE',terminal=terminal)})
    settled=raw(terminal);lease=dict(status='CLOSED',run_uuid=row['run_uuid'],
        manifest_sha256=row['run_manifest_sha256'],terminal_receipt_sha256=e.sha(settled),closed_epoch=101.)
    obs=dict(epoch=103.,GLOBAL_sha256=e.sha(raw(g)),same_owner_live=False,same_wrapper_live=False,
        effects={k:row[k]for k in('request_id','request_sha256','job_id','source_sha')},
        **{'OS_EXIT.json':dict(returncode=0,foreground_wait_completed=True,finished_epoch=102.,source_sha='d'*40),
           'CONTINUATION_PROCESS.json':dict(science_source_sha=row['source_sha'],controller_source_sha='d'*40,pid=999,procstart='123')})
    calls=[];monkeypatch.setattr(transfer.remaining,'validate',lambda value:calls.append(value))
    if change=='live_owner':obs['same_owner_live']=True
    elif change=='live_wrapper':obs['same_wrapper_live']=True
    elif change=='OS1':obs['OS_EXIT.json']['returncode']=1
    elif change=='foreign_effects':obs['effects']['job_id']='54321.stdct-mgmt-02'
    elif change=='foreign_owner':obs['CONTINUATION_PROCESS.json']['science_source_sha']='e'*40
    elif change=='wrong_lease':lease['manifest_sha256']='e'*64
    elif change=='wrong_settlement':settled=raw(dict(terminal,native={'different':True}))
    elif change=='close_after_OS':lease['closed_epoch']=104.
    elif change=='nonfinite_clock':obs['epoch']=float('inf')
    if change:
        with pytest.raises(PermissionError):transfer.closed_endpoint(raw(g),raw(lease),settled,obs,arm=packet['arm'])
    else:
        result=transfer.closed_endpoint(raw(g),raw(lease),settled,obs,arm=packet['arm'])
        assert result['endpoint']==packet['endpoint']and result['job_id']==row['job_id']
    assert calls==[g]
