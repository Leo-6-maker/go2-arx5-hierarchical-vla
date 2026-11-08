"""Synthetic site/worker seam checks; never a GPU capacity qualification."""
import json,os,time
from pathlib import Path
import pytest
from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
from go2arx5_vla.world_model import target_gr00t_core_v1 as core
from go2arx5_vla.world_model.storage import digest


def spec(mode='CAPACITY',variant='INTERFACES_LAST2_DIT',endpoint=2,microbatch=1):
    return dict(recipe=profile.recipe('a'*64,'b'*64,mode=mode,variant=variant,endpoint=endpoint,microbatch=microbatch),
        profile=profile.PROFILE,kind=profile.KIND,execution_site='CORELAB_4080S',start_step=0,
        GR00T_updates=endpoint,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
        physics=False,WBC_changes=False,parent_manifest_sha256=profile.PARENT,resume=None,deadline_epoch=time.time()+3600)


def test_core_updates_are_internal_same_effective_batch_and_not_vanda_profile_relabel():
    p=profile.validate(spec());assert p['effective_batch']==32 and p['accumulation']==32
    assert p['visual_and_language_backbone_frozen']and p['external_adapter_updates']==0
    profile.validate(spec('FIT4',endpoint=200))


@pytest.mark.parametrize('args',[dict(mode='FORMAL',endpoint=2000),dict(variant='FULL_ACTION_SIDE'),
    dict(variant='INTERFACES_LAST6_DIT'),dict(microbatch=2)])
def test_core_first_qualification_cannot_silently_start_formal_or_larger_scope(args):
    with pytest.raises(PermissionError,match='CORE_FIRST'):profile.validate(spec(**args))


def test_core_failure_preserves_actual_updates_and_closes_without_fake_pbs(tmp_path,monkeypatch):
    from scripts.wm02 import train_target_gr00t_v1 as entry
    from go2arx5_vla.world_model import target_gr00t_training_v1 as training
    root=tmp_path/'child';root.mkdir();(root/'control').mkdir()
    s=dict(spec(),registered_root=str(root),source_sha='f'*40,allocation_request_id='own-core-scope')
    path=root/'EXECUTION_SPEC.json';path.write_text(json.dumps(s))
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE',spec_sha256=digest(path),job_id='own-core-scope')))
    monkeypatch.setattr(os,'getuid',lambda:2009)
    monkeypatch.setattr(os,'uname',lambda:type('U',(),dict(nodename='corelab-4080s'))())
    monkeypatch.delenv('PBS_JOBID',raising=False)
    def fail(s):
        (root/'PROGRESS.json').write_text(json.dumps(dict(global_step=1,GR00T_updates=1)))
        raise RuntimeError('SYNTHETIC_AFTER_REAL_UPDATE_FAILURE')
    monkeypatch.setattr(training,'run',fail)
    with pytest.raises(RuntimeError,match='AFTER_REAL_UPDATE'):entry.main(['--spec',str(path)])
    actual=json.loads((root/'RESULT.json').read_bytes());lease=json.loads((root/'control/LEASE.json').read_bytes())
    assert actual['GR00T_updates']==1 and lease['status']=='CLOSED'and lease['worker_exit_code']==1
    assert json.loads((root/'WORKER_EXIT.private.json').read_bytes())['job_id']=='own-core-scope'


def test_core_worker_cannot_run_under_a_pbs_identity(tmp_path,monkeypatch):
    from scripts.wm02 import train_target_gr00t_v1 as entry
    root=tmp_path/'child';root.mkdir();(root/'control').mkdir()
    s=dict(spec(),registered_root=str(root),source_sha='f'*40,allocation_request_id='own-core-scope')
    path=root/'EXECUTION_SPEC.json';path.write_text(json.dumps(s))
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE',spec_sha256=digest(path),job_id='own-core-scope')))
    monkeypatch.setattr(os,'getuid',lambda:2009);monkeypatch.setenv('PBS_JOBID','other-job')
    with pytest.raises(PermissionError,match='ENVELOPE'):entry.main(['--spec',str(path)])
    assert not(root/'WORKER_STARTED.private.json').exists()


def test_gpu_requires_original_uuid_mask_and_preserves_two_gib_margin(monkeypatch):
    from types import SimpleNamespace
    uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd';monkeypatch.setenv('CUDA_VISIBLE_DEVICES',uuid)
    monkeypatch.setattr(core.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=uuid+', NVIDIA GeForce RTX 4080 SUPER, 14719, 16376\n'))
    actual=core.gpu(dict(GPU_uuid=uuid));assert actual['free_bytes']==14719*1024**2
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(PermissionError,match='SINGLE_SHARED'):core.gpu(dict(GPU_uuid=uuid))


def test_repaired_successors_use_distinct_global_registration_requests(tmp_path,monkeypatch):
    from scripts.pi0294.night10 import native_relay
    requests=[];owner=tmp_path/'owner';owner.mkdir();registry=tmp_path/'registry.json'
    registry.write_text('{"entries":[]}')
    monkeypatch.setattr(core,'scope',lambda c:(Path(c['root']),{}))
    monkeypatch.setattr(core,'gpu',lambda c:dict(free_bytes=16*1024**3))
    def stop_after_once(path,value,key,**kwargs):
        requests.append(key);raise RuntimeError('SYNTHETIC_STOP_AFTER_REGISTER_REQUEST')
    monkeypatch.setattr(native_relay,'once',stop_after_once)
    for suffix in ('first','repaired'):
        root=tmp_path/suffix;root.mkdir();(root/'control').mkdir()
        (root/'ALLOCATION_CONFIG.json').write_text('{}')
        c=dict(root=str(root),owner_root=str(owner),generation_id=suffix,GPU_uuid='synthetic-uuid',
            allocator_limit_bytes=12*1024**3,dataset_sha256='a'*64,statistics_sha256='b'*64,
            source_root='unused',source_sha='f'*40,source_files={},bindings=dict(cosmos='unused'),
            parent={},registry=str(registry),deadline_epoch=time.time()+3600)
        with pytest.raises(RuntimeError,match='STOP_AFTER_REGISTER_REQUEST'):core.run(c)
    assert requests==['first-capacity2-REGISTER','repaired-capacity2-REGISTER']


def test_torch_bare_uuid_matches_same_nvml_gpu_without_accepting_other_device():
    from types import SimpleNamespace
    from go2arx5_vla.world_model.target_gr00t_training_v1 import gpu_identity
    bare='845348f1-1511-dc19-0877-cc97658e3edd';limits=[]
    cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:'NVIDIA GeForce RTX 4080 SUPER',
        get_device_properties=lambda i:SimpleNamespace(uuid=bare,total_memory=16*1024**3),
        set_per_process_memory_fraction=limits.append)
    torch=SimpleNamespace(cuda=cuda)
    s=dict(execution_site='CORELAB_4080S',GPU_uuid='GPU-'+bare,allocator_limit_bytes=12*1024**3)
    assert gpu_identity(torch,s)==s['GPU_uuid']and limits==[.75]
    with pytest.raises(PermissionError,match='EXACT_GPU_UUID'):
        gpu_identity(torch,dict(s,GPU_uuid='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'))


def test_cpu_activation_execution_is_explicit_preserving_recipe_and_core_only():
    s=spec();s['recipe']=profile.recipe('a'*64,'b'*64,mode='CAPACITY',variant='INTERFACES_LAST2_DIT',
        endpoint=2,memory_policy='CORE_LOSSLESS_CPU_ACTIVATIONS_2G_V1')
    r=profile.validate(s);assert r['effective_batch']==32 and r['forward_precision']=='ORIGINAL_BF16_AUTOCAST'
    assert core.activation_policy()['per_microbatch_activation_offload_bytes']==2*1024**3
    assert core.activation_policy()['quantization']is False
    with pytest.raises(PermissionError,match='A40_ORIGINAL'):
        profile.validate(dict(s,execution_site='VANDA_A40'))


def test_unused_logits_option_preserves_all_hidden_tokens_and_frozen_weights():
    import torch
    from types import SimpleNamespace
    class Language(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(1,8).requires_grad_(False)
        def forward(self,input_ids,output_hidden_states=True,labels=None,logits_to_keep=0):
            hidden=input_ids.float().unsqueeze(-1)
            selected=hidden[:,-logits_to_keep:]if logits_to_keep else hidden
            return SimpleNamespace(hidden_states=(hidden,),logits=self.head(selected))
    language=Language();model=SimpleNamespace(backbone=SimpleNamespace(model=language))
    x=torch.arange(20).reshape(1,20);before=language(x,output_hidden_states=True)
    weights={n:t.detach().clone()for n,t in language.named_parameters()}
    core.trim_unused_logits(model);after=language(x,output_hidden_states=True)
    assert torch.equal(before.hidden_states[-1],after.hidden_states[-1])
    assert before.logits.shape==(1,20,8)and after.logits.shape==(1,1,8)
    assert all(torch.equal(weights[n],t)and not t.requires_grad for n,t in language.named_parameters())
    with pytest.raises(PermissionError,match='FEATURES_ONLY'):
        language(x,output_hidden_states=True,labels=x)


@pytest.mark.parametrize('change',[None,'prediction','noise'])
def test_actual_memory_execution_gate_requires_bit_exact_matched_predictions(tmp_path,change):
    import numpy as np
    folders=[tmp_path/'original',tmp_path/'current'];rows=[]
    for index,folder in enumerate(folders):
        folder.mkdir();records=[]
        for i in range(4):
            arrays=dict(physical_H40=np.zeros((40,10)),normalized=np.zeros((40,132)),
                actual_state10=np.zeros(10),expert_policy_H40=np.zeros((40,10)))
            if index==1 and i==0 and change=='prediction':arrays['physical_H40'][0,0]=1e-12
            name='episode'+str(i);file=folder/(name+'.npz');np.savez(file,**arrays)
            records.append(dict(episode=name,anchor=0,seed=i,noise='different'if index==1 and i==0 and change=='noise'else'same',
                command='SYNTHETIC',sha256=digest(file)))
        (folder/'RESULT.json').write_text(json.dumps(dict(records=records)))
    ref=dict(path=str(folders[0]/'RESULT.json'),sha256=digest(folders[0]/'RESULT.json'))
    if change is None:assert core.verify_memory_predictions(ref,folders[1])['observations']==4
    else:
        with pytest.raises(PermissionError):core.verify_memory_predictions(ref,folders[1])
