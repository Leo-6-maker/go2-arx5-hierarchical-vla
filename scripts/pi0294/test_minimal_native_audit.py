"""Real CPU serialization/files, synthetic device tags; no native GPU audit."""
import json,os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import torch
from scripts.pi0294.generalization import minimal_native_audit as n,minimal_resume as r,minimal_study as s
from scripts.pi0294.dec1.readonly_checkpoint_hash import StreamHashes


def optimizer():
    return dict(state={0:dict(step=torch.tensor(2.),exp_avg=torch.arange(8,dtype=torch.float32),
        exp_avg_sq=torch.ones(8))},param_groups=[dict(params=[0],lr=1e-5,weight_decay=.01)])


def test_real_CPU_AdamW_nested_state_matches_original_saved_identity(tmp_path):
    v=optimizer();p=tmp_path/'optimizer.pt';torch.save(v,p)
    loaded,proof=n.serialized_state(p)
    assert proof['sha256']==r.fingerprint(v)==r.fingerprint(loaded)
    assert proof['serialized_devices']==['cpu'] and not torch.cuda.is_initialized()


def item(tree,key):return next(row['value']for row in tree['items']if row['key']==key)


def test_CPU_mapping_preserves_mixed_original_device_tags_not_all_CUDA_or_all_CPU(tmp_path,monkeypatch):
    v=optimizer();moments=v['state'][0];cuda={int(moments[k].untyped_storage()._cdata)for k in('exp_avg','exp_avg_sq')}
    registry=torch.serialization._package_registry
    # Storage tags are deliberately synthesized in this CPU-only fixture.
    # They are not evidence of a CUDA model, allocation or optimizer run.
    monkeypatch.setattr(torch.serialization,'_package_registry',[
        (0,lambda storage:'cuda:0'if int(storage._cdata)in cuda else None,lambda storage,location:None),*registry])
    p=tmp_path/'mixed-tag.pt';torch.save(v,p);expected=r.tree_identity(v)
    state=item(item(expected,'state'),0)
    for key in('exp_avg','exp_avg_sq'):item(state,key)['device']='cuda:0'
    loaded,proof=n.serialized_state(p)
    assert proof['serialized_devices']==['cpu','cuda:0'] and proof['sha256']==s.digest(expected)
    assert proof['sha256']!=r.fingerprint(loaded)
    assert str(loaded['state'][0]['step'].device)=='cpu'and not torch.cuda.is_initialized()


def test_unexpected_serialized_second_GPU_is_not_silently_mapped_to_own_state(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.serialization,'_package_registry',[
        (0,lambda storage:'cuda:1',lambda storage,location:None),*torch.serialization._package_registry])
    p=tmp_path/'foreign.pt';torch.save(optimizer(),p)
    with pytest.raises(PermissionError,match='SINGLE_GPU_SERIALIZED_DEVICE'):n.serialized_state(p)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind',['extra','missing','symlink','writable','changed'])
def test_full_saved_member_inventory_rejects_hidden_alias_or_unsealed_bytes(tmp_path,kind):
    (tmp_path/'model.safetensors').write_bytes(b'CPU-fixture-not-a-neural-model')
    full=dict(files={'model.safetensors':__import__('hashlib').sha256((tmp_path/'model.safetensors').read_bytes()).hexdigest()})
    (tmp_path/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(full))
    for p in tmp_path.iterdir():p.chmod(0o400)
    if kind=='extra':(tmp_path/'hidden.pt').write_bytes(b'extra')
    elif kind=='missing':(tmp_path/'model.safetensors').unlink()
    elif kind=='symlink':
        (tmp_path/'model.safetensors').unlink();(tmp_path/'model.safetensors').symlink_to(tmp_path/'CHECKPOINT_COMPLETE.json')
    elif kind=='writable':(tmp_path/'model.safetensors').chmod(0o600)
    else:
        (tmp_path/'model.safetensors').chmod(0o600);(tmp_path/'model.safetensors').write_bytes(b'tamper');(tmp_path/'model.safetensors').chmod(0o400)
    with pytest.raises(PermissionError):n.own_members(tmp_path,full,StreamHashes())


def test_real_saved_draw_checks_full132_padding_and_input_noise(tmp_path):
    p=tmp_path/'draw.npz'
    def save(padding=0,seed=29400,signature='actual-input-fixture'):
        a=np.zeros((1,40,132));a[0,1,131]=padding
        np.savez(p,physical_H40=np.zeros((40,10)),normalized=a,noise_seed=np.asarray(seed),observation_fingerprint=np.asarray(signature))
    save();assert len(n.draw(p,StreamHashes(),seed=29400,input_fingerprint='actual-input-fixture'))==64
    for seed,signature in((29401,'actual-input-fixture'),(29400,'other-input')):
        with pytest.raises(PermissionError):n.draw(p,StreamHashes(),seed=seed,input_fingerprint=signature)
    save(padding=np.nan)
    with pytest.raises(PermissionError):n.draw(p,StreamHashes(),seed=29400,input_fingerprint='actual-input-fixture')
