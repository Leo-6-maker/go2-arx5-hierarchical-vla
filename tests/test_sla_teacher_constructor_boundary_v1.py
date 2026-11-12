"""I/O boundary unit checks; no teacher model, inference, or optimization."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from scripts.sla.cache_teacher_v1 import offline_constructor


def inputs(tmp_path,monkeypatch):
    asset=tmp_path/'asset';asset.mkdir()
    payload=asset/'patch.pth';payload.write_bytes(b'unit-test-serialization-placeholder')
    manifest=dict(schema='SLA_TEACHER_OWN_PATCH_EMBED_PREINIT_ASSET_V1',passes=True,
        source_teacher_weight_sha256='c9c3772b9bbfc648fa95ebffb3d9ff856f21e9b9712685eb0f98add53897969d',
        external_DINO_weights_used=False,asset_roundtrip_bit_exact=True,
        file=payload.name,file_sha256=sha(payload),arrays=[dict(key='component')])
    m=asset/'MANIFEST.json';m.write_text(json.dumps(manifest))
    run=tmp_path/'run';run.mkdir()
    calls=[]
    def load(selected,*args,**kwargs):
        calls.append((Path(selected).resolve(),kwargs))
        return dict(component=object())
    stub=SimpleNamespace(load=load,hub=SimpleNamespace(load=lambda *args,**kwargs:None))
    monkeypatch.setitem(sys.modules,'torch',stub)
    spec=dict(registered_root=str(run),teacher_preinit=dict(path=str(m),sha256=sha(m)))
    counts=dict(constructor_preinit_load_attempts=0,constructor_preinit_loads=0,constructor_hub_load_attempts=0)
    return stub,spec,counts,calls,payload


def test_offline_constructor_reads_only_the_pinned_local_alias(tmp_path,monkeypatch):
    torch,spec,counts,calls,payload=inputs(tmp_path,monkeypatch)
    before=Path.cwd();result=object()
    def factory():
        torch.load('checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth')
        return result
    assert offline_constructor(spec,counts,factory)is result
    assert Path.cwd()==before and calls==[(payload,dict(weights_only=True))]
    assert counts==dict(constructor_preinit_load_attempts=1,constructor_preinit_loads=1,constructor_hub_load_attempts=0)
    assert not(Path(spec['registered_root'])/'checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth').exists()


def test_constructor_cannot_fall_back_to_a_hub_download(tmp_path,monkeypatch):
    torch,spec,counts,calls,_=inputs(tmp_path,monkeypatch)
    before=Path.cwd()
    with pytest.raises(RuntimeError,match='REQUIRES_NO_HUB_DOWNLOAD'):
        offline_constructor(spec,counts,lambda:torch.hub.load('another-repo','another-model'))
    assert Path.cwd()==before and not calls and counts['constructor_hub_load_attempts']==1


def test_constructor_cannot_read_another_local_checkpoint(tmp_path,monkeypatch):
    torch,spec,counts,calls,_=inputs(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNDECLARED_CONSTRUCTOR_WEIGHT_READ'):
        offline_constructor(spec,counts,lambda:torch.load('/unregistered/model.pth'))
    assert not calls and counts['constructor_preinit_loads']==0
