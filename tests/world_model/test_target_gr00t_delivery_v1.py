"""Small tensor fixtures for delivery seams; no GR00T/CUDA/PBS qualification."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from go2arx5_vla.world_model import target_gr00t_delivery_v1 as d
from go2arx5_vla.world_model import target_gr00t_profile_v1 as p
from go2arx5_vla.world_model.storage import digest
from scripts.pi0294.generalization.minimal_adapter import LEAVES


class Category(nn.Module):
    def __init__(self):
        super().__init__();self.W=nn.Parameter(torch.arange(44.).reshape(11,2,2));self.b=nn.Parameter(torch.zeros(11,2))


def original():
    m=nn.Module();m.backbone=nn.Linear(2,2);m.action_head=nn.Module()
    for path in LEAVES:
        parent,_,name=path.rpartition('.')
        try:module=m.get_submodule(parent)
        except AttributeError:
            _,_,part=parent.rpartition('.');setattr(m.action_head,part,nn.Module());module=m.get_submodule(parent)
        setattr(module,name,Category())
    m.action_head.model=nn.Module();m.action_head.model.transformer_blocks=nn.ModuleList([nn.Linear(2,2)for _ in range(32)])
    m.action_head.projector=nn.Linear(2,2)
    return m


def make(tmp_path,variant='INTERFACES_LAST2_DIT',endpoint=2000):
    m=original();cp=tmp_path/('checkpoint-'+str(endpoint));cp.mkdir();(cp/'processor').mkdir()
    (cp/'statistics.json').write_text('{"TRAIN_only":true}')
    (cp/'DATASET_MANIFEST.json').write_text('{"test_fixture":true}')
    (cp/'processor/processor_config.json').write_text('{"test_fixture":true}')
    virtual=[];targets=[];table={}
    for name,tensor in m.named_parameters():
        path,_,leaf=name.rpartition('.')
        if path in LEAVES:
            name=path+'.target_'+leaf;targets.append(name);tensor=tensor[10]
        virtual.append(name);table[name]=tensor
    names=p.whitelist(tuple(virtual),tuple(targets),variant)
    rows=[dict(name=n,shape=list(table[n].shape),numel=table[n].numel(),dtype=str(table[n].dtype))for n in names]
    state={n:table[n].detach().clone()+.25 for n in names}
    torch.save(state,cp/'target_parameters.pt')
    (cp/'PARAMETER_CONTRACT.json').write_text(json.dumps(dict(variant=variant,target_category=10,trainable=rows)))
    recipe=p.recipe(digest(cp/'DATASET_MANIFEST.json'),digest(cp/'statistics.json'),mode='FORMAL',variant=variant,endpoint=endpoint)
    start={2000:0,4000:2000,8000:4000}[endpoint]
    spec=dict(recipe=recipe,profile=p.PROFILE,kind=p.KIND,execution_site='VANDA_A40',
        physics=False,WBC_changes=False,start_step=start,GR00T_updates=endpoint-start,WM_updates=0,
        Actor_updates=0,TASK_ADAPTER_updates=0,parent_manifest_sha256=p.PARENT,
        generation_id='synthetic-formal',resume=None if not start else dict(path='synthetic-prior',sha256='a'*64),deadline_epoch=1794200000.,
        source_sha='f'*40,parent=dict(files={'fixed-parent':'synthetic-parent-pin'}))
    files={str(f.relative_to(cp)):digest(f)for f in cp.rglob('*')if f.is_file()}
    (cp/'CHECKPOINT_COMPLETE.json').write_text(json.dumps(dict(files=files,recipe=recipe,
        global_step=endpoint,generation_id=spec['generation_id'])))
    restore=dict(checkpoint_sha256=digest(cp/'CHECKPOINT_COMPLETE.json'),model_parameters_restored=True,
        optimizer_restored=True,scheduler_restored=True,RNG_restored=True,fixed_batch_generation_restored=True)
    package=d.create(cp,tmp_path/'package',spec,restore,guard=lambda:None)
    return m,cp,spec,restore,package,state


@pytest.mark.parametrize('variant',p.VARIANTS)
def test_package_reconstructs_ordinary_parameters_without_changing_frozen_rows(tmp_path,variant):
    m,cp,spec,restore,package,state=make(tmp_path,variant)
    before={n:t.detach().clone()for n,t in m.named_parameters()}
    manifest=d.install(m,package['path'],package['sha256'],spec['parent']['files'])
    for n,t in m.named_parameters():
        path,_,leaf=n.rpartition('.')
        if path in LEAVES:
            assert torch.equal(t[:10],before[n][:10])
            assert torch.equal(t[10],state[path+'.target_'+leaf])
        elif n in state:assert torch.equal(t,state[n])
        else:assert torch.equal(t,before[n])
    assert not any(t.requires_grad for t in m.parameters())
    assert manifest['external_inference_adapter']is False
    assert not (Path(package['path'])/'optimizer.pt').exists()


def test_diagnostic_or_incomplete_restore_cannot_be_delivered(tmp_path):
    m,cp,spec,restore,package,state=make(tmp_path)
    restore['optimizer_restored']=False
    with pytest.raises(PermissionError,match='ACTUAL_COMPLETE_RESTORE'):
        d.create(cp,tmp_path/'invalid',spec,restore,guard=lambda:None)
    assert not (tmp_path/'invalid').exists()
    spec['recipe']=p.recipe('a'*64,'b'*64,mode='FIT4',variant=p.VARIANTS[0],endpoint=200)
    spec['GR00T_updates']=200
    with pytest.raises(PermissionError,match='FORMAL_ENDPOINT'):
        d.create(cp,tmp_path/'diagnostic',spec,restore,guard=lambda:None)


def test_corrupt_asset_and_wrong_parent_are_rejected_before_any_weight_copy(tmp_path):
    m,cp,spec,restore,package,state=make(tmp_path)
    before={n:t.detach().clone()for n,t in m.named_parameters()}
    with pytest.raises(PermissionError,match='EXACT_PARENT'):
        d.install(m,package['path'],package['sha256'],{'different':'parent'})
    (Path(package['path'])/'statistics.json').write_text('{}')
    with pytest.raises(PermissionError,match='PACKAGE_MEMBER'):
        d.install(m,package['path'],package['sha256'],spec['parent']['files'])
    assert all(torch.equal(t,before[n])for n,t in m.named_parameters())


def test_undeclared_backbone_delta_is_not_an_internal_action_package(tmp_path):
    m,cp,spec,restore,package,state=make(tmp_path);root=Path(package['path'])
    c=json.loads((root/'PARAMETER_CONTRACT.json').read_bytes())
    c['trainable'].append(dict(name='backbone.weight',shape=[2,2],numel=4,dtype='torch.float32'))
    (root/'PARAMETER_CONTRACT.json').write_text(json.dumps(c))
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_bytes())
    manifest['files']['PARAMETER_CONTRACT.json']=digest(root/'PARAMETER_CONTRACT.json')
    (root/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(PermissionError,match='INTERNAL_ACTION_WHITELIST'):
        d.install(m,root,digest(root/'PACKAGE_MANIFEST.json'),spec['parent']['files'])


@pytest.mark.parametrize('endpoint',[4000,8000])
def test_later_completed_endpoint_exports_and_reconstructs_its_actual_internal_weights(tmp_path,endpoint):
    m,cp,spec,restore,package,state=make(tmp_path,endpoint=endpoint)
    manifest=d.install(m,package['path'],package['sha256'],spec['parent']['files'])
    assert manifest['global_step']==manifest['recipe']['endpoint']==endpoint
    assert manifest['recipe']['scheduler_horizon']==8000 and manifest['generation_id']=='synthetic-formal'
    assert not (Path(package['path'])/'optimizer.pt').exists()
    for name,value in state.items():
        path,_,leaf=name.rpartition('.')
        target=m.get_parameter(path+'.'+leaf.removeprefix('target_'))[10]if '.target_'in name else m.get_parameter(name)
        assert torch.equal(target,value)


def generations(root,value=0.):
    root.mkdir();rows=[]
    for i in range(4):
        rows.append(dict(episode=str(i),split='TRAIN',anchor=0,seed=i,noise={'identity':i}))
        np.savez(root/(str(i)+'.npz'),physical_H40=np.full((40,10),value),normalized=np.full((1,40,32),value))
    (root/'RESULT.json').write_text(json.dumps(dict(records=rows)))


def test_reconstruction_requires_same_input_noise_and_predeclared_tolerance(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';generations(a);generations(b,1e-7)
    assert d.compare_generations(a,b)['absolute_component_tolerance']==1e-6
    np.savez(b/'3.npz',physical_H40=np.full((40,10),2e-6),normalized=np.zeros((1,40,32)))
    with pytest.raises(PermissionError,match='PREDICTION_MISMATCH'):d.compare_generations(a,b)
    r=json.loads((b/'RESULT.json').read_bytes());r['records'][0]['noise']={'identity':'different'}
    (b/'RESULT.json').write_text(json.dumps(r))
    with pytest.raises(PermissionError,match='SAME_REAL_INPUT_AND_NOISE'):d.compare_generations(a,b)
