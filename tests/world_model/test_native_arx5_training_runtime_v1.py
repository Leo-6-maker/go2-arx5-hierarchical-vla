"""Real optimizer/RNG save-restore seams, without claiming GR00T qualification."""
import copy
import random
import json
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from go2arx5_vla.world_model import native_arx5_training_runtime_v1 as runtime
from go2arx5_vla.world_model import native_arx5_training_profile_v1 as profile
from go2arx5_vla.world_model import task_learning_features_v1 as features
from scripts.pi0294.native_train import tree_identity,rng_identity


class Contract:
    trainable_names=('weight','bias')
    def __init__(self,model):self.model=model
    def describe(self):return dict(identity='CPU_TEST_ONLY_NOT_GR00T',names=list(self.trainable_names))


def session():
    model=torch.nn.Linear(3,2)
    contract=Contract(model)
    recipe=profile.recipe('A_TARGET','a'*64,'b'*64)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,profile.lr_multiplier)
    return model,contract,optimizer,scheduler,recipe


def next_update(model,optimizer,scheduler):
    optimizer.zero_grad(set_to_none=True)
    inputs=torch.randn(32,3)+random.random()+np.random.random()
    loss=model(inputs).square().mean();loss.backward();optimizer.step();scheduler.step()
    return dict(loss=loss.item().hex(),parameters=tree_identity(model.state_dict()),
                optimizer=tree_identity(optimizer.state_dict()),scheduler=scheduler.state_dict(),RNG=rng_identity())


def test_fresh_session_restore_preserves_next_update_and_rejects_wrong_stage(tmp_path):
    random.seed(199);np.random.seed(199);torch.manual_seed(199)
    model,contract,opt,sch,recipe=session()
    next_update(model,opt,sch)
    checkpoint=tmp_path/'step1';cache=dict(path='new-cache',sha256='c'*64)
    frozen=dict(identity='CPU_TEST_FROZEN')
    runtime.save_checkpoint(checkpoint,model,contract,opt,sch,1,recipe,cache,frozen,'d'*64)
    reference=next_update(model,opt,sch)
    rebuilt,c2,o2,s2,_=session()
    assert runtime.restore_checkpoint(checkpoint,rebuilt,c2,o2,s2,recipe,cache,frozen,'d'*64)==1
    assert next_update(rebuilt,o2,s2)==reference
    wrong=profile.recipe('B_TARGET','a'*64,'b'*64)
    with pytest.raises(ValueError,match='SAME_NEW_STAGE'):
        runtime.restore_checkpoint(checkpoint,rebuilt,c2,o2,s2,wrong,cache,frozen,'d'*64)
    with pytest.raises(ValueError,match='SAME_NEW_STAGE'):
        runtime.restore_checkpoint(checkpoint,rebuilt,c2,o2,s2,recipe,cache,frozen,'e'*64)


def test_corrupted_optimizer_cannot_be_restored(tmp_path):
    model,contract,opt,sch,recipe=session()
    next_update(model,opt,sch)
    cp=tmp_path/'step1';cache=dict(path='new-cache',sha256='c'*64)
    runtime.save_checkpoint(cp,model,contract,opt,sch,1,recipe,cache,{},'d'*64)
    with (cp/'optimizer.pt').open('ab')as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='CHECKPOINT_FILES'):
        runtime.restore_checkpoint(cp,model,contract,opt,sch,recipe,cache,{},'d'*64)


def test_rejects_legacy_parent_audit_before_file_access():
    with pytest.raises(ValueError,match='ACTUAL_OFFICIAL_PARENT_AUDIT'):
        runtime.checked_parent('/unread',dict(schema='ARX5_RAW_PARENT_ZERO_STEP_IDENTITY_V1',passes=True))


def test_processor_save_preserves_training_rng_and_checkpoint_endpoint_identity(tmp_path):
    from go2arx5_vla.world_model import native_arx5_endpoint_v1 as endpoint
    import hashlib
    statistics=tmp_path/'statistics.json';statistics.write_text('{}\n')
    stats_sha=hashlib.sha256(statistics.read_bytes()).hexdigest()
    class Processor:
        def save_pretrained(self,path):
            path.mkdir();(path/'processor_config.json').write_text('{}\n')
            random.random();np.random.random();torch.randn(3)
    model,contract,opt,sch,_=session()
    recipe=profile.recipe('A_TARGET','a'*64,stats_sha)
    for _ in range(4):next_update(model,opt,sch)
    before=rng_identity();cp=tmp_path/'step4'
    runtime.save_checkpoint(cp,model,contract,opt,sch,4,recipe,{}, {},'d'*64,
                            processor=Processor(),statistics_file=statistics)
    assert rng_identity()==before
    pin=dict(path=str(cp),sha256=hashlib.sha256((cp/'COMPLETE.json').read_bytes()).hexdigest())
    spec=dict(recipe=recipe,dataset_sha256='a'*64,statistics_sha256=stats_sha,qualification_only=True)
    path,value=endpoint.read_checkpoint(pin,contract,spec)
    assert path==cp and value['global_step']==4 and value['Processor_saved']
    values=torch.load(cp/'trainable.pt',weights_only=True)
    rebuilt=torch.nn.Linear(3,2);c2=Contract(rebuilt)
    endpoint.install(values,rebuilt,c2)
    assert tree_identity(rebuilt.state_dict())==tree_identity(model.state_dict())
    with pytest.raises(ValueError,match='COMPLETE_OWN'):
        endpoint.read_checkpoint(pin,contract,dict(spec,qualification_only=False))
