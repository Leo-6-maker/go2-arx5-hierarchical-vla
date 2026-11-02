from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.pi0294.multitask import shared_training as shared,shared_checkpoint as checkpoint
from scripts.pi0294.multitask.activation_offload import LOW_RESIDENCY_POLICY
from scripts.pi0294.multitask.book_gpu_capacity import TRAINING_REQUEST
from scripts.pi0294.multitask.book_fit import validate_manifest,select_rows


def raw(v):return(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(raw(v)).hexdigest()
def recipe(steps=8192):return shared.recipe(*[str(i)*64 for i in range(1,5)],steps=steps)


def manifest(steps=10):
    p=recipe(steps);memory=sha(LOW_RESIDENCY_POLICY)
    return dict(kind=shared.KIND,recipe=p,raw_parent_files_identity=p['raw_parent_files_identity'],GPU_uuid='GPU-actual-one',
        training_GPU_entry_MiB=15360,memory_execution_sha256=memory,sealed_inputs={'M32_MEMORY_EXECUTION.json':memory},
        GPU_training_qualification=TRAINING_REQUEST if steps==10 else None,
        limits=dict(model_loads=1,optimizer_updates=steps,physics_integrations=0,
                    generation_queries=p['generation_queries'],RGB_pairs=0,source_constructors=0))


def test_explicit_actual_train16_are_balanced_and_short_recipe_keeps_long_schedule():
    p,q=recipe(),recipe(10)
    assert shared.is_recipe(p)and shared.is_recipe(q)
    assert p['diagnostic_windows']==q['diagnostic_windows']
    assert len(p['diagnostic_windows'])==16
    assert {t:sum(r['task']==t for r in p['diagnostic_windows'])for t in shared.TASKS}==dict.fromkeys(shared.TASKS,4)
    rows=[(dict(id=name,task=task,split='TRAIN'),a)
          for task,name in zip(shared.TASKS,shared.DIAGNOSTIC_EPISODES)for a in range(601)]
    assert len(select_rows(rows,p))==16
    assert p['warmup_updates']==q['warmup_updates']==256
    assert p['scheduler_horizon_steps']==q['scheduler_horizon_steps']==8192
    assert p['generation_queries']==80 and q['generation_queries']==32
    # The delivered input recipe V1 remains separate from a runnable recipe.
    from scripts.pi0294.multitask.shared_task_sampling import recipe as previous
    assert not shared.is_recipe(previous(*['1'*64]*4))


@pytest.mark.parametrize('change',[('initialization','B248'),('effective_batch_size',64),('warmup_updates',128),
    ('loss','WEIGHTED'),('continuous_gripper_decode',False),('state_history_length',3)])
def test_old_book_warmstart_or_changed_recipe_does_not_enter_existing_fit_loop(change):
    m=manifest();p=deepcopy(m['recipe']);p[change[0]]=change[1];m['recipe']=p
    with pytest.raises(PermissionError):validate_manifest(m,p)


def test_same_device_raw_native10_is_distinct_from_book_capacity(tmp_path):
    q=manifest();assert shared.qualified_entry(q)==15*1024**3
    values={'RUN_MANIFEST.json':q,'RESULT.json':dict(status='COMPLETE_SHARED_RAW_FINITE_LEARNING',error=None,
        model_load_attempts=1,optimizer_updates=10,TRAIN_visits=320,generation_attempts=32,completed_generations=32,
        physics_integrations=0,DEV_conditions=0),
        'M32_CAPACITY_RESULT.json':dict(capacity_pass=True,minimum_observed_free_bytes=2*1024**3,
            actual_optimizer_updates=10,saved_and_restored=True),
        'ACTUAL_DELTA_CHECKPOINT_RESTORE.json':dict(required_raw_parent_files_identity=checkpoint.PARENT_ID,
            model_reconstruction=True,optimizer_updates=10,optimizer_state_entries=42,scheduler_restored=True,
            RNG_restored=True,sampler_cursor=320,frozen_parent_unchanged=True),
        'ACTUAL_ACTIVATION_OFFLOAD.json':dict(policy=LOW_RESIDENCY_POLICY,copied_tensors=1,restored_tensors=1),
        'RAW_ZERO_STEP_PARAMETERS.json':dict(operation='COMPARE_FRESH_LOADED_MODEL',
            raw_parent_files_identity=checkpoint.PARENT_ID,trainable_tensor_count=42,trainable_numel=77847684)}
    values['control/LEASE.json']=dict(status='CLOSED',manifest_sha256=sha(q),result_sha256=sha(values['RESULT.json']))
    for name,value in values.items():
        f=tmp_path/name;f.parent.mkdir(exist_ok=True);f.write_bytes(raw(value))
    m=manifest(8192);m['GPU_training_admission']=dict(schema='SHARED4_RAW_SAME_GPU_NATIVE10_TRAINING_V1',
        qualification_root=str(tmp_path),files={n:sha(v)for n,v in values.items()})
    assert shared.qualified_entry(m)==15*1024**3
    for field,value in [('GPU_uuid','OTHER_GPU'),('memory_execution_sha256','wrong'),
                        ('checkpoint_manifest_sha256','BOOK128'),('GPU_inference_admission',{})]:
        bad=deepcopy(m);bad[field]=value
        with pytest.raises(PermissionError):shared.qualified_entry(bad)
    # Even freshly re-pinned negative native evidence cannot admit training.
    for name,field,value in [('RESULT.json','completed_generations',31),
        ('RESULT.json','status','COMPLETE_BOOK5_FINITE_TARGET_LEARNING'),
        ('M32_CAPACITY_RESULT.json','minimum_observed_free_bytes',2*1024**3-1),
        ('ACTUAL_DELTA_CHECKPOINT_RESTORE.json','optimizer_state_entries',14),
        ('RAW_ZERO_STEP_PARAMETERS.json','operation','RESTORE_RAW_TRAINABLES')]:
        bad=deepcopy(m);v=deepcopy(values[name]);v[field]=value;(tmp_path/name).write_bytes(raw(v))
        bad['GPU_training_admission']['files'][name]=sha(v)
        with pytest.raises(PermissionError):shared.qualified_entry(bad)
        (tmp_path/name).write_bytes(raw(values[name]))


def test_raw_checkpoint_header_requires_all_data_and_full_state():
    p=recipe();members={n:'x'*64 for n in ['target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth',
        'trainer_state.json','statistics.json','TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json',
        'SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json',
        'RAW_PARENT_REFERENCE.json','M32_MEMORY_EXECUTION.json']}
    members.update({'statistics.json':p['statistics_sha256'],'TRAIN_POPULATION.json':p['TRAIN_population_sha256'],
                    'TASK_DEV_POPULATION.json':p['task_DEV_population_sha256']})
    v=dict(schema=checkpoint.SCHEMA,global_step=2048,cursor=65536,standalone_model=False,
        full_resume_with_required_parent=True,required_raw_parent_files_identity=checkpoint.PARENT_ID,recipe=p,files=members)
    assert checkpoint.validate_header(v)==v
    for key,value in [('global_step',1024),('cursor',32),('standalone_model',True),
                      ('required_raw_parent_files_identity','BOOK128')]:
        bad=deepcopy(v);bad[key]=value
        with pytest.raises(PermissionError):checkpoint.validate_header(bad)
    for name in members:
        bad=deepcopy(v);del bad['files'][name]
        with pytest.raises(PermissionError):checkpoint.validate_header(bad)


def test_raw_mapping_keeps_tensor_orientation_and_selects_original_category10():
    from scripts.pi0294.generalization.minimal_adapter import LEAVES
    index={p+'.'+s:'shard'for p in LEAVES for s in('W','b')}
    index.update({f'action_head.model.transformer_blocks.{i}.leaf{j}':'shard'for i in range(32)for j in range(14)})
    mapping=checkpoint.parameter_map(index)
    assert len(mapping)==42
    for p in LEAVES:
        assert mapping[p+'.target_W']==(p+'.W',10)
        assert mapping[p+'.target_b']==(p+'.b',10)
    assert all(mapping[n]==(n,None)for n in mapping if '.transformer_blocks.'in n)
    del index['action_head.model.transformer_blocks.31.leaf0']
    with pytest.raises(PermissionError):checkpoint.parameter_map(index)


def test_raw_compare_and_restore_never_changes_frozen_parameters(monkeypatch):
    import torch
    model=torch.nn.Linear(3,2);original={n:p.detach().clone()for n,p in model.named_parameters()}
    contract=SimpleNamespace(trainable_names=tuple(original),validate=lambda:None)
    monkeypatch.setattr(checkpoint,'raw_trainables',lambda *args,**kwargs:{n:v.clone()for n,v in original.items()})
    zero=checkpoint.raw_parameter_identity(model,contract,'unused',{},guard=lambda:None)
    assert zero['operation']=='COMPARE_FRESH_LOADED_MODEL'
    with torch.no_grad():model.weight.add_(1)
    with pytest.raises(ValueError):checkpoint.raw_parameter_identity(model,contract,'unused',{},guard=lambda:None)
    restored=checkpoint.raw_parameter_identity(model,contract,'unused',{},guard=lambda:None,restore=True)
    assert restored['trainable_tensors']==zero['trainable_tensors']
    assert torch.equal(model.weight,original['weight'])


def test_raw_inference_reconstruction_applies_only_finite_declared_delta(tmp_path,monkeypatch):
    import torch
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    model=torch.nn.Module();model.params=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2))for _ in range(42)])
    model.register_buffer('kept',torch.tensor([7.]))
    names=tuple(dict(model.named_parameters()));description={'fixture':'CPU_PARAMETER_CONTAINER_NOT_GR00T'}
    contract=SimpleNamespace(trainable_names=names,validate=lambda:None,describe=lambda:description)
    def compare(m,c,*args,**kwargs):
        assert all(torch.equal(p,torch.zeros_like(p))for p in m.parameters())
    monkeypatch.setattr(checkpoint,'raw_parameter_identity',compare)
    torch.save({n:torch.ones_like(p)for n,p in model.named_parameters()},tmp_path/'target_parameters.pt')
    p=recipe()
    docs={'RAW_PARENT_REFERENCE.json':checkpoint.raw_reference(),'ADAPTER_PARAMETER_CONTRACT.json':description,
        'FROZEN_PARENT_TENSORS.json':frozen_fingerprint(model)}
    for n in ['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','statistics.json',
              'TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json','SAMPLER_STATE.json','M32_MEMORY_EXECUTION.json']:
        docs[n]={'CPU_fixture_not_native_state':n}
    for n,v in docs.items():(tmp_path/n).write_bytes(raw(v))
    # Recipe pins bind the exact fixture metadata; native producer qualification
    # is deliberately absent and is never claimed by this consumer test.
    p=shared.recipe('1'*64,sha(docs['statistics.json']),sha(docs['TRAIN_POPULATION.json']),sha(docs['TASK_DEV_POPULATION.json']))
    members={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in tmp_path.iterdir()}
    v=dict(schema=checkpoint.SCHEMA,global_step=8192,cursor=262144,standalone_model=False,
        full_resume_with_required_parent=True,required_raw_parent_files_identity=checkpoint.PARENT_ID,recipe=p,files=members)
    def seal(value):
        (tmp_path/'CHECKPOINT_COMPLETE.json').write_bytes(raw(value));return sha(value)
    digest=seal(v)
    result=checkpoint.load_delta(model,contract,tmp_path,manifest_sha256=digest,raw_parent='unused',audit={},guard=lambda:None)
    assert result['frozen_parent_unchanged']and torch.equal(model.kept,torch.tensor([7.]))
    assert all(torch.equal(t,torch.ones_like(t))for t in model.parameters())
    with torch.no_grad():
        for t in model.parameters():t.zero_()
    (tmp_path/'target_parameters.pt').write_bytes(b'changed')
    with pytest.raises(PermissionError):checkpoint.load_delta(model,contract,tmp_path,manifest_sha256=digest,raw_parent='unused',audit={},guard=lambda:None)
    assert all(torch.equal(t,torch.zeros_like(t))for t in model.parameters())
    short=deepcopy(v);short.update(recipe=shared.recipe('1'*64,p['statistics_sha256'],p['TRAIN_population_sha256'],p['task_DEV_population_sha256'],steps=10),global_step=10,cursor=320)
    with pytest.raises(PermissionError,match='NOT_A_POLICY'):
        checkpoint.load_delta(model,contract,tmp_path,manifest_sha256=seal(short),raw_parent='unused',audit={},guard=lambda:None)
