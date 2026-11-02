from copy import deepcopy
import hashlib,json
import pytest
from scripts.pi0294.multitask.book_gpu_capacity import GIB,TRAINING_REQUEST,entry_requirement
from scripts.pi0294.multitask.activation_offload import POLICY
from scripts.pi0294.multitask.book_bc_dagger import plan
from scripts.pi0294.multitask.book_m32_export import COUNTS


def fixture_manifest(steps,policy=POLICY):
    names=[*[f'BOOK_SOURCE_SEED{i:03}'for i in range(15)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',*[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX'for i in range(5)]]
    rows=[dict(id=n,windows=c)for n,c in zip(names,COUNTS)]+[dict(id='BOOK_LATE_RECOVERY_TRAIN000_EXPERT_SUFFIX',windows=100)]
    p=plan('D2','a'*64,'b'*64,'c'*64,rows,steps=steps)
    pin=hashlib.sha256((json.dumps(policy,indent=2)+'\n').encode()).hexdigest()
    return dict(kind='V7_BOOK_M32_FINITE_LEARNING',GPU_uuid='GPU1',recipe=p,
        checkpoint_manifest_sha256='PARENT',delta_checkpoint_manifest_sha256='B248',memory_execution_sha256=pin,
        sealed_inputs={'M32_MEMORY_EXECUTION.json':pin},training_GPU_entry_MiB=15360,
        limits=dict(model_loads=1,optimizer_updates=steps,physics_integrations=0,generation_queries=p['generation_queries'],RGB_pairs=0,source_constructors=0))


def test_worker_actual_entry_accepts_only_explicit_native10_memory_probe():
    m=fixture_manifest(10);m['GPU_training_qualification']=deepcopy(TRAINING_REQUEST)
    assert entry_requirement(m)==15*GIB
    for key,value in [('memory_execution_sha256','x'),('training_GPU_entry_MiB',1),('kind','V7_BOOK_FROZEN_DEV10')]:
        wrong=deepcopy(m);wrong[key]=value
        with pytest.raises(PermissionError):entry_requirement(wrong)
    wrong=fixture_manifest(4096);wrong['GPU_training_qualification']=deepcopy(TRAINING_REQUEST)
    with pytest.raises(PermissionError):entry_requirement(wrong)


def test_v2_has_an_explicit_separate_execution_identity():
    from scripts.pi0294.multitask.activation_offload import LOW_RESIDENCY_POLICY
    m=fixture_manifest(10);m['GPU_training_qualification']=deepcopy(TRAINING_REQUEST)
    new_pin=hashlib.sha256((json.dumps(LOW_RESIDENCY_POLICY,indent=2)+'\n').encode()).hexdigest()
    m['memory_execution_sha256']=new_pin
    with pytest.raises(PermissionError):entry_requirement(m)
    m['sealed_inputs']['M32_MEMORY_EXECUTION.json']=new_pin
    assert entry_requirement(m)==15*GIB


def test_formal_requires_same_gpu_recipe_closed_restore_and_two_gib(tmp_path):
    selected_policy=[POLICY]
    q=fixture_manifest(10);q['GPU_training_qualification']=deepcopy(TRAINING_REQUEST)
    result=dict(status='COMPLETE_BOOK5_FINITE_TARGET_LEARNING',error=None,model_load_attempts=1,optimizer_updates=10,TRAIN_visits=320,
                generation_attempts=32,completed_generations=32,physics_integrations=0,DEV_conditions=0)
    cap=dict(capacity_pass=True,minimum_observed_free_bytes=2*GIB,actual_optimizer_updates=10,saved_and_restored=True)
    restore=dict(optimizer_updates=10,sampler_cursor=320,model_reconstruction=True,scheduler_restored=True,RNG_restored=True,frozen_parent_unchanged=True)
    copies=dict(policy=POLICY,copied_tensors=1,restored_tensors=1)
    def make():
        values={'RUN_MANIFEST.json':q,'RESULT.json':result,'M32_CAPACITY_RESULT.json':cap,
                'ACTUAL_DELTA_CHECKPOINT_RESTORE.json':restore,'ACTUAL_ACTIVATION_OFFLOAD.json':copies}
        pins={}
        for n,v in values.items():
            body=json.dumps(v).encode();(tmp_path/n).write_bytes(body);pins[n]=hashlib.sha256(body).hexdigest()
        body=json.dumps(dict(status='CLOSED',manifest_sha256=pins['RUN_MANIFEST.json'],result_sha256=pins['RESULT.json'])).encode()
        (tmp_path/'control').mkdir(exist_ok=True);(tmp_path/'control/LEASE.json').write_bytes(body);pins['control/LEASE.json']=hashlib.sha256(body).hexdigest()
        m=fixture_manifest(4096,selected_policy[0]);m['GPU_training_admission']=dict(schema='M32_SAME_GPU_LOSSLESS_NATIVE10_CERTIFIED_TRAINING_V1',qualification_root=str(tmp_path),files=pins)
        return m
    m=make();assert entry_requirement(m)==15*GIB
    for key,value in [('GPU_uuid','GPU0'),('delta_checkpoint_manifest_sha256','OTHER')]:
        wrong=deepcopy(m);wrong[key]=value
        with pytest.raises(PermissionError):entry_requirement(wrong)
    cap['minimum_observed_free_bytes']=2*GIB-1
    with pytest.raises(PermissionError):entry_requirement(make())
    cap['minimum_observed_free_bytes']=2*GIB;restore['RNG_restored']=False
    with pytest.raises(PermissionError):entry_requirement(make())
    restore['RNG_restored']=True;m=make();(tmp_path/'RESULT.json').write_text('{}')
    with pytest.raises(PermissionError,match='CERTIFICATE_CHANGED'):entry_requirement(m)
    from scripts.pi0294.multitask.activation_offload import LOW_RESIDENCY_POLICY
    selected_policy[0]=LOW_RESIDENCY_POLICY
    with pytest.raises(PermissionError):entry_requirement(make())
    q=fixture_manifest(10,LOW_RESIDENCY_POLICY);q['GPU_training_qualification']=deepcopy(TRAINING_REQUEST)
    copies['policy']=LOW_RESIDENCY_POLICY
    assert entry_requirement(make())==15*GIB
