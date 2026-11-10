import copy
import json
import pytest

from go2arx5_vla.world_model.task_learning_hopper_v1 import allocated
from go2arx5_vla.world_model.target_gr00t_profile_v1 import recipe


def test_explicit_job_read_includes_history_for_real_terminal_delivery(monkeypatch):
    from types import SimpleNamespace
    from go2arx5_vla.world_model import task_learning_hopper_v1 as hopper
    calls=[]
    def query(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps(dict(Jobs={'123.hopper-m-02':dict(job_state='F',Exit_status=0)})))
    monkeypatch.setattr(hopper.subprocess,'run',query)
    assert hopper.job('123.hopper-m-02','fixture-server')['job_state']=='F'
    assert '-x'in calls[0]and calls[0][-1]=='123.hopper-m-02@fixture-server'


def test_restore4_keeps_same_semantic_recipe_across_split_boundary():
    from go2arx5_vla.world_model.target_gr00t_training_v1 import semantic_recipe
    a=recipe('a'*64,'b'*64,mode='RESTORE4',variant='FULL_ACTION_SIDE',endpoint=2)
    b=recipe('a'*64,'b'*64,mode='RESTORE4',variant='FULL_ACTION_SIDE',endpoint=4)
    assert semantic_recipe(a)==semantic_recipe(b)


@pytest.mark.parametrize('change',[None,'ngpus','project','host','owner','state'])
def test_hopper_requires_actual_own_single_gpu_job(change):
    record=dict(Job_Owner='e1192548@hopper-l-01',job_state='R',project='CFP04-CF-007',
        Job_Name='gr00t_task_hopper',exec_host='hopper-g-01/0*12',
        Resource_List=dict(ngpus=1,nodect=1,select='1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB'))
    value=copy.deepcopy(record)
    if change=='ngpus':value['Resource_List']['ngpus']=3
    if change=='project':value['project']='unrelated'
    if change=='host':value['exec_host']='another-node/0*12'
    if change=='owner':value['Job_Owner']='another-user@hopper-l-01'
    if change=='state':value['job_state']='Q'
    arguments=dict(job_id='123.hopper-m-02',uid=22182,hostname='hopper-g-01',user='e1192548')
    if change:
        with pytest.raises(PermissionError):allocated(value,**arguments)
    else:allocated(value,**arguments)


def test_fit4_only_job_name_remains_bound_to_actual_allocation():
    record=dict(Job_Owner='e1192548@hopper-l-01',job_state='R',project='CFP04-CF-007',
        Job_Name='gr00t_fit4_200',exec_host='hopper-g-01/0*12',
        Resource_List=dict(ngpus=1,nodect=1,select='1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB'))
    arguments=dict(job_id='new-job',uid=22182,hostname='hopper-g-01',user='e1192548')
    allocated(record,**arguments,expected_job_name='gr00t_fit4_200')
    with pytest.raises(PermissionError):allocated(record,**arguments)


def test_h100_needs_explicit_restore_identity_and_actual_matching_allocation():
    from go2arx5_vla.world_model.task_learning_hopper_v1 import allocation_model
    c=dict(allocation_gpu_model='H100',pbs_job_name='gr00t_h100_check',phase='RESTORE_ONLY')
    assert allocation_model(c)=='H100' and allocation_model({})=='H200'
    record=dict(Job_Owner='e1192548@hopper-l-02',job_state='R',project='CFP04-CF-007',
        Job_Name='gr00t_h100_check',exec_host='hopper-compute/0*12',
        Resource_List=dict(ngpus=1,nodect=1,select='1:ngpus=1:gpu_model=H100:ncpus=12:mem=225GB'))
    arguments=dict(job_id='new-job',uid=22182,hostname='hopper-compute',user='e1192548')
    allocated(record,**arguments,expected_job_name=c['pbs_job_name'],expected_gpu_model=allocation_model(c))
    with pytest.raises(PermissionError):allocated(record,**arguments)
    record['Resource_List']['select']=record['Resource_List']['select'].replace('H100','H200')
    with pytest.raises(PermissionError):
        allocated(record,**arguments,expected_job_name=c['pbs_job_name'],expected_gpu_model='H100')


@pytest.mark.parametrize('change',['implicit_name','fit','formal','larger_microbatch','other_gpu'])
def test_h100_cannot_silently_authorize_formal_or_different_recipe(change):
    from go2arx5_vla.world_model.task_learning_hopper_v1 import allocation_model
    c=dict(allocation_gpu_model='H100',pbs_job_name='gr00t_h100_check',profile='TARGET_GR00T_V1',
        recipe=dict(mode='RESTORE4',microbatch=1))
    assert allocation_model(c)=='H100'
    if change=='implicit_name':c.pop('pbs_job_name')
    elif change=='fit':c['recipe']['mode']='FIT4'
    elif change=='formal':c['profile']='TASK_LEARNING_GR00T_V1'
    elif change=='larger_microbatch':c['recipe']['microbatch']=2
    else:c['allocation_gpu_model']='A100'
    with pytest.raises(PermissionError):allocation_model(c)


@pytest.mark.parametrize('model,name,gib,major,passes',[
    ('H100','NVIDIA H100 80GB HBM3',79,9,True),
    ('H100','NVIDIA H100 PCIe',79,9,True),
    ('H100','NVIDIA H200',140,9,False),
    ('H100','NVIDIA H100',60,9,False),
    ('H100','NVIDIA H100',79,8,False),
    ('H200','NVIDIA H200',140,9,True),
    ('H200','NVIDIA H100',79,9,False),
])
def test_actual_device_capacity_is_bound_to_declared_model(model,name,gib,major,passes):
    from go2arx5_vla.world_model.task_learning_hopper_v1 import actual_gpu
    gpu=dict(count=1,name=name,total_memory=gib*1024**3,major=major)
    if passes:actual_gpu(gpu,model)
    else:
        with pytest.raises(PermissionError):actual_gpu(gpu,model)


def fit4_successor(tmp_path):
    from go2arx5_vla.world_model.storage import digest
    old=tmp_path/'old';old.mkdir()
    sources={name:'unchanged'for name in (
        'src/go2arx5_vla/world_model/target_gr00t_training_v1.py',
        'src/go2arx5_vla/world_model/target_gr00t_profile_v1.py',
        'src/go2arx5_vla/world_model/target_gr00t_inference_v1.py',
        'scripts/wm02/train_target_gr00t_v1.py')}
    c=dict(root=str(tmp_path/'new'),pbs_server='official-server',dataset_sha256='data',statistics_sha256='stats',
        bindings={'dataset':'unchanged'},parent={'files':{}},python='same interpreter',source_files=sources)
    files={}
    def save(name,value,lines=False):
        p=old/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join(json.dumps(v)for v in value)+'\n'if lines else json.dumps(value))
        files[name]=digest(p)
    save('ALLOCATION_CONFIG.json',c)
    save('RESULT.private.json',dict(status='HOPPER_ORIGINAL_TRAINER_RESTORE_READY_WAIT_FIT4_WINDOW',wall_s=9999))
    save('control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,job_id='old-job',result_sha256=files['RESULT.private.json']))
    save('WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    save('FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',dict(
        status='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE',comparison='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2',
        next_update_parameters_optimizer_scheduler_RNG_sampler_exact=True,checks={name:True for name in(
            'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','SAMPLER_STATE.json',
            'PARAMETER_CONTRACT.json','FROZEN_PARENT_TENSORS.json')}))
    for name,steps in(('continuous4',[1,2,3,4]),('split2',[1,2]),('resume4',[3,4])):
        save(name+'/OPTIMIZER_STEPS.jsonl',[dict(step=i,optimizer_update_wall_s=7.5)for i in steps],lines=True)
    c['qualified_restore']=dict(root=str(old),files=files)
    return c,save


def test_finite_fit4_forecast_uses_actual_update_time_and_preserves_closeout(tmp_path):
    from go2arx5_vla.world_model.task_learning_hopper_v1 import qualified_fit4_forecast
    c,_=fit4_successor(tmp_path);v=qualified_fit4_forecast(c)
    assert v['completed_optimizer_calls']==8 and v['conservative_predicted_s']==2550
    assert v['closeout_reserve_s']==1200 and not v['qualification_weights_used_for_fit4']


@pytest.mark.parametrize('change',['open','failed_restore','changed_model','changed_code','missing_timing','already_fit','missing_server'])
def test_fit4_only_requires_completed_matching_restore_and_no_previous_fit(tmp_path,change):
    from pathlib import Path
    from go2arx5_vla.world_model.task_learning_hopper_v1 import qualified_fit4_forecast
    c,save=fit4_successor(tmp_path)
    if change=='open':save('control/LEASE.json',dict(status='ACTIVE'))
    elif change=='failed_restore':save('WORKER_EXIT.private.json',dict(exit_code=1))
    elif change=='changed_model':c['parent']={'different':'model'}
    elif change=='changed_code':c['source_files']=dict(c['source_files']);c['source_files']['scripts/wm02/train_target_gr00t_v1.py']='changed'
    elif change=='missing_timing':save('resume4/OPTIMIZER_STEPS.jsonl',[dict(step=4,optimizer_update_wall_s=7.5)],lines=True)
    elif change=='already_fit':(Path(c['qualified_restore']['root'])/'fit4-200').mkdir()
    elif change=='missing_server':c.pop('pbs_server')
    with pytest.raises(PermissionError):qualified_fit4_forecast(c)
def test_same_child_name_in_two_allocations_has_distinct_global_registry_intents(tmp_path):
    from go2arx5_vla.world_model.task_learning_hopper_v1 import child_registration_key
    from scripts.pi0294.night10.native_relay import snapshot, once
    registry = tmp_path / 'registry.json'
    registry.write_text('{"entries": []}')
    keys = []
    for allocation in ('original-h200', 'successor-h100'):
        root = tmp_path / allocation
        reg, before = snapshot(registry)
        reg['entries'].append(dict(path=str(root / 'continuous4')))
        key = child_registration_key(root, 'continuous4'); keys.append(key)
        once(registry, reg, key, expected_before_sha256=before)
    assert len(set(keys)) == 2
    assert len(snapshot(registry)[0]['entries']) == 2
    assert all((tmp_path / (key + '.INTENT.json')).is_file()
               and (tmp_path / (key + '.RECEIPT.json')).is_file() for key in keys)
