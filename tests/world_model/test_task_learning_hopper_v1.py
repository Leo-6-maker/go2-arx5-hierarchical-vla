import copy
import json
import pytest

from go2arx5_vla.world_model.task_learning_hopper_v1 import allocated
from go2arx5_vla.world_model.target_gr00t_profile_v1 import recipe


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


def fit4_successor(tmp_path):
    from go2arx5_vla.world_model.storage import digest
    old=tmp_path/'old';old.mkdir()
    sources={name:'unchanged'for name in (
        'src/go2arx5_vla/world_model/target_gr00t_training_v1.py',
        'src/go2arx5_vla/world_model/target_gr00t_profile_v1.py',
        'src/go2arx5_vla/world_model/target_gr00t_inference_v1.py',
        'scripts/wm02/train_target_gr00t_v1.py')}
    c=dict(root=str(tmp_path/'new'),dataset_sha256='data',statistics_sha256='stats',
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


@pytest.mark.parametrize('change',['open','failed_restore','changed_model','changed_code','missing_timing','already_fit'])
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
    with pytest.raises(PermissionError):qualified_fit4_forecast(c)
