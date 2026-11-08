import copy
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
