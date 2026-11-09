"""CPU allocation/data fixtures never establish an actual PBS or Processor pass."""
import copy,json
import pytest
from test_task_learning_transfer_v1 import bundle
from go2arx5_vla.world_model import task_learning_hopper_processor_v1 as worker
from go2arx5_vla.world_model.task_learning_transfer_v1 import inventory
from go2arx5_vla.world_model.storage import digest


def config():
    return dict(profile=worker.PROFILE,physics=False,policy_models=0,optimizer_updates=0,
        pbs_server='official-server-fixture',ncpus=4,mem_resource='32gb',
        root=str(worker.ROOT/'runs/WM02/processor-fixture'),predicted_work_s=1000,wall_s=3600)


def allocation():
    return dict(Job_Owner='e1192548@hopper-l-01',job_state='R',project='CFP04-CF-007',
        Job_Name=worker.JOB_NAME,exec_host='hopper-26/0*4',Resource_List=dict(
            ncpus=4,nodect=1,ngpus=0,select='1:ncpus=4:mem=32gb',mem='32gb'))


def test_only_actual_reserved_cpu_resources_for_the_owned_job_are_accepted():
    worker.allocated_cpu(allocation(),config(),uid=22182,user='e1192548',hostname='hopper-26')
    for change in('GPU','login','owner','project','queued','cpus','memory','job_name'):
        record=allocation();host='hopper-26'
        if change=='GPU':record['Resource_List']['ngpus']=1
        elif change=='login':host='hopper-l-01';record['exec_host']=host+'/0*4'
        elif change=='owner':record['Job_Owner']='other@hopper-l-01'
        elif change=='project':record['project']='other-project'
        elif change=='queued':record['job_state']='Q'
        elif change=='cpus':record['Resource_List']['ncpus']=12
        elif change=='memory':record['Resource_List']['mem']='1gb'
        else:record['Job_Name']='gr00t_task_sft'
        with pytest.raises(PermissionError):
            worker.allocated_cpu(record,config(),uid=22182,user='e1192548',hostname=host)


@pytest.mark.parametrize('key,value',[('pbs_server',''),('policy_models',1),('optimizer_updates',1),
    ('physics',True),('ncpus',0),('predicted_work_s',3500),('root','/unregistered/root')])
def test_cpu_job_cannot_load_models_train_or_silently_outlive_its_window(key,value):
    c=config();c[key]=value
    with pytest.raises(PermissionError):worker.configuration(c)


def test_cpu_processor_requires_actual_all_member_transfer_and_preserves_prior_attempts(tmp_path,monkeypatch):
    root=tmp_path/'data';root.mkdir();bundle(root)
    actual=inventory(root);receipt=tmp_path/'TRANSFER.json'
    receipt.write_text(json.dumps(dict(actual,status='PASS_PRIVATE_TASK_LEARNING_DATA_TRANSFER')))
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    c=dict(dataset=str(root),dataset_sha256=actual['manifest_sha256'],
        statistics_sha256=actual['statistics_sha256'],data_transfer=dict(path=str(receipt),sha256=digest(receipt)))
    assert worker.inputs(c)['files']==actual['files']
    (root/'PROCESSOR_HOPPER_RECONSTRUCTION.json').write_text('prior evidence')
    with pytest.raises(PermissionError,match='NEVER_REPLAYED'):worker.inputs(c)
    assert(root/'PROCESSOR_HOPPER_RECONSTRUCTION.json').read_text()=='prior evidence'


def test_formal_admission_needs_closed_cpu_worker_not_just_a_saved_processor_file(tmp_path,monkeypatch):
    from go2arx5_vla.world_model.task_learning_training_admission_v1 import processor_qualification
    root=tmp_path/'worker';(root/'control').mkdir(parents=True)
    data=tmp_path/'data';data.mkdir()
    admission=data/'PROCESSOR_HOPPER_RECONSTRUCTION.json';admission.write_text('fixture no real qualification')
    names=('scripts/wm02/qualify_target_gr00t_processor_v1.py',
        'src/go2arx5_vla/world_model/task_learning_native_dataset_v1.py',
        'src/go2arx5_vla/world_model/target_gr00t_dataset_v1.py',
        'scripts/pi0292/train_adapter.py','scripts/pi0292/eval_contract.py')
    c=dict(profile=worker.PROFILE,dataset=str(data),pbs_server='fixture-server',python='/fixture-python',
        bindings=dict(source='/fixture-source',cosmos='/fixture-cosmos',cosmos_hub='/fixture-hub'),
        source_files={n:'a'*64 for n in names})
    cp=root/'ALLOCATION_CONFIG.json';cp.write_text(json.dumps(c))
    result=dict(status='PASS_ACTUAL_HOPPER_CONTINUOUS_PROCESSOR_RECONSTRUCTION',
        exit_code=0,failure=None,job_id='fixture-cpu-job',config_sha256=digest(cp),
        manifest_sha256='b'*64,statistics_sha256='c'*64,Processor_admission_sha256=digest(admission),
        new_policy_models=0,new_optimizer_updates=0,new_native_integrations=0,GPU_allocations=0)
    rp=root/'RESULT.private.json';rp.write_text(json.dumps(result))
    (root/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0,failure=None,job_id='fixture-cpu-job')))
    lp=root/'control/LEASE.json';lease=dict(status='ACTIVE',worker_exit_code=0,job_id='fixture-cpu-job',
        result_sha256=digest(rp),config_sha256=digest(cp));lp.write_text(json.dumps(lease))
    proof=dict(path=str(rp),sha256=digest(rp))
    spec=dict(pbs_server=c['pbs_server'],python=c['python'],bindings=dict(c['bindings'],dataset=str(data)),
        recipe=dict(dataset_sha256='b'*64,statistics_sha256='c'*64),source_files=dict(c['source_files']))
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    with pytest.raises(PermissionError,match='CLOSED_ZERO_EXIT'):processor_qualification(proof,spec)
    lease['status']='CLOSED';lp.write_text(json.dumps(lease))
    assert processor_qualification(proof,spec)['job_id']=='fixture-cpu-job'
    spec['source_files'][names[0]]='d'*64
    with pytest.raises(PermissionError,match='SAME_ACTUALLY_QUALIFIED'):processor_qualification(proof,spec)
