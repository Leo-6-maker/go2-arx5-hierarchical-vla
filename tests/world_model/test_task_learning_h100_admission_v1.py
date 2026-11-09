"""Metadata fixtures exercise rejection paths; these are not native GPU results."""
import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_h100_admission_v1 as h100
from go2arx5_vla.world_model.storage import digest


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'restore';root.mkdir()
    monkeypatch.setattr(h100,'ROOT',tmp_path)
    import go2arx5_vla.world_model.task_learning_training_admission_v1 as admission
    monkeypatch.setattr(admission,'unchanged_training_math',lambda a,b:dict(status='FIXTURE_ONLY'))
    def save(name,value):
        f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(value))
    c=dict(root=str(root),phase='RESTORE_ONLY',allocation_gpu_model='H100',pbs_job_name='gr00t_h100_check',
        source_sha='a'*40,source_files={},parent={'fixture':'original-parent'},python='/fixture-python',
        pbs_server='fixture-server',bindings={k:'/fixture/'+k for k in ('source','cosmos','cosmos_hub')})
    save('ALLOCATION_CONFIG.json',c)
    save('WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    save('ACTUAL_GPU.private.json',dict(count=1,name='NVIDIA H100 80GB HBM3',major=9,total_memory=79*1024**3))
    proof=dict(status='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE',
        next_update_parameters_optimizer_scheduler_RNG_sampler_exact=True,checks={k:True for k in h100.CHECKS})
    save('FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',dict(proof,comparison='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2'))
    result=dict(status='HOPPER_ORIGINAL_TRAINER_RESTORE_ONLY_SEALED',allocation_gpu_model='H100',
        optimizer_calls=8,physics_integrations=0,formal_training=False,proof=proof,children=[])
    for name,end,cursor in [('continuous4',4,128),('split2',2,64),('resume4',4,128)]:
        r=dict(global_step=end,sample_visits=cursor,real_action_parameter_changed=True,backbone_frozen=True)
        save(name+'/RESULT.json',r);save(name+'/control/LEASE.json',dict(status='CLOSED',worker_exit_code=0))
        save(name+'/WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
        save(name+'/EXECUTION_SPEC.json',dict(registered_root=str(root/name),source_sha=c['source_sha'],
            source_files=c['source_files'],parent=c['parent'],allocation_gpu_model='H100',recipe=dict(
                mode='RESTORE4',variant='FULL_ACTION_SIDE',effective_batch=32,microbatch=1,endpoint=end)))
        save(name+'/PARAMETER_WHITELIST.json',dict(trainable_numel=1304913028,
            visual_and_language_backbone_frozen=True,external_adapter_trainables=0))
        result['children'].append(dict(name=name,exit_code=0,result=r,result_sha256=digest(root/name/'RESULT.json')))
    def seal():
        save('RESULT.private.json',result)
        save('control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,job_id='fixture-job',
            config_sha256=digest(root/'ALLOCATION_CONFIG.json'),result_sha256=digest(root/'RESULT.private.json')))
        return dict(path=str(root/'RESULT.private.json'),sha256=digest(root/'RESULT.private.json'))
    current=dict(copy.deepcopy(c),recipe=dict(microbatch=1))
    return root,c,current,result,save,seal


def test_only_completed_h100_full_action_restore_is_admitted(tmp_path,monkeypatch):
    _,_,current,_,_,seal=fixture(tmp_path,monkeypatch)
    x=h100.qualification(seal(),current)
    assert x['job_id']=='fixture-job'and x['microbatch']==1 and not x['qualification_weights_used_as_formal_parent']


@pytest.mark.parametrize('change',['queued','wrong_device','incomplete_restore','last2','unchanged_weights','runtime','microbatch','wrong_result'])
def test_queued_failed_partial_or_reduced_h100_evidence_cannot_admit_sft(tmp_path,monkeypatch,change):
    root,c,current,result,save,seal=fixture(tmp_path,monkeypatch)
    if change=='wrong_device':save('ACTUAL_GPU.private.json',dict(count=1,name='NVIDIA H200',major=9,total_memory=140*1024**3))
    elif change=='incomplete_restore':
        result['proof']['checks']['optimizer.pt']=False
        save('FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',dict(result['proof'],comparison='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2'))
    elif change=='last2':save('continuous4/PARAMETER_WHITELIST.json',dict(trainable_numel=123,visual_and_language_backbone_frozen=True,external_adapter_trainables=0))
    elif change=='unchanged_weights':
        child=result['children'][0];child['result']['real_action_parameter_changed']=False
        save('continuous4/RESULT.json',child['result']);child['result_sha256']=digest(root/'continuous4/RESULT.json')
    elif change=='runtime':current['python']='/different-runtime'
    elif change=='microbatch':current['recipe']['microbatch']=2
    elif change=='wrong_result':result['status']='HOPPER_FAILURE_PRESERVED'
    proof=seal()
    if change=='queued':
        lease=json.loads((root/'control/LEASE.json').read_text());lease['status']='PBS_SUBMITTED';save('control/LEASE.json',lease)
    with pytest.raises(PermissionError):h100.qualification(proof,current)
