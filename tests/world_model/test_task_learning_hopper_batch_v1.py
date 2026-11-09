"""Microbatch evidence fixtures; never an actual H200 performance claim."""
import copy,json
import pytest
from go2arx5_vla.world_model import task_learning_hopper_batch_v1 as batch


def trial(root,micro=4):
    root.mkdir();(root/'control').mkdir()
    values={'EXECUTION_SPEC.json':dict(recipe=dict(microbatch=micro,effective_batch=32,accumulation=32//micro,
        variant='FULL_ACTION_SIDE',mode='RESTORE4')),
        'RESULT.json':dict(global_step=4,sample_visits=128,real_action_parameter_changed=True,backbone_frozen=True),
        'control/LEASE.json':dict(status='CLOSED',worker_exit_code=0),
        'WORKER_EXIT.private.json':dict(exit_code=0,failure=None,wall_s=200),
        'GPU_MEMORY.json':dict(measurements=[dict(peak_allocated_bytes=32*1024**3,peak_reserved_bytes=34*1024**3,free_bytes=90*1024**3)])}
    for name,value in values.items():(root/name).write_text(json.dumps(value))
    rows=[dict(step=i,optimizer_update_wall_s=s)for i,s in enumerate((8.,2.,2.1,1.9),1)]
    (root/'OPTIMIZER_STEPS.jsonl').write_text('\n'.join(json.dumps(v)for v in rows))
    return values


def test_steady_rate_excludes_first_update_and_preserves_effective_batch(tmp_path):
    root=tmp_path/'trial';trial(root)
    r=batch.summarize_trial(root,4)
    assert r['steady_update_wall_s']==2 and r['maximum_update_wall_s']==8
    assert r['TRAIN_visits_per_s']==16 and r['sample_visits']==128
    assert r['effective_batch']==32


@pytest.mark.parametrize('file,key,value',[
    ('RESULT.json','sample_visits',64),('RESULT.json','real_action_parameter_changed',False),
    ('WORKER_EXIT.private.json','exit_code',1),('control/LEASE.json','status','ACTIVE')])
def test_partial_failed_or_zero_learning_trial_cannot_supply_capacity(tmp_path,file,key,value):
    root=tmp_path/'trial';values=trial(root);values[file][key]=value;(root/file).write_text(json.dumps(values[file]))
    with pytest.raises(PermissionError):batch.summarize_trial(root,4)


def test_selection_requires_every_trial_and_original_memory_margin():
    rows=[dict(microbatch=m,effective_batch=32,optimizer_calls=4,sample_visits=128,
        steady_update_wall_s=s,minimum_free_GPU_bytes=10*1024**3)for m,s in zip((1,2,4,8),(7.,4.,2.,2.))]
    result=batch.select(rows)
    assert result['selected_microbatch']==4 and result['measured_steady_update_speedup_over_micro1']==3.5
    assert result['physics_environments']==0 and result['larger_batch_is_not_claimed_bit_exact_to_micro1']
    with pytest.raises(PermissionError):batch.select(rows[:-1])
    bad=copy.deepcopy(rows);bad[-1]['minimum_free_GPU_bytes']=1024**3
    with pytest.raises(PermissionError):batch.select(bad)


def scan_fixture(root,monkeypatch):
    """Synthetic metadata for admission rejection tests, never native evidence."""
    from go2arx5_vla.world_model import task_learning_training_admission_v1 as admission
    from go2arx5_vla.world_model.storage import digest
    root.mkdir();(root/'control').mkdir()
    monkeypatch.setattr(batch,'ROOT',root.parent)
    monkeypatch.setattr(admission,'unchanged_training_math',lambda a,b:dict(status='FIXTURE_ONLY'))
    config=dict(root=str(root),phase='BATCH_THROUGHPUT_SCAN',pbs_job_name='gr00t_batchscan',
        python='fixture-python',parent=dict(path='fixture-parent'),pbs_server='fixture-server',
        bindings=dict(source='fixture-source',cosmos='fixture-cosmos',cosmos_hub='fixture-hub'))
    current={k:copy.deepcopy(config[k])for k in('python','parent','pbs_server','bindings')}
    current['recipe']=dict(microbatch=1)
    children=[]
    for name,endpoint in [('batch'+str(m)+'-continuous4',4)for m in(1,2,4,8)]+[
        ('selected-split2',2),('selected-resume4',4)]:
        m=int(name[5])if name.startswith('batch')else 1
        values=trial(root/name,m)
        result=values['RESULT.json'];result['global_step']=endpoint
        (root/name/'RESULT.json').write_text(json.dumps(result))
        children.append(dict(name=name,exit_code=0,result=result,result_sha256=digest(root/name/'RESULT.json')))
    trials=[batch.summarize_trial(root/('batch'+str(m)+'-continuous4'),m)for m in(1,2,4,8)]
    selection=batch.select(trials)
    full=dict(status='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE',
        next_update_parameters_optimizer_scheduler_RNG_sampler_exact=True,
        checks={n:True for n in('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth',
            'SAMPLER_STATE.json','PARAMETER_CONTRACT.json','FROZEN_PARENT_TENSORS.json')})
    values={'ALLOCATION_CONFIG.json':config,'BATCH_TRIALS_ACTUAL.json':dict(trials=trials),
        'BATCH_SELECTION_ACTUAL.json':selection,'SELECTED_FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json':dict(full,
            comparison='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2',microbatch=1),
        'WORKER_EXIT.private.json':dict(exit_code=0,failure=None),
        'RESULT.private.json':dict(status='HOPPER_BATCH_THROUGHPUT_AND_SELECTED_FULL_RESTORE_SEALED',
            optimizer_calls=20,effective_batch=32,physics_integrations=0,formal_training=False,
            trials=trials,selection=selection,selected_microbatch=1,selected_full_restore=full,children=children)}
    for name,value in values.items():(root/name).write_text(json.dumps(value))
    proof=dict(path=str(root/'RESULT.private.json'),sha256=digest(root/'RESULT.private.json'))
    lease=dict(status='CLOSED',worker_exit_code=0,job_id='fixture-not-a-PBS-job',
        config_sha256=digest(root/'ALLOCATION_CONFIG.json'),result_sha256=proof['sha256'])
    (root/'control/LEASE.json').write_text(json.dumps(lease))
    return current,proof


def test_batch_admission_binds_actual_selection_full_restore_and_runtime(tmp_path,monkeypatch):
    root=tmp_path/'fixture';current,proof=scan_fixture(root,monkeypatch)
    result=batch.qualification(proof,current)
    assert result['selected_microbatch']==1 and result['effective_batch']==32
    current['recipe']['microbatch']=4
    with pytest.raises(PermissionError,match='ACTUAL_FASTEST'):batch.qualification(proof,current)
    current['recipe']['microbatch']=1;current['python']='changed-runtime'
    with pytest.raises(PermissionError,match='PARENT_RUNTIME'):batch.qualification(proof,current)


@pytest.mark.parametrize('file,key,value,reason',[
    ('control/LEASE.json','status','ACTIVE','CLOSED_ZERO_EXIT'),
    ('WORKER_EXIT.private.json','exit_code',1,'CLOSED_ZERO_EXIT'),
    ('selected-resume4/WORKER_EXIT.private.json','exit_code',1,'TERMINAL_TRIAL'),
    ('SELECTED_FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',
        'next_update_parameters_optimizer_scheduler_RNG_sampler_exact',False,'SEVEN_STATE')])
def test_pending_failed_or_partial_restore_cannot_admit_batch(tmp_path,monkeypatch,file,key,value,reason):
    root=tmp_path/'fixture';current,proof=scan_fixture(root,monkeypatch)
    path=root/file;v=json.loads(path.read_text());v[key]=value;path.write_text(json.dumps(v))
    with pytest.raises(PermissionError,match=reason):batch.qualification(proof,current)
