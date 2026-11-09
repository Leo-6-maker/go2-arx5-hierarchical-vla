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
