"""Scientific gate and finite A40 wrapper regressions; no PBS submission."""
import json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model.target_gr00t_allocation_v1 import pbs_script,fit4_gate,compare_next_update


def test_job_is_explicit_single_a40_without_restarts_or_credential_exports():
    text=pbs_script('/scratch/projects/CFP04/CFP04-CF-007/test','/scratch/source','/scratch/python',21600)
    assert 'select=1:ncpus=8:mem=64GB:ngpus=1:gpu_model=A40' in text
    assert '#PBS -r n' in text and '#PBS -m n' in text
    assert 'H200' not in text and 'qsub' not in text and 'export HOME=' not in text
    with pytest.raises(PermissionError):pbs_script('/scratch/test','/scratch/src','/scratch/python',72*3600)


def panel(tmp_path,*,wrong_direction=False,rot_after=.0001):
    for folder in ('before','after'):(tmp_path/folder).mkdir()
    rows=[]
    for i in range(4):
        sign=1 if i%2 else -1;name=f'episode{i}'
        expert=np.zeros((40,10));expert[:,0]=sign*np.linspace(0,.004,40)
        pred=expert.copy()
        if wrong_direction:pred[-1,0]*=-1
        np.savez(tmp_path/'after'/(name+'.npz'),physical_H40=pred,expert_policy_H40=expert,actual_state10=np.zeros(10))
        rows.append(dict(episode=name,command='APPROACH'if sign==1 else'RETREAT',seed=i,noise={'sha256':str(i)},
            errors={h:dict(XYZ_euclidean_m={'mean':.002},SO3_rad={'mean':.0001},gripper_MAE=0.)for h in ('H40','prefix8')}))
    (tmp_path/'before/RESULT.json').write_text(json.dumps(dict(records=rows)))
    for r in rows:
        for h in ('H40','prefix8'):
            r['errors'][h]['XYZ_euclidean_m']['mean']=.0002
            r['errors'][h]['SO3_rad']['mean']=rot_after
    (tmp_path/'after/RESULT.json').write_text(json.dumps(dict(records=rows)))


def test_fit4_requires_both_actual_directions_and_no_rotation_tradeoff(tmp_path):
    panel(tmp_path);assert fit4_gate(tmp_path)['passes']
    bad=tmp_path/'wrong';bad.mkdir();panel(bad,wrong_direction=True)
    assert not fit4_gate(bad)['passes']
    bad=tmp_path/'rotation';bad.mkdir();panel(bad,rot_after=.02)
    assert not fit4_gate(bad)['passes']


def test_next_update_checks_optimizer_rng_and_every_learned_tensor(tmp_path):
    torch=pytest.importorskip('torch');first=tmp_path/'first';second=tmp_path/'resumed';first.mkdir();second.mkdir()
    for root in (first,second):
        for name in ('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth'):
            torch.save({'real_test_tensor':torch.arange(4),'numpy_state':np.arange(3)},root/name)
        for name in ('SAMPLER_STATE.json','PARAMETER_CONTRACT.json','FROZEN_PARENT_TENSORS.json'):
            (root/name).write_text(json.dumps({'cursor':64}))
    assert compare_next_update(first,second)['next_update_parameters_optimizer_scheduler_RNG_sampler_exact']
    torch.save({'real_test_tensor':torch.zeros(4),'numpy_state':np.arange(3)},second/'optimizer.pt')
    with pytest.raises(PermissionError,match='NEXT_UPDATE_NOT_EQUAL'):compare_next_update(first,second)


def test_child_preserves_pbs_device_mask_instead_of_reinterpreting_torch_uuid(monkeypatch):
    from go2arx5_vla.world_model.target_gr00t_allocation_v1 import child_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    env=child_environment('/scratch/source','/scratch/run')
    assert env['CUDA_VISIBLE_DEVICES']=='1' and env['TMPDIR']=='/scratch/run/tmp'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert 'CUDA_VISIBLE_DEVICES' not in child_environment('/scratch/source','/scratch/run')


@pytest.mark.parametrize('failure',[False,True])
def test_restore_only_never_runs_fit4_or_searches_smaller_variants(tmp_path,monkeypatch,failure):
    import time
    from go2arx5_vla.world_model import target_gr00t_allocation_v1 as allocation
    first=tmp_path/'first';first.mkdir()
    (first/'EXECUTION_SPEC.json').write_text(json.dumps(dict(generation_id='site-qualification')))
    cp1=first/'checkpoint-1';cp1.mkdir();(cp1/'CHECKPOINT_COMPLETE.json').write_text('{}')
    calls=[];children=[]
    def child(*args,**kwargs):
        calls.append((args,kwargs))
        row=dict(root=str(first),exit_code=int(failure),endpoint_complete=not failure,
            checkpoint=str(first/('checkpoint-2'if len(calls)==1 else'restored-2')),
            GR00T_updates=2 if len(calls)==1 else 1)
        children.append(row);return row
    def allocated(config,operation):
        return operation(tmp_path,allocation.profile.need,time.time()+3600,{},child,children,time.monotonic(),[])
    monkeypatch.setattr(allocation,'with_allocation',allocated)
    compared=[]
    def compare(a,b):
        compared.append((a,b));return dict(next_update_parameters_optimizer_scheduler_RNG_sampler_exact=True)
    monkeypatch.setattr(allocation,'compare_next_update',compare)
    if failure:
        with pytest.raises(PermissionError,match='NO_VARIANT_SEARCH'):
            allocation.run(dict(allocation_mode='RESTORE_ONLY'))
        assert len(calls)==1 and not compared
        return
    result=allocation.run(dict(allocation_mode='RESTORE_ONLY'))
    assert result['status']=='PASS_A40_FULL_ACTION_NEXT_UPDATE_RESTORE'
    assert result['actual_GR00T_updates']==3 and len(calls)==2 and len(compared)==1
    assert all(a[1:]==('CAPACITY','FULL_ACTION_SIDE',2)for a,k in calls)
    assert calls[1][1]['start_step']==1 and calls[1][1]['generation']=='site-qualification'
    assert result['qualification_weights_used_for_formal_training']is False
    assert (tmp_path/'NEXT_UPDATE_RESTORE_ACTUAL.json').is_file()


def test_reconciler_closed_failure_requires_independent_pbs_proof(tmp_path):
    from scripts.wm02.submit_target_gr00t_a40_v1 import settled_predecessor
    from go2arx5_vla.world_model.storage import digest
    (tmp_path/'control').mkdir()
    result=tmp_path/'ALLOCATION_NATIVE_FINAL.json';result.write_text('{}')
    receipt=tmp_path/'PBS_TERMINAL_ACTUAL.json'
    receipt.write_text(json.dumps(dict(job_id='1.server',PBS_state='F',PBS_exit=1,native_exit=1)))
    owner=dict(root=str(tmp_path),status='CLOSED',job_id='1.server',PBS_exit_code=1,
        PBS_terminal_receipt_sha256=digest(receipt))
    lease=dict(status='CLOSED',job_id='1.server',actual_PBS_terminal='F',PBS_exit_code=1,
        worker_exit_code=1,PBS_terminal_receipt_sha256=digest(receipt),result_sha256=digest(result))
    path=tmp_path/'control/LEASE.json';path.write_text(json.dumps(lease))
    assert settled_predecessor(owner)
    assert not settled_predecessor(dict(owner,status='NATIVE_CLOSED_PENDING_PBS_TERMINAL'))
    lease['PBS_exit_code']=0;path.write_text(json.dumps(lease))
    assert not settled_predecessor(owner)
    lease['PBS_exit_code']=1;path.write_text(json.dumps(lease))
    receipt.write_text(json.dumps(dict(job_id='2.server',PBS_state='F',PBS_exit=1,native_exit=1)))
    assert not settled_predecessor(owner)
