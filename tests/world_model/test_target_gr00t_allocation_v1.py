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
