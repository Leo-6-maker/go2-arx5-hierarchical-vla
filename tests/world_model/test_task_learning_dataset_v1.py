import copy
import json
import numpy as np
import pytest
from go2arx5_vla.world_model.task_learning_dataset_v1 import packet_arrays,exposure_schedule,fit_statistics


def packet():
    policy=np.tile([.26,-.014,.7,1.,0.,0.,0.,1.,0.,1.],(600,1))
    policy[:,2]-=np.arange(600)*1e-5
    installed=policy.copy();installed[:,0]+=.003
    stages=np.asarray(['PREGRASP']+['DESCEND']*7+['CLOSE','CLOSE']+['LIFT']*3+['HOLD']*2)
    valid=np.ones(15,bool);valid[8]=False
    checked=dict(native_command_anchor=np.arange(0,600,40),policy_absolute10=policy,
        installed_audit_only=installed,valid_window=valid,stages=stages)
    states=policy[::40].copy()
    return packet_arrays(checked,states,np.eye(4))


def test_equal_episode_visits_and_quarter_boundary_visits_exclude_intervention():
    p=packet();schedule=exposure_schedule(p)
    assert len(schedule)==32 and len(set(schedule))<32
    assert 8 not in schedule
    assert sum(i in(9,10,13)for i in schedule)==8
    assert p['valid_window'].sum()==14


def test_state_is_not_transformed_twice_and_nominal_actions_remain_distinct():
    p=packet();frame=np.eye(4);frame[:3,3]=[.1,.2,.3]
    checked=dict(native_command_anchor=p['native_command_anchor'],policy_absolute10=p['committed_absolute10'],
        installed_audit_only=p['installed_consumed_50ms_audit_only'],valid_window=p['valid_window'],
        stages=p['expert_stage_audit_only'])
    mapped=packet_arrays(checked,p['actual_state10'],frame)
    assert np.array_equal(mapped['actual_state10'],p['actual_state10'])
    assert np.allclose(mapped['committed_absolute10'][:,:3],p['committed_absolute10'][:,:3]+frame[:3,3])
    assert np.array_equal(mapped['committed_absolute10'][:,9],p['committed_absolute10'][:,9])
    assert not np.array_equal(mapped['committed_absolute10'],mapped['installed_consumed_50ms_audit_only'])


def test_statistics_exclude_DEV_and_intervention_commands_and_inputs():
    train=dict(row=dict(split='TRAIN',purpose='CONTINUOUS_TEACHER_DEMONSTRATION'),packet=packet())
    expected=json.dumps(fit_statistics([train]),sort_keys=True)
    train['packet']['actual_state10'][8,:3]=1e6
    train['packet']['committed_absolute10'][320:360,:3]=1e6
    dev=copy.deepcopy(train);dev['row']['split']='DEV'
    dev['packet']['actual_state10'][:,:3]=1e9
    dev['packet']['committed_absolute10'][:,:3]=1e9
    assert json.dumps(fit_statistics([train,dev]),sort_keys=True)==expected


def test_construction_cannot_be_used_as_new_statistics_or_partial_H40_padded():
    row=dict(split='TRAIN',purpose='TASK_CONSTRUCTION_NOT_TEACHER_QUALIFICATION_OR_BC')
    with pytest.raises(ValueError):fit_statistics([dict(row=row,packet=packet())])
    row['purpose']='CONTINUOUS_TEACHER_DEMONSTRATION';p=packet();p['committed_absolute10']=p['committed_absolute10'][:-1]
    with pytest.raises(ValueError):fit_statistics([dict(row=row,packet=p)])
