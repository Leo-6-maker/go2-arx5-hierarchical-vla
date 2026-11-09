"""Recipe fixtures grant no data, site, model or native admission."""
import copy
import pytest
from go2arx5_vla.world_model import task_learning_training_profile_v1 as task
from go2arx5_vla.world_model import target_gr00t_profile_v1 as original


def specification(endpoint=2000,start=0,seed=20261109):
    return dict(profile=task.PROFILE,kind=task.KIND,execution_site='HOPPER_SINGLE_GPU',
        recipe=task.recipe('a'*64,'b'*64,endpoint=endpoint,seed=seed),start_step=start,
        GR00T_updates=endpoint-start,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
        physics=False,WBC_changes=False,parent_manifest_sha256=original.PARENT,deadline_epoch=1234.,
        resume=None if not start else dict(path='fixture-checkpoint-not-native',sha256='c'*64),
        task_learning_admission=dict(path='fixture-not-actual-admission',sha256='d'*64),
        project_storage_admission=dict(path='fixture-not-actual-storage',sha256='a'*64))


def test_new_recipe_keeps_original_math_and_changes_explicit_task_identity():
    s=specification();p=original.validate(s)
    assert p['seed']==20261109 and p['mode']=='FORMAL'and p['variant']=='FULL_ACTION_SIDE'
    assert (p['effective_batch'],p['microbatch'],p['scheduler_horizon'],p['warmup_updates'])==(32,1,8000,200)
    assert p['learning_rate']==1e-4 and p['weight_decay']==1e-5
    assert p['policy_execution_prefix_atoms']==40
    with pytest.raises(PermissionError):original.validate_original(s)


def test_restore_has_the_same_full_schedule_and_later_endpoints_need_a_decision():
    first=specification();resumed=specification(start=1750)
    assert task.validate(first)==task.validate(resumed)
    later=specification(endpoint=4000,start=2000)
    with pytest.raises(PermissionError,match='NATIVE_DEV_DECISION'):task.validate(later)
    later['endpoint_decision']=dict(path='not-actual-decision',sha256='e'*64)
    assert task.validate(later)['scheduler_horizon']==8000
    with pytest.raises(PermissionError):task.validate(specification(endpoint=4000))


def test_old_fit4_no_admission_other_sites_or_unconfirmed_seeds_cannot_start_formal():
    for update in (dict(execution_site='CORELAB_4080S'),dict(task_learning_admission=None),
                   dict(kind=original.KIND)):
        s=specification();s.update(update)
        with pytest.raises(PermissionError):task.validate(s)
    second=specification(seed=20261110)
    with pytest.raises(PermissionError,match='FIRST_SEED_GAINS'):task.validate(second)
    second['first_seed_learning_and_native_gain']=dict(path='not-actual-gain',sha256='f'*64)
    assert task.validate(second)['seed']==20261110
    with pytest.raises(PermissionError):task.recipe('a'*64,'b'*64,endpoint=2000,microbatch=2)
