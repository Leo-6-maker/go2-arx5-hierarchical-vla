"""Allocation envelope fixtures never prove a GPU or training-data admission."""
import pytest
from go2arx5_vla.world_model import task_learning_hopper_bootstrap_v1 as boot
from go2arx5_vla.world_model import task_learning_training_profile_v1 as task


def config():
    return dict(root=task.PROJECT_ROOT+'/runs/fixture',source_root='/not-an-actual-capsule',
        source_sha='a'*40,source_files={},bindings={},parent={},python='/not-actual-python',
        pbs_server='private-official-server-fixture',recipe=task.recipe('a'*64,'b'*64,endpoint=2000),
        start_step=0,generation_id='fixture-only',role='fixture',registry='/fixture-registry',
        task_learning_admission=dict(path='/not-an-admission',sha256='c'*64),
        project_storage_admission=dict(path='/not-storage',sha256='d'*64),
        predicted_work_s=18000,wall_s=21600,absolute_deadline_epoch=123456.)


def test_missing_server_or_closeout_window_fails_before_a_submission_or_model():
    c=config();c.pop('pbs_server')
    with pytest.raises(PermissionError,match='SERVER_BEFORE_FORMAL_SUBMISSION'):boot.configuration(c)
    c=config();c['predicted_work_s']=21000
    with pytest.raises(PermissionError,match='MEASURED_FINITE_FORMAL_WINDOW'):boot.configuration(c)


def test_actual_allocation_fields_do_not_change_the_frozen_schedule_or_data():
    c=config();before=boot.configuration(c)
    s=boot.execution_spec(c,job_id='fixture-job',gpu=dict(uuid='fixture-uuid',name='NVIDIA H200'),deadline=123456.)
    assert s['recipe']==before['recipe']==c['recipe']
    assert s['profile']==task.PROFILE and s['kind']==task.KIND
    assert s['GR00T_updates']==2000 and s['physics']is False
    assert s['GPU_uuid']=='fixture-uuid'and s['pbs_job_name']=='gr00t_task_sft'
    with pytest.raises(FileNotFoundError):boot.preflight(c)


def test_midpoint_recovery_keeps_same_generation_seed_and_eight_thousand_schedule():
    c=config();c.update(start_step=1750,resume=dict(path='/fixture-checkpoint',sha256='e'*64))
    s=boot.configuration(c)
    assert s['GR00T_updates']==250 and s['recipe']['scheduler_horizon']==8000
    assert s['recipe']['seed']==20261109 and s['generation_id']==c['generation_id']
