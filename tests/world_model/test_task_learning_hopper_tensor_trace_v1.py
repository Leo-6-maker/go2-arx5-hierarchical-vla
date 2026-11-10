"""Metadata/PBS contract fixtures do not qualify an actual H200 allocation."""
import json
import shlex
from pathlib import Path

import pytest
from go2arx5_vla.world_model import task_learning_hopper_tensor_trace_v1 as trace
from go2arx5_vla.world_model.task_learning_tensor_trace_v1 import MODE
from go2arx5_vla.world_model.storage import digest


@pytest.fixture
def configured(tmp_path, monkeypatch):
    project = tmp_path / 'project'; old = project / 'runs/WM02/producer'; root = old.parent/'trace'
    (old/'control').mkdir(parents=True); root.mkdir(); (old/'inference-package').mkdir()
    monkeypatch.setattr(trace, 'PROJECT_ROOT', project)
    manifest = old/'inference-package/PACKAGE_MANIFEST.json'; manifest.write_text('{}')
    package = dict(path=str(manifest.parent), sha256=digest(manifest))
    common = dict(bindings={}, parent={}, python='/fake/python', registry='/fake/registry', ledger_lock='/fake/lock')
    (old/'ALLOCATION_CONFIG.json').write_text(json.dumps(common))
    result = dict(status='CHECKPOINT_SEALED', global_step=2000, GR00T_updates=2000, sample_visits=64000,
        backbone_frozen=True, real_action_parameter_changed=True, inference_package=package)
    (old/'RESULT.json').write_text(json.dumps(result))
    (old/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0, failure=None)))
    (old/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED', worker_exit_code=0,
        result_sha256=digest(old/'RESULT.json'), config_sha256=digest(old/'ALLOCATION_CONFIG.json'))))
    (old/'PACKAGE_RECONSTRUCTION_ACTUAL.json').write_text(json.dumps(dict(
        status='PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION',actual_matched_queries=40,
        max_abs_error=dict(physical_H40=0.,normalized=0.))))
    ref = root/'reference.json'; ref.write_text(json.dumps(dict(actual_VLA_queries=40, Processor_decode_calls=40,
        model_identity=dict(candidate_package=dict(path='other-site',sha256=package['sha256'])))))
    storage = root/'storage.json'; storage.write_text(json.dumps(dict(project='CFP04-CF-007',own_root=str(project),
        own_uid=22182,project_group_membership_verified=True,quota_remaining_bytes=40*1024**3,
        predicted_new_peak_bytes=10*1024**3)))
    c = dict(common,root=str(root),producer_root=str(old),source_root='/fake/source',source_sha='a'*40,
        source_files={},profile=trace.PROFILE,project='CFP04-CF-007',pbs_job_name='gr00t_numtrace',pbs_server='configured-server',
        allocation_gpu_model='H200',wall_s=3600,predicted_work_s=1400,candidate_package=package,
        project_storage_admission=dict(path=str(storage),sha256=digest(storage)),
        tensor_trace=dict(mode=MODE,reference_result=dict(path=str(ref),sha256=digest(ref))))
    return c, old


def test_only_same_closed_first2000_package_has_zero_update_trace_spec(configured):
    c, old = configured; s = trace.configuration(c)
    assert s['max_queries']==40 and s['diagnostic_only']is True and s['physics']is False
    assert s['GR00T_updates']==s['WM_updates']==s['Actor_updates']==0
    assert s['candidate_package']==c['candidate_package']


@pytest.mark.parametrize('change',[dict(allocation_gpu_model='H100'),dict(wall_s=7200),
    dict(predicted_work_s=3000),dict(numerical_diagnostic={'mode':'other'})])
def test_wrong_gpu_window_or_combined_ablation_is_rejected(configured, change):
    c, old = configured
    with pytest.raises(PermissionError,match='FINITE_SINGLE_H200_TRACE'):
        trace.configuration(dict(c,**change))


def test_incomplete_or_active_parent_cannot_be_relabelled(configured):
    c, old = configured; p=old/'control/LEASE.json'; lease=json.loads(p.read_text())
    p.write_text(json.dumps(dict(lease,status='ACTIVE')))
    with pytest.raises(PermissionError,match='ACTUAL_FIRST2000_TERMINAL'):
        trace.configuration(c)
    p.write_text(json.dumps(lease)); r=old/'RESULT.json'; result=json.loads(r.read_text())
    r.write_text(json.dumps(dict(result,global_step=1750))); lease['result_sha256']=digest(r);p.write_text(json.dumps(lease))
    with pytest.raises(PermissionError,match='ACTUAL_FIRST2000_TERMINAL'):
        trace.configuration(c)


def test_pbs_entry_quotes_paths_and_requests_one_finite_project_gpu():
    script=trace.pbs_script('/private/root with spaces','/private/source','/private/python')
    assert '#PBS -P CFP04-CF-007'in script and '#PBS -l walltime=01:00:00'in script
    assert '#PBS -N gr00t_numtrace'in script
    assert 'select=1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB'in script
    argv=shlex.split(script.splitlines()[-1])
    assert argv[-1]=='/private/root with spaces/ALLOCATION_CONFIG.json'
    assert argv[argv.index('-m')+1]=='scripts.wm02.task_learning_hopper_tensor_trace_v1'
    assert 'train_target_gr00t'not in script


@pytest.mark.parametrize('gpu,passes',[('H200',True),('H100',False)])
def test_trace_name_is_bound_to_actual_h200_only(gpu, passes):
    from go2arx5_vla.world_model.task_learning_hopper_v1 import allocated
    record=dict(Job_Owner='e1192548@login',job_state='R',project='CFP04-CF-007',
        Job_Name='gr00t_numtrace',exec_host='compute/0*12',
        Resource_List=dict(ngpus=1,nodect=1,select='1:ngpus=1:gpu_model='+gpu+':ncpus=12:mem=225GB'))
    args=dict(job_id='fixture',uid=22182,hostname='compute',user='e1192548',
        expected_job_name='gr00t_numtrace',expected_gpu_model=gpu)
    if passes: allocated(record,**args)
    else:
        with pytest.raises(PermissionError): allocated(record,**args)
