"""Synthetic closed-file evidence checks, not site or native qualification."""
import json
import copy
from pathlib import Path
import pytest
from go2arx5_vla.world_model import stable_task_prefix_qualification_v1 as q
from go2arx5_vla.world_model.storage import digest


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def sealed(tmp_path):
    root=tmp_path/'run';spec=root/'EXECUTION_SPEC.json';result=root/'native-01/RESULT.json'
    put(spec,dict(phase='SYNTHETIC_NATIVE',max_ticks=20))
    put(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=digest(spec)))
    put(root/'WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    value=dict(status='NATIVE_STAGE_PASSED',phase='SYNTHETIC_NATIVE',actual_target_integrations=20,
        native_binary_sha256=q.dataset.NATIVE,failure=None,shutdown_clean=True,
        owned_process_exits=[dict(name=n,exit_code=0)for n in ('sim','controller','solver','planner')],
        original_joint_or_warning_samples=0,native_outer_guard_rejection_count=0,physics_stops=[],trace_audit_error=None,
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS',native_step_records=20,writer=dict(dropped=0,events=180,written=180)))
    put(result,value)
    return root,result,value


def test_closed_receipt_and_unchanged_result_are_required(tmp_path):
    root,path,_=sealed(tmp_path)
    actual,spec,result=q.sealed_native(dict(path=str(path),sha256=digest(path)))
    assert actual==root and spec['max_ticks']==result['actual_target_integrations']==20
    put(root/'control/LEASE.json',dict(status='ACTIVE',worker_exit_code=None,spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    with pytest.raises(ValueError,match='CLOSED_SAFE_COMPLETE'):q.sealed_native(dict(path=str(path),sha256=digest(path)))


@pytest.mark.parametrize('fault',['writer_drop','missing_child','failed_child','unsafe','partial','unclean'])
def test_failed_or_incomplete_native_evidence_never_qualifies(tmp_path,fault):
    _,path,value=sealed(tmp_path)
    if fault=='writer_drop':value['trace_audit']['writer']['dropped']=1
    if fault=='missing_child':value['owned_process_exits'].pop()
    if fault=='failed_child':value['owned_process_exits'][0]['exit_code']=1
    if fault=='unsafe':value['physics_stops']=[dict(reason='ILLEGAL_CONTACT')]
    if fault=='partial':value['actual_target_integrations']=19
    if fault=='unclean':value['shutdown_clean']=False
    put(path,value)
    with pytest.raises(ValueError,match='CLOSED_SAFE_COMPLETE'):q.sealed_native(dict(path=str(path),sha256=digest(path)))


def test_cpu_probe_cannot_use_an_unrelated_machine(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(q.os,'uname',lambda:SimpleNamespace(nodename='not-target-site'))
    with pytest.raises(ValueError,match='ACTUAL_TARGET_CPU_FOR_TIMING'):q.qualify({}, {}, {})
