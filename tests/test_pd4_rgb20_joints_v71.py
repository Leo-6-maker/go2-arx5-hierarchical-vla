import json
from pathlib import Path

import pytest

from scripts.sla.diagnose_pd4_rgb20_joints_v71 import admission,OPERATION,PROBES
from scripts.wm02 import sla_pd4_rgb20_clone_queue_v71 as queue


@pytest.mark.parametrize('key,value',[('maximum_clone_calls',128),('planned_variants',96),('sampling_steps',40),
    ('indices',list(reversed(PROBES))),('noise_ids',[2,3]),('baseline64_rerun',True),('label_checks_rerun',True),
    ('new_model_generations',1),('optimizer_updates',1),('physics',True),('native_execution_admitted',True)])
def test_only_new20_outputs_are_admitted_without_rerunning_baseline(key,value):
    s=dict(operation=OPERATION,indices=PROBES.copy(),noise_ids=[0,1],planned_variants=64,maximum_clone_calls=64,
        sampling_steps=20,baseline64_rerun=False,label_checks_rerun=False,new_model_generations=0,
        optimizer_updates=0,physics=False,native_execution_admitted=False)
    admission(s);s[key]=value
    with pytest.raises(PermissionError):admission(s)


def upstream(tmp_path,monkeypatch,pid='0',exit_code=0):
    root=tmp_path/'sampling';(root/'control').mkdir(parents=True);(root/'inference').mkdir()
    (root/'EXECUTION_SPEC.json').write_text('{}')
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=exit_code)))
    (root/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=exit_code)))
    p=root/'inference/RESULT.private.json';p.write_text('{}')
    (root/'RESULT.private.json').write_text(json.dumps(dict(exit_code=exit_code,actual_generations=64,inference_result_sha256=queue.sha(p))))
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:pid+'\n')
    return root,dict(root=str(root),spec_sha256=queue.sha(root/'EXECUTION_SPEC.json'))


def test_live_upstream_is_not_treated_as_complete(tmp_path,monkeypatch):
    _,pin=upstream(tmp_path,monkeypatch,pid='123')
    assert queue.completed_sampling(pin)is None


def test_failed_upstream_never_triggers_replacement_inference(tmp_path,monkeypatch):
    root,pin=upstream(tmp_path,monkeypatch,exit_code=1)
    with pytest.raises(RuntimeError,match='FAILURE_RETAINED_NO_REPLACEMENT'):
        queue.completed_sampling(pin)
    assert not(root/'INFERENCE_INTENT.private.json').exists()


def test_complete_upstream_returns_exact_existing_result_pin(tmp_path,monkeypatch):
    root,pin=upstream(tmp_path,monkeypatch)
    actual=queue.completed_sampling(pin)
    assert Path(actual['path'])==root/'inference/RESULT.private.json'
    assert actual['sha256']==queue.sha(actual['path'])
    (root/'inference/RESULT.private.json').write_text('{"changed":true}')
    with pytest.raises(PermissionError,match='COMPLETED_RGB20_BYTES_REQUIRED'):
        queue.completed_sampling(pin)
