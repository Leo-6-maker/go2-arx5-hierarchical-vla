"""Count actual execution calls and preserve the ordinary input function."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.sla.diagnose_pd4_sampling_v71 import admission, OPERATION, PROBES
from scripts.sla.sampling_timing_v71 import SamplingTimer
from scripts.wm02 import sla_pd4_sampling_queue_v71 as queue


def specification():
    return dict(operation=OPERATION,scope=OPERATION,physics=False,native_execution_admitted=False,
        optimizer_updates=0,planned_generations=64,sampling_steps=20,original_sampler_steps=10,
        original_generations_reused=64,per_window_best_selection=False,selected_indices=PROBES.copy(),noise_ids=[0,1])


@pytest.mark.parametrize('field,value', [('planned_generations',128),('sampling_steps',40),
    ('original_generations_reused',0),('optimizer_updates',1),('physics',True),
    ('per_window_best_selection',True),('noise_ids',[2,3]),('selected_indices',list(reversed(PROBES)))])
def test_only_fixed64_new20_and_original10_reuse_are_admitted(field,value):
    spec=specification();admission(spec);spec[field]=value
    with pytest.raises(PermissionError):admission(spec)


class DeviceArray:
    def __init__(self,value):
        self.value=np.asarray(value);self.ready=False


class FakeJax:
    def __init__(self):
        self.device_put=self.put
        self.tree_util=SimpleNamespace(tree_leaves=lambda value:[value],
            tree_flatten=lambda value:(list(value),len(value)),
            tree_unflatten=lambda structure,values:tuple(values))
        self.gets=[]
        self.lowerings=self.compilations=0

    def jit(self,function):
        def lower(*args):
            self.lowerings+=1
            def compile():
                self.compilations+=1
                return function
            return SimpleNamespace(compile=compile)
        return SimpleNamespace(lower=lower)

    def put(self,value):
        return DeviceArray(value)

    def block_until_ready(self,value):
        if isinstance(value,tuple):
            for x in value:self.block_until_ready(x)
        elif isinstance(value,DeviceArray):value.ready=True
        return value

    def device_get(self,value):
        assert value.ready,'Host transfer occurred before device completion'
        self.gets.append(value)
        return value.value.copy()


class Function:
    def __init__(self):
        self.executions=0

    def __call__(self,active,frozen,obs,noise):
        self.executions+=1
        assert obs.ready and noise.ready
        return DeviceArray(noise.value+active+frozen+obs.value.sum())


def test_one_compile_no_extra_warmup_and_ready_before_each_D2H():
    jax=FakeJax();function=Function()
    def observation(item,tokenizer):return jax.device_put(item),jax.device_put(np.zeros((1,40,32),np.float32))
    model=SimpleNamespace(observation=observation)
    timer=SamplingTimer(function,jax_module=jax,model_module=model)
    original_put=jax.device_put
    for active in (1.,2.):
        item=np.ones((1,2),np.float32);noise=DeviceArray(np.zeros((1,40,32),np.float32))
        with timer.prediction()as timing:
            obs,_=model.observation(item,None)
            result=timer(active,3.,obs,noise)
        np.testing.assert_array_equal(result,np.full((1,40,32),active+5.,np.float32))
        assert timing['input_H2D_calls']==2
        assert timing['input_H2D_bytes']==item.nbytes+noise.value.nbytes
        assert timing['output_D2H_bytes']==noise.value.nbytes
        assert model.observation is observation and jax.device_put is original_put
    assert jax.lowerings==jax.compilations==timer.compilations==1
    assert function.executions==timer.submitted==timer.completed==2
    assert len(jax.gets)==4  # One noise audit and one result per real execution.


def test_transfer_wrapper_is_restored_when_observation_fails():
    jax=FakeJax();original_put=jax.device_put
    def observation(*args,**kwargs):
        jax.device_put(np.zeros(3));raise ValueError('input failed')
    model=SimpleNamespace(observation=observation)
    timer=SamplingTimer(Function(),jax_module=jax,model_module=model)
    with pytest.raises(ValueError,match='input failed'):
        with timer.prediction():model.observation()
    assert model.observation is observation and jax.device_put is original_put
    assert timer.submitted==timer.completed==0


def cold_dependency(tmp_path, monkeypatch, main_pid='0'):
    root=tmp_path/'P0';(root/'control').mkdir(parents=True);(root/'runs').mkdir()
    (root/'EXECUTION_SPEC.json').write_text('{}')
    (root/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=1)))
    for name in ('WORKER_EXIT.private.json','RESULT.private.json'):
        (root/name).write_text(json.dumps(dict(exit_code=1)))
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:f'MainPID={main_pid}\nSubState=failed\n')
    return root,dict(P0_dependency=dict(root=str(root),spec_sha256=queue.sha(root/'EXECUTION_SPEC.json')))


def test_closed_failure_is_only_a_resource_dependency_not_a_science_gate(tmp_path,monkeypatch):
    _,spec=cold_dependency(tmp_path,monkeypatch)
    actual=queue.dependency_cold(spec)
    assert actual['worker_exit_code']==1 and actual['scope_closed']
    assert actual['science_completion_not_used_as_diagnostic_qualification']


def test_a_live_parent_overrides_closed_files(tmp_path,monkeypatch):
    _,spec=cold_dependency(tmp_path,monkeypatch,main_pid='123')
    assert queue.dependency_cold(spec)is None


def test_missing_receipt_or_active_actor_cannot_admit_GPU_work(tmp_path,monkeypatch):
    root,spec=cold_dependency(tmp_path,monkeypatch)
    p=root/'WORKER_EXIT.private.json';p.unlink()
    assert queue.dependency_cold(spec)is None
    p.write_text(json.dumps(dict(exit_code=1)))
    actor=root/'runs/actor';(actor/'control').mkdir(parents=True)
    (actor/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE')))
    assert queue.dependency_cold(spec)is None


def test_mismatched_terminal_receipts_cannot_admit_GPU_work(tmp_path,monkeypatch):
    root,spec=cold_dependency(tmp_path,monkeypatch)
    (root/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0)))
    with pytest.raises(PermissionError,match='TERMINAL_RECEIPTS_MUST_AGREE'):
        queue.dependency_cold(spec)


def test_actor_bound_live_process_overrides_closed_receipts(tmp_path,monkeypatch):
    root,spec=cold_dependency(tmp_path,monkeypatch)
    actor=root/'runs/actor';(actor/'control').mkdir(parents=True)
    (actor/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0)))
    (actor/'WORKER_EXIT.private.json').write_text(json.dumps(dict(exit_code=0)))
    (actor/'SUPERVISOR.private.json').write_text(json.dumps(dict(pid=999)))
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p: str(actor/'EXECUTION_SPEC.json').encode()
        if p==Path('/proc/999/cmdline')else original(p))
    assert queue.dependency_cold(spec)is None


def test_failed_model_execution_is_submitted_but_never_counted_complete():
    jax=FakeJax();function=Function()
    def observation(item,tokenizer):return jax.device_put(item),jax.device_put(np.zeros(1))
    timer=SamplingTimer(function,jax_module=jax,model_module=SimpleNamespace(observation=observation))
    def failed(*args):raise RuntimeError('device execution failed')
    timer.compiled=failed
    timer.input_structure=4
    with pytest.raises(RuntimeError,match='device execution failed'):
        with timer.prediction():
            obs,_=timer.model_module.observation(np.ones(1),None)
            timer(1,2,obs,DeviceArray(np.zeros((1,40,32),np.float32)))
    assert timer.submitted==1 and timer.completed==0
