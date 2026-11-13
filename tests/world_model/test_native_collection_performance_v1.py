"""Synchronization seams only; these fixtures never qualify a native episode."""
import json
from types import SimpleNamespace

import pytest

from go2arx5_vla.world_model import native_collection_performance_v1 as p
from go2arx5_vla.world_model.task_learning_grasp_v1 import CONSTRUCTION_ROLE, COLLECTION_ROLE


def spec(mode='events'):
    return dict(phase='collect-target-task-learning-grasp-v1',task_learning_role=CONSTRUCTION_ROLE,
        collection_performance=dict(version=p.VERSION,mode=mode),task_learning_grasp=dict(start_step=5703))


def test_nested_times_are_not_added_twice_and_exceptions_are_retained():
    wall=iter([0,10,40,100]);cpu=iter([0,2,9,20])
    trace=p.PerformanceTrace(wall=lambda:next(wall),cpu=lambda:next(cpu))
    with pytest.raises(RuntimeError,match='original'):
        with trace.span('outer'):
            with trace.span('inner',native_step=5703):pass
            raise RuntimeError('original')
    report=trace.report();a,b=report['spans']['outer'],report['spans']['inner']
    assert (a['wall_ns'],a['self_wall_ns'],a['thread_cpu_ns'],a['self_thread_cpu_ns'])==(100,70,20,13)
    assert b['wall_ns']==30 and a['failures']==1 and report['active_spans']==0


def test_bounded_samples_do_not_truncate_counts_or_totals():
    tick=iter(range(100))
    trace=p.PerformanceTrace(wall=lambda:next(tick),cpu=lambda:0,sample_limit=2)
    for _ in range(5):
        with trace.span('x',native_step=5703):pass
    row=trace.report()['spans']['x']
    assert row['count']==5 and row['wall_ns']==5 and row['quantiles_first_n']==2
    assert len(trace.events)==2


def test_only_new_construction_can_opt_in():
    assert p.configuration({}) is None
    assert p.configuration(spec('profile'))['mode']=='profile'
    for changed in (dict(task_learning_role=COLLECTION_ROLE),dict(phase='g2-reference-qualification'),
                    dict(collection_performance=dict(version=p.VERSION,mode='unknown'))):
        with pytest.raises(ValueError,match='FRESH_TASK_CONSTRUCTION'):
            p.configuration(dict(spec(),**changed))


def adapter_fixture(tmp_path,monkeypatch):
    from go2arx5_vla.world_model import fusion_reference_qualification_native as reference
    from go2arx5_vla.world_model import fusion_trajectory_qualification_native as native
    before=dict(step=5703,time_s=11.406);after=dict(step=5728,time_s=11.456)
    ack=dict(after,running=False,clock_paused=True)
    (tmp_path/'pause_ack.json').write_text(json.dumps(ack))
    adapter=SimpleNamespace(spec=spec(),output=tmp_path,performance=p.PerformanceTrace(),
        latest=dict(mode=SimpleNamespace(data='WBC')),ros_evidence=[],calls=[])
    adapter.observe=lambda:dict(after)
    adapter.reference_events=lambda:[]
    adapter.measured_task_state=lambda:dict(native_step=5728,time_s=11.456)
    adapter.join=dict(solver_policies_with_exact_installed_target=[dict(derived=True)],arm_request_count=1)
    adapter.gripper_checks=0
    def confirm(*args):adapter.gripper_checks+=1
    monkeypatch.setattr(native,'confirm_gripper',confirm)
    monkeypatch.setattr(reference,'consumed_window',lambda *args:adapter.join)
    def spin(seconds,predicate=None):
        adapter.calls.append((seconds,predicate is not None))
        return predicate() if predicate else True
    adapter.spin=spin
    kwargs=dict(event_start=0,before=before,after=after,clock={},submitted={})
    return adapter,kwargs,ack


def feedback(adapter,stamp=11.456):
    adapter.ros_evidence.extend(dict(topic_key=k,message=dict(time=stamp))for k in ('ee','observation'))


def test_complete_current_evidence_returns_early_without_extra_clock_write(tmp_path,monkeypatch):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch);feedback(adapter)
    p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(.15,True)] and adapter.gripper_checks==1
    assert adapter.performance.report()['spans']['atom.event_wait']['failures']==0


def memory_read_wait(adapter):
    from go2arx5_vla.world_model.native_body_memory_v3 import VERSION
    from go2arx5_vla.world_model.native_camera_transfer_v1 import CAMERA_ID
    adapter.spec.update(body_camera_session_v3=dict(version='BODY_BASE_EPISODE_SESSION_V3',mode='PERSISTENT_COMPATIBILITY'),
        observation_camera_profile=CAMERA_ID,body_camera_transport_v3=dict(version=VERSION),
        collection_event_read_wait_v3=dict(version=p.EVENT_READ_WAIT_VERSION,max_wall_s=2.))


def test_delayed_current_feedback_can_arrive_at_same_pause_without_new_clock(tmp_path,monkeypatch):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch)
    def delayed(seconds,predicate):
        adapter.calls.append((seconds,True))
        if seconds>=.374:feedback(adapter,kwargs['after']['time_s']-.010)
        return predicate()
    adapter.spin=delayed
    with pytest.raises(RuntimeError,match='TIMEOUT_NO_RETRY'):p.settle_atom(adapter,**kwargs)
    memory_read_wait(adapter);p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(.15,True),(2.,True)]
    assert adapter.observe()['step']==kwargs['after']['step']


def test_longer_read_does_not_admit26ms_source_or_missing_consumption(tmp_path,monkeypatch):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch);memory_read_wait(adapter)
    feedback(adapter,kwargs['after']['time_s']-.026)
    with pytest.raises(RuntimeError,match='TIMEOUT_NO_RETRY'):p.settle_atom(adapter,**kwargs)
    feedback(adapter);adapter.join['solver_policies_with_exact_installed_target']=[]
    with pytest.raises(RuntimeError,match='TIMEOUT_NO_RETRY'):p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(2.,True),(2.,True)]


def test_extra_wall_read_is_not_a_foreign_role_or_arbitrary_timeout_grant(tmp_path,monkeypatch):
    adapter,_,_=adapter_fixture(tmp_path,monkeypatch);memory_read_wait(adapter)
    assert p.event_read_seconds(adapter.spec)==2.
    for changed in (dict(task_learning_role='SLA_TARGET500_NATIVE_EVALUATION_V1'),
                    dict(collection_event_read_wait_v3=dict(version=p.EVENT_READ_WAIT_VERSION,max_wall_s=10.))):
        with pytest.raises(ValueError):p.event_read_seconds(dict(adapter.spec,**changed))


@pytest.mark.parametrize('fault',['missing','stale','old_query','policy','arm_request'])
def test_missing_actual_evidence_times_out_without_retry(tmp_path,monkeypatch,fault):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch)
    if fault!='missing':feedback(adapter,10. if fault=='stale' else 11.456)
    if fault=='old_query':kwargs['event_start']=len(adapter.ros_evidence)
    if fault=='policy':adapter.join['solver_policies_with_exact_installed_target']=[]
    if fault=='arm_request':adapter.join['arm_request_count']=0
    with pytest.raises(RuntimeError,match='TIMEOUT_NO_RETRY'):p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(.15,True)]


@pytest.mark.parametrize('fault',['pause_step','pause_time','running','WBC','TTL','gripper'])
def test_invariant_failures_are_not_converted_into_wait_success(tmp_path,monkeypatch,fault):
    from go2arx5_vla.world_model import fusion_trajectory_qualification_native as native
    adapter,kwargs,ack=adapter_fixture(tmp_path,monkeypatch);feedback(adapter)
    if fault=='pause_step':ack['step']-=25
    if fault=='pause_time':ack['time_s']-=.05
    if fault=='running':ack['running']=True
    if fault=='WBC':adapter.latest['mode'].data='HOLD'
    if fault=='TTL':adapter.reference_events=lambda:[dict(reason='PREFIX_TTL_EXPIRED')]
    if fault=='gripper':
        def reject(*args):raise ValueError('original gripper mismatch')
        monkeypatch.setattr(native,'confirm_gripper',reject)
    (tmp_path/'pause_ack.json').write_text(json.dumps(ack))
    with pytest.raises((ValueError,RuntimeError)):p.settle_atom(adapter,**kwargs)
    assert len(adapter.calls)<=1


@pytest.mark.parametrize('mode,step',[('profile',5703),('camera_cache',5703),('events',4703)])
def test_profile_camera_and_startup_preserve_original_fixed_wait(tmp_path,monkeypatch,mode,step):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch)
    adapter.spec=spec(mode);kwargs['before']['step']=step
    p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(.15,False)] and adapter.gripper_checks==0


def new_live_expert(adapter):
    from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert
    from go2arx5_vla.world_model.native_live_control_v3 import VERSION
    from go2arx5_vla.world_model.native_body_memory_v3 import DENSE_VERSION
    memory_read_wait(adapter)
    adapter.spec.update(task_learning_role=expert.ROLE,
        native_live_control_v3=dict(version=VERSION),
        expert_recorded_success_tail_v3=expert.RECORDED_TAIL,
        expert_dense_observation_v3=dict(version=DENSE_VERSION,frame_period_ticks=50,anchor_period_ticks=250))


def test_new_expert_prefix_waits_for_delayed_same_clock_consumption(tmp_path,monkeypatch):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch);new_live_expert(adapter)
    # The measured failure was before5703: legacy fixed-spin drained no new
    # policy for one atom, then a later decision found unconfirmed history.
    kwargs['before'].update(step=3503,time_s=7.006)
    kwargs['after'].update(step=3528,time_s=7.056)
    (tmp_path/'pause_ack.json').write_text(json.dumps(dict(kwargs['after'],running=False,clock_paused=True)))
    adapter.observe=lambda:dict(kwargs['after'])
    def delayed(seconds,predicate):
        adapter.calls.append((seconds,True))
        if seconds>=18:feedback(adapter,kwargs['after']['time_s']-.010)
        return predicate()
    adapter.spin=delayed
    p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(90,True)]and adapter.observe()['step']==3528


@pytest.mark.parametrize('fault',['policy','arm_request','26ms'])
def test_new_expert_extended_prefix_wait_never_certifies_missing_or_stale_data(tmp_path,monkeypatch,fault):
    adapter,kwargs,_=adapter_fixture(tmp_path,monkeypatch);new_live_expert(adapter)
    kwargs['before']['step']=3503
    feedback(adapter,kwargs['after']['time_s']-(.026 if fault=='26ms'else 0))
    if fault=='policy':adapter.join['solver_policies_with_exact_installed_target']=[]
    if fault=='arm_request':adapter.join['arm_request_count']=0
    with pytest.raises(RuntimeError,match='TIMEOUT_NO_RETRY'):p.settle_atom(adapter,**kwargs)
    assert adapter.calls==[(90,True)]
