import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
from go2arx5_vla.world_model.sla_native_continuation_v2 import wait_seconds


def make_adapter(tmp_path, *, extended=True, matching=True):
    source=Path(__file__).resolve().parents[1]/'experiments/sla_go2_posttrain/native_pause_read_wait_v1.json'
    adapter=TargetSequentialAdapter.__new__(TargetSequentialAdapter)
    adapter.output=tmp_path
    adapter.trace_sealed=False
    adapter.spec=dict(max_ticks=21203,phase='collect-target-task-learning-grasp-v1',startup_pause_stride_ticks=25,
                      task_learning_role=task.SLA_ROLE)
    if extended:
        adapter.spec['native_pause_read_wait']=dict(path=str(source),sha256=digest(source))
    adapter.dynamic_phase=lambda:False
    adapter.observe=lambda:dict(step=2075,time_s=4.15)
    adapter.keys=SimpleNamespace(get_subscription_count=lambda:1)
    adapter.control=lambda:{}
    publications=[];waits=[]
    adapter.resume_clock_once=lambda ticks:publications.append(ticks)
    def spin(seconds,predicate):
        waits.append(seconds)
        # Simulate a read arriving after the old wall wait. No physics is run.
        (tmp_path/'pause_ack.json').write_text(json.dumps(dict(step=2075 if matching else 2050)))
        return seconds>=55 and predicate()
    adapter.spin=spin
    return adapter,publications,waits


def test_late_exact_ack_is_read_without_republishing_or_changing_quota(tmp_path):
    adapter,publications,waits=make_adapter(tmp_path)
    assert adapter.advance_to(2075)['step']==2075
    assert publications==[2075] and waits==[90]
    assert (tmp_path/'step_quota').read_text()=='2075\n'


def test_longer_read_wait_does_not_admit_a_stale_ack(tmp_path):
    adapter,publications,waits=make_adapter(tmp_path,matching=False)
    with pytest.raises(RuntimeError,match='EXACT_NATIVE_PAUSE_NOT_REACHED'):
        adapter.advance_to(2075)
    failure=json.loads((tmp_path/'CLOCK_RESUME_FAILURE.json').read_bytes())
    assert publications==[2075] and waits==[90]
    assert failure['automatic_ENTER_retries']==0 and failure['requested_step']==2075


def test_original_clock_wait_remains_35_seconds(tmp_path):
    adapter,publications,waits=make_adapter(tmp_path,extended=False)
    adapter.spec['task_learning_role']=task.NATIVE_A500_ROLE
    with pytest.raises(RuntimeError,match='EXACT_NATIVE_PAUSE_NOT_REACHED'):
        adapter.advance_to(2075)
    assert publications==[2075] and waits==[35]


def test_another_role_cannot_import_the_sla_wait_grant(tmp_path):
    adapter,_,_=make_adapter(tmp_path)
    with pytest.raises(ValueError,match='SLA_ONLY_SEALED_READ_WAIT'):
        wait_seconds(dict(adapter.spec,task_learning_role=task.NATIVE_A500_ROLE))


def test_cold_suffix_cannot_replay_either_retained_failure(monkeypatch):
    from go2arx5_vla.world_model import sla_native_v1 as sla
    from go2arx5_vla.world_model import sla_native_continuation_v2 as cold
    rows=[dict(layout=x)for x in('TRAIN01','TRAIN07','TRAIN13','TRAIN25')]
    result=dict(path='/original/RESULT.private.json',sha256='closed-failure')
    panel=dict(source_sha='new-sealed-source',cold_continuation=dict(result=result))
    monkeypatch.setattr(sla,'validate_panel',lambda p:rows)
    monkeypatch.setattr(cold,'prefix',lambda p:rows[:2])
    inp=dict(schema=cold.INPUT_SCHEMA,source_sha=panel['source_sha'],assignment_slice=[2,4],population_predecessor=result)
    assert [r['layout']for r in sla.selection(inp,panel)]==['TRAIN13','TRAIN25']
    for span in([0,4],[1,4],[2,3]):
        with pytest.raises(ValueError,match='ONLY_NEVER_STARTED_SUFFIX'):
            sla.selection(dict(inp,assignment_slice=span),panel)
    with pytest.raises(ValueError,match='ONLY_NEVER_STARTED_SUFFIX'):
        sla.selection(dict(inp,population_predecessor=None),panel)


def test_changed_failure_prefix_is_rejected_before_any_continuation():
    from go2arx5_vla.world_model import sla_native_continuation_v2 as cold
    panel=dict(population='TRAIN4',cold_continuation=dict(schema=cold.INPUT_SCHEMA,
        result=dict(sha256=cold.ORIGINAL_RESULT_SHA),outcomes=dict(sha256='replaced-failure')))
    with pytest.raises(ValueError,match='EXACT_RETAINED_FAILURE_PREFIX'):
        cold.prefix(panel)
