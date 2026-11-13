"""Actual tmpfs rename and durable-journal routing; no model or physics."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2arx5_vla.world_model import native_live_control_v3 as live
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert


def spec():
    return dict(native_live_control_v3=dict(version=live.VERSION),
        phase='collect-target-task-learning-grasp-v1', task_learning_role=expert.ROLE,
        expert_recorded_success_tail_v3=expert.RECORDED_TAIL,
        expert_dense_observation_v3={'version': 'DENSE'})


def adapter(tmp_path):
    if not Path('/dev/shm').exists():
        pytest.skip('Linux tmpfs required')
    return SimpleNamespace(spec=spec(),output=tmp_path,processes=[])


def test_original_capsules_keep_original_paths_and_reject_wrong_opt_in(tmp_path):
    a=SimpleNamespace(spec={},output=tmp_path)
    live.initialize(a)
    assert live.configuration({}) is None and live.live_path(a,'state.bin')==tmp_path/'state.bin'
    assert not list(tmp_path.iterdir())
    for changed in (dict(task_learning_role='OLD'),dict(expert_recorded_success_tail_v3=None),
                    dict(native_live_control_v3=dict(version='OTHER'))):
        with pytest.raises(ValueError):live.configuration(dict(spec(),**changed))


def test_atomic_current_snapshot_and_original_journal_survive_archive(tmp_path):
    a=adapter(tmp_path);live.initialize(a);root=a.live_control_output
    try:
        assert root.is_relative_to('/dev/shm')
        assert (root/'state.bin.journal').resolve()==tmp_path/'state.bin.journal'
        (root/'state.bin.journal').write_bytes(b'actual journal')
        assert (tmp_path/'state.bin.journal').read_bytes()==b'actual journal'
        for value in (b'first native packet',b'next native packet'):
            pending=root/'state.bin.pending';pending.write_bytes(value)
            os.replace(pending,root/'state.bin')
            assert (tmp_path/'state.bin').read_bytes()==value
            assert live.live_path(a,'state.bin').read_bytes()==value
        (root/'pause_ack.json').write_text('{"step":25}')
        live.check(a)
        live.archive(a)
        assert not root.exists() and a.live_control_output is None
        assert not (tmp_path/'state.bin').is_symlink()
        assert (tmp_path/'state.bin').read_bytes()==b'next native packet'
        assert json.loads((tmp_path/'LIVE_CONTROL_ARCHIVE.private.json').read_text())['children_terminal']
        assert (tmp_path/'state.bin.journal').read_bytes()==b'actual journal'
    finally:
        if root.exists():live.archive(a)


def test_live_child_prevents_archive_and_failed_copy_retains_tmpfs(tmp_path,monkeypatch):
    a=adapter(tmp_path);live.initialize(a);root=a.live_control_output
    try:
        (root/'state.bin').write_bytes(b'actual state')
        a.processes=[('sim',SimpleNamespace(poll=lambda:None))]
        with pytest.raises(RuntimeError,match='CHILDREN_TERMINAL'):live.archive(a)
        assert root.exists() and (tmp_path/'state.bin').is_symlink()
        a.processes=[]
        original=os.replace
        def fail(source,destination):
            if Path(source).name=='state.bin.archive-pending':raise OSError('archive disk failure')
            return original(source,destination)
        with monkeypatch.context() as m:
            m.setattr(os,'replace',fail)
            with pytest.raises(OSError,match='disk failure'):live.archive(a)
        assert (root/'state.bin').read_bytes()==b'actual state'
        assert not (tmp_path/'LIVE_CONTROL_ARCHIVE.private.json').exists()
        (tmp_path/'state.bin.archive-pending').unlink()
    finally:live.archive(a)


def test_unknown_live_members_and_oversized_snapshots_are_rejected(tmp_path,monkeypatch):
    a=adapter(tmp_path);live.initialize(a);root=a.live_control_output
    try:
        (root/'unexpected').write_bytes(b'unknown')
        with pytest.raises(RuntimeError,match='UNEXPECTED'):live.check(a)
        (root/'unexpected').unlink()
        (root/'state.bin').write_bytes(b'12345')
        with monkeypatch.context() as m:
            m.setattr(live,'MAX_BYTES',4)
            with pytest.raises(MemoryError,match='BOUNDED'):live.check(a)
    finally:live.archive(a)


@pytest.mark.parametrize('fresh,mode,passes',[(True,'WBC',True),(False,'WBC',False),(True,'HOLD',False)])
def test_delayed_mode_notification_never_republishes_or_accepts_old_receipt(fresh,mode,passes):
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    calls=[]
    a=SimpleNamespace(spec=dict(spec(),phase='collect-target-probe'),counts={'mode':1},
        latest={'mode':SimpleNamespace(data='HOLD')},modes=object(),
        dynamic_phase=lambda:True,advance_initialization=lambda step:calls.append(('initial',step)),
        publish=lambda pub,text:calls.append(('publish',text)),
        advance_to=lambda step:calls.append(('advance',step)),
        wait_source=lambda:None,capture_loaded_identity=lambda:None)
    # Keep the production phase for its configuration; use a phase proxy only
    # after the mode wait to omit unrelated geometry setup from this seam.
    a.spec=spec()
    def spin(seconds,predicate):
        calls.append(('wait',seconds))
        if seconds>=18:
            a.counts['mode']+=int(fresh);a.latest['mode']=SimpleNamespace(data=mode)
        result=predicate()
        if result:a.spec=dict(a.spec,phase='probe')
        return result
    a.spin=spin
    if passes:TargetSequentialAdapter.prepare_dynamic(a)
    else:
        with pytest.raises(RuntimeError,match='MODE_ACK_MISSING'):TargetSequentialAdapter.prepare_dynamic(a)
    assert calls.count(('publish','WBC'))==1 and ('wait',90)in calls
    assert (('advance',2503)in calls)==passes
    assert live.acknowledgement_wait({},3)==3 and live.acknowledgement_wait({},10)==10
