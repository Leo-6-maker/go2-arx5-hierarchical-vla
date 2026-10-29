"""Real dtype and durable-command-order regressions; no native effects."""
import json
import os

import numpy as np
import pytest

from .task_path_worker import encode_record,publish_intent


def test_actual_numpy_predicate_and_nested_state_round_trip():
    milestones=dict(position_progress=False,safe_precontact=False,actual_contact=False,manipulation=False)
    milestones['position_progress'] |= np.float64(0)-np.linalg.norm(np.array([.1,0,0])) >= .02
    assert isinstance(milestones['position_progress'],np.generic)
    with pytest.raises(TypeError):json.dumps(milestones)
    original=dict(milestones=milestones,qpos=np.array([.5,-.7]),native_step=np.int64(2500),
        nested=[dict(safe=np.bool_(True),score=np.float64(.01))])
    saved=json.loads(encode_record(original))
    assert saved==dict(milestones=dict(position_progress=False,safe_precontact=False,actual_contact=False,manipulation=False),
        qpos=[.5,-.7],native_step=2500,nested=[dict(safe=True,score=.01)])
    assert original['qpos'].dtype==np.float64


def test_native_effect_observes_fsynced_exact_intent(tmp_path,monkeypatch):
    path=tmp_path/'commands.jsonl';sync=[];real=os.fsync
    monkeypatch.setattr(os,'fsync',lambda fd:(real(fd),sync.append(fd)))
    intent=dict(sequence=2,wire_sha256='a'*64,milestones=dict(progress=np.bool_(False)))
    def effects():
        assert len(sync)==1
        assert json.loads(path.read_text())==dict(sequence=2,wire_sha256='a'*64,
            milestones=dict(progress=False),phase='SUBMISSION_INTENT_BEFORE_EFFECT')
    with path.open('x')as log:assert publish_intent(log,intent,effects)is intent
    rows=[json.loads(s)for s in path.read_text().splitlines()]
    assert len(sync)==2 and [r['phase']for r in rows]==[
        'SUBMISSION_INTENT_BEFORE_EFFECT','SUBMITTED_REQUIRES_INSTALL_ACK']


@pytest.mark.parametrize('bad',[object(),np.float64(float('nan')),np.array([float('inf')])])
def test_invalid_record_cannot_publish_any_effect(tmp_path,bad):
    called=[];path=tmp_path/'commands.jsonl'
    with path.open('x')as log:
        with pytest.raises((TypeError,ValueError)):
            publish_intent(log,dict(bad=bad),lambda:called.append(True))
    assert not called and path.read_bytes()==b''


def test_persistence_failure_cannot_publish_any_effect(monkeypatch,tmp_path):
    called=[]
    def broken(fd):raise OSError('SIMULATED_STORAGE_FAILURE')
    monkeypatch.setattr(os,'fsync',broken)
    with (tmp_path/'commands.jsonl').open('x')as log:
        with pytest.raises(OSError):publish_intent(log,dict(sequence=2),lambda:called.append(True))
    assert not called
