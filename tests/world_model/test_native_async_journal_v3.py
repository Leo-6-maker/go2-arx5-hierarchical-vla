"""Bounded handoff, slow-storage and failure checks; no native qualification."""
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from go2arx5_vla.world_model import native_async_journal_v3 as j


class SlowFile:
    def __init__(self, path, entered, release, *, fail=False):
        self.file = path.open('xb', buffering=0)
        self.entered, self.release, self.fail = entered, release, fail
        self.thread_ids = []

    def write(self, payload):
        self.thread_ids.append(threading.get_ident())
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError('test storage stalled')
        if self.fail:
            raise OSError('injected write failure')
        # Exercise successful short writes, too.
        return self.file.write(payload[:3])

    def __getattr__(self, name):
        return getattr(self.file, name)


def blocked_writer(tmp_path, *, records=3, bytes_=12, fail=False):
    entered, release = threading.Event(), threading.Event()
    files = []
    def open_file(path):
        f = SlowFile(path, entered, release, fail=fail)
        files.append(f)
        return f
    writer = j.BoundedJournal(tmp_path/'records.jsonl', max_records=records,
                              max_bytes=bytes_, open_file=open_file)
    writer.append(b'1\n')
    assert entered.wait(2)
    return writer, files[0], release


def test_callback_returns_while_disk_is_blocked_and_every_short_write_is_kept(tmp_path):
    writer, file, release = blocked_writer(tmp_path)
    try:
        writer.append(b'22\n')
        writer.append(b'333\n')
        assert writer.pending_records == 3 and writer.pending_bytes == 9
        assert not writer.report()['committed'] and writer.written == 0
        assert all(t != threading.get_ident() for t in file.thread_ids)
    finally:
        release.set()
        result = writer.close()
    expected = b'1\n22\n333\n'
    assert writer.path.read_bytes() == expected
    assert result['committed'] and result['accepted_records'] == result['written_records'] == 3
    assert result['sha256'] == hashlib.sha256(expected).hexdigest()
    assert result['pending_records'] == result['pending_bytes'] == 0
    assert result['peak_payload_bytes'] == 9


@pytest.mark.parametrize('records,bytes_,payload', [(1,100,b'2\n'),(5,4,b'333\n')])
def test_inflight_payload_counts_against_both_bounds(tmp_path,records,bytes_,payload):
    writer, _, release = blocked_writer(tmp_path,records=records,bytes_=bytes_)
    try:
        with pytest.raises(BufferError,match='NO_RECORD_DROP_OR_NEW_CLOCK'):
            writer.append(payload)
        assert writer.accepted == 1
    finally:
        release.set();result=writer.close()
    assert result['written_records'] == 1 and writer.path.read_bytes() == b'1\n'


def test_storage_failure_is_propagated_and_never_publishes_a_complete_hash(tmp_path):
    writer, _, release = blocked_writer(tmp_path, fail=True)
    release.set()
    with pytest.raises(RuntimeError,match='WRITE_FAILED'):
        writer.close()
    result=writer.report()
    assert not result['committed'] and result['sha256'] is None
    assert result['accepted_records'] == 1 and result['written_records'] == 0
    with pytest.raises(RuntimeError,match='WRITE_FAILED'):
        writer.append(b'2\n')


def test_finite_close_does_not_claim_stalled_records_committed(tmp_path):
    writer, _, release = blocked_writer(tmp_path)
    try:
        with pytest.raises(TimeoutError,match='FINITE_CLOSE_INCOMPLETE'):
            writer.close(timeout=.01)
        assert not writer.report()['committed']
    finally:
        release.set();writer.close()


def test_safe_pause_is_checked_before_backpressure_wait(tmp_path):
    writer, _, release = blocked_writer(tmp_path,records=1)
    adapter=SimpleNamespace(output=tmp_path,ros_journal=writer,performance=None,
        observe=lambda:dict(step=25,time_s=.05),guard=lambda:None)
    ack=dict(step=25,time_s=.05,running=True,clock_paused=False)
    (tmp_path/'pause_ack.json').write_text(json.dumps(ack))
    try:
        with pytest.raises(RuntimeError,match='REQUIRES_EXACT_SAFE_PAUSE'):
            j.before_clock_grant(adapter)
        with pytest.raises(TimeoutError,match='BACKPRESSURE_TIMEOUT'):
            writer.wait_low_water(.01)
        ack.update(running=False,clock_paused=True)
        (tmp_path/'pause_ack.json').write_text(json.dumps(ack))
        release.set()
        j.before_clock_grant(adapter)
    finally:
        release.set();writer.close()


def test_unknown_file_and_mutable_payload_are_rejected(tmp_path):
    path=tmp_path/'old.jsonl';path.write_bytes(b'preserve\n')
    with pytest.raises(FileExistsError):j.BoundedJournal(path)
    assert path.read_bytes()==b'preserve\n'
    writer=j.BoundedJournal(tmp_path/'new.jsonl')
    try:
        for value in (bytearray(b'1\n'),b'partial',b'1\n2\n'):
            with pytest.raises(ValueError,match='IMMUTABLE_COMMITTED'):
                writer.append(value)
    finally:writer.close()


def test_a_rejected_callback_cannot_make_a_partial_journal_complete(tmp_path):
    writer=j.BoundedJournal(tmp_path/'records.jsonl')
    writer.append(b'1\n')
    adapter=SimpleNamespace(ros_journal=writer,output=tmp_path,ros_evidence=[{},{}])
    with pytest.raises(RuntimeError,match='CALLBACK_COUNT_MISMATCH_NO_COMMIT'):
        j.close_journal(adapter)
    report=json.loads((tmp_path/'ASYNC_ROS_JOURNAL_ACTUAL.private.json').read_bytes())
    assert not report['committed'] and report['sha256']is None
    assert report['observed_callback_records']==2 and report['written_records']==1


def test_opt_in_is_limited_to_fresh_expert_performance_roles():
    from go2arx5_vla.world_model.native_collection_performance_v1 import VERSION
    from go2arx5_vla.world_model.task_learning_grasp_v1 import CONSTRUCTION_ROLE,SLA_ROLE
    spec=dict(collection_async_journal_v3=dict(version=j.VERSION),
        collection_performance=dict(version=VERSION,mode='events'),
        phase='collect-target-task-learning-grasp-v1',task_learning_role=CONSTRUCTION_ROLE)
    assert j.configuration({})is None and j.configuration(spec)['version']==j.VERSION
    with pytest.raises(ValueError):j.configuration(dict(spec,task_learning_role=SLA_ROLE))
    with pytest.raises(ValueError):j.configuration(dict(spec,collection_performance=None))
