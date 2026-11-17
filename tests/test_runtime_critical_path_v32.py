import json
from pathlib import Path
import os
import subprocess
import sys
from threading import Event

import numpy as np
import pytest

from scripts.pi0294.e1.journal import Journal, MutationOutcomeUnknown
from go2arx5_vla.world_model.async_request_archive_v1 import AsyncRequestArchive, unknown_reservations
from go2arx5_vla.world_model.hybrid_ipc_storage_v31 import archive_request
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver, load_input
from go2arx5_vla.world_model.verified_observation_v1 import VerifiedObservation
from test_hybrid_nonphysical_v3 import fixture, strict_request


def memory_receiver(tmp_path):
    legacy, q, a = fixture(tmp_path)
    observation = VerifiedObservation.from_file(q['input'])
    s = legacy.state
    r = Receiver(tmp_path/'memory', s['episode'], s['native_step'], s['previous'],
                 s['previous_evidence'], s['bindings'], observation=observation, cache_compiled=True)
    request = r.request(q['input'], q['language'], q['noise_id'])
    return r, request, dict(a, request_id=request['request_id']), legacy, q, observation


def test_single_read_snapshot_is_immutable_and_new_boundary_rechecks(tmp_path, monkeypatch):
    _, q, _ = fixture(tmp_path)
    original = Path.open
    reads = []
    def counted(path, *args, **kwargs):
        if str(path) == q['input']['path']:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    snapshot = VerifiedObservation.from_file(q['input'])
    assert len(reads) == 1
    a = snapshot.values_for(q['input'])['body_rgb']
    with pytest.raises(ValueError):
        a.setflags(write=True)
    Path(q['input']['path']).write_bytes(b'changed after receive boundary')
    assert snapshot.values_for(q['input'])['body_rgb'].tobytes() == a.tobytes()
    with pytest.raises(ValueError, match='BYTES_CHANGED'):
        load_input(q)
    with pytest.raises(ValueError, match='IDENTITY'):
        snapshot.values_for(dict(q['input'], path='different'))


def test_cached_H40_wire_and_atoms_equal_legacy_without_reopening_observation(tmp_path, monkeypatch):
    r, q, a, old, oq, snapshot = memory_receiver(tmp_path)
    old.install(oq, dict(a, request_id=oq['request_id']))
    ack = r.install(q, a)
    original = Path.open
    def forbid_live_reads(path, *args, **kwargs):
        if str(path) == q['input']['path'] or path.name.startswith('compiled-'):
            raise AssertionError('REPEATED_INPUT_OR_PER_ATOM_COMPILED_READ')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', forbid_live_reads)
    assert r.install(q, a) == ack
    assert r.consume(10)['full_P10']
    monkeypatch.setattr(Path, 'open', original)
    old.consume(10)
    new_doc = json.loads((r.root/'install-0001.json').read_text())['bundle']['eef_wire']
    old_doc = json.loads((old.root/'install-0001.json').read_text())['bundle']['eef_wire']
    assert new_doc == old_doc
    for i in range(10):
        new = json.loads((r.root/f'emitted-0001-{i:02d}.json').read_text())
        prior = json.loads((old.root/f'emitted-0001-{i:02d}.json').read_text())
        assert new['EEF10_mechanical'] == prior['EEF10_mechanical']
        assert new['source_native_s'] == prior['source_native_s']


@pytest.mark.parametrize('fault', ['compiled_identity', 'STOP', 'TTL', 'suffix'])
def test_cached_receiver_preserves_original_guards(tmp_path, monkeypatch, fault):
    r, q, a, _, _, _ = memory_receiver(tmp_path)
    if fault == 'suffix':
        from go2arx5_vla.world_model.storage import digest
        with np.load(a['action']['path']) as z:
            raw = z['physical_H40'].copy()
        raw[30, 0] += .1
        np.savez(a['action']['path'], physical_H40=raw, normalized=np.zeros((1,40,32)))
        a['action']['sha256'] = digest(a['action']['path'])
        with pytest.raises(ValueError, match='FULL_PROPOSAL_REJECTED'):
            r.install(q, a)
        assert not list(r.root.glob('install-*'))
        return
    if fault == 'TTL':
        strict_request(r, q, a)
        monkeypatch.setattr('time.monotonic', lambda: q['created_monotonic']+.02)
    r.install(q, a)
    if fault == 'compiled_identity':
        r.state['installed']['compiled']['sha256'] = '0'*64
    elif fault == 'STOP':
        r.stop()
    else:
        monkeypatch.setattr('time.monotonic', lambda: q['created_monotonic']+.36)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        r.consume(1)
    assert r.state['emitted_atoms'] == 0


def archive_fixture(tmp_path, name):
    source = tmp_path/('live-'+name)
    source.mkdir()
    (source/'data').write_bytes(name.encode())
    return source, tmp_path/('archive-'+name)


def test_pending_includes_active_and_reserved_and_applies_backpressure(tmp_path):
    started, release = Event(), Event()
    def slow(source, destination):
        started.set()
        assert release.wait(2)
        return archive_request(source, destination)
    q = AsyncRequestArchive(Journal(tmp_path/'journal', 'one'), archive=slow, max_bytes=40)
    tickets = []
    for i in range(4):
        ticket = q.reserve(str(i), dict(new_generations=0), max_bytes=10, timeout_s=.2)
        tickets.append(ticket)
        q.require_dispatch(ticket)
        if i < 3:
            q.submit(ticket, *archive_fixture(tmp_path, str(i)))
    assert started.wait(.2)
    with pytest.raises(TimeoutError, match='BACKPRESSURE'):
        q.reserve('fifth', {}, max_bytes=1, timeout_s=.03)
    with pytest.raises(TimeoutError, match='STOP'):
        q.reserve('stop', {}, max_bytes=1, timeout_s=1, stopped=lambda: True)
    q.submit(tickets[-1], *archive_fixture(tmp_path, '3'))
    release.set()
    result = q.close(timeout_s=3)
    assert result == dict(archived=4, peak_pending=4, peak_reserved_bytes=40, max_pending=4, max_reserved_bytes=40)
    assert not unknown_reservations(q.journal)


def test_byte_bound_can_backpressure_before_four_requests(tmp_path):
    q = AsyncRequestArchive(Journal(tmp_path/'journal', 'one'), max_bytes=10)
    ticket = q.reserve('one', {}, max_bytes=10, timeout_s=1)
    with pytest.raises(TimeoutError):
        q.reserve('two', {}, max_bytes=1, timeout_s=.03)
    q.submit(ticket, *archive_fixture(tmp_path, 'one'))
    assert q.close(timeout_s=2)['archived'] == 1


def test_archive_failure_latches_new_dispatch_but_drains_existing_evidence(tmp_path):
    release, failed = Event(), Event()
    def broken(source, destination):
        assert release.wait(2)
        if Path(source).name.endswith('bad'):
            failed.set()
            raise OSError('INJECTED_DISK_FULL')
        return archive_request(source, destination)
    journal = Journal(tmp_path/'journal', 'one')
    q = AsyncRequestArchive(journal, archive=broken)
    a = q.reserve('bad', {}, max_bytes=100, timeout_s=1)
    b = q.reserve('good', {}, max_bytes=100, timeout_s=1)
    q.submit(a, *archive_fixture(tmp_path, 'bad'))
    q.submit(b, *archive_fixture(tmp_path, 'good'))
    release.set()
    with pytest.raises(MutationOutcomeUnknown):
        q.close(timeout_s=2)
    assert failed.is_set() and 'good' in q.completed
    assert unknown_reservations(journal) == ['bad']
    with pytest.raises(MutationOutcomeUnknown, match='DISPATCH'):
        q.require_dispatch()
    with pytest.raises(MutationOutcomeUnknown, match='NO_BLIND_REPLAY'):
        AsyncRequestArchive(journal)


def test_crash_before_submission_remains_unknown_and_cannot_replay(tmp_path):
    journal = Journal(tmp_path/'journal', 'one')
    q = AsyncRequestArchive(journal)
    q.reserve('reserved', dict(branch_upper_bound=2), max_bytes=100, timeout_s=1)
    with pytest.raises(MutationOutcomeUnknown):
        q.close(timeout_s=1)
    assert unknown_reservations(journal) == ['reserved']
    with pytest.raises(MutationOutcomeUnknown):
        AsyncRequestArchive(journal)


def test_source_mutation_during_archive_is_unknown_not_success(tmp_path):
    source, dest = archive_fixture(tmp_path, 'one')
    def changed(s, d):
        result = archive_request(s, d)
        (Path(s)/'data').write_bytes(b'mutated')
        return result
    q = AsyncRequestArchive(Journal(tmp_path/'journal', 'one'), archive=changed)
    ticket = q.reserve('one', {}, max_bytes=100, timeout_s=1)
    q.submit(ticket, source, dest)
    with pytest.raises(MutationOutcomeUnknown):
        q.close(timeout_s=2)
    assert not q.completed


def test_issued_pin_mismatch_before_seal_is_preserved_unknown(tmp_path):
    q = AsyncRequestArchive(Journal(tmp_path/'journal', 'one'))
    ticket = q.reserve('one', {}, max_bytes=100, timeout_s=1)
    with pytest.raises(ValueError, match='ISSUED'):
        q.submit(ticket, *archive_fixture(tmp_path, 'one'), expected_pins={'data':'0'*64})
    with pytest.raises(MutationOutcomeUnknown):
        q.close(timeout_s=1)


def test_queue_owner_and_completed_identity_are_not_reused(tmp_path):
    journal = Journal(tmp_path/'journal', 'one')
    q = AsyncRequestArchive(journal)
    with pytest.raises(BlockingIOError):
        AsyncRequestArchive(journal)
    ticket = q.reserve('one', {}, max_bytes=100, timeout_s=1)
    q.submit(ticket, *archive_fixture(tmp_path, 'one'))
    q.close(timeout_s=2)
    successor = AsyncRequestArchive(journal)
    with pytest.raises(MutationOutcomeUnknown, match='ALREADY_USED'):
        successor.reserve('one', {}, max_bytes=100, timeout_s=1)
    successor.close(timeout_s=1)


@pytest.mark.parametrize('crash_boundary', ['reserved', 'archive_entered'])
def test_real_process_crash_releases_owner_but_never_replays_unknown(tmp_path, crash_boundary):
    source, destination = archive_fixture(tmp_path, 'crash')
    program = '''
import os,sys,time
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.async_request_archive_v1 import AsyncRequestArchive
def crash(source,destination):os._exit(17)
q=AsyncRequestArchive(Journal(sys.argv[1],'crash'),archive=crash)
t=q.reserve('request',dict(branch_upper_bound=2),max_bytes=100,timeout_s=1)
if sys.argv[4]=='reserved':os._exit(17)
q.submit(t,sys.argv[2],sys.argv[3])
time.sleep(5)
'''
    proc = subprocess.run([sys.executable, '-c', program, str(tmp_path/'journal'),
                           str(source), str(destination), crash_boundary], timeout=8,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert proc.returncode == 17
    journal = Journal(tmp_path/'journal', 'crash')
    assert unknown_reservations(journal) == ['request']
    with pytest.raises(MutationOutcomeUnknown, match='NO_BLIND_REPLAY'):
        AsyncRequestArchive(journal)


def test_completed_archive_loss_is_unknown_at_next_owner_boundary(tmp_path):
    journal = Journal(tmp_path/'journal', 'one')
    q = AsyncRequestArchive(journal)
    ticket = q.reserve('one', {}, max_bytes=100, timeout_s=1)
    source, dest = archive_fixture(tmp_path, 'one')
    q.submit(ticket, source, dest)
    q.close(timeout_s=2)
    (dest/'data').write_bytes(b'corrupted after completion')
    with pytest.raises(MutationOutcomeUnknown):
        AsyncRequestArchive(journal)
