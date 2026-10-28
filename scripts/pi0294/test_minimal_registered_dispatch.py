"""Captured real pre-qsub CAS/OS/queue evidence; no native dispatch in tests."""
import base64
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

import pytest
from scripts.pi0294.generalization import minimal_registered_dispatch as d


@pytest.fixture
def captured():
    name = os.environ.get('ARX5_REGISTERED_DISPATCH_CAPTURE')
    if not name:
        pytest.skip('actual registered pre-qsub full-byte capture required')
    c = json.loads(Path(name).read_bytes())
    gb = gzip.decompress(base64.b64decode(c['GLOBAL_gzip_base64']))
    actual = c['actual']
    assert d.digest(gb) == actual['GLOBAL_sha256']
    return gb, base64.b64decode(actual['request_base64']), actual


def test_actual_register_only_is_explained_without_a_second_payment_or_fake_job(captured):
    gb, rb, c = captured
    before = json.loads(gb)
    out = d.predispatch_queue(gb, rb, c['evidence'], c['full_queue'], now_epoch=c['observed_epoch'])
    assert out['reconciliation']['old_UNKNOWN_preserved'] is True
    assert out['reconciliation']['registration_repeated'] is False
    assert out['reconciliation']['submission_repeated'] is False
    assert out['reconciliation']['native_permission'] is False
    assert out['reconciliation']['old_owner_CPU_upper_s']+out['reconciliation']['new_owner_CPU_upper_s'] == 3000
    assert out['full_queue'] == c['full_queue'] and out['owned'] == []
    assert out['original_classification'] == [dict(job_id=None, reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF')]
    assert json.loads(gb) == before


@pytest.mark.parametrize('bad', ['qsub_intent', 'qsub_script', 'bound', 'release', 'effects',
                                'live_owner', 'live_wrapper', 'different_phase', 'journal_cleared',
                                'missing_exit', 'wrong_CAS', 'CAS_temp', 'expired', 'CPU_bank'])
def test_no_dispatch_from_incomplete_or_ambiguous_reconciliation(captured, bad):
    gb, rb, c = captured
    evidence = deepcopy(c['evidence'])
    if bad == 'qsub_intent': evidence['old_absent']['SOURCE_QSUB_INTENT.private.json'] = False
    elif bad == 'qsub_script': evidence['allocation_files'].append('job.pbs')
    elif bad == 'bound': evidence['old_absent']['SOURCE_JOB_BOUND.private.json'] = False
    elif bad == 'release': evidence['initial_release_exists'] = True
    elif bad == 'effects': evidence['effect_ledger_exists'] = True
    elif bad.startswith('live_'): evidence['old_processes'][bad[5:]]['same_live_procstart'] = True
    elif bad == 'different_phase': evidence['old_journal']['intent']['phase'] = 'QSUB'
    elif bad == 'journal_cleared': evidence['old_journal']['status'] = 'CONFIRMED'
    elif bad == 'missing_exit': evidence['old_exit']['foreground_wait_completed'] = False
    elif bad == 'wrong_CAS':
        value = json.loads(base64.b64decode(evidence['native_once']['receipt_base64']))
        value['after_sha256'] = '0'*64
        evidence['native_once']['receipt_base64'] = base64.b64encode(d.raw(value)).decode()
    elif bad == 'CAS_temp': evidence['native_once']['temporary_exists'] = True
    elif bad == 'expired': evidence['observed_epoch'] -= 121
    elif bad == 'CPU_bank': evidence['old_launch_epoch'] -= 200
    with pytest.raises((PermissionError, KeyError, ValueError)):
        d.reconcile(gb, rb, evidence, now_epoch=c['observed_epoch'])


def test_changed_scientific_request_never_relabels_the_existing_reservation(captured):
    gb, rb, c = captured
    request = json.loads(rb)
    request['submission_nonce'] = 'ffffffff-ffff-4fff-8fff-ffffffffffff'
    with pytest.raises((PermissionError, KeyError, ValueError)):
        d.reconcile(gb, d.raw(request), c['evidence'], now_epoch=c['observed_epoch'])


def test_no_original_generic_queue_gate_or_scientific_owner_method_replaced():
    from scripts.pi0294.generalization.minimal_native_owner import Owner
    assert d.Owner.release is Owner.release
    assert d.Owner.release_science_session is Owner.release_science_session
    assert d.Owner.native_material is Owner.native_material
    assert d.Owner.finish is Owner.finish
    assert d.Owner.registration is Owner.registration
    assert d.OLD_CPU_UPPER == 800


def test_actual_temporary_once_CAS_lost_reply_is_read_only_and_never_reissued(tmp_path):
    from scripts.pi0294.dec1.source_cpu_once_io import commit
    from scripts.pi0294.dec1.owner_transport import Journal, UnknownMutation
    path = tmp_path/'GLOBAL.json'
    before, after = d.raw({'CPU_ONLY_FIXTURE': 1}), d.raw({'CPU_ONLY_FIXTURE': 2})
    path.write_bytes(before)
    key = 'V5_CD_CPU_BIND'
    jroot = tmp_path/'journal'; jroot.mkdir()
    journal = Journal(jroot, dict(run_uuid='cpu-bind', job_id='CPU_ONLY',
                                  source_sha='f'*40, request_sha256='a'*64))
    calls = []
    def lost_reply(intent, limit):
        calls.append(intent['operation_id'])
        commit(path, expected_before_sha256=d.digest(before), proposed_bytes=after,
               request_id='CPU_BIND', intent_path=tmp_path/(key+'_INTENT.json'),
               receipt_path=tmp_path/(key+'_RECEIPT.json'), lock_path=tmp_path/'GLOBAL.lock')
        raise TimeoutError('CPU_LOST_REPLY_AFTER_REAL_TEMP_CAS')
    with pytest.raises(UnknownMutation):journal.call('BIND', 'MUTATION', lost_reply)
    original = journal.state_path.read_bytes()
    proof = d.exact_current_cas(path, before_sha256=d.digest(before), after_sha256=d.digest(after))
    assert proof['read_only'] is True and proof['write_repeated'] is False
    assert journal.state_path.read_bytes() == original
    with pytest.raises(UnknownMutation):journal.call('BIND', 'MUTATION', lost_reply)
    assert len(calls) == 1
    with pytest.raises(PermissionError):d.exact_current_cas(path, before_sha256='0'*64, after_sha256=d.digest(after))
