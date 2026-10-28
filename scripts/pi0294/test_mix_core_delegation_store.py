"""Actual temporary onceIO files; CPU process facts remain explicit fixtures."""
import json
from pathlib import Path

import pytest

from scripts.pi0294.test_v5_core_full_admission import full, captured_global
from scripts.pi0294.test_mix_core_delegation import activated
from scripts.pi0294.pv1 import mix_accounting as a
from scripts.pi0294.pv1 import mix_core_delegation_store as store


@pytest.fixture
def local_store(full, tmp_path):
    v = activated(full, tmp_path)
    root = tmp_path / 'registered-core-child'
    evidence = root / 'evidence'; evidence.mkdir(parents=True)
    (evidence / store.GRANT).write_bytes(v['grant_bytes'])
    (evidence / store.STATE).write_bytes(v['state_bytes'])
    return root, v


def attempt(root, v, kind='model_loads', count=1, *, sequence=None):
    state = json.loads((root / 'evidence' / store.STATE).read_bytes())
    p = json.loads(v['inputs']['native_facts_bytes']); p['observed_epoch'] = 1207.
    token = '{}:{}:{}:DEBIT'.format(p['pid'], p['proc_start'],
                                   state['sequence'] if sequence is None else sequence)
    return store.consume_once(root, expected_grant_sha256=a.sha(v['grant_bytes']),
        effect_id=token, kind=kind, count=count,
        native_facts_bytes=a.raw(p), now_epoch=1207.)


def test_confirmed_local_debits_have_original_once_receipts_and_no_GLOBAL_write(local_store):
    root, v = local_store
    before = v['global_bytes']
    one = attempt(root, v)
    two = attempt(root, v, 'generation_queries')
    assert one['state']['sequence'] == 1 and two['state']['sequence'] == 2
    assert two['state']['row']['charged_query_attempts'] == 1
    assert not two['canonical_GLOBAL_is_fresh'] and not two['native_dispatch_admitted']
    assert v['global_bytes'] == before
    assert len(list((root / 'evidence').glob('DELEGATED-EFFECT-*.RECEIPT.json'))) == 2
    with pytest.raises(FileExistsError, match='NO_REPLAY'):
        attempt(root, v, sequence=0)


def test_lost_reply_after_receipt_is_readonly_reconciled_without_replaying(local_store, monkeypatch):
    root, v = local_store; original = store.once.commit
    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError('SIMULATED_REPLY_LOSS_AFTER_DURABLE_RECEIPT')
    monkeypatch.setattr(store.once, 'commit', lost)
    with pytest.raises(TimeoutError):
        attempt(root, v)
    read = store.read_confirmed(root, expected_grant_sha256=a.sha(v['grant_bytes']))
    assert read['state']['sequence'] == 1 and not read['automatic_retry_allowed']
    with pytest.raises(FileExistsError):
        attempt(root, v, sequence=0)


def test_applied_state_with_missing_receipt_blocks_the_next_token(local_store, monkeypatch):
    root, v = local_store; original = store.once._exclusive
    def missing_receipt(path, raw, mode=0o400):
        if str(path).endswith('.RECEIPT.json'):
            raise OSError('SIMULATED_RECEIPT_STORAGE_FAILURE')
        return original(path, raw, mode)
    monkeypatch.setattr(store.once, '_exclusive', missing_receipt)
    with pytest.raises(OSError):
        attempt(root, v)
    assert json.loads((root / 'evidence' / store.STATE).read_bytes())['sequence'] == 1
    with pytest.raises(FileNotFoundError):
        attempt(root, v, 'generation_queries')
    assert len(list((root / 'evidence').glob('DELEGATED-EFFECT-*.INTENT.json'))) == 1


def test_unknown_intent_before_state_change_blocks_new_effect(local_store):
    root, v = local_store
    (root / 'evidence' / ('DELEGATED-EFFECT-' + 'a' * 64 + '.INTENT.json')).write_text('{}')
    with pytest.raises(PermissionError, match='UNKNOWN_OR_ORPHAN'):
        attempt(root, v)
    assert json.loads((root / 'evidence' / store.STATE).read_bytes())['sequence'] == 0


def test_receipt_tampering_and_changed_grant_are_rejected(local_store):
    root, v = local_store; attempt(root, v)
    rp = next((root / 'evidence').glob('DELEGATED-EFFECT-*.RECEIPT.json'))
    value = json.loads(rp.read_bytes()); value['after_sha256'] = '0' * 64
    rp.chmod(0o600); rp.write_bytes(a.raw(value))
    with pytest.raises(PermissionError, match='EVERY_DEBIT'):
        attempt(root, v, 'generation_queries')
    with pytest.raises(PermissionError, match='IMMUTABLE_GRANT'):
        store.read_confirmed(root, expected_grant_sha256='0' * 64)


def test_symlinked_state_is_rejected_before_any_write(local_store, tmp_path):
    root, v = local_store; path = root / 'evidence' / store.STATE
    other = tmp_path / 'other-state.json'; path.rename(other); path.symlink_to(other)
    with pytest.raises(PermissionError, match='SYMLINK'):
        attempt(root, v)
    assert not list((root / 'evidence').glob('DELEGATED-EFFECT-*'))
