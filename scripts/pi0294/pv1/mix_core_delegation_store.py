"""Original onceIO storage for one explicit finite Core allocation.

This is a lower-level owner helper, not an admission or execution entry. The
registered owner must first validate its manifest, process, grant adoption,
leases and scientific gates. These files never represent a fresh GLOBAL read.
"""
import json
import re
from pathlib import Path

from scripts.pi0294.dec1 import source_cpu_once_io as once
from . import mix_accounting as a
from . import mix_core_delegation as contract

GRANT = 'CORE_FINITE_DELEGATION.json'
STATE = 'CORE_DELEGATED_EFFECTS.json'
PREFIX = 'DELEGATED-EFFECT-'


def need(ok, why):
    if not ok:
        raise PermissionError('CORE_DELEGATED_STORE_' + why)


def _paths(root, expected_grant_sha256):
    root = Path(root)
    need(root.is_absolute() and root.is_dir()
         and not any(p.is_symlink() for p in (root, *root.parents)),
         'EXACT_OWN_REGISTERED_ROOT_REQUIRED')
    need(type(expected_grant_sha256) is str
         and re.fullmatch('[0-9a-f]{64}', expected_grant_sha256), 'PINNED_GRANT')
    grant, state = root / 'evidence' / GRANT, root / 'evidence' / STATE
    gb, sb = once._file(grant), once._file(state)
    need(a.sha(gb) == expected_grant_sha256, 'IMMUTABLE_GRANT_BYTES')
    value = contract.validate_state(gb, sb)
    return state, gb, sb, value


def _operation(effect_id):
    need(type(effect_id) is str and 0 < len(effect_id) <= 160, 'BOUNDED_EFFECT_TOKEN')
    return PREFIX + a.sha(effect_id.encode())


def _confirmed_chain(path, state_bytes, state):
    """Check every durable debit; an orphan/partial intent stops new writes."""
    events = state['events']
    expected = set()
    for index, event in enumerate(events):
        operation = _operation(event['effect_id'])
        intent = path.with_name(operation + '.INTENT.json')
        receipt = path.with_name(operation + '.RECEIPT.json')
        expected.update((intent.name, receipt.name))
        ib, rb = once._file(intent), once._file(receipt)
        iv, rv = json.loads(ib), json.loads(rb)
        before = event['previous_state_sha256']
        after = events[index + 1]['previous_state_sha256'] if index + 1 < len(events) else a.sha(state_bytes)
        need(iv.get('schema') == 'DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1'
             and rv.get('schema') == 'DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
             and iv.get('target') == rv.get('target') == str(path.absolute())
             and iv.get('request_id') == rv.get('request_id') == operation
             and iv.get('before_sha256') == rv.get('before_sha256') == before
             and iv.get('proposed_sha256') == rv.get('after_sha256') == after
             and iv.get('temporary') == str(path.with_name(path.name + '.source-once-' + operation).absolute())
             and rv.get('intent_sha256') == a.sha(ib)
             and rv.get('status') == 'CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
             and iv.get('native_dispatch_admitted') is rv.get('native_dispatch_admitted') is False,
             'EVERY_DEBIT_HAS_ORIGINAL_ONCE_RECEIPT')
    present = {p.name for p in path.parent.glob(PREFIX + '*')}
    need(present == expected, 'UNKNOWN_OR_ORPHAN_INTENT_NO_RETRY_OR_NEXT_EFFECT')
    need(not any(path.parent.glob(path.name + '.source-once-' + PREFIX + '*')),
         'UNKNOWN_TEMPORARY_NO_NEW_EFFECT')


def read_confirmed(root, *, expected_grant_sha256):
    """Read-only reconciliation of local charges, never proof of a model call."""
    path, gb, sb, state = _paths(root, expected_grant_sha256)
    _confirmed_chain(path, sb, state)
    return dict(status='CONFIRMED_FINITE_CORE_LOCAL_CHARGES_ONLY',
                state=state, state_sha256=a.sha(sb), grant_sha256=a.sha(gb),
                canonical_GLOBAL_is_fresh=False, native_dispatch_admitted=False,
                automatic_retry_allowed=False, read_only=True)


def consume_once(root, *, expected_grant_sha256, effect_id, kind, count,
                 native_facts_bytes, now_epoch, stopped=False):
    """Persist before effect through the existing owner CAS. Never retry."""
    path, gb, sb, state = _paths(root, expected_grant_sha256)
    _confirmed_chain(path, sb, state)
    operation = _operation(effect_id)
    ip, rp = path.with_name(operation + '.INTENT.json'), path.with_name(operation + '.RECEIPT.json')
    if ip.exists() or rp.exists():
        raise FileExistsError('CORE_DELEGATED_STORE_EFFECT_ALREADY_ATTEMPTED_NO_REPLAY')
    proposed = contract.consume(gb, sb, effect_id=effect_id, kind=kind, count=count,
                                native_facts_bytes=native_facts_bytes, now_epoch=now_epoch,
                                stopped=stopped)
    receipt = once.commit(path, expected_before_sha256=a.sha(sb), proposed_bytes=a.raw(proposed),
                          request_id=operation, intent_path=ip, receipt_path=rp,
                          lock_path=path.with_name('CORE_DELEGATED_EFFECTS.owner.lock'))
    confirmed = read_confirmed(root, expected_grant_sha256=expected_grant_sha256)
    need(confirmed['state_sha256'] == receipt['after_sha256'], 'EXACT_COMMITTED_STATE')
    return dict(confirmed, receipt=receipt, read_only=False)
