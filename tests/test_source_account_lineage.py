"""History stays lossless; current grants and costs never inherit cache trust."""
from copy import deepcopy
import hashlib
import pytest
from scripts.pi0294.multitask import night_scope as n
from scripts.pi0294.test_night_source_v7_carry import closed,policy,raw
from scripts.pi0294.test_night_source_v7_finite import profile


def test_finite_descendant_keeps_closed_history_once_losslessly_and_retains_costs():
    parent,lease,terminal=closed();now=n.SUCCESSOR_DEADLINE+100
    first=n.development_account(raw(parent),raw(lease),raw(terminal),policy(),destination='/owned/first',now=now,job=profile(now))
    first.update(native_adopted=True,custody='ONE_NATIVE_CORE_SOURCE_OWNER')
    # A historical payload carried on an already valid V7 node is evidence,
    # never a new grant. The next node keeps its complete exact bytes once.
    first['closed_book_unrun_cabinet_carry']={'historical_evidence':'abc'*10000}
    n.validate_account(first)
    lease=dict(lease,run_uuid='first');terminal=dict(terminal,closed_lease_sha256=hashlib.sha256(raw(lease)).hexdigest())
    next=n.development_account(raw(first),raw(lease),raw(terminal),policy(),destination='/owned/next',now=now+1,job=profile(now+1))
    assert 'closed_book_unrun_cabinet_carry'not in next
    assert n.closed_carry_bytes(next[n.V7_CARRY],'account')==raw(first)
    assert next['charged_upper']==first['charged_upper']and next['operations']==first['operations']
    assert n.validate_account(next)==next
    for key in('caps','approval_sha256','operations'):
        bad=deepcopy(next);bad.pop(key)
        with pytest.raises((PermissionError,KeyError)):n.validate_account(bad)


def test_bounded_hash_cache_revalidates_mutated_dictionary_and_repeated_invalid_values():
    value,_,_=closed();n.validate_account(value)
    value['caps']['attempts']+=1
    for _ in range(2):
        with pytest.raises(PermissionError):n.validate_account(value)
    for revision in range(80):
        fresh,_,_=closed();fresh['revision']=revision;n.validate_account(fresh)
    assert len(n._VALIDATED_ACCOUNT_IDENTITIES)<=64
