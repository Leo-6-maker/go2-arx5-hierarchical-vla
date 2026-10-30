"""Pure continuation fixtures; native actors/custody are not admitted here."""
from copy import deepcopy
import hashlib,json
from pathlib import Path

import pytest
from scripts.pi0294.multitask import night_scope as n

raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
policy=lambda:(Path(__file__).resolve().parents[2]/'docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md').read_bytes()


def closed():
    old=dict(scope=n.SCOPE,caps=dict(n.CAPS),approval_sha256=n.APPROVAL_SHA,
        stop_new_epoch=n.STOP_NEW,deadline_epoch=n.DEADLINE,revision=1,native_adopted=True,
        native_destination='/owned/closed-source',custody='ONE_NATIVE_CORE_SOURCE_OWNER',
        charged_upper={k:0 for k in n.CAPS},operations=[dict(operation='OLD_PAID_ATTEMPT',kind='ATTEMPT')])
    old['charged_upper'].update(attempts=1,physics_integrations=50000,RGB_pairs=2000,Core_CPU_s=2800)
    lease=dict(run_uuid='closed-source',status='CLOSED',all_children_terminal=True,approval_sha256=n.APPROVAL_SHA)
    terminal=dict(status='FINITE_SOURCE_CHAIN_TERMINAL',closed_lease_sha256=hashlib.sha256(raw(lease)).hexdigest())
    return old,lease,terminal


def develop(old=None,lease=None,terminal=None):
    a,l,t=closed()
    return n.development_account(raw(a if old is None else old),raw(l if lease is None else lease),
        raw(t if terminal is None else terminal),policy(),destination='/owned/new-v7-book',now=n.STOP_NEW-1)


def test_v7_reuses_one_closed_counter_and_original_clock_without_a_fresh_pool():
    old,lease,terminal=closed();new=develop()
    assert new['caps']==old['caps']and new['charged_upper']==old['charged_upper']
    assert new['operations']==old['operations']and new['deadline_epoch']==old['deadline_epoch']
    assert new['native_adopted']is False
    new.update(native_adopted=True,qualification_sha256='a'*64,custody='ONE_NATIVE_CORE_SOURCE_OWNER')
    later=n.reserve(new,operation='LIBERO_LEFT_PORT03_TOP_EDGE_ROUTE',kind='ATTEMPT',task='LIBERO_BOOK_LEFT',
        upper=dict(attempts=1,physics_integrations=50000,RGB_pairs=2000,Core_CPU_s=2800,render_device_s=600),
        now=n.STOP_NEW-1,qualification='a'*64)
    assert later['charged_upper']['attempts']==2 and later['charged_upper']['physics_integrations']==100000
    assert old==closed()[0]


@pytest.mark.parametrize('fault',['active_lease','foreign_lease','unclosed_chain','refunded_cost','changed_history','new_pool','new_clock'])
def test_v7_carry_keeps_cold_identity_cost_and_window_fences(fault):
    old,lease,terminal=closed()
    if fault=='active_lease':lease['status']='ACTIVE'
    if fault=='foreign_lease':lease['run_uuid']='another-owner'
    if fault=='unclosed_chain':terminal['closed_lease_sha256']='f'*64
    if fault in ('active_lease','foreign_lease','unclosed_chain'):
        with pytest.raises(PermissionError):develop(old,lease,terminal)
        return
    new=develop()
    if fault=='refunded_cost':new['charged_upper']['Core_CPU_s']-=1
    if fault=='changed_history':new['operations']=[]
    if fault=='new_pool':new['caps']['attempts']+=12
    if fault=='new_clock':new['deadline_epoch']+=3600
    with pytest.raises(PermissionError):n.validate_account(new)


def test_custody_transfer_requires_exact_preimage_and_leaves_historical_costs_untouched():
    old,lease,terminal=closed();new=develop()
    transferred=n.transfer_development_custody(raw(old),raw(new),destination='/owned/new-v7-book')
    assert transferred['charged_upper']==old['charged_upper']and transferred['operations']==old['operations']
    assert transferred['custody']=='CLOSED_CARRIED_TO_ONE_V7_SOURCE_DEVELOPMENT'
    with pytest.raises(PermissionError):
        n.transfer_development_custody(raw(transferred),raw(new),destination='/owned/new-v7-book')
    with pytest.raises(PermissionError):
        n.development_account(raw(old),raw(lease),raw(terminal),policy()+b' ',
                              destination='/owned/new-v7-book',now=n.STOP_NEW-1)
