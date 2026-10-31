"""A fresh finite source job is not a reset of the old night pool/clock."""
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.pi0294.multitask import night_scope as n, night_source_stage as stage
from scripts.pi0294.test_night_source_v7_carry import closed, policy, raw


def profile(now):
    return dict(policy_sha256=n.V7_POLICY,created_epoch=now,stop_new_epoch=now+300,
                deadline_epoch=now+1200,upper=dict(n.FINITE_UPPER))


def future():
    old,lease,terminal=closed();old['charged_upper']=dict(n.CAPS)
    now=n.SUCCESSOR_DEADLINE+100
    p=profile(now)
    new=n.development_account(raw(old),raw(lease),raw(terminal),policy(),
        destination='/owned/finite-seed13',now=now,job=p)
    return old,new,p,now


def test_all_old_costs_caps_operations_and_clocks_remain_in_the_closed_predecessor():
    old,new,p,now=future()
    assert new['charged_upper']==old['charged_upper']==n.CAPS
    assert new['operations']==old['operations']and new['caps']==old['caps']
    assert old['deadline_epoch']==n.DEADLINE
    assert new['deadline_epoch']==p['deadline_epoch']
    assert n.account_window(new)==(p['stop_new_epoch'],p['deadline_epoch'])
    with pytest.raises(PermissionError):
        n.development_account(raw(old),raw(closed()[1]),raw(closed()[2]),policy(),
            destination='/owned/late-old-night',now=now)
    new.update(native_adopted=True,custody='ONE_NATIVE_CORE_SOURCE_OWNER',qualification_sha256='a'*64)
    upper=dict(n.FINITE_UPPER);upper['Core_CPU_s']=2800
    after=n.reserve(new,operation='FINITE_SEED13',kind='ATTEMPT',upper=upper,now=now+1,
                    task='LIBERO_BOOK_LEFT',qualification='a'*64)
    assert after['charged_upper']['attempts']==13
    assert after['charged_upper']['physics_integrations']==650000
    audit=n.reserve(after,operation='AUDIT13',kind='PREPARATION',upper={'Core_CPU_s':40},now=now+2)
    assert audit['charged_upper']['Core_CPU_s']==old['charged_upper']['Core_CPU_s']+2840
    with pytest.raises(PermissionError):
        n.reserve(after,operation='SECOND_SAME_JOB',kind='ATTEMPT',upper=upper,now=now+3,
                  task='LIBERO_BOOK_LEFT',qualification='a'*64)
    with pytest.raises(PermissionError):
        n.reserve(audit,operation='OVER_CPU',kind='PREPARATION',upper={'Core_CPU_s':1},now=now+3)


@pytest.mark.parametrize('fault',['wide_start','wide_end','loose_policy','bigger_physics','typed_attempt','refund','no_carry'])
def test_finite_authority_does_not_remove_job_bounds_or_history(fault):
    old,new,p,now=future()
    if fault=='wide_start':new[n.V7_FINITE]['stop_new_epoch']+=1
    elif fault=='wide_end':new[n.V7_FINITE]['deadline_epoch']+=1
    elif fault=='loose_policy':new[n.V7_FINITE]['policy_sha256']='0'*64
    elif fault=='bigger_physics':new[n.V7_FINITE]['upper']['physics_integrations']+=1
    elif fault=='typed_attempt':new[n.V7_FINITE]['upper']['attempts']=True
    elif fault=='refund':new['charged_upper']['attempts']-=1
    else:new.pop(n.V7_CARRY)
    with pytest.raises(PermissionError):n.validate_account(new)


def test_runtime_uses_sealed_policy_and_finite_clock_without_renewing_old_authority(tmp_path):
    _,new,p,now=future()
    approval=(Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/NIGHT_BOUNDED_AUTONOMY_DIRECT_AUTHORITY_20261030.json').read_bytes()
    with pytest.raises(PermissionError):n.authority(approval,now=now)
    (tmp_path/'SOURCE_V7_EXECUTION_POLICY.md').write_bytes(policy())
    (tmp_path/'SOURCE_V7_FINITE_JOB.json').write_bytes(raw(p))
    runtime=n.for_root(tmp_path);runtime.authority(approval,now=now+1)
    with pytest.raises(PermissionError):runtime.authority(approval,now=p['deadline_epoch'])
    with pytest.raises(PermissionError):n.authority(approval,now=now,job=p,v7_policy_body=policy()+b' ')
    assert runtime.DEADLINE==p['deadline_epoch']and runtime.STOP_NEW==p['stop_new_epoch']
    assert 'SOURCE_V7_EXECUTION_POLICY.md' in stage.PROGRAM and 'SOURCE_V7_FINITE_JOB.json' in stage.PROGRAM
    compile(stage.PROGRAM,'v7-finite-cold-source','exec')
