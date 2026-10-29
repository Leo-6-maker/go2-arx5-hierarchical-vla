"""Shared aggregate caps, distinct old resources and the actual night cutoff."""
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.multitask import night_scope as n


def account():
    return dict(scope=n.SCOPE,caps=dict(n.CAPS),approval_sha256=n.APPROVAL_SHA,
        stop_new_epoch=n.STOP_NEW,deadline_epoch=n.DEADLINE,
        charged_upper={k:0 for k in n.CAPS},operations=[],native_adopted=False,revision=0)


def attempt(v,number,*,now=None,task='LIBERO_BOOK_LEFT'):
    return n.reserve(v,operation='attempt-'+str(number),kind='ATTEMPT',task=task,
        upper=dict(attempts=1,physics_integrations=50000,RGB_pairs=2000,
            Core_CPU_s=3200,render_device_s=600),qualification='a'*64,
        now=n.STOP_NEW-1 if now is None else now)


def test_real_direct_approval_and_preparation_do_not_authorize_physics():
    body=(Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/NIGHT_BOUNDED_AUTONOMY_DIRECT_AUTHORITY_20261030.json').read_bytes()
    n.authority(body,now=n.STOP_NEW-1)
    with pytest.raises(PermissionError):n.authority(body+b' ',now=n.STOP_NEW-1)
    with pytest.raises(PermissionError):n.authority(body,now=n.DEADLINE)
    v=account()
    with pytest.raises(PermissionError):attempt(v,0)
    with pytest.raises(PermissionError):n.reserve(v,operation='hidden-step',kind='PREPARATION',upper={'physics_integrations':1},now=n.STOP_NEW-1)


def test_all_tasks_share_exactly_twelve_attempts_and_all_counters():
    v=account();v.update(native_adopted=True,qualification_sha256='a'*64)
    for number in range(12):v=attempt(v,number,task=n.TASKS[number%3])
    assert v['charged_upper']['attempts']==12
    assert v['charged_upper']['physics_integrations']==600000
    assert v['charged_upper']['RGB_pairs']==24000
    assert v['charged_upper']['render_device_s']==7200
    with pytest.raises(PermissionError):attempt(v,12)
    assert account()['charged_upper']['attempts']==0


def test_no_new_starts_after08_and_all_effects_finish_by09():
    v=account();v.update(native_adopted=True,qualification_sha256='a'*64)
    with pytest.raises(PermissionError):attempt(v,0,now=n.STOP_NEW)
    v=n.reserve(v,operation='finish',kind='CLOSEOUT',upper={'Core_CPU_s':10},now=n.DEADLINE-1)
    with pytest.raises(PermissionError):n.reserve(v,operation='late',kind='CLOSEOUT',upper={'Core_CPU_s':1},now=n.DEADLINE)


def test_duplicate_operation_and_unapproved_task_cannot_refresh_limits():
    v=account();v.update(native_adopted=True,qualification_sha256='a'*64)
    with pytest.raises(PermissionError):attempt(v,0,task='S1_T2')
    v=attempt(v,0)
    with pytest.raises(PermissionError):attempt(v,0)
    with pytest.raises(PermissionError):n.reserve(v,operation='borrow',kind='PREPARATION',upper={'GPU_training_s':1},now=n.STOP_NEW-1)


def test_old_shortfall_is_never_filled_by_new_named_allowance():
    from scripts.pi0294.pv1.future_budget import _vector
    original=dict(dimensions={k:dict(cap=1000000000000,projected=5)for k in _vector({})},cap_changes_required={})
    original['dimensions']['PBS_submissions']=dict(cap=None,projected=41,count_still_accounted=True)
    result=n.add_to_original(original)
    assert result['old_work_may_spend_new_allowance']is False
    assert result['dimensions']['physics_ticks']['new_named_source_allowance']==600000
    assert result['dimensions']['H200_s']['new_named_source_allowance']==0
    assert result['dimensions']['PBS_submissions']['combined_projected']==41
    bad=deepcopy(original);bad['cap_changes_required']={'Core_CPU_s':1}
    with pytest.raises(PermissionError):n.add_to_original(bad)
    bad=deepcopy(original);bad['dimensions']['Core_CPU_s']['cap']=4
    with pytest.raises(PermissionError):n.add_to_original(bad)


def test_cpu_and_storage_are_shared_caps_not_per_node_or_per_task():
    v=account();v=n.reserve(v,operation='Core-prep',kind='PREPARATION',upper={'Core_CPU_s':41400},now=n.STOP_NEW-1)
    v=n.reserve(v,operation='PC-prep',kind='PREPARATION',upper={'PC_CPU_s':1800},now=n.STOP_NEW-1)
    assert sum(v['charged_upper'][k]for k in ('Core_CPU_s','PC_CPU_s'))==43200
    for key in ['Core_CPU_s','PC_CPU_s','Core_peak_bytes','PC_evidence_bytes','transfer_bytes']:
        bad=deepcopy(v);bad['charged_upper'][key]=n.CAPS[key]
        with pytest.raises(PermissionError):n.reserve(bad,operation='over-'+key,kind='PREPARATION',upper={key:1},now=n.STOP_NEW-1)
