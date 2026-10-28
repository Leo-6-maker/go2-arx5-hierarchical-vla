"""Existing-ledger callback fixtures; no physics, admission or remote calls."""
from contextlib import contextmanager
import threading
import pytest
from scripts.pi0294.dec1.source_component_meter import SourceComponentMeters,SourceComponentUnknown

class Ledger:
    def __init__(self):self.reserved={};self.used=0;self.unknown=[];self.events=[]
    def guard(self):self.events.append('guard')
    @contextmanager
    def steps(self,kind,maximum,identifier):
        assert kind=='shadow_integrations';self.reserved[identifier]=maximum;m=dict(actual=0)
        try:yield m
        except BaseException:
            self.unknown.append(identifier);raise
        else:self.used+=m['actual'];del self.reserved[identifier]

def fixture(record=None):
    l=Ledger();out=dict(actual=0);events=[]
    c=SourceComponentMeters(l,episode_id='CPU',outer_live_meter=out,record=record or events.append)
    return c,l,out,events

@pytest.mark.parametrize('count,status',[(200,'PASS'),(37,'S1_REJECT'),(0,'REFERENCE_REJECT')])
def test_known_shadow_partial_closes_actual_count_before_reject_propagation(count,status):
    c,l,out,events=fixture();t=c.reserve('shadow',200,1);c.charge('shadow',count,False,{})
    c.finish('shadow',t,count,False,status)
    assert l.used==count and not l.reserved and not l.unknown and not c.unknown
    assert c.reconcile_outer(0)==0


def test_live_prefix_has_no_new_ledger_context_or_double_debit():
    c,l,out,events=fixture();t=c.reserve('live',200,0);c.charge('live',200,False,{})
    c.finish('live',t,200,False,'PASS');assert l.used==0 and not l.reserved
    assert c.reconcile_outer(200)==200 and out['actual']==200

@pytest.mark.parametrize('kind',['shadow','live'])
def test_unknown_keeps_original_bound_and_reconciled_live_lower_bound(kind):
    c,l,out,events=fixture();t=c.reserve(kind,200,int(kind=='shadow'));c.charge(kind,7,True,{})
    c.finish(kind,t,7,True,'ENGINEERING_MISSING')
    if kind=='shadow':assert l.used==0 and l.reserved[t]==200 and l.unknown==[t]
    with pytest.raises(SourceComponentUnknown):c.reconcile_outer(3)
    assert out['actual']==3 and out['component_unknown']
    with pytest.raises(PermissionError):c.reserve('shadow',200,1)


def test_unknown_shadow_reserve_callback_never_enters_twice_or_refunds():
    def bad(event):raise KeyboardInterrupt('INTENT_WRITE_UNKNOWN')
    c,l,out,events=fixture(bad)
    with pytest.raises(KeyboardInterrupt):c.reserve('shadow',200,1)
    assert not l.reserved and c.unknown and out['component_unknown']
    with pytest.raises(PermissionError):c.reserve('shadow',200,1)


def test_charge_callback_interruption_keeps_actual_count_and_unknown_bound():
    c,l,out,events=fixture();t=c.reserve('shadow',200,1)
    def bad(event):raise KeyboardInterrupt('EFFECT_RECORD_UNKNOWN')
    c.record=bad
    with pytest.raises(KeyboardInterrupt):c.charge('shadow',1,False,{})
    assert c.entries[t]['known']==1 and c.unknown
    c.record=events.append;c.finish('shadow',t,1,True,'ENGINEERING_MISSING')
    assert l.reserved[t]==200 and l.used==0


def test_finish_reply_interruption_is_not_retried_or_redebit():
    c,l,out,events=fixture();t=c.reserve('shadow',200,1);c.charge('shadow',37,False,{})
    def bad(event):raise KeyboardInterrupt('FINISH_RECORD_UNKNOWN')
    c.record=bad
    with pytest.raises(KeyboardInterrupt):c.finish('shadow',t,37,False,'S1_REJECT')
    assert l.used==37 and not l.reserved and c.unknown
    with pytest.raises(SourceComponentUnknown):c.finish('shadow',t,37,False,'S1_REJECT')
    assert l.used==37

@pytest.mark.parametrize('bad',[True,-1,201,None])
def test_wrong_finish_counts_cannot_refund_real_shadow_effects(bad):
    c,l,out,events=fixture();t=c.reserve('shadow',200,1);c.charge('shadow',7,False,{})
    with pytest.raises(SourceComponentUnknown):c.finish('shadow',t,bad,False,'PASS')
    assert l.reserved[t]==200 and l.used==0 and c.unknown


def test_active_context_cannot_be_bypassed_by_outer_return():
    c,l,out,events=fixture();c.reserve('shadow',200,1)
    with pytest.raises(SourceComponentUnknown):c.reconcile_outer(4)
    assert out['actual']==4 and l.reserved


def test_second_prefix_refused_until_first_finish_and_original_clone_counter_not_touched():
    c,l,out,events=fixture();t=c.reserve('shadow',200,1)
    with pytest.raises(PermissionError):c.reserve('live',200,0)
    c.finish('shadow',t,0,False,'REFERENCE_REJECT')
    assert not any(v=='constructor' for v in l.events)


def test_foreign_thread_cannot_charge_or_finish():
    c,l,out,events=fixture();t=c.reserve('shadow',200,1);errors=[]
    def invoke():
        try:c.charge('shadow',1,False,{})
        except BaseException as e:errors.append(e)
    th=threading.Thread(target=invoke);th.start();th.join()
    assert len(errors)==1 and isinstance(errors[0],SourceComponentUnknown)
    assert c.entries[t]['known']==0 and l.reserved[t]==200
    assert c.unknown and out['component_unknown']


def test_real_existing_ledger_settles_known_partial_reject_without_live_double_charge(tmp_path):
    import json
    from scripts.pi0294.test_pv1_core_ledger import fixture as real_ledger
    root,l=real_ledger(tmp_path);events=[]
    with l.steps('live_and_recovery_integrations',16000,'outer') as live:
        c=SourceComponentMeters(l,episode_id='CPU',outer_live_meter=live,record=events.append)
        t=c.reserve('shadow',200,1);c.charge('shadow',37,False,{})
        c.finish('shadow',t,37,False,'S1_REJECT');c.reconcile_outer(0)
    v=json.loads(l.path.read_text())
    assert v['shadow_integrations']==37 and v['live_and_recovery_integrations']==0
    assert v['reservations']=={} and v['unknown_effects']==[]


def test_real_existing_ledger_retains_both_bounds_on_unknown_shadow_and_outer(tmp_path):
    import json
    from scripts.pi0294.test_pv1_core_ledger import fixture as real_ledger
    root,l=real_ledger(tmp_path);events=[]
    with pytest.raises(SourceComponentUnknown):
        with l.steps('live_and_recovery_integrations',16000,'outer') as live:
            c=SourceComponentMeters(l,episode_id='CPU',outer_live_meter=live,record=events.append)
            t=c.reserve('shadow',200,1);c.charge('shadow',7,True,{})
            c.finish('shadow',t,7,True,'ENGINEERING_MISSING');c.reconcile_outer(3)
    v=json.loads(l.path.read_text())
    assert v['shadow_integrations']==v['live_and_recovery_integrations']==0
    assert {p['maximum'] for p in v['reservations'].values()}=={16000,200}
    assert len(v['unknown_effects'])==2 and live['actual']==3
    assert any(e.get('known_ticks')==7 for e in events)
    with pytest.raises(PermissionError):l.charge('engine_starts',1)
